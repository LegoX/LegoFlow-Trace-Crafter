from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_REPOS_FILE = REPO_ROOT / "artifacts" / "excluded_repos.txt"

PANGUML_VERSION = "2.0.0"
DEFAULT_TOKENIZER_NAME = "Qwen/Qwen3-8B"
DEFAULT_SYSTEM_SOURCE_MODEL = "GLM-5-FP8"
DEFAULT_SYSTEM_TARGET_MODEL = "Qwen3-8B"
DEFAULT_TOKEN_BATCH_SIZE = 64
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_PANGUML_TOP_LEVEL_KEYS = frozenset({
    "version",
    "meta_info",
    "tools",
    "messages",
    "pseudo_turns",
    "think_mode",
    "_instance_id",
    "_agent_type",
    "_score",
})
_PANGUML_COMPAT_UNIQUE_INFO_KEYS = (
    "_instance_id",
    "_agent_type",
    "_score",
    "_gen_params",
    "_usage",
)


@dataclass(frozen=True)
class ModelProcessingConfig:
    """Shared model/tokenizer settings used by conversion utilities."""

    tokenizer_name: str = DEFAULT_TOKENIZER_NAME
    system_source_model: str = DEFAULT_SYSTEM_SOURCE_MODEL
    system_target_model: str = DEFAULT_SYSTEM_TARGET_MODEL
    token_batch_size: int = DEFAULT_TOKEN_BATCH_SIZE


DEFAULT_MODEL_PROCESSING_CONFIG = ModelProcessingConfig()


def _resolve_model_processing_config(
    model_config: ModelProcessingConfig | None = None,
    *,
    tokenizer_name: str | None = None,
    source_model: str | None = None,
    target_model: str | None = None,
    token_batch_size: int | None = None,
) -> ModelProcessingConfig:
    base = model_config or DEFAULT_MODEL_PROCESSING_CONFIG
    return ModelProcessingConfig(
        tokenizer_name=tokenizer_name
        if tokenizer_name is not None
        else base.tokenizer_name,
        system_source_model=source_model
        if source_model is not None
        else base.system_source_model,
        system_target_model=target_model
        if target_model is not None
        else base.system_target_model,
        token_batch_size=token_batch_size
        if token_batch_size is not None
        else base.token_batch_size,
    )


def is_im_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    return all(
        isinstance(message, dict) and isinstance(message.get("role"), str)
        for message in messages
    )


def infer_think_mode_from_messages(messages: list[dict[str, Any]]) -> str:
    has_reasoning = any(
        message.get("role") == "assistant" and bool(message.get("reasoning_content"))
        for message in messages
    )
    return "slow" if has_reasoning else "fast"


def get_record_think_mode(record: dict[str, Any]) -> str:
    think_mode = record.get("think_mode")
    if think_mode in {"fast", "slow"}:
        return think_mode

    meta_info = record.get("meta_info")
    if isinstance(meta_info, dict):
        unique_info = meta_info.get("unique_info")
        if isinstance(unique_info, dict):
            unique_think_mode = unique_info.get("think_mode")
            if unique_think_mode in {"fast", "slow"}:
                return unique_think_mode

    return infer_think_mode_from_messages(record.get("messages") or [])


def count_assistant_rounds(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def infer_language_code(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return "zh" if _CJK_RE.search(content) else "en"

    return "en"


def _normalize_content_to_string(content: Any) -> str:
    if isinstance(content, str):
        return content

    if content is None:
        return ""

    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if text_parts:
            return "\n\n".join(part for part in text_parts if part)

    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _extract_reasoning_from_content(content: str) -> tuple[str, str]:
    stripped = content.strip()
    if not stripped.startswith("<think>") or "</think>" not in stripped:
        return "", content

    reasoning_block = stripped[len("<think>"):]
    reasoning, remainder = reasoning_block.split("</think>", 1)
    return reasoning.strip(), remainder.strip()


def _normalize_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = arguments
    elif arguments is None:
        parsed = {}
    else:
        parsed = arguments

    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}

    normalized = {
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": _normalize_tool_arguments(function.get("arguments")),
        },
    }

    call_id = tool_call.get("id")
    if isinstance(call_id, str) and call_id:
        normalized["id"] = call_id

    return normalized


def _restore_tool_arguments_for_lf(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    restored_messages: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            restored_messages.append(message)
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            restored_messages.append(dict(message))
            continue

        restored_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                restored_tool_calls.append(tool_call)
                continue

            restored_tool_call = dict(tool_call)
            function = tool_call.get("function")
            if not isinstance(function, dict):
                restored_tool_calls.append(restored_tool_call)
                continue

            restored_function = dict(function)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_arguments = arguments
                restored_function["arguments"] = parsed_arguments

            restored_tool_call["function"] = restored_function
            restored_tool_calls.append(restored_tool_call)

        restored_message = dict(message)
        restored_message["tool_calls"] = restored_tool_calls
        restored_messages.append(restored_message)

    return restored_messages


def _align_tool_call_ids(messages: list[dict[str, Any]]) -> None:
    next_call_index = 1

    for msg_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        following_tools: list[dict[str, Any]] = []
        cursor = msg_index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            following_tools.append(messages[cursor])
            cursor += 1

        following_tool_ids = [
            tool_message.get("tool_call_id")
            for tool_message in following_tools
            if isinstance(tool_message.get("tool_call_id"), str) and tool_message["tool_call_id"]
        ]

        assigned_ids: list[str] = []
        for tool_call_index, tool_call in enumerate(tool_calls):
            call_id = tool_call.get("id")
            if not isinstance(call_id, str) or not call_id:
                if tool_call_index < len(following_tool_ids):
                    call_id = following_tool_ids[tool_call_index]
                else:
                    call_id = f"call_{next_call_index:06d}"
                    next_call_index += 1
                tool_call["id"] = call_id
            assigned_ids.append(call_id)

        assign_cursor = 0
        for tool_message in following_tools:
            tool_call_id = tool_message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                while assign_cursor < len(assigned_ids) and assigned_ids[assign_cursor] != tool_call_id:
                    assign_cursor += 1
                if assign_cursor < len(assigned_ids):
                    assign_cursor += 1
                continue

            if assign_cursor < len(assigned_ids):
                tool_message["tool_call_id"] = assigned_ids[assign_cursor]
                assign_cursor += 1


def _normalize_messages_for_panguml(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []

    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue

        role = raw_message.get("role")
        if not isinstance(role, str) or not role:
            continue

        normalized_message: dict[str, Any] = {"role": role}

        name = raw_message.get("name")
        if isinstance(name, str) and name:
            normalized_message["name"] = name

        if role == "assistant":
            content = _normalize_content_to_string(raw_message.get("content"))
            extracted_reasoning, stripped_content = _extract_reasoning_from_content(content)
            normalized_message["content"] = stripped_content if extracted_reasoning else content

            reasoning = raw_message.get("reasoning_content")
            if reasoning is None:
                reasoning = extracted_reasoning
            normalized_message["reasoning_content"] = _normalize_content_to_string(reasoning)

            tool_calls = raw_message.get("tool_calls")
            if isinstance(tool_calls, list):
                normalized_tool_calls = [
                    normalized_tool_call
                    for tool_call in tool_calls
                    if (normalized_tool_call := _normalize_tool_call(tool_call)) is not None
                ]
                if normalized_tool_calls:
                    normalized_message["tool_calls"] = normalized_tool_calls

            weight = raw_message.get("weight")
            if isinstance(weight, (int, float)):
                normalized_message["weight"] = weight
        else:
            normalized_message["content"] = _normalize_content_to_string(raw_message.get("content"))

            if role == "tool":
                tool_call_id = raw_message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    normalized_message["tool_call_id"] = tool_call_id

        normalized_messages.append(normalized_message)

    if normalized_messages and normalized_messages[0].get("role") != "system":
        normalized_messages.insert(0, {"role": "system", "content": ""})

    _align_tool_call_ids(normalized_messages)
    return normalized_messages


def _normalize_tools_for_panguml(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    normalized_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(function, dict):
            continue

        normalized_tools.append({
            "type": "function",
            "function": {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters") or function.get("input_schema") or {},
            },
        })

    return normalized_tools


def _build_meta_info(record: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    existing_meta = record.get("meta_info")
    meta_info = dict(existing_meta) if isinstance(existing_meta, dict) else {}

    unique_info = meta_info.get("unique_info")
    normalized_unique_info = dict(unique_info) if isinstance(unique_info, dict) else {}

    for key in _PANGUML_COMPAT_UNIQUE_INFO_KEYS:
        if key in record and key not in normalized_unique_info:
            normalized_unique_info[key] = record[key]

    for key, value in record.items():
        if key in _PANGUML_TOP_LEVEL_KEYS:
            continue
        normalized_unique_info.setdefault(key, value)

    today = date.today().isoformat()
    return {
        "teacher": meta_info.get("teacher") or "glm-5-thinking",
        "query_source": meta_info.get("query_source") or "synthesized",
        "response_generate_time": meta_info.get("response_generate_time") or today,
        "response_update_time": meta_info.get("response_update_time") or today,
        "owner": meta_info.get("owner") or "00000000",
        "language": meta_info.get("language") or infer_language_code(messages),
        "category": meta_info.get("category") or "code",
        "rounds": meta_info.get("rounds") if meta_info.get("rounds") is not None else count_assistant_rounds(messages),
        "unique_info": normalized_unique_info,
    }


def to_panguml_v2_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = _normalize_messages_for_panguml(record.get("messages") or [])
    tools = _normalize_tools_for_panguml(record.get("tools") or [])

    return {
        "version": PANGUML_VERSION,
        "meta_info": _build_meta_info(record, messages),
        "tools": tools,
        "messages": messages,
    }


def expand_panguml_compat_fields(record: dict[str, Any]) -> dict[str, Any]:
    if not is_im_record(record):
        return record

    expanded = dict(record)
    meta_info = expanded.get("meta_info")
    unique_info = meta_info.get("unique_info") if isinstance(meta_info, dict) else None
    if isinstance(unique_info, dict):
        for key in _PANGUML_COMPAT_UNIQUE_INFO_KEYS:
            if key in unique_info and key not in expanded:
                expanded[key] = unique_info[key]
    return expanded


# ---------------------------------------------------------------------------
# ProcessSummary — shared across all claudecode_opencode converter scripts
# ---------------------------------------------------------------------------

@dataclass
class ProcessSummary:
    """汇总处理统计信息。"""

    role_filtered: int = 0
    reasoning_filtered: int = 0
    failed_instances: int = 0


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------

def print_lf_token_stats_from_texts(
    texts_for_token_stats: list[str],
    n_turns: list[int],
    token_batch_size: int | None = None,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
    model_config: ModelProcessingConfig | None = None,
) -> dict[str, Any]:
    if not texts_for_token_stats:
        print("No LF records for token statistics.")
        return {}

    config = _resolve_model_processing_config(
        model_config,
        tokenizer_name=tokenizer_name,
        token_batch_size=token_batch_size,
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    token_lens: list[int] = []
    for start in tqdm(range(0, len(texts_for_token_stats), config.token_batch_size), desc="Token stats"):
        batch_texts = texts_for_token_stats[start:start + config.token_batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_length=True,
            padding=False,
            truncation=False,
        )
        token_lens.extend(encoded['length'])

    if tokenizer_name is not None or model_config is not None:
        print(f"Tokenizer: {config.tokenizer_name}")
    total_tokens = int(sum(token_lens))
    print(f"[token_lens]\nMax: {int(np.max(token_lens))}\nMin: {int(np.min(token_lens))}\nMean: {int(np.mean(token_lens))}\nTotal: {total_tokens}")
    print("num of token len larger than 128k: ", sum(length > 131072 for length in token_lens), "\n")
    print(f"[n_turn]\nMax: {int(np.max(n_turns))}\nMin: {int(np.min(n_turns))}\nMean: {int(np.mean(n_turns))}")
    print("num of n_turn >= 100: ", sum(turns >= 100 for turns in n_turns), "\n")

    return {
        "token_lens": {"max": int(np.max(token_lens)), "min": int(np.min(token_lens)), "mean": int(np.mean(token_lens)), "gt_128k": int(sum(length > 131072 for length in token_lens))},
        "n_turns": {"max": int(np.max(n_turns)), "min": int(np.min(n_turns)), "mean": int(np.mean(n_turns)), "gte_100": int(sum(turns >= 100 for turns in n_turns))},
        "total_tokens": total_tokens,
        "count": len(texts_for_token_stats),
    }


def print_lf_token_stats(
    lf_records: list[dict[str, Any]],
    tokenizer_name: str | None = None,
    token_batch_size: int | None = None,
    stats_output_path: Path | None = None,
    model_config: ModelProcessingConfig | None = None,
) -> dict[str, Any]:
    if not lf_records:
        print("No LF records for token statistics.")
        return {}

    config = _resolve_model_processing_config(
        model_config,
        tokenizer_name=tokenizer_name,
        token_batch_size=token_batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    texts_for_token_stats: list[str] = []
    n_turns: list[int] = []

    for item in tqdm(lf_records, desc="Prepare token stats"):
        text = tokenizer.apply_chat_template(
            item['messages'],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts_for_token_stats.append(text)
        n_turns.append(sum(1 for m in item['messages'] if m.get('role') == 'assistant'))

    stats = print_lf_token_stats_from_texts(
        texts_for_token_stats,
        n_turns,
        token_batch_size=config.token_batch_size,
        tokenizer=tokenizer,
        tokenizer_name=config.tokenizer_name,
    )

    scores_list = []
    for item in lf_records:
        score = item.get('_score')
        if score and isinstance(score, dict):
            cs = score.get('composite_score')
            if cs is not None:
                scores_list.append(cs)
    if scores_list:
        stats["scores"] = {"max": round(max(scores_list), 4), "min": round(min(scores_list), 4), "mean": round(float(np.mean(scores_list)), 4)}

    if stats_output_path is not None:
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Stats saved: {stats_output_path}")

    return stats


# ---------------------------------------------------------------------------
# Role / reasoning validation
# ---------------------------------------------------------------------------

def check_roles(messages: list[dict[str, Any]]) -> bool:
    """
    角色顺序：
        1. 第一个角色一定是用户，最后一个角色一定是助手；
        2. 助手之后只能是工具或用户；
        3. 工具之后只能是工具或助手；
        4. 用户之后只能是助手。
    """
    if not messages:
        print("  [check_roles] 异常数据，messages 为空")
        return False

    start_idx = 1 if messages[0]["role"] == "system" else 0
    if start_idx >= len(messages):
        print("  [check_roles] 异常数据，system 之后没有有效对话")
        return False

    first_role = messages[start_idx]["role"]
    if first_role != "user":
        print(f"  [check_roles] 异常数据，messages第一轮对话必须是用户，实际为: {first_role}")
        return False

    last_role = messages[-1]["role"]
    if last_role != "assistant":
        print(f"  [check_roles] 异常数据，messages最后一轮对话必须是助手，实际为: {last_role}")
        return False

    for idx in range(start_idx + 1, len(messages)):
        role = messages[idx]["role"]
        pre_role = messages[idx - 1]["role"]
        rel_idx = idx - start_idx

        if pre_role == "assistant":
            if role != "tool" and role != "user":
                print(f"  [check_roles] 异常数据，助手之后只能是工具或用户，实际为: {role} (idx={rel_idx})")
                return False
        elif pre_role == "tool":
            if role != "tool" and role != "assistant":
                print(f"  [check_roles] 异常数据，工具之后只能是工具或助手，实际为: {role} (idx={rel_idx})")
                return False
        elif pre_role == "user":
            if role != "assistant":
                print(f"  [check_roles] 异常数据，用户之后只能是助手，实际为: {role} (idx={rel_idx})")
                return False
        else:
            print(f"  [check_roles] 未知角色: role={role}, pre_role={pre_role} (idx={rel_idx})")
            return False
    return True


def check_tool_calls(messages: list[dict[str, Any]]) -> bool:
    """检查除最后一轮 assistant 外，其余 assistant 轮次是否都有 tool_calls。"""
    assistant_indices = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if not assistant_indices:
        return True
    for idx in assistant_indices[:-1]:
        if not messages[idx].get("tool_calls"):
            print(f"  [check_tool_calls] 异常数据，assistant 轮次缺少 tool_calls (idx={idx})")
            return False
    return True


def check_reasoning_content(
    messages: list[dict[str, Any]],
    think_mode: str,
    pseudo_turns: int | None,
    reasoning_check_mode: Literal["strict", "adaptive"] = "strict",
    reasoning_content_ratio_threshold: float = 0.5,
) -> bool:
    if think_mode == "fast":
        return True

    check_turns = pseudo_turns if pseudo_turns else 0
    if reasoning_check_mode == "adaptive":
        assistant_count = 0
        reasoning_count = 0
        for idx in range(check_turns, len(messages)):
            msg = messages[idx]
            if msg.get("role") != "assistant":
                continue
            assistant_count += 1
            if msg.get("reasoning_content"):
                reasoning_count += 1

        if assistant_count == 0:
            return True
        return reasoning_count / assistant_count >= reasoning_content_ratio_threshold

    for idx in range(check_turns, len(messages)):
        msg = messages[idx]
        if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
            print(f"  [check_reasoning_content] 异常数据，assistant 轮次缺少 reasoning_content (idx={idx})")
            return False
    return True


# ---------------------------------------------------------------------------
# LF format conversion
# ---------------------------------------------------------------------------

def convert_json_to_lf_format(
    all_json_data: list[dict[str, Any]],
    tokenizer_name: str | None = None,
    compute_token_stats: bool = True,
    token_batch_size: int | None = None,
    model_config: ModelProcessingConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not all_json_data:
        print("No records to convert; saving empty LF dataset.")
        return [], {}

    config = _resolve_model_processing_config(
        model_config,
        tokenizer_name=tokenizer_name,
        token_batch_size=token_batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    lf_all_json_data: list[dict[str, Any]] = []
    texts_for_token_stats: list[str] | None = [] if compute_token_stats else None
    n_turns: list[int] = []
    scores_list: list[float] = []

    for i in tqdm(range(len(all_json_data))):
        item = expand_panguml_compat_fields(all_json_data[i])
        text = tokenizer.apply_chat_template(
            _restore_tool_arguments_for_lf(item['messages']),
            tools=item.get('tools') or [],
            tokenize=False,
            add_generation_prompt=False,
        )
        if compute_token_stats:
            texts_for_token_stats.append(text)

        messages: list[dict[str, str]] = []
        think_mode_fast = get_record_think_mode(item) == 'fast'
        for turn in text.split("<|im_end|>"):
            turn_str = turn.strip()
            if not turn_str:
                continue

            header, _, content = turn_str.partition('\n')
            if not header.startswith('<|im_start|>'):
                continue

            role = header.split('<|im_start|>')[-1]
            if not role:
                continue

            if think_mode_fast and "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()

            messages.append({
                'role': role,
                'content': content,
            })
        n_turn = sum(1 for m in messages if m['role'] == 'assistant')
        n_turns.append(n_turn)

        lf_record = {'messages': messages}
        if '_score' in item:
            lf_record['_score'] = item['_score']
            score = item['_score']
            if score and isinstance(score, dict):
                cs = score.get('composite_score')
                if cs is not None:
                    scores_list.append(cs)
        if '_instance_id' in item:
            lf_record['_instance_id'] = item['_instance_id']
        if '_agent_type' in item:
            lf_record['_agent_type'] = item['_agent_type']
        if '_gen_params' in item:
            lf_record['_gen_params'] = item['_gen_params']
        if '_usage' in item:
            lf_record['_usage'] = item['_usage']
        lf_all_json_data.append(lf_record)

    stats: dict[str, Any] = {}
    if compute_token_stats:
        stats = print_lf_token_stats_from_texts(
            texts_for_token_stats,
            n_turns,
            token_batch_size=config.token_batch_size,
            tokenizer=tokenizer,
            tokenizer_name=config.tokenizer_name,
        )
    else:
        print(f"[n_turn]\nMax: {int(np.max(n_turns))}\nMin: {int(np.min(n_turns))}\nMean: {int(np.mean(n_turns))}")
        print("num of n_turn >= 100: ", sum(i >= 100 for i in n_turns), "\n")
        stats = {
            "n_turns": {"max": int(np.max(n_turns)), "min": int(np.min(n_turns)), "mean": int(np.mean(n_turns)), "gte_100": int(sum(i >= 100 for i in n_turns))},
            "count": len(n_turns),
        }
    if scores_list:
        stats["scores"] = {"max": round(max(scores_list), 4), "min": round(min(scores_list), 4), "mean": round(float(np.mean(scores_list)), 4)}
    print(f"final save {len(lf_all_json_data)} samples.")
    return lf_all_json_data, stats


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------

def load_json(file_path: Path) -> Any:
    """Read and parse a single JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Save records as JSONL (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            output_record = to_panguml_v2_record(record) if is_im_record(record) else record
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件（每行一个 JSON 对象）。"""
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(expand_panguml_compat_fields(record) if is_im_record(record) else record)
            except json.JSONDecodeError as e:
                print(f"  [WARN] 跳过 {file_path.name} 第 {lineno} 行: {e}")
    return records


def save_lf_json(
    output_path: Path,
    records: list[dict[str, Any]],
    tokenizer_name: str | None = None,
    token_batch_size: int | None = None,
    model_config: ModelProcessingConfig | None = None,
) -> None:
    """Convert IM records to LF format and save as JSON + stats sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf_all_json_data, stats = convert_json_to_lf_format(
        records,
        tokenizer_name=tokenizer_name,
        token_batch_size=token_batch_size,
        model_config=model_config,
    )

    # 工具调用错误率基于 IM 记录计算（此时 role="tool" 结果尚未被 LF 合并），
    # 局部导入避免 utils <-> rule_score 循环依赖。
    from swe_data_process.rule_score import (
        compute_tool_call_error_rate,
        print_tool_call_error_summary,
    )
    tool_call_errors = compute_tool_call_error_rate(records)
    print_tool_call_error_summary(tool_call_errors)
    stats = stats or {}
    stats["tool_call_errors"] = tool_call_errors

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lf_all_json_data, f, ensure_ascii=False, indent=4)
    if stats:
        stats_path = output_path.with_suffix(".stats.json")
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Stats saved: {stats_path}")


# ---------------------------------------------------------------------------
# Shared filtering / instance helpers
# ---------------------------------------------------------------------------

def should_keep_instance(role_filtered: int, reasoning_filtered: int) -> bool:
    """保持原始筛选逻辑：两个过滤计数都为 0 则保留该实例的记录。"""
    return role_filtered == 0 and reasoning_filtered == 0


# ---------------------------------------------------------------------------
# Main agent / subagent detection (CC/OC only)
# ---------------------------------------------------------------------------

_MAIN_AGENT_TOOLS = frozenset({"edit", "write"})


def detect_agent_type(record: dict[str, Any]) -> str:
    """检测 IM 记录来自 main agent 还是 subagent。

    Main agent 拥有写操作工具 (Edit, Write)；subagent (context-gatherer) 只有只读工具。
    无 tools 字段的记录（如 Terminus2）默认为 "main"。
    """
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        return "main"

    tool_names_lower = set()
    for t in tools:
        func = t.get("function", {}) if isinstance(t, dict) else {}
        name = func.get("name", "")
        if name:
            tool_names_lower.add(name.lower())

    if tool_names_lower and not (tool_names_lower & _MAIN_AGENT_TOOLS):
        return "subagent"
    return "main"


def tag_instance_records(
    records: list[dict[str, Any]],
    instance_id: str,
) -> None:
    """为同一 instance 的所有 converted records 就地添加 _instance_id 和 _agent_type。"""
    for record in records:
        record["_instance_id"] = instance_id
        record["_agent_type"] = detect_agent_type(record)


def filter_by_score_bundled(
    records: list[dict[str, Any]],
    min_score: float,
    score_key: str = "composite_score",
) -> list[dict[str, Any]]:
    """按分数筛选记录，同 instance 的 main + subagent 捆绑保留/丢弃。

    判定逻辑：instance 内 main agent 的最高分 >= min_score 则保留整个 instance。
    无 _instance_id 的记录按自身分数独立判定。
    """
    instance_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ungrouped: list[dict[str, Any]] = []

    for r in records:
        iid = r.get("_instance_id")
        if iid:
            instance_groups[iid].append(r)
        else:
            ungrouped.append(r)

    result: list[dict[str, Any]] = []

    for group in instance_groups.values():
        main_scores = [
            r["_score"][score_key]
            for r in group
            if r.get("_agent_type") == "main"
            and isinstance(r.get("_score"), dict)
            and score_key in r["_score"]
        ]
        if main_scores and max(main_scores) >= min_score:
            result.extend(group)

    for r in ungrouped:
        score = r.get("_score")
        if isinstance(score, dict) and score.get(score_key, 0) >= min_score:
            result.append(r)

    return result


def replace_system_model_name(
    messages: list[dict[str, Any]],
    source_model: str | None = None,
    target_model: str | None = None,
    model_config: ModelProcessingConfig | None = None,
) -> None:
    """Replace model name in the system message (first message) in-place."""
    if not messages:
        return

    config = _resolve_model_processing_config(
        model_config,
        source_model=source_model,
        target_model=target_model,
    )
    first_message = messages[0]
    if first_message.get("role") != "system":
        return

    content = first_message.get("content")
    if isinstance(content, str):
        first_message["content"] = content.replace(
            config.system_source_model,
            config.system_target_model,
        )


InstanceStatus = Literal["resolved", "unresolved", "all"]

_INSTANCE_STATUS_REWARD_KEYS: dict[InstanceStatus, tuple[str, ...] | None] = {
    "resolved": ("1.0",),
    "unresolved": ("0.0",),
    "all": None,
}


def get_instances_from_job_dir(
    job_dir: Path,
    instance_status: InstanceStatus = "resolved",
) -> list[str]:
    """Read selected folder names from job_dir/result.json.

    Returns folder names (with hash suffix, e.g. 'astropy__astropy-7606__nCRsfSp').
    Use extract_instance_id() to get the bare instance_id for tagging.
    """
    reward_keys = _INSTANCE_STATUS_REWARD_KEYS[instance_status]
    result_json = job_dir / "result.json"
    with result_json.open("r", encoding="utf-8") as f:
        result = json.load(f)
    evals = result["stats"]["evals"]
    selected: list[str] = []
    for key in evals:
        rewards = evals[key]["reward_stats"]["reward"]
        if reward_keys is None:
            for folders in rewards.values():
                selected.extend(folders)
        else:
            for reward_key in reward_keys:
                selected.extend(rewards.get(reward_key, []))
    return selected


def get_resolved_instances_from_job_dir(job_dir: Path) -> list[str]:
    """Read resolved (reward=1.0) folder names from job_dir/result.json."""
    return get_instances_from_job_dir(job_dir, instance_status="resolved")


def extract_instance_id(folder_name: str) -> str:
    """Extract instance_id from folder name by removing the hash suffix.

    e.g. 'astropy__astropy-7606__nCRsfSp' -> 'astropy__astropy-7606'
    """
    return folder_name.rsplit("__", 1)[0]


def extract_instance_id_from_config(job_dir: Path, folder_name: str) -> str:
    """Read the authoritative instance_id from a Harbor instance config."""
    config_path = job_dir / folder_name / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return Path(config["task"]["path"]).name


# ---------------------------------------------------------------------------
# Reference-repo filtering — exclude instances belonging to reference datasets
# ---------------------------------------------------------------------------

# Default reference datasets whose repos should be excluded from training data
DEFAULT_REFERENCE_DATASETS: list[dict[str, str]] = [
    {"name": "SWE-bench/SWE-bench_Verified", "split": "test"},
    {"name": "ScaleAI/SWE-bench_Pro", "split": "test"},
    {"name": "SWE-bench/SWE-bench_Multilingual", "split": "test"},
]


def load_reference_repos_from_hf(
    reference_datasets: list[dict[str, str]] | None = None,
) -> set[str]:
    """Load unique repo names from HuggingFace reference datasets.

    Returns a set of ``"owner/repo"`` strings (original casing preserved).
    """
    from datasets import load_dataset as _hf_load_dataset

    if reference_datasets is None:
        reference_datasets = DEFAULT_REFERENCE_DATASETS

    all_repos: set[str] = set()
    for spec in reference_datasets:
        ds_name = spec["name"]
        split = spec.get("split", "test")
        try:
            ds = _hf_load_dataset(ds_name, split=split)
            repos = {str(row["repo"]).strip() for row in ds if row.get("repo")}
            all_repos.update(repos)
            print(f"  [ref] {ds_name}[{split}]: {len(repos)} unique repos")
        except Exception as exc:
            print(f"  [ref] WARNING: failed to load {ds_name}: {exc}")
    print(f"  [ref] Total unique reference repos: {len(all_repos)}")
    return all_repos


def load_excluded_repos_from_file(exclude_repos_file: Path) -> set[str]:
    """Read excluded repos from a text file (one ``owner/repo`` per line).

    Lines starting with ``#`` and blank lines are ignored.
    """
    repos: set[str] = set()
    with exclude_repos_file.open("r", encoding="utf-8") as f:
        for line in f:
            repo = line.strip()
            if repo and not repo.startswith("#"):
                repos.add(repo)
    return repos


def build_repo_exclusion_patterns(repos: Iterable[str]) -> list[re.Pattern[str]]:
    """Build compiled regex patterns to match instance_ids from given repos.

    Each reference repo ``"owner/repo_name"`` produces a pattern that matches
    instance_ids of the form ``owner__repo_name-<digits>`` with an optional
    ``__<hash>`` suffix (.

    Returns a list of compiled regex patterns.
    """
    patterns: list[re.Pattern[str]] = []
    for repo in repos:
        owner, _, repo_name = repo.partition("/")
        if not (owner and repo_name):
            continue
        # Match: owner__repo_name-<issue_number> with optional __<hash> suffix
        pat = re.compile(
            rf"^{re.escape(owner)}__{re.escape(repo_name)}-\d+(__\w+)?$",
            re.IGNORECASE,
        )
        patterns.append(pat)
    return patterns


def instance_id_matches_excluded_repos(
    instance_id: str,
    patterns: list[re.Pattern[str]],
) -> bool:
    """Return True if *instance_id* belongs to any excluded repo."""
    for pat in patterns:
        if pat.match(instance_id):
            return True
    return False


def filter_instance_ids_by_repo(
    instance_ids: list[str],
    patterns: list[re.Pattern[str]],
    *,
    label: str = "",
) -> list[str]:
    """Return *instance_ids* that do **not** belong to any excluded repo.

    Prints a summary of how many were filtered.
    """
    if not patterns:
        return instance_ids
    original = len(instance_ids)
    filtered = [
        iid for iid in instance_ids
        if not instance_id_matches_excluded_repos(iid, patterns)
    ]
    excluded = original - len(filtered)
    tag = f" ({label})" if label else ""
    print(f"  [repo-filter]{tag} excluded {excluded}/{original} instances, "
          f"kept {len(filtered)}")
    return filtered


def filter_paths_by_repo(
    paths: list[Path],
    patterns: list[re.Pattern[str]],
    *,
    name_func: Callable[[Path], str] | None = None,
    label: str = "",
) -> list[Path]:
    """Return *paths* whose derived instance name does **not** match excluded repos.

    *name_func* extracts the instance-id string from a path; defaults to
    ``path.stem`` (filename without extension).
    """
    if not patterns:
        return paths
    if name_func is None:
        name_func = lambda p: p.stem
    original = len(paths)
    filtered = [
        p for p in paths
        if not instance_id_matches_excluded_repos(name_func(p), patterns)
    ]
    excluded = original - len(filtered)
    tag = f" ({label})" if label else ""
    print(f"  [repo-filter]{tag} excluded {excluded}/{original} paths, "
          f"kept {len(filtered)}")
    return filtered


def load_exclusion_patterns(
    exclude_repos_file: Path | None = None,
) -> list[re.Pattern[str]]:
    """Convenience: load exclusion patterns from a repos file.

    Returns an empty list when *exclude_repos_file* is ``None``.
    """
    if exclude_repos_file is None:
        return []
    repos = load_excluded_repos_from_file(exclude_repos_file)
    patterns = build_repo_exclusion_patterns(repos)
    print(f"  [repo-filter] Loaded {len(repos)} excluded repos "
          f"({len(patterns)} patterns) from {exclude_repos_file}")
    return patterns
