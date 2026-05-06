from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_REPOS_FILE = REPO_ROOT / "artifacts" / "excluded_repos.txt"


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
    token_batch_size: int = 64,
    tokenizer: Any = None,
    tokenizer_name: str | None = None,
) -> dict[str, Any]:
    if not texts_for_token_stats:
        print("No LF records for token statistics.")
        return {}

    if tokenizer is None:
        if tokenizer_name is None:
            raise ValueError("Either tokenizer or tokenizer_name must be provided.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_lens: list[int] = []
    for start in tqdm(range(0, len(texts_for_token_stats), token_batch_size), desc="Token stats"):
        batch_texts = texts_for_token_stats[start:start + token_batch_size]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_length=True,
            padding=False,
            truncation=False,
        )
        token_lens.extend(encoded['length'])

    if tokenizer_name is not None:
        print(f"Tokenizer: {tokenizer_name}")
    print(f"[token_lens]\nMax: {int(np.max(token_lens))}\nMin: {int(np.min(token_lens))}\nMean: {int(np.mean(token_lens))}")
    print("num of token len larger than 128k: ", sum(length > 131072 for length in token_lens), "\n")
    print(f"[n_turn]\nMax: {int(np.max(n_turns))}\nMin: {int(np.min(n_turns))}\nMean: {int(np.mean(n_turns))}")
    print("num of n_turn >= 100: ", sum(turns >= 100 for turns in n_turns), "\n")

    return {
        "token_lens": {"max": int(np.max(token_lens)), "min": int(np.min(token_lens)), "mean": int(np.mean(token_lens)), "gt_128k": int(sum(length > 131072 for length in token_lens))},
        "n_turns": {"max": int(np.max(n_turns)), "min": int(np.min(n_turns)), "mean": int(np.mean(n_turns)), "gte_100": int(sum(turns >= 100 for turns in n_turns))},
        "count": len(texts_for_token_stats),
    }


def print_lf_token_stats(
    lf_records: list[dict[str, Any]],
    tokenizer_name: str = "Qwen/Qwen3-8B",
    token_batch_size: int = 64,
    stats_output_path: Path | None = None,
) -> dict[str, Any]:
    if not lf_records:
        print("No LF records for token statistics.")
        return {}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
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
        token_batch_size=token_batch_size,
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
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


def check_reasoning_content(
    messages: list[dict[str, Any]],
    think_mode: str,
    pseudo_turns: int | None,
) -> bool:
    if think_mode == "fast":
        return True

    check_turns = pseudo_turns if pseudo_turns else 0
    for idx in range(check_turns, len(messages)):
        msg = messages[idx]
        if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
            return False
    return True


# ---------------------------------------------------------------------------
# LF format conversion
# ---------------------------------------------------------------------------

def convert_json_to_lf_format(
    all_json_data: list[dict[str, Any]],
    tokenizer_name: str = "Qwen/Qwen3-8B",
    compute_token_stats: bool = True,
    token_batch_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not all_json_data:
        print("No records to convert; saving empty LF dataset.")
        return [], {}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lf_all_json_data: list[dict[str, Any]] = []
    texts_for_token_stats: list[str] | None = [] if compute_token_stats else None
    n_turns: list[int] = []
    scores_list: list[float] = []

    for i in tqdm(range(len(all_json_data))):
        item = all_json_data[i]
        text = tokenizer.apply_chat_template(
            item['messages'],
            tools=item['tools'],
            tokenize=False,
            add_generation_prompt=False,
        )
        if compute_token_stats:
            texts_for_token_stats.append(text)

        messages: list[dict[str, str]] = []
        think_mode_fast = item['think_mode'] == 'fast'
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
        lf_all_json_data.append(lf_record)

    stats: dict[str, Any] = {}
    if compute_token_stats:
        stats = print_lf_token_stats_from_texts(
            texts_for_token_stats,
            n_turns,
            token_batch_size=token_batch_size,
            tokenizer=tokenizer,
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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件（每行一个 JSON 对象）。"""
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] 跳过 {file_path.name} 第 {lineno} 行: {e}")
    return records


def save_lf_json(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Convert IM records to LF format and save as JSON + stats sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf_all_json_data, stats = convert_json_to_lf_format(records)
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
    source_model: str = "GLM-5-FP8",
    target_model: str = "Qwen3-8B",
) -> None:
    """Replace model name in the system message (first message) in-place."""
    if not messages:
        return

    first_message = messages[0]
    if first_message.get("role") != "system":
        return

    content = first_message.get("content")
    if isinstance(content, str):
        first_message["content"] = content.replace(source_model, target_model)


def get_resolved_instances(result_json: Path) -> list[str]:
    """Read resolved (reward=1.0) instance IDs from Harbor result.json."""
    with result_json.open("r", encoding="utf-8") as f:
        result = json.load(f)
    evals = result["stats"]["evals"]
    resolved: list[str] = []
    for key in evals:
        resolved.extend(evals[key]["reward_stats"]["reward"]["1.0"])
    return resolved


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
    ``__<hash>`` suffix (jierun format).

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
