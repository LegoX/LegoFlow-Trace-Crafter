"""
Convert dataclaw conversations.jsonl files to IM (intermediate) format
for LLaMA-Factory SFT training.

Dataclaw format (per record):
  session_id, model, git_branch, start_time, end_time, messages, stats, project, source

Each message has: role (user|assistant), timestamp, and some of:
  content (str), thinking (str), tool_uses (list of {tool, input})

Key differences from CC session / LiteLLM formats:
  - No tool results stored (stripped at export time)
  - Consecutive assistant messages are split by field type
    (thinking, tool_uses, content each get their own message object)
  - No system prompt, no tool definitions stored
  - tool_uses.input is always a plain string (not a dict)

Conversion strategy:
  - Only keep Claude model records
  - Filter internal CC user messages (slash commands, local-command-* tags)
  - Merge consecutive assistant messages into one IM assistant message
  - Inject offline system prompt (cc_system_prompt.json) as first message
  - think_mode="slow" if ANY merged assistant turn has thinking;
    reasoning_content kept only on turns that have it (partial is allowed)
  - No tool result messages produced (not available)
  - Tool definitions loaded from cc_tool_definitions.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import (
    normalize_tool_definition,
)
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    ProcessSummary,
    check_roles,
    filter_paths_by_repo,
    load_exclusion_patterns,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)

_SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# System prompt loading
# ---------------------------------------------------------------------------

def load_system_prompt(path: Path) -> str:
    with open(path) as f:
        data = json.load(f)
    return "\n\n".join(b["text"] for b in data["blocks"] if b.get("text"))


# ---------------------------------------------------------------------------
# Tool input parsing
# ---------------------------------------------------------------------------

_SINGLE_ARG = {
    "Bash": "command",
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "Agent": "prompt",
    "Task": "prompt",
    "WebFetch": "url",
    "WebSearch": "query",
    "TodoWrite": "todos",
    "CronDelete": "id",
    "EnterWorktree": "name",
    "Skill": "skill",
}

_KV_TOOLS = {
    "Glob": ["pattern", "path"],
    "Grep": ["pattern", "path", "glob", "output_mode", "type",
             "context", "-A", "-B", "-C", "-n", "-i", "head_limit", "offset", "multiline"],
}


def _parse_kv_input(input_str: str, param_names: list[str]) -> dict:
    positions: list[tuple[int, int, str]] = []
    for param in param_names:
        for m in re.finditer(r"(?:^|(?<=\s))" + re.escape(param) + r"=", input_str):
            positions.append((m.start(), m.end(), param))
    positions.sort()

    result: dict = {}
    for i, (_, val_start, param) in enumerate(positions):
        if i + 1 < len(positions):
            val_end = positions[i + 1][0]
            value = input_str[val_start:val_end].rstrip()
        else:
            value = input_str[val_start:]
        if value:
            result[param] = value
    return result


def parse_tool_input(tool_name: str, input_val) -> dict:
    if isinstance(input_val, dict):
        return input_val
    input_str = input_val if isinstance(input_val, str) else str(input_val)
    if tool_name in _SINGLE_ARG:
        return {_SINGLE_ARG[tool_name]: input_str}
    if tool_name in _KV_TOOLS:
        parsed = _parse_kv_input(input_str, _KV_TOOLS[tool_name])
        return parsed if parsed else {"input": input_str}
    return {"input": input_str}


def make_tool_call(tool_name: str, input_str: str, call_id: str) -> dict:
    args = parse_tool_input(tool_name, input_str)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": args,
        },
    }


# ---------------------------------------------------------------------------
# User message filtering
# ---------------------------------------------------------------------------

_SKIP_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-name>",
    "[Request interrupted",
    "<system-reminder>",
)


def is_internal_user_message(content: str) -> bool:
    s = content.strip()
    return any(s.startswith(p) for p in _SKIP_PREFIXES)


# ---------------------------------------------------------------------------
# Claude model detection
# ---------------------------------------------------------------------------

def is_claude_model(model: str) -> bool:
    return model.startswith("claude-") or "anthropic/claude" in model


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def merge_assistant_messages(raw_messages: list[dict]) -> list[dict]:
    result: list[dict] = []
    i = 0
    call_counter = 0

    while i < len(raw_messages):
        msg = raw_messages[i]
        if msg["role"] != "assistant":
            result.append(msg)
            i += 1
            continue

        thinking_parts: list[str] = []
        tool_calls: list[dict] = []
        content_parts: list[str] = []

        while i < len(raw_messages) and raw_messages[i]["role"] == "assistant":
            m = raw_messages[i]
            if "thinking" in m:
                thinking_parts.append(m["thinking"])
            if "tool_uses" in m:
                for tu in m["tool_uses"]:
                    call_id = f"toolu_{call_counter:06d}"
                    call_counter += 1
                    tool_calls.append(make_tool_call(tu["tool"], tu.get("input", ""), call_id))
            if "content" in m:
                content_parts.append(m["content"])
            i += 1

        merged: dict = {"role": "assistant"}
        merged["content"] = "\n\n".join(content_parts) if content_parts else ""
        if thinking_parts:
            merged["reasoning_content"] = "\n\n".join(thinking_parts)
        if tool_calls:
            merged["tool_calls"] = tool_calls
        result.append(merged)

    return result


def build_im_messages(raw_messages: list[dict]) -> list[dict] | None:
    filtered: list[dict] = []
    for m in raw_messages:
        if m["role"] == "user":
            content = m.get("content", "")
            if not is_internal_user_message(content):
                filtered.append({"role": "user", "content": content})
        else:
            filtered.append(m)

    merged = merge_assistant_messages(filtered)

    deduped: list[dict] = []
    for m in merged:
        if deduped and deduped[-1]["role"] == "user" and m["role"] == "user":
            deduped[-1]["content"] = deduped[-1]["content"] + "\n\n" + m["content"]
        else:
            deduped.append(m)
    merged = deduped

    while merged and merged[-1]["role"] == "user":
        merged.pop()

    if not merged:
        return None
    if not any(m["role"] == "assistant" for m in merged):
        return None
    if not any(m["role"] == "user" for m in merged):
        return None

    return merged


def process_one_record(
    record: dict,
    tools: list[dict],
    system_prompt: str,
) -> tuple[dict | None, int, int]:
    messages = build_im_messages(record.get("messages", []))
    if messages is None:
        return None, 1, 0

    messages = [{"role": "system", "content": system_prompt}] + messages

    asst_messages = [m for m in messages if m["role"] == "assistant"]
    any_have_thinking = any("reasoning_content" in m for m in asst_messages)
    think_mode = "slow" if any_have_thinking else "fast"

    if think_mode == "fast":
        for m in messages:
            m.pop("reasoning_content", None)

    if not check_roles(messages):
        return None, 1, 0

    im = {
        "messages": messages,
        "tools": tools,
        "pseudo_turns": None,
        "think_mode": think_mode,
    }
    return im, 0, 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = Path("/mnt/haoli/data/dataclaw")
DEFAULT_IM_OUTPUT = Path(
    "/mnt/haoli/code/swe_data_process/output/20260408/dataclaw_cc_all.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/mnt/haoli/code/swe_data_process/output/20260408/dataclaw_cc_all.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert dataclaw CC sessions to IM format"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help="Root directory containing user subdirs with conversations.jsonl",
    )
    parser.add_argument("--im-output", type=Path, default=DEFAULT_IM_OUTPUT)
    parser.add_argument("--lf-output", type=Path, default=DEFAULT_LF_OUTPUT)
    parser.add_argument(
        "--system-prompt-json", type=Path,
        default=_SCRIPT_DIR / "cc_system_prompt.json",
    )
    parser.add_argument(
        "--tools-json", type=Path,
        default=_SCRIPT_DIR / "cc_tool_definitions.json",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_im_data(
    all_records: list[dict],
    tools: list[dict],
    system_prompt: str,
    max_instances: int | None = None,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], ProcessSummary]:
    summary = ProcessSummary()
    im_data: list[dict[str, Any]] = []

    if max_instances:
        all_records = all_records[:max_instances]

    for record in tqdm(all_records, disable=quiet, desc="Converting"):
        model = record.get("model", "")
        if not is_claude_model(model):
            continue

        try:
            im, role_f, reason_f = process_one_record(record, tools, system_prompt)
            summary.role_filtered += role_f
            summary.reasoning_filtered += reason_f

            if im is not None and should_keep_instance(role_f, reason_f):
                converted_records = [im]
                session_id = record.get("session_id", "unknown")
                tag_instance_records(converted_records, str(session_id))
                im_data.extend(converted_records)
        except Exception as exc:
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] record {record.get('session_id', '?')} failed: {exc}")

    return im_data, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"source-dir does not exist: {args.source_dir}")

    system_prompt = load_system_prompt(args.system_prompt_json)

    with open(args.tools_json) as f:
        raw_tools = json.load(f)
    tools = [normalize_tool_definition(t) for t in raw_tools]

    jsonl_files = sorted(args.source_dir.rglob("conversations.jsonl"))
    if not jsonl_files:
        print(f"No conversations.jsonl found under {args.source_dir}", file=sys.stderr)
        return

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        jsonl_files = filter_paths_by_repo(
            jsonl_files, exclusion_patterns, label="dataclaw",
        )

    all_records: list[dict] = []
    for f in jsonl_files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))

    print(f"Total raw records: {len(all_records)}")

    im_data, summary = collect_im_data(
        all_records, tools, system_prompt,
        max_instances=args.max_instances,
        quiet=args.quiet,
    )

    im_data = score_dataset(im_data, quiet=True)

    args.im_output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.im_output, im_data)

    print(f"Total IM records : {len(im_data)}")
    print(f"Role-filtered    : {summary.role_filtered}")
    print(f"Reasoning-filter : {summary.reasoning_filtered}")
    print(f"Failed/skipped   : {summary.failed_instances}")
    print(f"Saved IM  -> {args.im_output}")

    save_lf_json(args.lf_output, im_data)
    print(f"Saved LF  -> {args.lf_output}")


if __name__ == "__main__":
    main()
