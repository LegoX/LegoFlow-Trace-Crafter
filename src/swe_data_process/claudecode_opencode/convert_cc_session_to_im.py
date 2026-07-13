"""
Convert Claude Code exported session JSONL files to IM (intermediate) format
for LLaMA-Factory SFT training.

Background
----------
Claude Code writes every conversation turn to a per-session JSONL file under:
  <project-dir>/.claude/sessions/projects/<slug>/<session-uuid>.jsonl

Each line is a JSON event with a `type` field:
  queue-operation  — enqueue/dequeue of the initial prompt (2 lines, skipped)
  user             — user message or tool result
  assistant        — model response
  last-prompt      — truncated copy of initial prompt (1 line, skipped)

Key differences from LiteLLM trajectory files
----------------------------------------------
  - No system prompt — not stored in session files; loaded from offline JSON.
  - No tool definitions — not stored; loaded from offline JSON.
  - No thinking blocks — Claude Code strips them before writing; think_mode="fast".
  - No system-reminder injections — injected at API-call time; loaded from offline JSON.
  - Multi-tool calls: when an assistant makes N tool calls, N separate user lines
    follow (one tool_result each), rather than one user message with N blocks.

These differences mean the converter is simpler: no chain splitting, no thinking
recovery, no cross-chain fingerprint maps. The conversation is a flat event log.

Content identity
----------------
Text content, tool call names/ids/inputs, and tool result strings are
byte-for-byte identical to the corresponding LiteLLM trajectory data (verified).
The produced IM records match litellm-converted IM in all respects except:
  - No reasoning_content on assistant turns (think_mode="fast")
  - One record per task (no split chains from API-level context resets)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import (
    build_tool_properties_order_map,
    convert_assistant_blocks,
    infer_think_mode,
    join_text_parts,
    normalize_assistant_empty_content,
    normalize_tool_content,
    normalize_tool_definition,
    remove_tool_use_error_turns,
    trim_trailing_tools,
)
from swe_data_process.rule_score import score_dataset
from swe_data_process.utils import (
    EXCLUDED_REPOS_FILE,
    ProcessSummary,
    check_reasoning_content,
    check_roles,
    filter_paths_by_repo,
    load_exclusion_patterns,
    save_jsonl,
    save_lf_json,
    should_keep_instance,
    tag_instance_records,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCE_DIR = Path(
    "/mnt/haoli/code/harbor-rl/haoli-memory/traj_comparison/"
    "run_swebench-verified-cc-ascend-10task-20260403230853/jobs/"
    "swebench-verified-cc-ascend-10task-20260403230853"
)
DEFAULT_IM_OUTPUT = Path(
    "/mnt/haoli/code/swe_data_process/output/20260406/"
    "cc_session_swebench_verified_10tasks.jsonl"
)
DEFAULT_LF_OUTPUT = Path(
    "/mnt/haoli/code/swe_data_process/output/20260406/"
    "cc_session_swebench_verified_10tasks.json"
)
DEFAULT_SYSTEM_PROMPT = _SCRIPT_DIR / "cc_system_prompt.json"
DEFAULT_SYSTEM_REMINDER = _SCRIPT_DIR / "cc_system_reminder.json"
DEFAULT_TOOL_DEFS = _SCRIPT_DIR / "cc_tool_definitions.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Claude Code session JSONL files to IM training data. "
            "System prompt, tool definitions, and system-reminder injections are "
            "loaded from offline JSON files (extracted once from a LiteLLM trajectory)."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Directory containing CC session files. Accepts two layouts: "
            "(1) harbor job dir with */agent/sessions/projects/-testbed/*.jsonl, "
            "(2) flat dir with *.jsonl session files directly."
        ),
    )
    parser.add_argument("--im-output", type=Path, default=DEFAULT_IM_OUTPUT)
    parser.add_argument("--lf-output", type=Path, default=DEFAULT_LF_OUTPUT)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Path to cc_system_prompt.json (offline system prompt blocks).",
    )
    parser.add_argument(
        "--system-reminder",
        type=Path,
        default=DEFAULT_SYSTEM_REMINDER,
        help="Path to cc_system_reminder.json (offline system-reminder blocks).",
    )
    parser.add_argument(
        "--tool-definitions",
        type=Path,
        default=DEFAULT_TOOL_DEFS,
        help="Path to cc_tool_definitions.json (offline tool definitions).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of session files to convert (default: unlimited).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file warnings.")
    parser.add_argument(
        "--exclude-repos-file", type=lambda s: Path(s) if s else None,
        default=EXCLUDED_REPOS_FILE,
        help="排除 repo 列表文件路径（由 generate_excluded_repos.py 生成）",
    )
    parser.add_argument(
        "--reasoning-check-mode",
        choices=("strict", "adaptive"),
        default="strict",
        help="slow 轨迹 reasoning_content 过滤模式",
    )
    parser.add_argument(
        "--reasoning-content-ratio-threshold",
        type=float,
        default=0.2,
        help="adaptive 模式下 assistant 轮次包含 reasoning_content 的最低比例",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Session file discovery
# ---------------------------------------------------------------------------

def discover_session_files(source_dir: Path) -> list[Path]:
    """
    Find CC session JSONL files under source_dir.

    Tries harbor job layout first:
      source_dir/*/agent/sessions/projects/-testbed/*.jsonl

    Falls back to flat layout:
      source_dir/*.jsonl

    Returns one Path per discovered session file, sorted.
    """
    harbor_files = sorted(
        source_dir.glob("*/agent/sessions/projects/-testbed/*.jsonl")
    )
    if harbor_files:
        return harbor_files

    flat_files = sorted(source_dir.glob("*.jsonl"))
    return flat_files


def _instance_id_from_session_path(session_path: Path, source_dir: Path) -> str:
    """Extract instance ID from a session file path.

    Harbor layout: source_dir/<instance_id>/agent/sessions/.../*.jsonl
      -> returns the <instance_id> directory name
    Flat layout: source_dir/<name>.jsonl
      -> returns the file stem
    """
    try:
        rel = session_path.relative_to(source_dir)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
    except ValueError:
        pass
    return session_path.stem


# ---------------------------------------------------------------------------
# Offline data loaders
# ---------------------------------------------------------------------------

def load_system_prompt(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = data["blocks"]
    return join_text_parts([b["text"] for b in blocks])


def load_system_reminders(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [b["text"] for b in data["blocks"]]


def load_tool_definitions(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """
    Load cc_tool_definitions.json (Anthropic format with input_schema),
    normalize to OpenAI format (with parameters), and build the
    tool-property-order map for consistent argument key ordering.
    """
    with path.open("r", encoding="utf-8") as f:
        raw_tools = json.load(f)
    normalized = [normalize_tool_definition(t) for t in raw_tools]
    order_map = build_tool_properties_order_map(normalized)
    return normalized, order_map


# ---------------------------------------------------------------------------
# Session line parsing
# ---------------------------------------------------------------------------

def parse_session_lines(session_path: Path) -> list[dict[str, Any]]:
    """
    Read session JSONL and return only user/assistant event lines.
    Skips: queue-operation, last-prompt, and any unrecognised types.
    """
    lines: list[dict[str, Any]] = []
    with session_path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] {session_path.name} line {lineno}: invalid JSON – {exc}")
                continue
            if obj.get("type") in ("user", "assistant"):
                lines.append(obj)
    return lines


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def build_first_user_content(task_text: str, system_reminder_texts: list[str]) -> str:
    return join_text_parts(system_reminder_texts + [task_text])


def build_im_messages(
    conversation_lines: list[dict[str, Any]],
    system_prompt_text: str,
    system_reminder_texts: list[str],
    tool_properties_order: dict[str, list[str]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    messages.append({"role": "system", "content": system_prompt_text})

    for line in conversation_lines:
        msg = line.get("message", {})
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                user_content = build_first_user_content(content, system_reminder_texts)
                messages.append({"role": "user", "content": user_content})
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_text = normalize_tool_content(block.get("content", ""))
                        messages.append({"role": "tool", "content": tool_text})
                    elif block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            messages.append({"role": "user", "content": text})

        elif role == "assistant":
            if isinstance(content, list):
                converted = convert_assistant_blocks(content, tool_properties_order)
                if converted:
                    messages.append(converted)
            elif isinstance(content, str) and content.strip():
                messages.append({"role": "assistant", "content": content})

    messages = remove_tool_use_error_turns(messages)
    messages = trim_trailing_tools(messages)
    normalize_assistant_empty_content(messages)

    return messages


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_one_session(
    session_path: Path,
    system_prompt_text: str,
    system_reminder_texts: list[str],
    normalized_tools: list[dict[str, Any]],
    tool_properties_order: dict[str, list[str]],
    source_dir: Path,
    reasoning_check_mode: Literal["strict", "adaptive"] = "strict",
    reasoning_content_ratio_threshold: float = 0.2,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Convert one CC session file → list of IM records (with instance tagging).
    Returns (converted_records, role_filtered, reasoning_filtered).
    """
    conversation_lines = parse_session_lines(session_path)
    if not conversation_lines:
        return [], 0, 0

    messages = build_im_messages(
        conversation_lines,
        system_prompt_text,
        system_reminder_texts,
        tool_properties_order,
    )

    role_filtered = 0
    reasoning_filtered = 0

    if not check_roles(messages):
        role_filtered += 1
        return [], role_filtered, reasoning_filtered

    think_mode = infer_think_mode(messages)
    record: dict[str, Any] = {
        "messages": messages,
        "tools": normalized_tools,
        "pseudo_turns": None,
        "think_mode": think_mode,
    }

    if not check_reasoning_content(
        messages,
        think_mode=think_mode,
        pseudo_turns=None,
        reasoning_check_mode=reasoning_check_mode,
        reasoning_content_ratio_threshold=reasoning_content_ratio_threshold,
    ):
        reasoning_filtered += 1
        return [], role_filtered, reasoning_filtered

    converted_records = [record]
    instance_id = _instance_id_from_session_path(session_path, source_dir)
    tag_instance_records(converted_records, instance_id)
    return converted_records, role_filtered, reasoning_filtered


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect_im_data(
    session_files: list[Path],
    system_prompt_text: str,
    system_reminder_texts: list[str],
    normalized_tools: list[dict[str, Any]],
    tool_properties_order: dict[str, list[str]],
    source_dir: Path,
    max_instances: int | None,
    quiet: bool,
    reasoning_check_mode: Literal["strict", "adaptive"] = "strict",
    reasoning_content_ratio_threshold: float = 0.2,
) -> tuple[list[dict[str, Any]], ProcessSummary]:
    im_data: list[dict[str, Any]] = []
    summary = ProcessSummary()
    kept = 0

    for path in tqdm(session_files, desc="Processing sessions"):
        if max_instances is not None and kept >= max_instances:
            break
        try:
            records, role_f, reason_f = process_one_session(
                path,
                system_prompt_text,
                system_reminder_texts,
                normalized_tools,
                tool_properties_order,
                source_dir,
                reasoning_check_mode=reasoning_check_mode,
                reasoning_content_ratio_threshold=reasoning_content_ratio_threshold,
            )
            summary.role_filtered += role_f
            summary.reasoning_filtered += reason_f

            if records and should_keep_instance(role_f, reason_f):
                im_data.extend(records)
                kept += 1
        except Exception as exc:
            summary.failed_instances += 1
            if not quiet:
                print(f"[WARN] {path.name} failed: {exc}")

    return im_data, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"source-dir does not exist: {args.source_dir}")

    session_files = discover_session_files(args.source_dir)
    print(f"Found {len(session_files)} session file(s) in {args.source_dir}")
    if not session_files:
        print("No session files found. Check --source-dir and directory layout.")
        return

    exclusion_patterns = load_exclusion_patterns(args.exclude_repos_file)
    if exclusion_patterns:
        session_files = filter_paths_by_repo(
            session_files,
            exclusion_patterns,
            name_func=lambda p: _instance_id_from_session_path(p, args.source_dir),
            label="cc-session",
        )

    system_prompt_text = load_system_prompt(args.system_prompt)
    system_reminder_texts = load_system_reminders(args.system_reminder)
    normalized_tools, tool_properties_order = load_tool_definitions(args.tool_definitions)

    print(f"System prompt length : {len(system_prompt_text)} chars")
    print(f"System reminders     : {len(system_reminder_texts)} blocks")
    print(f"Tool definitions     : {len(normalized_tools)} tools")

    im_data, summary = collect_im_data(
        session_files=session_files,
        system_prompt_text=system_prompt_text,
        system_reminder_texts=system_reminder_texts,
        normalized_tools=normalized_tools,
        tool_properties_order=tool_properties_order,
        source_dir=args.source_dir,
        max_instances=args.max_instances,
        quiet=args.quiet,
        reasoning_check_mode=args.reasoning_check_mode,
        reasoning_content_ratio_threshold=args.reasoning_content_ratio_threshold,
    )

    im_data = score_dataset(im_data, quiet=True)

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
