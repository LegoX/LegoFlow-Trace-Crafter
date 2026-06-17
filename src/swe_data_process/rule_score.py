#!/usr/bin/env python3
"""轨迹质量打分模块 — TQS V2 (Trajectory Quality Score).

对 IM 格式的 JSONL 轨迹数据逐条打分，支持所有脚手架类型：
  Claude Code, OpenCode, OpenHands, OpenHands SDK, Terminus2

TQS V2 最终综合分 (Fail-soft 加权):
  TQS = Σ(weight_i × transformed_component_i) / Σ(weight_i)
  仅对有数据且权重非 0 的组件求和。

最终采用的 5 个非零权重指标:
  SUB (0.33) — 提交完整性: 是否正常收尾，并结合后期错误率/后期测试
  STP (0.27) — 步数效率: assistant turn 数是否落在合理范围
  TVR (0.23) — 测试验证: 是否写测试、跑测试、最后一次测试是否成功
  FEC (0.10) — 文件编辑集中度: 平均每个文件被编辑的次数，综合分中使用 FEC^5
  DPI (0.07) — 脏模式惩罚: 截断/从未成功写入/循环/重复错误，综合分中使用 DPI^3

OEC/IAC/PED/PSN/TTE/SCP 仍会计算并输出，作为诊断指标保留；
它们当前权重为 0，不参与 composite_score。

用法:
  conda activate swelf
  python -m swe_data_process.rule_score --input <im.jsonl> [--output <scored.jsonl>] [--max-instances N]
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from swe_data_process.utils import load_jsonl, save_jsonl


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# --- TQS V2 weights ---
# Non-zero components are used by composite_score.
# Diagnostic components are still computed and emitted, but have weight 0.
# Note: _aggregate_tqs applies FEC^5 and DPI^3 before weighting.
TQS_WEIGHTS = {
    "oec": 0.00, "iac": 0.00, "dpi": 0.07,
    "ped": 0.00, "psn": 0.00, "tte": 0.00, "scp": 0.00,
    "sub": 0.33, "fec": 0.10, "stp": 0.27, "tvr": 0.23,
}

# --- OEC ---
_OEC_SCAN_LIMIT = 5000
_OEC_BASELINE_WINDOW = 5
_OEC_SMOOTH_WINDOW = 3

# --- IAC ---
_IAC_MIN_TURNS = 3

# --- DPI ---
_DPI_NEVER_COMMITTED = 0.40
_DPI_EARLY_STOP = 0.40
_DPI_LOOP_FRAC_SCALE = 3.0
_DPI_LOOP_FRAC_CAP = 0.30
_DPI_ERR_REPEAT_SCALE = 0.60
_DPI_ERR_REPEAT_CAP = 0.30
_LOOP_RUN_MIN_LEN = 5

# --- PSN ---
_PSN_WINDOW = 5

# --- TTE ---
_TTE_BUCKET_COUNT = 7

# --- SCP ---
_SCP_SWEET_LO = 0.20
_SCP_SWEET_HI = 0.50
_SCP_SIGMA = 0.20

# --- Scaffold detection ---
_OPENHANDS_SDK_TOOL_NAMES = frozenset({
    "terminal", "file_editor", "task_tracker", "finish", "think",
})
_CC_OC_TOOL_NAMES_LOWER = frozenset({
    "bash", "read", "edit", "write", "glob", "grep", "task",
    "webfetch", "websearch", "notebookedit", "todowrite", "taskoutput",
    "taskstop", "askuserquestion", "skill", "enterplanmode",
    "exitplanmode", "enterworktree",
})

# --- Error detection ---
_ERROR_SCAN_LIMIT = 3000
_ERROR_PATTERNS_HARD: list[re.Pattern[str]] = [
    re.compile(r"command not found"),
    re.compile(r"Permission denied"),
    re.compile(r"exit code[:\s]+[1-9]", re.IGNORECASE),
    re.compile(r"returned non-zero exit status"),
    re.compile(r"<tool_use_error>"),
    re.compile(r"The arguments provided to the tool are invalid"),
]
_ERROR_PATTERNS_SOFT: list[re.Pattern[str]] = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(
        r"\b(?:SyntaxError|TypeError|ValueError|KeyError|IndexError"
        r"|AttributeError|ImportError|ModuleNotFoundError|FileNotFoundError"
        r"|NameError|RuntimeError|OSError|IOError|PermissionError"
        r"|ZeroDivisionError|NotImplementedError|StopIteration"
        r"|RecursionError|AssertionError|UnicodeDecodeError)\b"
    ),
    re.compile(r"No such file or directory"),
    re.compile(r"\bFAILED\b"),
]

# --- Tool classification ---
_PURE_EDIT_TOOL_NAMES = frozenset({"edit", "write"})
_MULTI_EDITOR_TOOL_NAMES = frozenset({"file_editor", "str_replace_editor"})
_EDITOR_WRITE_COMMANDS = frozenset({"str_replace", "create", "insert"})
_EDIT_TOOL_NAMES = _PURE_EDIT_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES
_BASH_TOOL_NAMES = frozenset({"bash", "terminal", "execute_bash"})
_FILE_PATH_KEYS = ("file_path", "filePath", "path")

# --- Test detection ---
_TEST_RUN_RE = re.compile(
    r"\b(?:"
    r"pytest|py\.test|python\s+-m\s+pytest|python\s+-m\s+unittest"
    r"|unittest|python\s+test_|nosetests"
    r"|go\s+test|cargo\s+test|ctest|gtest_filter"
    r"|mvn\s+(?:test|verify|surefire)|gradle\s+test|gradlew\s+test"
    r"|jest|mocha|vitest|npx\s+jest|npx\s+vitest|npx\s+mocha"
    r"|(?:npm|yarn|pnpm)\s+(?:run\s+)?test"
    r"|make\s+(?:test|check)"
    r")\b"
    r"|\.\/[^\s]*(?:_test|test_)\S*",
    re.IGNORECASE,
)

# --- Bash edit path extraction ---
_WORKSPACE_PREFIXES = ("/workspace/", "/testbed/", "/repo/", "/home/swe-bench/")
_REDIRECT_WRITE_RE = re.compile(r">>?\s*(\S+)")
_CAT_REDIRECT_RE = re.compile(r"\bcat\s[^|]*>>?\s*(\S+)")
_TEE_RE = re.compile(r"\btee\s+(?:-[a-zA-Z]+\s+)*(\S+)")
_SED_INPLACE_RE = re.compile(r"\bsed\s+-i\b")
_PATCH_FILE_RE = re.compile(r"\bpatch\s+(?:-\S+\s+)*(\S+)")

# --- IAC intent keywords ---
_INTENT_KEYWORDS = {
    "read":   {"read", "look", "check", "examine", "open", "view", "inspect", "show"},
    "write":  {"write", "edit", "modify", "fix", "change", "update", "patch", "add", "insert", "replace"},
    "bash":   {"run", "execute", "test", "try", "install", "pip", "npm", "build"},
    "search": {"search", "find", "grep", "locate", "list", "explore"},
    "submit": {"finish", "submit", "complete", "done"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold detection
# ═══════════════════════════════════════════════════════════════════════════

SCAFFOLD_TYPES = ("claudecode", "opencode", "openhands", "openhands_sdk", "terminus2")


def detect_scaffold(record: dict[str, Any]) -> str:
    """根据 IM 记录的结构自动检测脚手架类型。"""
    tools = record.get("tools")

    if tools is None:
        return "terminus2"

    tool_names: set[str] = set()
    if isinstance(tools, list):
        for t in tools:
            func = t.get("function", {}) if isinstance(t, dict) else {}
            name = func.get("name", "")
            if name:
                tool_names.add(name)

    if not tool_names:
        return "terminus2"

    if tool_names <= _OPENHANDS_SDK_TOOL_NAMES:
        return "openhands_sdk"

    tool_names_lower = {n.lower() for n in tool_names}
    if tool_names_lower & _CC_OC_TOOL_NAMES_LOWER:
        messages = record.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                head = (msg.get("content") or "")[:500].lower()
                if "claude code" in head or "anthropic" in head:
                    return "claudecode"
                break
        return "opencode"

    return "openhands"


# ═══════════════════════════════════════════════════════════════════════════
# Shared utility functions
# ═══════════════════════════════════════════════════════════════════════════

def _get_file_path(args: dict) -> str:
    """从 tool_call arguments 中提取文件路径。"""
    for key in _FILE_PATH_KEYS:
        val = args.get(key)
        if val:
            return val
    return ""


def _is_write_operation(name_lower: str, args: dict) -> bool:
    """判断一个 tool_call 是否为文件写操作。"""
    if name_lower in _PURE_EDIT_TOOL_NAMES:
        return True
    if name_lower in _MULTI_EDITOR_TOOL_NAMES:
        return args.get("command", "") in _EDITOR_WRITE_COMMANDS
    return False


def _parse_tool_call(tc: Any) -> tuple[str, dict]:
    """从 tool_call dict 中提取 (name_lower, parsed_args)。"""
    func = tc.get("function", {}) if isinstance(tc, dict) else {}
    name = (func.get("name") or "").lower()
    args = func.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    if not isinstance(args, dict):
        args = {}
    return name, args


def _normalize_path(p: str) -> str:
    """归一化文件路径。"""
    for prefix in _WORKSPACE_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return posixpath.normpath(p)


def _extract_bash_edit_paths(cmd_str: str) -> list[str]:
    """从 bash 命令字符串中提取文件编辑目标路径。"""
    paths: list[str] = []
    m = _CAT_REDIRECT_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))
    if not m and (">" in cmd_str):
        if re.search(r"\b(?:echo|printf)\b", cmd_str):
            rm = _REDIRECT_WRITE_RE.search(cmd_str)
            if rm:
                paths.append(rm.group(1))
    m = _TEE_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))
    if _SED_INPLACE_RE.search(cmd_str):
        after_sed = _SED_INPLACE_RE.split(cmd_str, 1)[-1].strip()
        tokens = after_sed.split()
        skip_next = False
        in_expr = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if tok.startswith("-"):
                if tok in ("-e", "-E"):
                    skip_next = True
                continue
            if not in_expr and (tok.startswith("'") or tok.startswith('"') or tok.startswith("s")):
                in_expr = True
                continue
            if in_expr:
                paths.append(tok)
    m = _PATCH_FILE_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))
    return paths


def _count_assistant_turns(messages: list[dict]) -> int:
    """统计 assistant turns 数量。"""
    return sum(1 for m in messages if m.get("role") == "assistant")


# ═══════════════════════════════════════════════════════════════════════════
# Action / Observation extraction
# ═══════════════════════════════════════════════════════════════════════════

def _strip_think_tags(content: str) -> str:
    """去除 <think>...</think> 标签。"""
    if not content:
        return ""
    text = content.strip()
    if text.startswith("<think>") and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _parse_t2_assistant(msg: dict) -> dict | None:
    """解析 Terminus2 assistant 消息的 JSON 内容。"""
    content = msg.get("content", "")
    json_text = _strip_think_tags(content)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_tool_call_actions(messages: list[dict]) -> list[dict[str, Any]]:
    """从 tool-call 脚手架提取 actions。"""
    actions: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = func.get("name", "")
            if name:
                actions.append({"tool_name": name, "msg_idx": idx})
    return actions


def _extract_terminus2_actions(messages: list[dict]) -> list[dict[str, Any]]:
    """从 Terminus2 脚手架提取 actions。"""
    actions: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        parsed = _parse_t2_assistant(msg)
        if parsed is None:
            continue
        for cmd in parsed.get("commands", []) or []:
            if not isinstance(cmd, dict):
                continue
            keystrokes = (cmd.get("keystrokes") or "").strip()
            if not keystrokes:
                continue
            parts = keystrokes.split()
            tool_name = parts[0] if parts else keystrokes
            tool_name = tool_name.rsplit("/", 1)[-1]
            actions.append({"tool_name": tool_name, "msg_idx": idx})
    return actions


def _extract_actions(messages: list[dict], scaffold: str) -> list[dict[str, Any]]:
    """统一接口：提取 actions。"""
    if scaffold == "terminus2":
        return _extract_terminus2_actions(messages)
    return _extract_tool_call_actions(messages)


def _extract_observations(
    messages: list[dict],
    scaffold: str,
) -> list[dict[str, Any]]:
    """提取 observations。"""
    observations: list[dict[str, Any]] = []

    if scaffold == "terminus2":
        for idx in range(1, len(messages)):
            msg = messages[idx]
            if msg.get("role") == "user" and messages[idx - 1].get("role") == "assistant":
                observations.append({
                    "content": msg.get("content", ""),
                    "msg_idx": idx,
                    "prev_assistant_idx": idx - 1,
                    "tool_call_position": None,
                })
    else:
        _position_counter: dict[int, int] = {}
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            prev_assistant_idx = None
            for j in range(idx - 1, -1, -1):
                if messages[j].get("role") == "assistant":
                    prev_assistant_idx = j
                    break
            pos = 0
            if prev_assistant_idx is not None:
                pos = _position_counter.get(prev_assistant_idx, 0)
                _position_counter[prev_assistant_idx] = pos + 1
            observations.append({
                "content": msg.get("content", ""),
                "msg_idx": idx,
                "prev_assistant_idx": prev_assistant_idx,
                "tool_call_position": pos,
            })

    return observations


# ═══════════════════════════════════════════════════════════════════════════
# Error detection
# ═══════════════════════════════════════════════════════════════════════════

def _is_error_result(content: str, is_test_output: bool = False) -> bool:
    """检测 observation 是否包含错误信息。"""
    if not content:
        return False
    text = content[:_ERROR_SCAN_LIMIT]
    for pat in _ERROR_PATTERNS_HARD:
        if pat.search(text):
            return True
    if not is_test_output:
        for pat in _ERROR_PATTERNS_SOFT:
            if pat.search(text):
                return True
    return False


def _is_test_running_turn(msg: dict, scaffold: str) -> bool:
    """判断一个 assistant turn 是否包含测试执行命令。"""
    if scaffold == "terminus2":
        parsed = _parse_t2_assistant(msg)
        if parsed is None:
            return False
        for cmd in parsed.get("commands", []) or []:
            if isinstance(cmd, dict) and _TEST_RUN_RE.search(cmd.get("keystrokes", "")):
                return True
        return False

    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        if name_lower in _BASH_TOOL_NAMES:
            if _TEST_RUN_RE.search(args.get("command", "")):
                return True
    return False


def _get_per_toolcall_test_flags(msg: dict) -> list[bool]:
    """返回 assistant 消息中每个 tool_call 是否为测试执行命令。"""
    flags: list[bool] = []
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        is_test = False
        if name_lower in _BASH_TOOL_NAMES:
            is_test = bool(_TEST_RUN_RE.search(args.get("command", "")))
        flags.append(is_test)
    return flags


def _resolve_is_test_for_obs(obs: dict, messages: list[dict], scaffold: str, _test_flags_cache: dict) -> bool:
    """复用逐 tool_call test-output 判定。"""
    prev_idx = obs.get("prev_assistant_idx")
    if prev_idx is None:
        return False
    if scaffold == "terminus2":
        return _is_test_running_turn(messages[prev_idx], scaffold)
    if prev_idx not in _test_flags_cache:
        _test_flags_cache[prev_idx] = _get_per_toolcall_test_flags(messages[prev_idx])
    flags = _test_flags_cache[prev_idx]
    pos = obs.get("tool_call_position", 0)
    return flags[pos] if pos < len(flags) else False


def _is_obs_error(obs: dict, messages: list[dict], scaffold: str, _test_flags_cache: dict) -> bool:
    """判断 observation 是否为错误（排除测试输出误判）。"""
    is_test = _resolve_is_test_for_obs(obs, messages, scaffold, _test_flags_cache)
    return _is_error_result(obs.get("content", ""), is_test_output=is_test)


# ═══════════════════════════════════════════════════════════════════════════
# Action classification (shared by IAC, TTE, PED)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_action(name_lower: str, args: dict) -> str:
    """把 tool_call 映射到 7 类标准 action_type。"""
    if name_lower in ("think",):
        return "think"
    if name_lower in ("finish",):
        return "submit"
    if name_lower in _BASH_TOOL_NAMES:
        return "bash"
    if name_lower in ("grep", "glob", "search", "find", "websearch"):
        return "search"
    if name_lower in ("read", "webfetch"):
        return "read"
    if name_lower in _PURE_EDIT_TOOL_NAMES:
        return "write"
    if name_lower in _MULTI_EDITOR_TOOL_NAMES:
        cmd = args.get("command", "")
        if cmd == "view":
            return "read"
        if cmd in _EDITOR_WRITE_COMMANDS:
            return "write"
        return "read"
    return "other"


def _classify_action_from_action(a: dict, messages: list[dict]) -> str:
    """从 actions 列表元素获取 7 类 action_type。"""
    msg = messages[a["msg_idx"]]
    tool_name_lower = a["tool_name"].lower()
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        if name_lower == tool_name_lower:
            return _classify_action(name_lower, args)
    return _classify_action(tool_name_lower, {})


def _action_target_files(a: dict, messages: list[dict]) -> set[str]:
    """获取 action 操作的目标文件集合。"""
    msg = messages[a["msg_idx"]]
    tool_name_lower = a["tool_name"].lower()
    files: set[str] = set()

    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg.get("tool_calls", []) or []:
            name_lower, args = _parse_tool_call(tc)
            if name_lower == tool_name_lower:
                path = _get_file_path(args)
                if path:
                    files.add(_normalize_path(path))
                if name_lower in _BASH_TOOL_NAMES:
                    cmd = args.get("command", "")
                    for p in _extract_bash_edit_paths(cmd):
                        files.add(_normalize_path(p))
                break
    elif msg.get("role") == "assistant":
        parsed = _parse_t2_assistant(msg)
        if parsed:
            for cmd in parsed.get("commands", []) or []:
                if isinstance(cmd, dict):
                    ks = cmd.get("keystrokes", "")
                    for p in _extract_bash_edit_paths(ks):
                        files.add(_normalize_path(p))
    return files


# ═══════════════════════════════════════════════════════════════════════════
# 1. OEC — 观察熵坍缩
# ═══════════════════════════════════════════════════════════════════════════

def _char_entropy(text: str) -> float:
    if not text:
        return 0.0
    text = text[:_OEC_SCAN_LIMIT]
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _compute_oec(observations: list[dict]) -> float | None:
    contents = [o.get("content") or "" for o in observations]
    if len(contents) < _OEC_BASELINE_WINDOW:
        return None
    entropies = [_char_entropy(c) for c in contents]
    baseline = sum(entropies[:_OEC_BASELINE_WINDOW]) / _OEC_BASELINE_WINDOW
    if baseline <= 0:
        return None
    smoothed = []
    for i in range(len(entropies)):
        lo = max(0, i - 1)
        hi = min(len(entropies), i + 2)
        smoothed.append(sum(entropies[lo:hi]) / (hi - lo))
    rel = [s / baseline for s in smoothed]
    min_rel = min(rel)
    # min_rel 越低 → 坍缩越严重 → OEC 分数越低
    return max(0.0, min(min_rel, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# 2. IAC — 意图-行动一致性
# ═══════════════════════════════════════════════════════════════════════════

def _compute_iac(messages: list[dict], scaffold: str) -> float | None:
    matches = total = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        thought = ((msg.get("reasoning_content") or "") + " " + (msg.get("content") or "")).lower()
        if not thought.strip():
            continue

        if scaffold == "terminus2":
            parsed = _parse_t2_assistant(msg)
            action_type = "bash" if parsed and parsed.get("commands") else "other"
            if parsed and parsed.get("task_complete"):
                action_type = "submit"
        else:
            action_type = "other"
            for tc in msg.get("tool_calls") or []:
                name, args = _parse_tool_call(tc)
                cls = _classify_action(name, args)
                if cls not in ("other", "think"):
                    action_type = cls
                    break

        if action_type in ("other", "think"):
            continue
        kws = _INTENT_KEYWORDS.get(action_type, set())
        if any(kw in thought for kw in kws):
            matches += 1
        total += 1

    if total < _IAC_MIN_TURNS:
        return None
    return matches / total


# ═══════════════════════════════════════════════════════════════════════════
# 3. DPI — 脏模式惩罚
# ═══════════════════════════════════════════════════════════════════════════

def _has_successful_write(messages: list[dict], observations: list[dict], scaffold: str) -> bool:
    """检查是否有至少一次成功的写操作。"""
    _test_flags_cache: dict[int, list[bool]] = {}
    if scaffold == "terminus2":
        for obs in observations:
            prev_idx = obs.get("prev_assistant_idx")
            if prev_idx is None:
                continue
            prev_msg = messages[prev_idx]
            parsed = _parse_t2_assistant(prev_msg)
            if parsed is None:
                continue
            has_write = False
            for cmd in parsed.get("commands", []) or []:
                if isinstance(cmd, dict):
                    ks = cmd.get("keystrokes", "")
                    if _extract_bash_edit_paths(ks):
                        has_write = True
                        break
            if has_write and not _is_obs_error(obs, messages, scaffold, _test_flags_cache):
                return True
    else:
        obs_by_assistant: dict[int, list[dict]] = {}
        for obs in observations:
            if obs.get("prev_assistant_idx") is not None:
                obs_by_assistant.setdefault(obs["prev_assistant_idx"], []).append(obs)

        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for i, tc in enumerate(msg.get("tool_calls", []) or []):
                name_lower, args = _parse_tool_call(tc)
                if _is_write_operation(name_lower, args) or (
                    name_lower in _BASH_TOOL_NAMES and _extract_bash_edit_paths(args.get("command", ""))
                ):
                    obs_list = obs_by_assistant.get(idx, [])
                    if i < len(obs_list):
                        obs = obs_list[i]
                        if not _is_obs_error(obs, messages, scaffold, _test_flags_cache):
                            return True
    return False


def _is_truncated(messages: list[dict], scaffold: str) -> bool:
    """判断轨迹是否被截断（非正常结束）。"""
    if not messages:
        return True

    last_msg = messages[-1]
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        return True

    if scaffold in ("openhands_sdk", "openhands"):
        for tc in last_assistant.get("tool_calls", []) or []:
            name_lower, _ = _parse_tool_call(tc)
            if name_lower == "finish":
                return False
        if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
            return False
        return True

    if scaffold == "terminus2":
        parsed = _parse_t2_assistant(last_assistant)
        if parsed is not None and parsed.get("task_complete") is True:
            return False
        return True

    if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
        return False
    if last_msg.get("role") == "tool":
        return True
    return True


def _compute_loop_fraction(actions: list[dict], messages: list[dict], scaffold: str) -> float:
    """计算连续重复步骤占比。

    只有 tool+target 都相同且 target 非空的连续 run 才计为 loop。
    bash/terminal 无明确 target 时不参与 loop 检测（连续执行命令是正常行为）。
    """
    if len(actions) < _LOOP_RUN_MIN_LEN:
        return 0.0

    signatures = []
    for a in actions:
        tool_name = a["tool_name"].lower()
        files = _action_target_files(a, messages)
        target = sorted(files)[0] if files else ""
        # 无明确 target 的 action 用 None 标记，不参与 loop 匹配
        if not target:
            signatures.append(None)
        else:
            signatures.append((tool_name, target))

    loop_steps = 0
    i = 0
    while i < len(signatures):
        if signatures[i] is None:
            i += 1
            continue
        j = i + 1
        while j < len(signatures) and signatures[j] == signatures[i]:
            j += 1
        run_len = j - i
        if run_len >= _LOOP_RUN_MIN_LEN:
            loop_steps += run_len
        i = j

    n_with_target = sum(1 for s in signatures if s is not None)
    return loop_steps / n_with_target if n_with_target > 0 else 0.0


def _compute_error_repeat_rate(observations: list[dict], messages: list[dict], scaffold: str) -> float:
    """计算相邻 observation 重复错误的比率。"""
    _test_flags_cache: dict[int, list[bool]] = {}
    error_count = 0
    repeat_pairs = 0

    prev_is_error = False
    prev_signature = ""

    for obs in observations:
        is_err = _is_obs_error(obs, messages, scaffold, _test_flags_cache)
        if is_err:
            error_count += 1
            content = obs.get("content", "")[:80]
            sig = content
            if prev_is_error and sig == prev_signature:
                repeat_pairs += 1
            prev_signature = sig
        else:
            prev_signature = ""
        prev_is_error = is_err

    if error_count == 0:
        return 0.0
    return repeat_pairs / error_count


def _compute_dpi(messages: list[dict], actions: list[dict], observations: list[dict], scaffold: str) -> float:
    penalty = 0.0

    if not _has_successful_write(messages, observations, scaffold):
        penalty += _DPI_NEVER_COMMITTED

    if _is_truncated(messages, scaffold):
        penalty += _DPI_EARLY_STOP

    loop_frac = _compute_loop_fraction(actions, messages, scaffold)
    penalty += min(_DPI_LOOP_FRAC_CAP, loop_frac * _DPI_LOOP_FRAC_SCALE)

    err_repeat = _compute_error_repeat_rate(observations, messages, scaffold)
    penalty += min(_DPI_ERR_REPEAT_CAP, err_repeat * _DPI_ERR_REPEAT_SCALE)

    return max(0.0, 1.0 - penalty)


# ═══════════════════════════════════════════════════════════════════════════
# 4. PED — 错误后策略多样性
# ═══════════════════════════════════════════════════════════════════════════

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _action_for_observation(obs: dict, actions: list[dict], messages: list[dict], scaffold: str) -> dict | None:
    """找到产生该 observation 的 action。"""
    prev_idx = obs.get("prev_assistant_idx")
    if prev_idx is None:
        return None
    pos = obs.get("tool_call_position") or 0
    matching = [a for a in actions if a["msg_idx"] == prev_idx]
    if pos < len(matching):
        return matching[pos]
    return matching[0] if matching else None


def _next_action_after_msg(actions: list[dict], msg_idx: int) -> dict | None:
    """找到 msg_idx 之后的第一个 action。"""
    for a in actions:
        if a["msg_idx"] > msg_idx:
            return a
    return None


def _compute_ped(actions: list[dict], observations: list[dict], messages: list[dict], scaffold: str) -> float | None:
    diversities: list[float] = []
    _test_flags_cache: dict[int, list[bool]] = {}

    for obs in observations:
        if not _is_obs_error(obs, messages, scaffold, _test_flags_cache):
            continue
        cur_action = _action_for_observation(obs, actions, messages, scaffold)
        nxt_action = _next_action_after_msg(actions, obs["msg_idx"])
        if cur_action is None or nxt_action is None:
            continue

        type_changed = 0 if cur_action["tool_name"].lower() == nxt_action["tool_name"].lower() else 1
        cur_files = _action_target_files(cur_action, messages)
        nxt_files = _action_target_files(nxt_action, messages)
        file_changed = 1.0 - _jaccard(cur_files, nxt_files)
        diversities.append((type_changed + file_changed) / 2.0)

    if not diversities:
        return None
    return sum(diversities) / len(diversities)


# ═══════════════════════════════════════════════════════════════════════════
# 5. PSN — 渐进式范围收窄
# ═══════════════════════════════════════════════════════════════════════════

def _per_step_target_files(messages: list[dict], scaffold: str) -> list[set[str]]:
    """每个 assistant turn 的 target_files 集合。"""
    per_step: list[set[str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        files: set[str] = set()
        if scaffold == "terminus2":
            parsed = _parse_t2_assistant(msg)
            if parsed:
                for cmd in parsed.get("commands", []) or []:
                    if isinstance(cmd, dict):
                        ks = cmd.get("keystrokes", "")
                        for p in _extract_bash_edit_paths(ks):
                            files.add(_normalize_path(p))
        else:
            for tc in msg.get("tool_calls", []) or []:
                name_lower, args = _parse_tool_call(tc)
                path = _get_file_path(args)
                if path:
                    files.add(_normalize_path(path))
                if name_lower in _BASH_TOOL_NAMES:
                    for p in _extract_bash_edit_paths(args.get("command", "")):
                        files.add(_normalize_path(p))
        per_step.append(files)
    return per_step


def _compute_psn(messages: list[dict], scaffold: str, window: int = _PSN_WINDOW) -> float | None:
    per_step_files = _per_step_target_files(messages, scaffold)
    if len(per_step_files) < 2 * window:
        return None

    active_scope: list[int] = []
    for t in range(len(per_step_files)):
        union: set[str] = set()
        for f in per_step_files[max(0, t - window + 1):t + 1]:
            union |= f
        active_scope.append(len(union))

    mid = len(active_scope) // 2
    late = active_scope[mid:]
    if len(late) < 3:
        return None

    n = len(late)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = late[j] - late[i]
            if d > 0:
                concordant += 1
            elif d < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.5
    tau = (concordant - discordant) / total
    psn_late = -tau
    return (psn_late + 1.0) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. TTE — 工具转移熵
# ═══════════════════════════════════════════════════════════════════════════

def _compute_tte(actions: list[dict], messages: list[dict]) -> float | None:
    if len(actions) < 3:
        return None
    types = [_classify_action_from_action(a, messages) for a in actions]
    transitions = Counter(zip(types[:-1], types[1:]))
    total = sum(transitions.values())
    if total == 0:
        return None
    h = -sum((c / total) * math.log2(c / total) for c in transitions.values())
    return min(1.0, h / math.log2(_TTE_BUCKET_COUNT * _TTE_BUCKET_COUNT))


# ═══════════════════════════════════════════════════════════════════════════
# 7. SCP — 首次有效编辑时机
# ═══════════════════════════════════════════════════════════════════════════

def _first_successful_write_turn(messages: list[dict], observations: list[dict], scaffold: str) -> int | None:
    """返回首个成功写操作的 1-based assistant turn rank。"""
    _test_flags_cache: dict[int, list[bool]] = {}
    obs_by_assistant: dict[int, list[dict]] = {}
    for obs in observations:
        if obs.get("prev_assistant_idx") is not None:
            obs_by_assistant.setdefault(obs["prev_assistant_idx"], []).append(obs)

    turn_rank = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        turn_rank += 1

        if scaffold == "terminus2":
            parsed = _parse_t2_assistant(msg)
            if parsed is None:
                continue
            has_write = False
            for cmd in parsed.get("commands", []) or []:
                if isinstance(cmd, dict):
                    ks = cmd.get("keystrokes", "")
                    if _extract_bash_edit_paths(ks):
                        has_write = True
                        break
            if has_write:
                obs_list = obs_by_assistant.get(idx, [])
                if obs_list and not _is_obs_error(obs_list[0], messages, scaffold, _test_flags_cache):
                    return turn_rank
        else:
            for i, tc in enumerate(msg.get("tool_calls", []) or []):
                name_lower, args = _parse_tool_call(tc)
                if _is_write_operation(name_lower, args) or (
                    name_lower in _BASH_TOOL_NAMES and _extract_bash_edit_paths(args.get("command", ""))
                ):
                    obs_list = obs_by_assistant.get(idx, [])
                    if i < len(obs_list):
                        if not _is_obs_error(obs_list[i], messages, scaffold, _test_flags_cache):
                            return turn_rank
    return None


def _compute_scp(messages: list[dict], observations: list[dict], scaffold: str) -> float:
    total_turns = _count_assistant_turns(messages)
    if total_turns == 0:
        return 0.0
    first_turn = _first_successful_write_turn(messages, observations, scaffold)
    if first_turn is None:
        return 0.0
    scp = first_turn / total_turns
    if _SCP_SWEET_LO <= scp <= _SCP_SWEET_HI:
        return 1.0
    dist = min(abs(scp - _SCP_SWEET_LO), abs(scp - _SCP_SWEET_HI))
    return math.exp(-(dist ** 2) / (2 * _SCP_SIGMA ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# 8. SUB — 提交完整性 (from v5 D1, r=0.33 with resolved)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_sub(messages: list[dict], observations: list[dict], scaffold: str) -> float:
    """提交完整性 + 最终状态质量。

    Base: 正常提交=1.0, 截断=0.0, 半成品=0.5
    Penalty: finish without any successful write = 0.3 (premature finish)
    Modifiers: late test execution, low late-error rate.
    """
    if not messages:
        return 0.0

    last_assistant = None
    last_msg = messages[-1]
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        return 0.0

    base = 0.0
    if scaffold in ("openhands_sdk", "openhands"):
        for tc in last_assistant.get("tool_calls", []) or []:
            name_lower, _ = _parse_tool_call(tc)
            if name_lower == "finish":
                base = 1.0
                break
        if base == 0.0:
            if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
                base = 0.5
    elif scaffold == "terminus2":
        parsed = _parse_t2_assistant(last_assistant)
        if parsed is not None and parsed.get("task_complete") is True:
            base = 1.0
    else:
        if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
            base = 1.0
        elif last_msg.get("role") == "tool":
            base = 0.0

    # Premature finish penalty: called finish but never successfully wrote
    # Disabled: _has_successful_write has false negatives on resolved instances
    # if base == 1.0 and not _has_successful_write(messages, observations, scaffold):
    #     base = 0.3

    # Late-trajectory quality: error rate in last 30% of observations
    if observations and base > 0:
        n_obs = len(observations)
        late_start = int(n_obs * 0.7)
        late_obs = observations[late_start:]
        if late_obs:
            _cache: dict = {}
            n_late_err = sum(1 for o in late_obs if _is_obs_error(o, messages, scaffold, _cache))
            late_success_rate = 1.0 - n_late_err / len(late_obs)
            base = base * (0.7 + 0.3 * late_success_rate)

    # Late test execution bonus
    n_assistant = _count_assistant_turns(messages)
    late_start_turn = max(0, int(n_assistant * 0.8))
    turn_idx = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            turn_idx += 1
            if turn_idx >= late_start_turn:
                if _is_test_running_turn(msg, scaffold):
                    base = min(1.0, base + 0.15)
                    break

    return min(1.0, base)


# ═══════════════════════════════════════════════════════════════════════════
# 9. FEC — 文件编辑集中度 (from v5 E1, r=0.17 with resolved)
# ═══════════════════════════════════════════════════════════════════════════

_FEC_THRESHOLD = 5


def _extract_edit_paths(messages: list[dict], scaffold: str) -> list[str]:
    """提取所有文件修改操作的目标路径列表（含重复），路径已归一化。"""
    paths: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if scaffold == "terminus2":
            parsed = _parse_t2_assistant(msg)
            if parsed is None:
                continue
            for cmd in parsed.get("commands", []) or []:
                if not isinstance(cmd, dict):
                    continue
                ks = cmd.get("keystrokes", "")
                paths.extend(_normalize_path(p) for p in _extract_bash_edit_paths(ks))
        else:
            for tc in msg.get("tool_calls", []) or []:
                name_lower, args = _parse_tool_call(tc)
                if _is_write_operation(name_lower, args):
                    path = _get_file_path(args)
                    if path:
                        paths.append(_normalize_path(path))
                elif name_lower in _BASH_TOOL_NAMES:
                    cmd_str = args.get("command", "")
                    paths.extend(_normalize_path(p) for p in _extract_bash_edit_paths(cmd_str))
    return paths


def _compute_fec(messages: list[dict], scaffold: str) -> float:
    """文件编辑集中度: 1 - clip((mean_edits - 1) / 4, 0, 1)。"""
    paths = _extract_edit_paths(messages, scaffold)
    if not paths:
        return 1.0
    file_counts = Counter(paths)
    unique_files = len(file_counts)
    if unique_files == 0:
        return 1.0
    mean_edits = len(paths) / unique_files
    normalized = max(0.0, min((mean_edits - 1) / (_FEC_THRESHOLD - 1), 1.0))
    return 1.0 - normalized


# ═══════════════════════════════════════════════════════════════════════════
# 10. STP — 步数效率 (r=-0.21 with resolved: fewer steps = better)
# ═══════════════════════════════════════════════════════════════════════════

_STP_OPTIMAL_RANGE = (5, 30)
_STP_MAX_STEPS = 90


def _compute_stp(messages: list[dict]) -> float:
    """步数效率: 在最优范围内得满分，超出范围递减。

    Uses quadratic decay (faster than linear) to penalize long trajectories more.
    """
    turns = _count_assistant_turns(messages)
    if turns == 0:
        return 0.0
    lo, hi = _STP_OPTIMAL_RANGE
    if lo <= turns <= hi:
        return 1.0
    if turns < lo:
        return turns / lo
    if turns >= _STP_MAX_STEPS:
        return 0.0
    # Quadratic decay: penalizes long trajectories more aggressively
    linear = 1.0 - (turns - hi) / (_STP_MAX_STEPS - hi)
    return linear ** 1.5


# ═══════════════════════════════════════════════════════════════════════════
# 11. TVR — 测试验证 (test writing + running correlates with success)
# ═══════════════════════════════════════════════════════════════════════════

_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    r"test_[^/]+\.py|[^/]+_test\.py|tests/[^/]+\.py|conftest\.py"
    r"|[^/]+_test\.go|tests/[^/]+\.rs|test_[^/]+\.rs"
    r"|[^/]+Tests?\.java|[^/]+IT\.java"
    r"|[^/]+\.(?:test|spec)\.(?:js|ts|jsx|tsx|mjs|cjs)"
    r")$",
    re.IGNORECASE,
)


def _compute_tvr(messages: list[dict], scaffold: str) -> float:
    """测试验证: 综合测试写入、执行次数和最终测试通过信号。

    0.3 * has_test_write + 0.3 * has_test_run + 0.4 * late_test_success
    late_test_success: 最后一次测试执行的 observation 不含错误
    """
    has_test_write = False
    test_run_count = 0
    last_test_obs_idx = -1

    for idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if scaffold == "terminus2":
            parsed = _parse_t2_assistant(msg)
            if parsed is None:
                continue
            for cmd in parsed.get("commands", []) or []:
                if not isinstance(cmd, dict):
                    continue
                ks = cmd.get("keystrokes", "")
                if _TEST_RUN_RE.search(ks):
                    test_run_count += 1
                    # Find next user msg as observation
                    for j in range(idx + 1, len(messages)):
                        if messages[j].get("role") == "user":
                            last_test_obs_idx = j
                            break
                for edited_path in _extract_bash_edit_paths(ks):
                    if _TEST_FILE_RE.search(edited_path):
                        has_test_write = True
        else:
            for tc_idx, tc in enumerate(msg.get("tool_calls", []) or []):
                name_lower, args = _parse_tool_call(tc)
                if _is_write_operation(name_lower, args):
                    path = _get_file_path(args)
                    if _TEST_FILE_RE.search(path):
                        has_test_write = True
                if name_lower in _BASH_TOOL_NAMES:
                    cmd = args.get("command", "")
                    if _TEST_RUN_RE.search(cmd):
                        test_run_count += 1
                        # Find corresponding tool response
                        tool_count = 0
                        for j in range(idx + 1, len(messages)):
                            if messages[j].get("role") == "tool":
                                if tool_count == tc_idx:
                                    last_test_obs_idx = j
                                    break
                                tool_count += 1
                            elif messages[j].get("role") == "assistant":
                                break

    has_test_run = test_run_count > 0
    # Late test success: last test observation doesn't contain errors
    late_test_success = 0.0
    if last_test_obs_idx >= 0:
        content = messages[last_test_obs_idx].get("content", "")
        if not _is_error_result(content, is_test_output=False):
            late_test_success = 1.0
        else:
            late_test_success = 0.3  # ran tests but they failed

    return (0.3 * float(has_test_write)
            + 0.3 * float(has_test_run)
            + 0.4 * late_test_success)


# ═══════════════════════════════════════════════════════════════════════════
# 12. LER — 后期错误率 (late error rate: fewer errors near end = better)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_ler(observations: list[dict], messages: list[dict], scaffold: str) -> float | None:
    """后期错误率: 轨迹后 40% 的 observation 中无错误的比例。

    Resolved 实例在后期应该错误更少（已找到正确方案）。
    """
    if len(observations) < 5:
        return None
    late_start = int(len(observations) * 0.6)
    late_obs = observations[late_start:]
    if not late_obs:
        return None
    _cache: dict = {}
    n_success = sum(1 for o in late_obs if not _is_obs_error(o, messages, scaffold, _cache))
    return n_success / len(late_obs)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation and scoring
# ═══════════════════════════════════════════════════════════════════════════

def _round_or_none(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v, 4)


def _desaturate(v: float, power: float = 3.0) -> float:
    """Apply power transform to spread saturated-high distributions.

    For values clustered near 1.0, v^power spreads them toward 0.
    E.g., [0.85, 0.90, 0.95, 1.0] -> [0.61, 0.73, 0.86, 1.0] with power=3.
    """
    return v ** power


def _sigmoid_stretch(x: float, center: float = 0.75, steepness: float = 8.0) -> float:
    """Apply sigmoid stretch centered at `center` to amplify discrimination."""
    z = steepness * (x - center)
    sig = 1.0 / (1.0 + math.exp(-z))
    # Normalize so that 0->~0 and 1->~1
    sig_0 = 1.0 / (1.0 + math.exp(steepness * center))
    sig_1 = 1.0 / (1.0 + math.exp(-steepness * (1.0 - center)))
    return (sig - sig_0) / (sig_1 - sig_0)


def _aggregate_tqs(components: dict[str, float | None]) -> float:
    num = den = 0.0
    for name, w in TQS_WEIGHTS.items():
        if w == 0:
            continue
        v = components.get(name)
        if v is None:
            continue
        # Desaturate components that cluster near 1.0
        if name in ("oec", "iac", "scp", "fec"):
            v = _desaturate(v, power=5.0)
        elif name == "dpi":
            v = _desaturate(v, power=3.0)
        num += w * v
        den += w
    if den == 0:
        return 0.0
    raw = num / den
    return raw


def score_record(
    record: dict[str, Any],
    median_steps: float | str | None = None,
    scaffold_override: str | None = None,
) -> dict[str, Any]:
    """对单条 IM 记录打分，返回包含所有 TQS V2 指标的字典。

    median_steps is accepted for compatibility with the previous v5 API,
    but TQS V2 does not use dataset-level median-step normalization.
    """
    if scaffold_override is None and isinstance(median_steps, str):
        scaffold_override = median_steps
    messages = record.get("messages", [])
    scaffold = scaffold_override or detect_scaffold(record)
    actions = _extract_actions(messages, scaffold)
    observations = _extract_observations(messages, scaffold)
    assistant_turns = _count_assistant_turns(messages)

    components = {
        "oec": _compute_oec(observations),
        "iac": _compute_iac(messages, scaffold),
        "dpi": _compute_dpi(messages, actions, observations, scaffold),
        "ped": _compute_ped(actions, observations, messages, scaffold),
        "psn": _compute_psn(messages, scaffold),
        "tte": _compute_tte(actions, messages),
        "scp": _compute_scp(messages, observations, scaffold),
        "sub": _compute_sub(messages, observations, scaffold),
        "fec": _compute_fec(messages, scaffold),
        "stp": _compute_stp(messages),
        "tvr": _compute_tvr(messages, scaffold),
    }
    composite = _aggregate_tqs(components)

    return {
        "scaffold": scaffold,
        "assistant_turns": assistant_turns,
        "total_tool_calls": len(actions),
        "oec_score": _round_or_none(components["oec"]),
        "iac_score": _round_or_none(components["iac"]),
        "dpi_score": _round_or_none(components["dpi"]),
        "ped_score": _round_or_none(components["ped"]),
        "psn_score": _round_or_none(components["psn"]),
        "tte_score": _round_or_none(components["tte"]),
        "scp_score": _round_or_none(components["scp"]),
        "sub_score": _round_or_none(components["sub"]),
        "fec_score": _round_or_none(components["fec"]),
        "stp_score": _round_or_none(components["stp"]),
        "tvr_score": _round_or_none(components["tvr"]),
        "composite_score": round(composite, 4),
    }


def score_dataset(
    records: list[dict[str, Any]],
    quiet: bool = False,
    scaffold_override: str | None = None,
) -> list[dict[str, Any]]:
    """对整个数据集打分（TQS V2，单遍扫描，无需数据集统计）。"""
    if not records:
        print("没有记录可以打分。")
        return []

    scored: list[dict[str, Any]] = []
    n_subagent = n_scored = 0
    for record in records:
        if record.get("_agent_type") == "subagent":
            scored.append({**record, "_score": None})
            n_subagent += 1
            continue
        scored_record = {**record, "_score": score_record(record, scaffold_override=scaffold_override)}
        n_scored += 1
        if not quiet and n_scored % 100 == 0:
            print(f"  已打分 {n_scored} 条 main agent")
        scored.append(scored_record)

    if not quiet:
        print(f"  打分完成: {n_scored} 条 main agent, {n_subagent} 条 subagent (跳过)")

    return scored


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

def _format_stats(values: list[float]) -> str:
    """格式化统计值: N, mean, std, min, max。"""
    if not values:
        return "   0      N/A      N/A      N/A      N/A"
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / n
    std_val = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    return f"{n:>4} {mean_val:>8.4f} {std_val:>8.4f} {min_val:>8.4f} {max_val:>8.4f}"


def _print_group_stats(group_name: str, records: list[dict[str, Any]]) -> None:
    """打印一个分组的统计表。"""
    metrics = [
        "oec_score", "iac_score", "dpi_score",
        "ped_score", "psn_score", "tte_score", "scp_score",
        "sub_score", "fec_score", "stp_score", "tvr_score",
        "composite_score",
    ]

    print(f"\n  [{group_name}] ({len(records)} 条)")
    print(f"  {'指标':<24} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─' * 60}")

    for m in metrics:
        vals = []
        for r in records:
            score = r.get("_score")
            if isinstance(score, dict) and m in score and score[m] is not None:
                vals.append(score[m])
        print(f"  {m:<24} {_format_stats(vals)}")

    scored_records_only = [r for r in records if isinstance(r.get("_score"), dict)]
    turns = [r["_score"]["assistant_turns"] for r in scored_records_only]
    calls = [r["_score"]["total_tool_calls"] for r in scored_records_only]
    if turns:
        print(f"  {'assistant_turns':<24} {len(turns):>4} "
              f"{sum(turns)/len(turns):>8.1f} {'':>8} {min(turns):>8} {max(turns):>8}")
    if calls:
        print(f"  {'total_tool_calls':<24} {len(calls):>4} "
              f"{sum(calls)/len(calls):>8.1f} {'':>8} {min(calls):>8} {max(calls):>8}")


def print_score_summary(scored_records: list[dict[str, Any]]) -> None:
    """按脚手架分组打印打分汇总统计。"""
    if not scored_records:
        print("没有记录可以汇总。")
        return

    by_scaffold: dict[str, list[dict]] = {}
    n_skipped = 0
    for r in scored_records:
        score = r.get("_score")
        if not isinstance(score, dict):
            n_skipped += 1
            continue
        scaffold = score.get("scaffold", "unknown")
        by_scaffold.setdefault(scaffold, []).append(r)

    print(f"\n{'═' * 72}")
    n_scored = len(scored_records) - n_skipped
    print(f"  TQS V2 轨迹质量打分汇总 — {n_scored} 条轨迹 (已跳过 {n_skipped} 条 subagent)")
    print(f"{'═' * 72}")

    for scaffold in sorted(by_scaffold.keys()):
        _print_group_stats(scaffold, by_scaffold[scaffold])

    if len(by_scaffold) > 1:
        all_scored = [r for rs in by_scaffold.values() for r in rs]
        _print_group_stats("ALL", all_scored)

    print()


# ═══════════════════════════════════════════════════════════════════════════
# Dataset-level tool-call error rate
# ═══════════════════════════════════════════════════════════════════════════

def _count_tool_call_errors(
    record: dict[str, Any],
    scaffold: str,
) -> tuple[int, int]:
    """统计单条 IM 记录的 (总 tool 调用数, 错误 tool 调用数)。

    口径与 _compute_c1 完全对齐：

    - tool-call 脚手架（CC/OC/OpenHands）：每个 role="tool" observation 记为
      一次 tool 调用；该 observation 报错则记为一次错误调用。
    - Terminus2：一个 observation（user 反馈）覆盖前一个 assistant turn 的所有
      commands，因此 total 按 len(commands) 加权；该 observation 报错时按 1 个
      command 失败计（与 _compute_c1 的保守估计一致）。

    错误判定复用 _is_error_result，并沿用 C1 的 test-output 处理：测试命令产生的
    observation 只匹配明确执行错误（Tier 1），避免 pytest 预期失败被误判为工具
    调用错误。
    """
    messages = record.get("messages", []) or []
    observations = _extract_observations(messages, scaffold)
    if not observations:
        return 0, 0

    if scaffold == "terminus2":
        total = 0
        errors = 0
        for obs in observations:
            n_cmds = 1
            is_test = False
            prev_idx = obs.get("prev_assistant_idx")
            if prev_idx is not None:
                prev_msg = messages[prev_idx]
                is_test = _is_test_running_turn(prev_msg, scaffold)
                parsed = _parse_t2_assistant(prev_msg)
                if parsed is not None:
                    cmds = parsed.get("commands") or []
                    if cmds:
                        n_cmds = len(cmds)
            total += n_cmds
            if _is_error_result(obs.get("content", ""), is_test_output=is_test):
                errors += 1
        return total, errors

    total = 0
    errors = 0
    _test_flags_cache: dict[int, list[bool]] = {}
    for obs in observations:
        is_test = False
        prev_idx = obs.get("prev_assistant_idx")
        if prev_idx is not None:
            if prev_idx not in _test_flags_cache:
                _test_flags_cache[prev_idx] = _get_per_toolcall_test_flags(messages[prev_idx])
            flags = _test_flags_cache[prev_idx]
            pos = obs.get("tool_call_position", 0)
            is_test = flags[pos] if pos < len(flags) else False
        total += 1
        if _is_error_result(obs.get("content", ""), is_test_output=is_test):
            errors += 1
    return total, errors


def compute_tool_call_error_rate(
    records: list[dict[str, Any]],
    scaffold_override: str | None = None,
) -> dict[str, Any]:
    """计算数据集级别的工具调用错误率（基于 IM 记录）。

    - 轮次维度: error_rate = 错误 tool 调用数 / 总 tool 调用数
    - 轨迹维度: trajectory_error_rate = 含错误的轨迹数 / 含 tool 调用的轨迹数

    错误判定复用 rule_score 的错误正则与 test-output 处理，因此该聚合指标与
    单条轨迹的 c1_tool_success_rate 口径一致。统计涵盖全部记录（含 subagent）。
    """
    total_tool_calls = 0
    error_tool_calls = 0
    trajectories_with_tool_calls = 0
    trajectories_with_error = 0
    by_scaffold: dict[str, dict[str, int]] = {}

    for record in records:
        scaffold = scaffold_override or detect_scaffold(record)
        total, errors = _count_tool_call_errors(record, scaffold)
        if total == 0:
            continue

        total_tool_calls += total
        error_tool_calls += errors
        trajectories_with_tool_calls += 1
        if errors > 0:
            trajectories_with_error += 1

        bucket = by_scaffold.setdefault(scaffold, {
            "total_tool_calls": 0,
            "error_tool_calls": 0,
            "trajectories_with_tool_calls": 0,
            "trajectories_with_error": 0,
        })
        bucket["total_tool_calls"] += total
        bucket["error_tool_calls"] += errors
        bucket["trajectories_with_tool_calls"] += 1
        if errors > 0:
            bucket["trajectories_with_error"] += 1

    for bucket in by_scaffold.values():
        bt = bucket["total_tool_calls"]
        btr = bucket["trajectories_with_tool_calls"]
        bucket["error_rate"] = round(bucket["error_tool_calls"] / bt, 6) if bt else 0.0
        bucket["trajectory_error_rate"] = (
            round(bucket["trajectories_with_error"] / btr, 6) if btr else 0.0
        )

    return {
        "total_tool_calls": total_tool_calls,
        "error_tool_calls": error_tool_calls,
        "error_rate": round(error_tool_calls / total_tool_calls, 6) if total_tool_calls else 0.0,
        "trajectories_with_tool_calls": trajectories_with_tool_calls,
        "trajectories_with_error": trajectories_with_error,
        "trajectory_error_rate": (
            round(trajectories_with_error / trajectories_with_tool_calls, 6)
            if trajectories_with_tool_calls else 0.0
        ),
        "by_scaffold": by_scaffold,
    }


def print_tool_call_error_summary(stats: dict[str, Any]) -> None:
    """打印数据集级别工具调用错误率汇总。"""
    if not stats or not stats.get("total_tool_calls"):
        print("  工具调用错误率: N/A (无 tool 调用)")
        return

    print(f"\n{'═' * 72}")
    print("  工具调用错误率统计 (基于 IM)")
    print(f"{'═' * 72}")
    print(f"  【按轮次维度】 错误 {stats['error_tool_calls']} / 总 {stats['total_tool_calls']} "
          f"= {stats['error_rate']:.4f} ({stats['error_rate'] * 100:.2f}%)")
    print(f"  【按轨迹维度】 含错误轨迹 {stats['trajectories_with_error']} / "
          f"含 tool 调用轨迹 {stats['trajectories_with_tool_calls']} "
          f"= {stats['trajectory_error_rate']:.4f} ({stats['trajectory_error_rate'] * 100:.2f}%)")

    by_scaffold = stats.get("by_scaffold") or {}
    if len(by_scaffold) > 1:
        print("  按脚手架:")
        for scaffold in sorted(by_scaffold):
            b = by_scaffold[scaffold]
            print(f"    [{scaffold}] 轮次错误率 {b['error_rate']:.4f} "
                  f"({b['error_tool_calls']}/{b['total_tool_calls']}), "
                  f"轨迹错误率 {b['trajectory_error_rate']:.4f} "
                  f"({b['trajectories_with_error']}/{b['trajectories_with_tool_calls']})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 IM 格式的轨迹 JSONL 文件逐条打分（TQS V2 quality scoring framework）"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="输入 IM JSONL 文件路径")
    parser.add_argument("--output", "-o", type=Path, default=None, help="输出打分后的 JSONL 文件路径")
    parser.add_argument("--max-instances", type=int, default=None, help="最多处理多少条记录")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    parser.add_argument("--scaffold", type=str, choices=SCAFFOLD_TYPES, default=None, help="强制指定脚手架类型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    print(f"读取: {input_path}")
    records = load_jsonl(input_path)
    print(f"加载了 {len(records)} 条记录")

    if not records:
        print("没有记录，退出。")
        sys.exit(0)

    if args.max_instances is not None and args.max_instances < len(records):
        records = records[:args.max_instances]
        print(f"截断到 {len(records)} 条记录")

    scored_records = score_dataset(records, quiet=args.quiet, scaffold_override=args.scaffold)
    print_score_summary(scored_records)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_rule_scored.jsonl")

    save_jsonl(output_path, scored_records)
    print(f"打分结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
