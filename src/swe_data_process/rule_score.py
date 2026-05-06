#!/usr/bin/env python3
"""轨迹质量打分模块 — 基于 v5 quality scoring framework.

对 IM 格式的 JSONL 轨迹数据逐条打分，支持所有脚手架类型：
  Claude Code, OpenCode, OpenHands, OpenHands SDK, Terminus2

打分公式 (v5 — 子指标加权):
  Score = 0.20*Efficiency + 0.15*Style + 0.25*ToolMastery + 0.25*Completion + 0.15*Precision

  Efficiency = 0.8*A1 + 0.2*A2
    A1: Error-Retry Cycles    — 1 - clip(adjusted_cycles / 10, 0, 1)
        adjusted_cycles = raw_cycles - n_error_turns * sum(p_i^2)
        where p_i is the empirical usage frequency of tool i
    A2: Step Count Ratio      — 1 - normalize(clip(steps/median, 0.5, 3.0))
  Style = 0.4*B1 + 0.6*B2
    B1: Action Diversity      — Shannon entropy of tool types, normalized
    B2: Obs. Utilization      — fraction of obs keywords reused in actions
  Tool Mastery = 0.9*C1 + 0.1*C2
    C1: Tool Call Success Rate — 非错误 observation 占比
    C2: Tool Call Parallelism  — (mean(calls_per_turn) - 1) / (cap - 1)
  Task Completion = 0.3*D1 + 0.7*D2
    D1: Submission Completeness — 正常提交=1, 截断=0
    D2: Test Verification       — 0.5*has_test_write + 0.5*has_test_run
  Code Precision = 0.85*E1 + 0.15*E2
    E1: File Edit Concentration — 1 - clip((mean_edits-1)/4, 0, 1)
    E2: Delete-then-Modify      — 1 - clip(count/3, 0, 1)

v4→v5 调整理由:
  - A2(步数比率) 降权: 轨迹长度≠质量，长但正确的轨迹不应被惩罚
  - C2(并行度) 降权: OH-SDK 91%+ 为 0，CC 32% 为 0，区分力极弱
  - D1(提交完整性) 降权: 97-100% 饱和在 1.0，无区分力
  - E2(删后再改) 降权: 97-100% 饱和在 1.0，无区分力
  - D2(测试验证) 升权: 写+跑测试是可学习的高质量行为，top/bot 差异显著
  - C1(工具成功率) 升权: 反映工具使用质量，对 SFT 学习信号最直接

用法:
  conda activate swelf
  cd ~/swe_data_process
  python -m swe_data_process.rule_score --input <im.jsonl> [--output <scored.jsonl>] [--max-instances N]
"""

from __future__ import annotations

import argparse
import json
import math
import posixpath
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from swe_data_process.utils import load_jsonl, save_jsonl


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# OpenHands SDK 的 5 个硬编码工具名
_OPENHANDS_SDK_TOOL_NAMES = frozenset({
    "terminal", "file_editor", "task_tracker", "finish", "think",
})

# Claude Code / OpenCode 的特征工具名（小写，用于大小写不敏感匹配）
_CC_OC_TOOL_NAMES_LOWER = frozenset({
    "bash", "read", "edit", "write", "glob", "grep", "task",
    "webfetch", "websearch", "notebookedit", "todowrite", "taskoutput",
    "taskstop", "askuserquestion", "skill", "enterplanmode",
    "exitplanmode", "enterworktree",
})

# Terminus2 没有 tools 字段，用固定常量对齐 CC/OC 的 breadth 归一化
_TERMINUS2_N_AVAILABLE_TOOLS = 18

# 错误检测正则（编译后缓存）
# Tier 1: 明确的执行错误 — 在任何上下文中都算错误
_ERROR_PATTERNS_HARD: list[re.Pattern[str]] = [
    re.compile(r"command not found"),
    re.compile(r"Permission denied"),
    re.compile(r"exit code[:\s]+[1-9]", re.IGNORECASE),
    re.compile(r"returned non-zero exit status"),
    re.compile(r"<tool_use_error>"),
    re.compile(r"The arguments provided to the tool are invalid"),
]
# Tier 2: 可能出现在测试输出中的错误信号 — 仅在非测试上下文中匹配
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

# B2 关键词提取：≥ 4 字符的标识符
_KEYWORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{3,}")

# B2 停用词
_STOPWORDS = frozenset({
    "that", "this", "with", "from", "were", "been", "being",
    "have", "does", "will", "would", "could", "should", "might",
    "your", "each", "every", "some", "more", "most", "other", "into",
    "over", "such", "than", "very", "just", "about", "also", "then",
    "them", "they", "their", "there", "these", "those", "what", "which",
    "whom", "when", "where", "true", "false", "none", "null",
    "return", "import", "class", "self", "print", "file", "line",
    "name", "type", "function", "value", "string", "number",
    "args", "kwargs", "param", "result", "output", "input",
    "default", "config", "options", "settings",
})

# 错误检测时只扫描前 N 个字符
_ERROR_SCAN_LIMIT = 3000

# B2 关键词提取时只扫描前 N 个字符
_KEYWORD_SCAN_LIMIT = 5000

# B2 每个 observation 最多提取 N 个关键词
_MAX_KEYWORDS_PER_OBS = 50

# B2 分母只取 observation 中频率最高的 K 个关键词
_B2_OBS_TOP_K = 10

# 测试文件路径匹配（多语言）
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    # Python: test_*.py, *_test.py, tests/*.py, conftest.py
    r"test_[^/]+\.py|[^/]+_test\.py|tests/[^/]+\.py|conftest\.py"
    # Go: *_test.go
    r"|[^/]+_test\.go"
    # C/C++: *_test.cc, *_test.cpp, *_test.c, test_*.cc, test_*.cpp, test_*.c
    r"|[^/]+_test\.(?:cc|cpp|c)|test_[^/]+\.(?:cc|cpp|c)"
    # Rust: tests/*.rs (Rust 惯例测试目录), test_*.rs
    r"|tests/[^/]+\.rs|test_[^/]+\.rs"
    # Java: *Test.java, *Tests.java, *IT.java (integration test)
    r"|[^/]+Tests?\.java|[^/]+IT\.java"
    # JS/TS: *.test.js, *.test.ts, *.test.jsx, *.test.tsx, *.spec.js, *.spec.ts
    r"|[^/]+\.(?:test|spec)\.(?:js|ts|jsx|tsx|mjs|cjs)"
    r")$",
    re.IGNORECASE,
)

# 测试执行命令匹配（多语言，用于 D2 和错误检测的 test-output 判断）
_TEST_RUN_RE = re.compile(
    r"\b(?:"
    # Python
    r"pytest|py\.test|python\s+-m\s+pytest|python\s+-m\s+unittest"
    r"|unittest|python\s+test_|nosetests"
    # Go
    r"|go\s+test"
    # Rust
    r"|cargo\s+test"
    # C/C++ (cmake/ctest, gtest)
    r"|ctest|gtest_filter"
    # Java (maven, gradle)
    r"|mvn\s+(?:test|verify|surefire)|gradle\s+test|gradlew\s+test"
    # JS/TS (jest, mocha, vitest, npm/yarn/pnpm test)
    r"|jest|mocha|vitest|npx\s+jest|npx\s+vitest|npx\s+mocha"
    r"|(?:npm|yarn|pnpm)\s+(?:run\s+)?test"
    # Generic
    r"|make\s+(?:test|check)"
    r")\b"
    # C/C++ test binary execution: ./bin/*_test, ./test_*, ./*_test (无 \b 边界)
    r"|\.\/[^\s]*(?:_test|test_)\S*",
    re.IGNORECASE,
)

# ── 工具名 / 参数键常量（跨脚手架统一匹配）──
# 新增脚手架时只需在此处添加对应名称，所有指标自动生效

# 纯编辑工具 — 调用即为编辑
_PURE_EDIT_TOOL_NAMES = frozenset({
    "edit", "write",
})
# 多功能编辑工具 — 需检查 args.command 是否为写操作
_MULTI_EDITOR_TOOL_NAMES = frozenset({
    "file_editor", "str_replace_editor",
})
# 写操作 command 值（str_replace_editor / file_editor 的 command 参数）
_EDITOR_WRITE_COMMANDS = frozenset({
    "str_replace", "create", "insert",
})
# 所有编辑工具名（用于快速判断是否需要进一步检查）
_EDIT_TOOL_NAMES = _PURE_EDIT_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES

_BASH_TOOL_NAMES = frozenset({
    "bash", "terminal", "execute_bash",
})
_FILE_PATH_KEYS = ("file_path", "filePath", "path")


def _get_file_path(args: dict) -> str:
    """从 tool_call arguments 中提取文件路径（兼容不同脚手架的键名）。"""
    for key in _FILE_PATH_KEYS:
        val = args.get(key)
        if val:
            return val
    return ""


def _is_write_operation(name_lower: str, args: dict) -> bool:
    """判断一个 tool_call 是否为文件写操作。

    对纯编辑工具（edit, write）直接返回 True；
    对多功能编辑工具（file_editor, str_replace_editor）检查 command 参数，
    只有 str_replace / create / insert 才算写操作，view / undo_edit 不算。
    """
    if name_lower in _PURE_EDIT_TOOL_NAMES:
        return True
    if name_lower in _MULTI_EDITOR_TOOL_NAMES:
        return args.get("command", "") in _EDITOR_WRITE_COMMANDS
    return False


def _parse_tool_call(tc: Any) -> tuple[str, dict]:
    """从 tool_call dict 中提取 (name_lower, parsed_args)。

    统一处理 arguments 可能是 str 或 dict 的情况，
    避免在每个指标函数中重复相同的解析逻辑。
    """
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


# ── Composite score 权重（5 组，总和 = 1.0）──
# 调整这里即可改变各组在最终分数中的占比
W_EFFICIENCY = 0.20   # A: Efficiency (A1, A2)
W_STYLE      = 0.15   # B: Style (B1, B2)
W_TOOL       = 0.25   # C: Tool Mastery (C1, C2)
W_COMPLETION = 0.25   # D: Task Completion (D1, D2)
W_PRECISION  = 0.15   # E: Code Precision (E1, E2)

# ── 子指标权重（每组内两个子指标的权重，总和 = 1.0）──
# v5: 降低饱和/低区分力指标的权重，提升有信号的指标
W_A1 = 0.80; W_A2 = 0.20   # A2(步数比率) 降权: 长度≠质量
W_B1 = 0.40; W_B2 = 0.60   # B2(观察利用率) 略升
W_C1 = 0.90; W_C2 = 0.10   # C2(并行度) 降权: OH-SDK 91%+ 为 0
W_D1 = 0.30; W_D2 = 0.70   # D1(提交完整性) 降权: 97%+ 饱和; D2(测试) 升权
W_E1 = 0.85; W_E2 = 0.15   # E2(删后再改) 降权: 97%+ 饱和


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold detection
# ═══════════════════════════════════════════════════════════════════════════

SCAFFOLD_TYPES = ("claudecode", "opencode", "openhands", "openhands_sdk", "terminus2")


def detect_scaffold(record: dict[str, Any]) -> str:
    """根据 IM 记录的结构自动检测脚手架类型。"""
    tools = record.get("tools")

    # ---- Terminus2: 没有 tools 字段（None 或缺失）----
    if tools is None:
        return "terminus2"

    # ---- 提取 tool names ----
    tool_names: set[str] = set()
    if isinstance(tools, list):
        for t in tools:
            func = t.get("function", {}) if isinstance(t, dict) else {}
            name = func.get("name", "")
            if name:
                tool_names.add(name)

    if not tool_names:
        return "terminus2"

    # ---- OpenHands SDK: 工具名是 5 个硬编码名称的子集 ----
    if tool_names <= _OPENHANDS_SDK_TOOL_NAMES:
        return "openhands_sdk"

    # ---- Claude Code / OpenCode: 包含 CC 特征工具（大小写不敏感） ----
    tool_names_lower = {n.lower() for n in tool_names}
    if tool_names_lower & _CC_OC_TOOL_NAMES_LOWER:
        # 只检查 system message 开头（scaffold 自我介绍段），避免 problem statement
        # 中偶然出现 "claude code" 等关键词导致误判
        messages = record.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                head = (msg.get("content") or "")[:500].lower()
                if "claude code" in head or "anthropic" in head:
                    return "claudecode"
                break  # 只检查第一个 system 消息
        return "opencode"

    # ---- 其他有 tools 的情况: OpenHands (native) ----
    return "openhands"


# ═══════════════════════════════════════════════════════════════════════════
# Action / Observation extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_tool_call_actions(messages: list[dict]) -> list[dict[str, Any]]:
    """从 tool-call 脚手架提取 actions。

    返回: [{tool_name, msg_idx}, ...]
    """
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


def _strip_think_tags(content: str) -> str:
    """去除 <think>...</think> 标签，返回剩余内容。"""
    if not content:
        return ""
    text = content.strip()
    if text.startswith("<think>") and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _parse_t2_assistant(msg: dict) -> dict | None:
    """解析 Terminus2 assistant 消息的 JSON 内容，返回 parsed dict 或 None。"""
    content = msg.get("content", "")
    json_text = _strip_think_tags(content)
    if not json_text:
        return None
    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_terminus2_actions(messages: list[dict]) -> list[dict[str, Any]]:
    """从 Terminus2 脚手架提取 actions。

    解析 assistant content JSON 中的 commands[].keystrokes，
    取第一个单词作为命令类型名。

    返回: [{tool_name, msg_idx}, ...]
    """
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
            # 提取命令名（第一个单词）
            parts = keystrokes.split()
            tool_name = parts[0] if parts else keystrokes
            # 去除路径前缀，只保留命令名
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
    """提取 observations（tool 结果 / 终端输出）。

    返回: [{content, msg_idx, prev_assistant_idx, tool_call_position}, ...]
    tool_call_position: 该 observation 对应的 tool_call 在 assistant turn 中的位置索引
                        （仅 tool-call 脚手架有效，T2 为 None）
    """
    observations: list[dict[str, Any]] = []

    if scaffold == "terminus2":
        # Terminus2: assistant 后的 user 消息即为 observation
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
        # Tool-call 脚手架: role="tool" 的消息
        # 按 prev_assistant_idx 分组计数，确定每个 observation 对应的 tool_call 位置
        _position_counter: dict[int, int] = {}
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            # 向前查找最近的 assistant 消息
            prev_assistant_idx = None
            for j in range(idx - 1, -1, -1):
                if messages[j].get("role") == "assistant":
                    prev_assistant_idx = j
                    break
            # 计算该 observation 在同一 assistant turn 的 tool 响应中的位置
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
    """检测 observation 是否包含错误信息。

    当 is_test_output=True 时，只匹配明确的执行错误（Tier 1），
    跳过可能出现在测试输出中的异常名/FAILED 等（Tier 2），
    避免 pytest 输出中的预期失败被误判为工具调用错误。
    """
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
    """判断一个 assistant turn 是否包含测试执行命令。

    仅用于 Terminus2（每 turn 只有一个 observation，无法按 tool call 拆分）。
    对 tool-call 脚手架，请使用 _get_per_toolcall_test_flags 做逐 tool call 判断。
    """
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
    """返回 assistant 消息中每个 tool_call 是否为测试执行命令。

    用于 tool-call 脚手架的逐 observation 测试检测。
    tool 响应与 tool_calls 按位置一一对应。
    """
    flags: list[bool] = []
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        is_test = False
        if name_lower in _BASH_TOOL_NAMES:
            is_test = bool(_TEST_RUN_RE.search(args.get("command", "")))
        flags.append(is_test)
    return flags


# ═══════════════════════════════════════════════════════════════════════════
# A1: Error-Retry Cycles
# ═══════════════════════════════════════════════════════════════════════════

def _compute_a1(
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    messages: list[dict],
    scaffold: str,
) -> tuple[float, int]:
    """A1: Error-Retry Cycles — 1 - clip(adjusted_cycles / 10, 0, 1)

    概率校正：用实际工具使用频率分布 sum(p_i^2) 估算随机重叠概率，
    扣除 expected_random = n_error_turns * overlap_prob 作为 baseline，
    避免少工具脚手架被系统性惩罚。

    返回: (a1_score, raw_error_retry_cycles)
    """
    if not actions or not observations:
        return 1.0, 0

    # 构建映射: assistant_msg_idx -> set of tool names
    assistant_tools: dict[int, set[str]] = {}
    for a in actions:
        assistant_tools.setdefault(a["msg_idx"], set()).add(a["tool_name"])

    # 构建映射: assistant_msg_idx -> list of (observation_content, is_test_output)
    assistant_obs: dict[int, list[tuple[str, bool]]] = {}
    # 预计算每个 assistant turn 的 per-tool-call test flags（仅 tool-call 脚手架）
    _test_flags_cache: dict[int, list[bool]] = {}
    for obs in observations:
        if obs["prev_assistant_idx"] is None:
            continue
        prev_idx = obs["prev_assistant_idx"]
        # 确定该 observation 是否为测试输出
        if scaffold == "terminus2":
            is_test = _is_test_running_turn(messages[prev_idx], scaffold)
        else:
            # 按位置匹配到具体的 tool_call
            if prev_idx not in _test_flags_cache:
                _test_flags_cache[prev_idx] = _get_per_toolcall_test_flags(messages[prev_idx])
            flags = _test_flags_cache[prev_idx]
            pos = obs.get("tool_call_position", 0)
            is_test = flags[pos] if pos < len(flags) else False
        assistant_obs.setdefault(prev_idx, []).append((obs["content"], is_test))

    # 按消息索引排序的 assistant turns（有 tool calls 的）
    sorted_indices = sorted(assistant_tools.keys())

    cycles = 0
    n_error_turns = 0
    for i in range(len(sorted_indices) - 1):
        curr_idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]

        # 检查当前 assistant turn 的 observation 是否有 error（逐 observation 判断）
        obs_for_curr = assistant_obs.get(curr_idx, [])
        has_error = any(
            _is_error_result(content, is_test_output=is_test)
            for content, is_test in obs_for_curr
        )

        if has_error:
            n_error_turns += 1
            # 检查下一个 assistant turn 是否使用了相同的 tool
            curr_tools = assistant_tools[curr_idx]
            next_tools = assistant_tools[next_idx]
            if curr_tools & next_tools:
                cycles += 1

    # 概率校正：用实际工具使用频率分布估算随机重叠概率 sum(p_i^2)，
    # 替代之前的均匀假设 1/n_available_tools，避免少工具脚手架被系统性惩罚
    if actions and n_error_turns > 0:
        tool_counts = Counter(a["tool_name"] for a in actions)
        total = sum(tool_counts.values())
        overlap_prob = sum((c / total) ** 2 for c in tool_counts.values())
        expected_random = n_error_turns * overlap_prob
        adjusted_cycles = max(0.0, cycles - expected_random)
    else:
        adjusted_cycles = float(cycles)

    a1 = 1.0 - min(adjusted_cycles / 10.0, 1.0)
    return a1, cycles


# ═══════════════════════════════════════════════════════════════════════════
# A2: Step Count Ratio
# ═══════════════════════════════════════════════════════════════════════════

def _count_assistant_turns(messages: list[dict]) -> int:
    """统计 assistant turns 数量。"""
    return sum(1 for m in messages if m.get("role") == "assistant")


def _compute_a2(steps: int, median_steps: float) -> float:
    """A2: Step Count Ratio — 1 - normalize(clip(steps/median, 0.5, 3.0))

    注意: median_steps 是数据集内部按脚手架分组计算的中位数，
    因此 A2 分数是相对于同数据集同脚手架的排名，不可跨数据集直接比较。
    """
    if median_steps <= 0:
        return 1.0
    ratio = steps / median_steps
    clipped = max(0.5, min(ratio, 3.0))
    normalized = (clipped - 0.5) / 2.5  # [0.5, 3.0] -> [0, 1]
    return 1.0 - normalized


# ═══════════════════════════════════════════════════════════════════════════
# B1: Action Diversity (Shannon Entropy)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_b1(
    actions: list[dict[str, Any]],
    n_available_tools: int | None = None,
) -> float:
    """B1: Action Diversity — Shannon entropy of tool types, normalized.

    当提供 n_available_tools 时，用 log2(n_available_tools) 归一化（衡量工具使用广度）；
    否则用 log2(n_unique_types_used) 归一化（衡量工具使用均匀度）。
    """
    if not actions:
        return 0.0

    tool_counts = Counter(a["tool_name"] for a in actions)
    n_types = len(tool_counts)

    if n_types <= 1:
        return 0.0

    total = sum(tool_counts.values())
    entropy = 0.0
    for count in tool_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # 优先用可用工具总数归一化（衡量广度），否则用实际使用种类数（衡量均匀度）
    denom_n = n_available_tools if (n_available_tools and n_available_tools > 1) else n_types
    max_entropy = math.log2(denom_n)
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# B2: Observation Utilization
# ═══════════════════════════════════════════════════════════════════════════

def _extract_keywords(text: str) -> set[str]:
    """从文本中提取有意义的关键词。"""
    if not text:
        return set()
    text_slice = text[:_KEYWORD_SCAN_LIMIT]
    tokens = _KEYWORD_RE.findall(text_slice)
    # 过滤停用词
    keywords = {t.lower() for t in tokens if t.lower() not in _STOPWORDS}
    # 限制数量：按频率取 top N
    if len(keywords) > _MAX_KEYWORDS_PER_OBS:
        counter = Counter(t.lower() for t in tokens if t.lower() not in _STOPWORDS)
        keywords = {k for k, _ in counter.most_common(_MAX_KEYWORDS_PER_OBS)}
    return keywords


def _get_assistant_text(msg: dict) -> str:
    """提取 assistant 消息的可检索文本（content + tool_call argument values）。

    只序列化 argument values（不含 keys），减少结构性噪声（如 file_path, command 等键名）。
    对结果应用 _KEYWORD_SCAN_LIMIT 截断，与 observation 侧对称。
    """
    parts: list[str] = []
    content = msg.get("content", "")
    if content:
        parts.append(content)
    for tc in msg.get("tool_calls", []) or []:
        _, args = _parse_tool_call(tc)
        if args:
            # 只取 values，避免 keys（file_path, command 等）成为关键词
            for v in args.values():
                if isinstance(v, str):
                    parts.append(v)
                else:
                    parts.append(json.dumps(v, ensure_ascii=False))
    text = " ".join(parts)
    return text[:_KEYWORD_SCAN_LIMIT]


def _compute_b2(
    observations: list[dict[str, Any]],
    messages: list[dict],
) -> float:
    """B2: Observation Utilization — fraction of top-K obs keywords reused in next action."""
    if not observations:
        return 0.0

    utilization_scores: list[float] = []

    for obs in observations:
        obs_text = obs["content"]
        if not obs_text:
            continue

        # 提取 top-K 高频关键词作为分母（而非全部关键词）
        text_slice = obs_text[:_KEYWORD_SCAN_LIMIT]
        tokens = _KEYWORD_RE.findall(text_slice)
        filtered = [t.lower() for t in tokens if t.lower() not in _STOPWORDS]
        if not filtered:
            continue
        counter = Counter(filtered)
        top_k_keywords = {k for k, _ in counter.most_common(_B2_OBS_TOP_K)}

        # 查找此 observation 之后的第一个 assistant 消息
        next_assistant_text = ""
        for idx in range(obs["msg_idx"] + 1, len(messages)):
            if messages[idx].get("role") == "assistant":
                next_assistant_text = _get_assistant_text(messages[idx])
                break

        if not next_assistant_text:
            continue

        next_keywords = _extract_keywords(next_assistant_text)
        overlap = top_k_keywords & next_keywords
        utilization = len(overlap) / len(top_k_keywords)
        utilization_scores.append(utilization)

    if not utilization_scores:
        return 0.0

    return sum(utilization_scores) / len(utilization_scores)


# ═══════════════════════════════════════════════════════════════════════════
# C1: Tool Call Success Rate
# ═══════════════════════════════════════════════════════════════════════════

def _compute_c1(
    observations: list[dict[str, Any]],
    messages: list[dict],
    scaffold: str,
) -> float:
    """C1: Tool Call Success Rate — 非错误 observation 占比。

    对 T2 做加权处理：每个 observation 覆盖前一个 assistant turn 的所有 commands，
    若该 observation 报错，按 1/n_commands 计为失败（而非整个 turn 失败），
    避免相对 tool-call 脚手架（每个 tool call 有独立 observation）的系统性偏高。
    """
    if not observations:
        return 1.0

    if scaffold != "terminus2":
        success = 0
        # 预计算每个 assistant turn 的 per-tool-call test flags
        _test_flags_cache: dict[int, list[bool]] = {}
        for obs in observations:
            is_test = False
            if obs["prev_assistant_idx"] is not None:
                prev_idx = obs["prev_assistant_idx"]
                if prev_idx not in _test_flags_cache:
                    _test_flags_cache[prev_idx] = _get_per_toolcall_test_flags(messages[prev_idx])
                flags = _test_flags_cache[prev_idx]
                pos = obs.get("tool_call_position", 0)
                is_test = flags[pos] if pos < len(flags) else False
            if not _is_error_result(obs["content"], is_test_output=is_test):
                success += 1
        return success / len(observations)

    # T2: 按 commands 数量加权
    total_commands = 0
    failed_commands = 0
    for obs in observations:
        # 找前一个 assistant turn 的 commands 数量
        n_cmds = 1
        is_test = False
        if obs["prev_assistant_idx"] is not None:
            prev_msg = messages[obs["prev_assistant_idx"]]
            is_test = _is_test_running_turn(prev_msg, scaffold)
            parsed = _parse_t2_assistant(prev_msg)
            if parsed is not None:
                cmds = parsed.get("commands") or []
                if cmds:
                    n_cmds = len(cmds)
        total_commands += n_cmds
        if _is_error_result(obs["content"], is_test_output=is_test):
            # 整个 observation 报错，但只算 1 个 command 失败（保守估计）
            failed_commands += 1

    if total_commands == 0:
        return 1.0
    return 1.0 - failed_commands / total_commands


# ═══════════════════════════════════════════════════════════════════════════
# C2: Tool Call Parallelism
# ═══════════════════════════════════════════════════════════════════════════

_PARALLELISM_CAP = 5  # 归一化上限

def _count_calls_per_turn_toolcall(messages: list[dict]) -> list[int]:
    """统计每个 assistant turn 的 tool_call 数量（tool-call 脚手架）。"""
    counts: list[int] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        if tcs:
            counts.append(len(tcs))
    return counts


def _count_calls_per_turn_terminus2(messages: list[dict]) -> list[int]:
    """统计每个 assistant turn 的 commands 数量（Terminus2）。"""
    counts: list[int] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        parsed = _parse_t2_assistant(msg)
        if parsed is None:
            continue
        cmds = parsed.get("commands") or []
        if cmds:
            counts.append(len(cmds))
    return counts


def _compute_c2(messages: list[dict], scaffold: str) -> float:
    """C2: Tool Call Parallelism — (mean(calls_per_turn) - 1) / (cap - 1), clipped to [0,1]。

    以 1 为基线（每 turn 至少 1 次调用），只衡量"额外"并行度，
    消除单次调用 turn 的 0.2 下限。
    """
    if scaffold == "terminus2":
        counts = _count_calls_per_turn_terminus2(messages)
    else:
        counts = _count_calls_per_turn_toolcall(messages)
    if not counts:
        return 0.0
    mean_parallel = sum(counts) / len(counts)
    return min(max((mean_parallel - 1.0) / (_PARALLELISM_CAP - 1.0), 0.0), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# D1: Submission Completeness
# ═══════════════════════════════════════════════════════════════════════════

def _compute_d1(messages: list[dict], scaffold: str) -> float:
    """D1: Submission Completeness — 1.0 正常提交, 0.0 截断。"""
    if not messages:
        return 0.0

    # last_msg: 轨迹的最后一条消息（任意 role）
    # last_assistant: 最后一条 assistant 消息（可能不是 last_msg，
    #   例如 OH/SDK 的 finish 调用后还有 tool 响应）
    last_assistant = None
    last_msg = messages[-1]
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break

    if last_assistant is None:
        return 0.0

    if scaffold == "openhands_sdk" or scaffold == "openhands":
        # 检查最后一个 assistant 是否调用了 finish
        for tc in last_assistant.get("tool_calls", []) or []:
            name_lower, _ = _parse_tool_call(tc)
            if name_lower == "finish":
                return 1.0
        # 最后消息是 assistant 且无 tool_calls → 可能是总结
        if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
            return 0.5
        return 0.0

    if scaffold == "terminus2":
        parsed = _parse_t2_assistant(last_assistant)
        if parsed is not None and parsed.get("task_complete") is True:
            return 1.0
        return 0.0

    # CC / OC: 最后消息是 assistant 且无 tool_calls → 正常总结
    if last_msg.get("role") == "assistant" and not (last_msg.get("tool_calls") or []):
        return 1.0
    # 最后是 tool 响应 → 截断
    if last_msg.get("role") == "tool":
        return 0.0
    # assistant 带 tool_calls 但没有后续 tool 响应 → 截断
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# D2: Test Writing & Verification
# ═══════════════════════════════════════════════════════════════════════════



def _compute_d2(messages: list[dict], scaffold: str) -> float:
    """D2: Test Writing & Verification — 0.5 * has_test_write + 0.5 * has_test_run。"""
    has_test_write = False
    has_test_run = False

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
                if _TEST_RUN_RE.search(ks):
                    has_test_run = True
                # 写入测试文件: 复用 _extract_bash_edit_paths + _TEST_FILE_RE
                for edited_path in _extract_bash_edit_paths(ks):
                    if _TEST_FILE_RE.search(edited_path):
                        has_test_write = True
        else:
            for tc in msg.get("tool_calls", []) or []:
                name_lower, args = _parse_tool_call(tc)

                # 测试编写检测
                if _is_write_operation(name_lower, args):
                    path = _get_file_path(args)
                    if _TEST_FILE_RE.search(path):
                        has_test_write = True

                # 测试执行检测
                if name_lower in _BASH_TOOL_NAMES:
                    cmd = args.get("command", "")
                    if _TEST_RUN_RE.search(cmd):
                        has_test_run = True

        if has_test_write and has_test_run:
            break

    return 0.5 * float(has_test_write) + 0.5 * float(has_test_run)


# ═══════════════════════════════════════════════════════════════════════════
# E1: File Edit Concentration
# ═══════════════════════════════════════════════════════════════════════════

# 常见工作区路径前缀（不同脚手架使用不同的基目录）
_WORKSPACE_PREFIXES = ("/workspace/", "/testbed/", "/repo/", "/home/swe-bench/")


def _normalize_path(p: str) -> str:
    """归一化文件路径，使 ./foo.py、foo.py、/workspace/foo.py 视为同一文件。"""
    for prefix in _WORKSPACE_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return posixpath.normpath(p)


# 重定向写入: echo ... > file, echo ... >> file, cat > file, tee file
_REDIRECT_WRITE_RE = re.compile(r">>?\s*(\S+)")
_CAT_REDIRECT_RE = re.compile(r"\bcat\s[^|]*>>?\s*(\S+)")
_TEE_RE = re.compile(r"\btee\s+(?:-[a-zA-Z]+\s+)*(\S+)")
# sed -i: 提取 expression 之后的所有非 flag 文件路径
_SED_INPLACE_RE = re.compile(r"\bsed\s+-i\b")
_PATCH_FILE_RE = re.compile(r"\bpatch\s+(?:-\S+\s+)*(\S+)")


def _extract_bash_edit_paths(cmd_str: str) -> list[str]:
    """从 bash 命令字符串中提取文件编辑目标路径。

    支持: echo/printf > file, cat > file, tee file, sed -i file..., patch file
    """
    paths: list[str] = []

    # cat > file / cat >> file
    m = _CAT_REDIRECT_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))

    # echo ... > file / echo ... >> file (不含 cat，避免重复)
    if not m and (">" in cmd_str):
        # 只在有 echo/printf 时匹配重定向
        if re.search(r"\b(?:echo|printf)\b", cmd_str):
            rm = _REDIRECT_WRITE_RE.search(cmd_str)
            if rm:
                paths.append(rm.group(1))

    # tee file
    m = _TEE_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))

    # sed -i [opts] 'expr' file1 file2 ...
    if _SED_INPLACE_RE.search(cmd_str):
        # 提取 sed -i 之后的部分，跳过 flags 和引号表达式，取剩余文件路径
        after_sed = _SED_INPLACE_RE.split(cmd_str, 1)[-1].strip()
        # 跳过 -e/-E 等 flags 和引号包裹的表达式
        tokens = after_sed.split()
        skip_next = False
        in_expr = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if tok.startswith("-"):
                if tok in ("-e", "-E"):
                    skip_next = True  # 下一个 token 是表达式
                continue
            # 跳过引号包裹的表达式（sed 的第一个非 flag 参数）
            if not in_expr and (tok.startswith("'") or tok.startswith('"') or tok.startswith("s")):
                in_expr = True
                continue
            # 剩余的非 flag token 是文件路径
            if in_expr:
                paths.append(tok)

    # patch file
    m = _PATCH_FILE_RE.search(cmd_str)
    if m:
        paths.append(m.group(1))
    return paths


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
                # Bash 命令中的文件编辑（sed -i, cat >, patch 等）
                elif name_lower in _BASH_TOOL_NAMES:
                    cmd_str = args.get("command", "")
                    paths.extend(_normalize_path(p) for p in _extract_bash_edit_paths(cmd_str))

    return paths


_E1_THRESHOLD = 5  # mean_edits_per_file >= 5 → 0 分


def _compute_e1(messages: list[dict], scaffold: str) -> float:
    """E1: File Edit Concentration — 1 - clip((mean_edits - 1) / 4, 0, 1)。"""
    paths = _extract_edit_paths(messages, scaffold)
    if not paths:
        return 1.0

    file_counts = Counter(paths)
    unique_files = len(file_counts)
    if unique_files == 0:
        return 1.0

    mean_edits = len(paths) / unique_files
    normalized = max(0.0, min((mean_edits - 1) / (_E1_THRESHOLD - 1), 1.0))
    return 1.0 - normalized


# ═══════════════════════════════════════════════════════════════════════════
# E2: Delete-then-Modify
# ═══════════════════════════════════════════════════════════════════════════

_RM_CMD_RE = re.compile(r"\brm\s+(?:-[rfi]+\s+)*(.+)")

def _parse_rm_targets(cmd_str: str) -> list[str]:
    """从 rm 命令中提取所有目标文件路径。"""
    m = _RM_CMD_RE.search(cmd_str)
    if not m:
        return []
    # 按空格拆分剩余部分，过滤掉 flag（-开头）
    return [t for t in m.group(1).split() if not t.startswith("-")]

_E2_THRESHOLD = 3  # 3 次以上扣满分（渐进式惩罚）


def _compute_e2(messages: list[dict], scaffold: str) -> float:
    """E2: Delete-then-Modify — 1 - clip(count/3, 0, 1)，渐进式惩罚。"""
    deleted_files: set[str] = set()
    dtm_count = 0

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
                # 检测删除（支持多文件）
                for target in _parse_rm_targets(ks):
                    deleted_files.add(_normalize_path(target))
                # 检测写入已删除文件
                for edited_path in _extract_bash_edit_paths(ks):
                    if _normalize_path(edited_path) in deleted_files:
                        dtm_count += 1
        else:
            for tc in msg.get("tool_calls", []) or []:
                name_lower, args = _parse_tool_call(tc)

                if name_lower in _BASH_TOOL_NAMES:
                    cmd_str = args.get("command", "")
                    # 检测删除: rm 命令（支持多文件）
                    for target in _parse_rm_targets(cmd_str):
                        deleted_files.add(_normalize_path(target))
                    # 检测 bash 命令中修改已删除文件（sed -i, cat >, patch 等）
                    for edited_path in _extract_bash_edit_paths(cmd_str):
                        if _normalize_path(edited_path) in deleted_files:
                            dtm_count += 1

                # 检测 edit/write 工具修改已删除文件
                if _is_write_operation(name_lower, args):
                    path = _get_file_path(args)
                    if path and _normalize_path(path) in deleted_files:
                        dtm_count += 1

    if not deleted_files:
        return 1.0

    return 1.0 - min(dtm_count / _E2_THRESHOLD, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Composite scoring
# ═══════════════════════════════════════════════════════════════════════════

def score_record(
    record: dict[str, Any],
    median_steps: float,
    scaffold_override: str | None = None,
) -> dict[str, Any]:
    """对单条 IM 记录打分，返回包含所有指标的字典。"""
    messages = record.get("messages", [])
    scaffold = scaffold_override or detect_scaffold(record)

    assistant_turns = _count_assistant_turns(messages)

    # 提取一次，复用于所有指标
    actions = _extract_actions(messages, scaffold)
    observations = _extract_observations(messages, scaffold)
    total_tool_calls = len(actions)

    # 可用工具数（用于 B1 归一化）：从 record.tools 获取，Terminus2 用固定常量
    tools = record.get("tools")
    n_available_tools: int | None = None
    if isinstance(tools, list) and tools:
        n_available_tools = len(tools)
    elif scaffold == "terminus2":
        n_available_tools = _TERMINUS2_N_AVAILABLE_TOOLS

    # --- Efficiency (A) ---
    a1, error_retry_cycles = _compute_a1(actions, observations, messages, scaffold)
    a2 = _compute_a2(assistant_turns, median_steps)

    # --- Style (B) ---
    b1 = _compute_b1(actions, n_available_tools)
    b2 = _compute_b2(observations, messages)

    # --- Tool Mastery (C) ---
    c1 = _compute_c1(observations, messages, scaffold)
    c2 = _compute_c2(messages, scaffold)

    # --- Task Completion (D) ---
    d1 = _compute_d1(messages, scaffold)
    d2 = _compute_d2(messages, scaffold)

    # --- Code Precision (E) ---
    e1 = _compute_e1(messages, scaffold)
    e2 = _compute_e2(messages, scaffold)

    # --- Group scores (v5: 子指标加权) ---
    efficiency = W_A1 * a1 + W_A2 * a2
    style = W_B1 * b1 + W_B2 * b2
    tool_mastery = W_C1 * c1 + W_C2 * c2
    completion = W_D1 * d1 + W_D2 * d2
    precision = W_E1 * e1 + W_E2 * e2

    # v3 composite (backward compat)
    composite_v3 = 0.5 * efficiency + 0.5 * style

    # v4 composite (backward compat — 旧的等权 mean 公式)
    composite_v4 = (
        0.30 * ((a1 + a2) / 2.0)
        + 0.20 * ((b1 + b2) / 2.0)
        + 0.20 * ((c1 + c2) / 2.0)
        + 0.15 * ((d1 + d2) / 2.0)
        + 0.15 * ((e1 + e2) / 2.0)
    )

    # v5 composite: 子指标加权
    composite = (
        W_EFFICIENCY * efficiency
        + W_STYLE * style
        + W_TOOL * tool_mastery
        + W_COMPLETION * completion
        + W_PRECISION * precision
    )

    return {
        "scaffold": scaffold,
        "assistant_turns": assistant_turns,
        "total_tool_calls": total_tool_calls,
        "error_retry_cycles": error_retry_cycles,
        # A: Efficiency
        "a1_error_retry": round(a1, 4),
        "a2_step_count_ratio": round(a2, 4),
        # B: Style
        "b1_action_diversity": round(b1, 4),
        "b2_observation_utilization": round(b2, 4),
        # C: Tool Mastery
        "c1_tool_success_rate": round(c1, 4),
        "c2_tool_parallelism": round(c2, 4),
        # D: Task Completion
        "d1_submission_completeness": round(d1, 4),
        "d2_test_verification": round(d2, 4),
        # E: Code Precision
        "e1_file_edit_concentration": round(e1, 4),
        "e2_delete_then_modify": round(e2, 4),
        # Group scores
        "efficiency_score": round(efficiency, 4),
        "style_score": round(style, 4),
        "tool_mastery_score": round(tool_mastery, 4),
        "completion_score": round(completion, 4),
        "precision_score": round(precision, 4),
        # Composite
        "composite_score_v3": round(composite_v3, 4),
        "composite_score_v4": round(composite_v4, 4),
        "composite_score": round(composite, 4),
    }


def score_dataset(
    records: list[dict[str, Any]],
    quiet: bool = False,
    scaffold_override: str | None = None,
) -> list[dict[str, Any]]:
    """对整个数据集打分（两遍扫描）。

    Pass 1: 按脚手架分组统计 assistant turns → 计算各组 median
    Pass 2: 逐条计算所有指标（使用对应脚手架的 median）

    注意: A2 的 median 是在本数据集内按脚手架分组计算的，
    因此 A2 分数仅在同一数据集内有可比性，不可跨数据集直接比较。
    """
    if not records:
        print("没有记录可以打分。")
        return []

    # Pass 1: 按脚手架分组计算 median steps（仅 main agent，排除 subagent）
    scaffold_steps: dict[str, list[int]] = {}
    for r in records:
        if r.get("_agent_type") == "subagent":
            continue
        scaffold = scaffold_override or detect_scaffold(r)
        steps = _count_assistant_turns(r.get("messages", []))
        scaffold_steps.setdefault(scaffold, []).append(steps)

    scaffold_median: dict[str, float] = {}
    for scaffold, steps_list in scaffold_steps.items():
        scaffold_median[scaffold] = statistics.median(steps_list) if steps_list else 1.0

    for scaffold in sorted(scaffold_median):
        n = len(scaffold_steps[scaffold])
        print(f"  [{scaffold}] {n} 条, median assistant turns = {scaffold_median[scaffold]:.1f}")

    # Pass 2: 逐条打分（subagent 记录跳过打分，_score 设为 None）
    scored: list[dict[str, Any]] = []
    n_subagent = 0
    n_scored = 0
    for record in records:
        if record.get("_agent_type") == "subagent":
            scored.append({**record, "_score": None})
            n_subagent += 1
            continue

        scaffold = scaffold_override or detect_scaffold(record)
        median_steps = scaffold_median.get(scaffold, 1.0)
        scores = score_record(record, median_steps, scaffold_override=scaffold_override)
        n_scored += 1

        if not quiet and n_scored % 100 == 0:
            print(f"  已打分 {n_scored} 条 main agent")

        scored_record = {**record, "_score": scores}
        scored.append(scored_record)

    if not quiet:
        print(f"  打分完成: {n_scored} 条 main agent, {n_subagent} 条 subagent (跳过)")

    return scored


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

def _format_stats(values: list[float]) -> str:
    """格式化统计值: mean, std, min, max。"""
    if not values:
        return "    N/A      N/A      N/A      N/A"
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((v - mean_val) ** 2 for v in values) / n
    std_val = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    return f"{mean_val:>8.4f} {std_val:>8.4f} {min_val:>8.4f} {max_val:>8.4f}"


def _print_group_stats(
    group_name: str,
    records: list[dict[str, Any]],
) -> None:
    """打印一个分组的统计表。"""
    metrics = [
        "a1_error_retry", "a2_step_count_ratio",
        "b1_action_diversity", "b2_observation_utilization",
        "c1_tool_success_rate", "c2_tool_parallelism",
        "d1_submission_completeness", "d2_test_verification",
        "e1_file_edit_concentration", "e2_delete_then_modify",
        "efficiency_score", "style_score",
        "tool_mastery_score", "completion_score", "precision_score",
        "composite_score_v3", "composite_score_v4", "composite_score",
    ]

    print(f"\n  [{group_name}] ({len(records)} 条)")
    print(f"  {'指标':<32} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─' * 66}")

    for m in metrics:
        vals = [r["_score"][m] for r in records if isinstance(r.get("_score"), dict) and m in r["_score"]]
        print(f"  {m:<32} {_format_stats(vals)}")

    # 额外统计 assistant_turns 和 total_tool_calls
    scored_records_only = [r for r in records if isinstance(r.get("_score"), dict)]
    turns = [r["_score"]["assistant_turns"] for r in scored_records_only]
    calls = [r["_score"]["total_tool_calls"] for r in scored_records_only]
    cycles = [r["_score"]["error_retry_cycles"] for r in scored_records_only]
    if turns:
        print(f"  {'assistant_turns':<32} "
              f"{sum(turns)/len(turns):>8.1f} {'':>8} {min(turns):>8} {max(turns):>8}")
    if calls:
        print(f"  {'total_tool_calls':<32} "
              f"{sum(calls)/len(calls):>8.1f} {'':>8} {min(calls):>8} {max(calls):>8}")
    if cycles:
        print(f"  {'error_retry_cycles':<32} "
              f"{sum(cycles)/len(cycles):>8.1f} {'':>8} {min(cycles):>8} {max(cycles):>8}")


def print_score_summary(scored_records: list[dict[str, Any]]) -> None:
    """按脚手架分组打印打分汇总统计。"""
    if not scored_records:
        print("没有记录可以汇总。")
        return

    # 分组（跳过 _score 为 None 的 subagent 记录）
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
    print(f"  轨迹质量打分汇总 — {n_scored} 条轨迹 (已跳过 {n_skipped} 条 subagent)")
    print(f"{'═' * 72}")

    # 各脚手架分组
    for scaffold in sorted(by_scaffold.keys()):
        _print_group_stats(scaffold, by_scaffold[scaffold])

    # 总体统计（多于一个脚手架时显示）
    if len(by_scaffold) > 1:
        _print_group_stats("ALL", scored_records)

    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 IM 格式的轨迹 JSONL 文件逐条打分（v5 quality scoring framework）"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="输入 IM JSONL 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出打分后的 JSONL 文件路径（默认: <input_stem>_scored.jsonl）",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="最多处理多少条记录",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    parser.add_argument(
        "--scaffold",
        type=str,
        choices=SCAFFOLD_TYPES,
        default=None,
        help="强制指定脚手架类型（跳过自动检测，用于调试）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 输入
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

    # 截断
    if args.max_instances is not None and args.max_instances < len(records):
        records = records[:args.max_instances]
        print(f"截断到 {len(records)} 条记录")

    # 打分
    scored_records = score_dataset(
        records, quiet=args.quiet, scaffold_override=args.scaffold,
    )

    # 汇总统计
    print_score_summary(scored_records)

    # 输出
    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_scored.jsonl")

    save_jsonl(output_path, scored_records)
    print(f"打分结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
