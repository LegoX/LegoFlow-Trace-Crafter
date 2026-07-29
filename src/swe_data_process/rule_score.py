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

多语言正确性要点:

  * 基于 observation 内容的判定先经 _scan_window：剥离 ANSI（转义序列会插进词内部，
    破坏 `\\bFAILED\\b` 之类的边界锚定），并同时取输出的开头与结尾（结论在末尾）。
  * 测试通过/失败由 _test_run_outcome 判定（pass / unknown / fail），覆盖 Go、Rust、
    Maven、Gradle、CTest、GoogleTest、Jest、Vitest、Mocha、PHPUnit、RSpec、pytest、
    unittest 等的摘要形态。它与 _is_error_result（工具调用错误率口径）**分离**：
    测试断言失败是有效验证行为，不是工具调用失败。
  * 判定「测试执行」逐 shell 段进行（_iter_run_segments），段首是 grep/rm/git/cat
    等只读命令时整段跳过 —— 否则 `git diff .../src/main/java/Foo.java` 会因路径里
    的 java、`rm -f vitest-tmp.config.ts` 会因文件名而被误判成跑测试。
  * 注意：脚手架回填的 `[The command completed with exit code N.]` 不可用作失败信号
    —— 命令常写成 `... | tail -40`，管道使 shell 退出码恒为 tail 的 0。可用的是
    agent 自己回显的 `${PIPESTATUS[0]}` 与输出内容本身。

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

# --- Observation text normalization ---
# 命令 observation 普遍含 ANSI 转义（grep --color、mvn/cargo/gradle 的彩色输出）。
# 转义序列会插进**词内部**，例如 Maven 的
#   "Tests run\x1b[m\x1b[K: 6, \x1b[01;31m\x1b[KFailures\x1b[m\x1b[K: 2"
# 其中 `\x1b[K` 的 `K` 是 word 字符，会让 `\bFAILED\b` 这类边界锚定失效。
# 因此所有基于内容的模式匹配都必须先剥掉 ANSI/回车。
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|\r")


def _strip_ansi(text: str) -> str:
    """剥离 ANSI 转义序列与裸 CR（终端输出普遍存在，会破坏词边界匹配）。"""
    if not text or "\x1b" not in text and "\r" not in text:
        return text
    return _ANSI_RE.sub("", text)


def _text_of(content: Any) -> str:
    """把 observation/message 的 content 归一为纯文本。

    兼容 str 与 content-block 列表两种格式（后者见 _assistant_text_content）。
    没有这层归一时，list 型 content 会让下游正则抛 TypeError。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    parts.append(str(t))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


# --- Error detection ---
# 扫描窗口：命令输出的**结论**通常在末尾（测试摘要、BUILD FAILURE、exit code 回显），
# 而中间是大段正常日志。只扫开头会让超长输出的失败摘要落在窗口外，因此取「头 + 尾」。
_ERROR_SCAN_LIMIT = 3000
_ERROR_SCAN_TAIL = 3000
# 尾窗对齐到行首时，最多向前丢弃多少字符的残行（超过则认为是超长单行，不丢）。
_ERROR_SCAN_ALIGN = 300


def _scan_window(content: Any) -> str:
    """取用于模式匹配的文本窗口：剥离 ANSI 后的「开头 + 结尾」。

    尾窗必须先对齐到行首：text[-N:] 从任意字符位置切开，拼接后被截断的**行中间**
    片段会变成伪造的行首，使 `^FAIL`、`^(?:FAILED|ERROR)\\s`、`^\\s*\\[ERROR\\]` 这些
    行首锚定的模式命中根本不在行首的文本。只在开头 _ERROR_SCAN_ALIGN 字符内找换行。
    """
    text = _strip_ansi(_text_of(content))
    if len(text) <= _ERROR_SCAN_LIMIT + _ERROR_SCAN_TAIL:
        return text
    tail = text[-_ERROR_SCAN_TAIL:]
    nl = tail.find("\n")
    tail = tail[nl + 1:] if 0 <= nl < _ERROR_SCAN_ALIGN else tail
    return text[:_ERROR_SCAN_LIMIT] + "\n" + tail

# 显式工具失败标记：由脚手架/运行时注入，独立于工具输出内容，因此对所有工具类型都
# 可靠。文件查看/编辑工具返回的源码内容不会把这些 wrapper 文本当作数据出现。
_TOOL_ERROR_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"<tool_use_error>"),
    re.compile(r"The arguments provided to the tool are invalid"),
    re.compile(r"\[An error occurred during execution\.\]"),  # OpenHands SDK 工具执行失败
    re.compile(r"Error validating args\b.*\bfor tool\b"),      # OpenHands SDK 参数校验失败
    re.compile(r"Error executing tool\b"),                     # OpenHands SDK 工具执行异常
    # 非 SDK OpenHands 的 str_replace_editor 失败：内容以 "ERROR:" 块开头，或带下列
    # OpenHands 特有的编辑失败措辞。均为起始锚定/高度特异，源码内容不会误命中。
    re.compile(r"^\s*ERROR:"),
    re.compile(r"No replacement was performed"),
    re.compile(r"parameter is required for ['\"]?\w+['\"]? command"),
    re.compile(r"Invalid `[^`]+` parameter"),
]

# 命令输出错误信号：仅对“执行类”工具（bash/terminal/ipython）的 observation 有意义，
# 因为此时 content 才是命令的 stdout/stderr。若对文件查看/编辑/搜索类工具套用，会把
# 返回的源码内容（其中合法地出现 except/raise/AssertionError、FAILED、exit code 等
# token）误判成工具调用错误 —— 这是历史上工具调用错误率假阳性的主因。
_ERROR_PATTERNS_HARD: list[re.Pattern[str]] = [
    re.compile(r"command not found"),
    re.compile(r"Permission denied"),
    # 允许负号：超时/信号杀死时脚手架回填 `exit code -1` / `-9`，缺了 `-?` 会整类漏判。
    re.compile(r"exit code[:\s]+-?[1-9]", re.IGNORECASE),
    re.compile(r"returned non-zero exit status"),
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

# --- 测试结果判定（多语言）---
# 供 TVR 的 late_test_success 使用，**不进入** _ERROR_PATTERNS_SOFT / 工具调用错误率
# 口径 —— 一次正常的「测试跑完但断言失败」是有效的验证行为，不是工具调用失败。
#
# 关键背景：agent 极少直接跑 runner，命令通常形如
#     <runner> ... 2>&1 | tail -40; echo "exit: ${PIPESTATUS[0]}"
# 管道使 shell 退出码恒为 tail 的 0，脚手架回填的
#     [The command completed with exit code 0.]
# 因而**不能**作为失败信号；必须靠输出内容本身与 agent 自己回显的 PIPESTATUS。
# 非零计数：必须排除 0，否则 "0 failed" / "0 failures"（**通过**时的正常输出，Rust、Go、
# pytest、jest、Maven 都会打印）会被判成失败 —— 这是本类模式最容易犯的反向错误。
_NZ = r"(?!0+\b)\d+"

_TEST_FAIL_PATTERNS: list[re.Pattern[str]] = [
    # 通用：agent 自己回显的非零 PIPESTATUS（exit: 1 / EXIT: 2 / ctest exit: 8 /
    # DOCTEST EXIT: 101）。刻意不含 `status:`，否则 HTTP 测试里的 `status: 200` 会误命中。
    re.compile(r"\bexit(?:\s*code)?\s*[:=]\s*-?(?!0+\b)\d{1,3}\b", re.IGNORECASE),
    # pytest / unittest
    re.compile(r"^(?:FAILED|ERROR)\s+\S+", re.MULTILINE),
    re.compile(rf"\b{_NZ}\s+failed\b", re.IGNORECASE),
    re.compile(r"^(?:FAIL|ERROR):\s", re.MULTILINE),
    # Go：独立成行的 FAIL，或 --- FAIL:
    re.compile(r"^\s*---\s*FAIL:", re.MULTILINE),
    re.compile(r"^FAIL\b", re.MULTILINE),
    # Rust：test result: FAILED / 编译错误 / doctest 失败
    re.compile(r"test result:\s*FAILED"),
    re.compile(r"^error(?:\[E\d+\])?:", re.MULTILINE),
    re.compile(r"Couldn't compile the test"),
    re.compile(r"\berror: (?:doctest failed|test failed)"),
    # Java / Maven / Gradle
    re.compile(rf"Tests run:[^\n]*?(?:Failures|Errors):\s*{_NZ}"),
    re.compile(r"\bBUILD FAILURE\b"),
    re.compile(r"^\s*\[ERROR\]\s", re.MULTILINE),
    re.compile(r"\b\w+Test\b\s*>\s*\S+\s+FAILED\b"),
    re.compile(rf"\b\d+\s+tests? completed,\s*{_NZ}\s+failed"),
    # CTest / GoogleTest / Catch2（C、C++ 主力）
    re.compile(r"The following tests FAILED"),
    re.compile(rf"\b{_NZ}\s+tests? failed out of\b"),
    re.compile(r"Errors while running CTest"),
    re.compile(r"\[\s*FAILED\s*\]"),
    # Jest / Vitest / Mocha / Jasmine
    re.compile(rf"^Tests:\s+[^\n]*?\b{_NZ}\s+failed", re.MULTILINE),
    re.compile(rf"^Test Suites:\s+[^\n]*?\b{_NZ}\s+failed", re.MULTILINE),
    re.compile(rf"\b{_NZ}\s+failing\b"),
    re.compile(r"No test files found"),
    re.compile(r"error Command failed with exit code [1-9]"),
    # PHPUnit / RSpec / Ruby minitest
    re.compile(r"^FAILURES!", re.MULTILINE),
    re.compile(rf"\b{_NZ}\s+failures?\b", re.IGNORECASE),
    re.compile(r"\bfail:\s*[1-9]"),
    # 通用编译/构建失败
    re.compile(r"\b(?:compilation|build) failed\b", re.IGNORECASE),
]

# 明确的「测试通过」信号。用于把「确实跑了测试且通过」与「输出里看不出结论」区分开。
# 与失败模式同理，所有计数都必须是**非零**：`0 passing` / `OK (0 tests)` /
# `Tests run: 0, Failures: 0, Errors: 0`（Maven 多模块里没有测试的模块）表示
# 一个测试都没跑，那是「无结论」而不是「通过」，判 pass 会白送 late_test_success=1.0。
_TEST_PASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"test result:\s*ok\."),                       # Rust
    re.compile(r"^ok\s+\S+", re.MULTILINE),                   # Go
    re.compile(r"^PASS\b", re.MULTILINE),                     # Go
    re.compile(rf"\b{_NZ}\s+passed\b", re.IGNORECASE),        # pytest / jest
    re.compile(r"\b100%\s+tests passed\b"),                   # CTest
    re.compile(r"\bBUILD SUCCESS(?:FUL)?\b"),                 # Maven / Gradle
    re.compile(rf"Tests run:\s*{_NZ},\s*Failures:\s*0,\s*Errors:\s*0"),
    re.compile(rf"\bOK\s*\({_NZ} tests?\)"),                  # JUnit / PHPUnit
    re.compile(r"^\s*\[\s*PASSED\s*\]", re.MULTILINE),        # GoogleTest
    re.compile(rf"\b{_NZ}\s+passing\b"),                      # Mocha
    re.compile(rf"\b{_NZ}\s+examples?,\s*0\s+failures\b"),    # RSpec
    re.compile(rf"\bPass:\s*{_NZ},\s*fail:\s*0\b"),           # Ruby minitest
    re.compile(r"\ball (?:tests|checks) (?:passed|PASS)\b", re.IGNORECASE),
    # Python unittest 的标准通过输出：`Ran 97 tests in 9.254s` + 空行 + `OK`
    re.compile(rf"Ran {_NZ} tests? in [\d.]+s\s*\n+\s*OK\b"),
    re.compile(r"^OK\s*(?:\(skipped=\d+\))?\s*$", re.MULTILINE),
    re.compile(r"\berror count:\s*0\b", re.IGNORECASE),       # agent 自建的 tsc 检查
]

# 脚手架注入的尾部标记。这些行里的 "exit code 0" **不能**当作通过信号：命令通常经过
# `| tail -N`，管道使 shell 退出码恒为 tail 的 0，与测试是否通过无关。
# 判定「通过」时先剥掉它们，只认 agent 自己回显的 PIPESTATUS。
_SCAFFOLD_MARKER_RE = re.compile(
    r"^\[(?:The command completed with exit code -?\d+\.|"
    r"Command finished with exit code -?\d+|"
    r"Current working directory:[^\]]*|"
    r"Python interpreter:[^\]]*|"
    r"Below is the output of the previous command\.?)\]\s*$",
    re.MULTILINE,
)
# agent 自己回显的零退出码：`exit: 0` / `exit:0` / `=== ctest exit: 0 ===`
_TEST_PASS_ECHO_RE = re.compile(r"\bexit(?:\s*code)?\s*[:=]\s*0+\b", re.IGNORECASE)

# 「一个测试都没跑起来」的显式标记。runner 此时仍会打印成功收尾（unittest 的裸 `OK`、
# Maven 的 `BUILD SUCCESS`、退出码 0），但什么都没验证 —— 判 pass 会白送满分，
# 必须单列为 unknown。刻意不含 Go 的 `[no test files]`：那是多包运行里没有测试的
# 单个包，同一次运行的其它包仍在正常跑。
_TEST_ZERO_RUN_RE = re.compile(
    r"\bRan 0+ tests?\b"
    r"|\bno tests? (?:ran|were found|to run|found)\b"
    r"|\bcollected 0+ items\b",
    re.IGNORECASE,
)


def _test_run_outcome(content: Any) -> str:
    """判定一次测试执行的结果：'fail' / 'pass' / 'unknown'。

    与 _is_error_result 分开：测试断言失败是有效的验证行为，不应计入工具调用错误率。

    判定顺序刻意如此：
    1. 工具层/运行时硬失败（脚手架失败标记、command not found、非零退出码）—— 命令
       根本没跑起来，优先于任何输出内容。
    2. 显式失败摘要（多语言）。失败优先于通过：`11 passed; 1 failed` 判 fail。
    3. 显式通过摘要。放在 soft error 之前 —— 一次通过的测试完全可能在输出里合法地
       打印 `TypeError` / `No such file or directory`（正是它在测的错误路径），
       此时 `100% tests passed` 应当压过这些 soft 信号。
    4. soft error 兜底。
    5. 都没有 → unknown（输出里看不出结论，例如只 `| tail -5` 截了几行日志）。
    """
    text = _scan_window(content)
    if not text:
        return "unknown"
    for pat in _TOOL_ERROR_MARKERS:
        if pat.search(text):
            return "fail"
    for pat in _ERROR_PATTERNS_HARD:
        if pat.search(text):
            return "fail"
    for pat in _TEST_FAIL_PATTERNS:
        if pat.search(text):
            return "fail"
    # 「零测试跑成」优先于任何通过信号：runner 仍会打印 OK / BUILD SUCCESS / exit 0
    if _TEST_ZERO_RUN_RE.search(text):
        return "unknown"
    for pat in _TEST_PASS_PATTERNS:
        if pat.search(text):
            return "pass"
    # agent 自己回显的零退出码（需先剥掉脚手架恒为 0 的尾部标记，见 _SCAFFOLD_MARKER_RE）
    if _TEST_PASS_ECHO_RE.search(_SCAFFOLD_MARKER_RE.sub("", text)):
        return "pass"
    for pat in _ERROR_PATTERNS_SOFT:
        if pat.search(text):
            return "fail"
    return "unknown"

# --- Tool classification ---
# 纯写工具：调用即写文件。须含 Claude Code 的 MultiEdit / NotebookEdit 与 OpenCode 的
# patch，否则这些脚手架的编辑会被 FEC 整体漏掉。
_PURE_EDIT_TOOL_NAMES = frozenset({
    "edit", "write", "multiedit", "notebookedit", "patch",
})
_MULTI_EDITOR_TOOL_NAMES = frozenset({"file_editor", "str_replace_editor"})
_EDITOR_WRITE_COMMANDS = frozenset({"str_replace", "create", "insert"})
_EDIT_TOOL_NAMES = _PURE_EDIT_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES
_BASH_TOOL_NAMES = frozenset({"bash", "terminal", "execute_bash"})
# 执行类工具：其 observation 内容是命令/代码的真实输出，可套用命令输出错误模式
# （_ERROR_PATTERNS_HARD/SOFT）。其余工具（文件查看/编辑/搜索等）仅依据显式工具
# 失败标记（_TOOL_ERROR_MARKERS）判定，避免把返回的源码内容误判为工具调用错误。
#
# 工具名常被各 SWE 数据源重命名（shell_exec/run_command/exec_cmd/code_editor/...），
# 因此执行/非执行的判别**以工具调用参数为准**（见 _tool_call_is_execution），下面的
# 名单只作为参数缺失/命令为空时的快速回退。
_EXECUTION_TOOL_NAMES = frozenset({
    "bash", "terminal", "execute_bash", "shell", "cmd",
    "execute_ipython_cell", "run_ipython", "ipython", "python", "run_python",
})
# 编辑器子命令：command 取这些值即文件编辑/查看工具（非执行类）。
_EDITOR_SUBCOMMANDS = frozenset({"view", "create", "str_replace", "insert", "undo_edit"})
# 编辑器专属参数键：bash 类工具从不携带，用作非执行类的辅助判据。
_EDITOR_ONLY_ARG_KEYS = ("old_str", "new_str", "file_text", "view_range", "insert_line")
# 同理的非执行类专属键。task_tracker 也带 command（取值是 plan/add 而非 shell 命令），
# 只看「command 非空」会把它判成执行类。
_NON_EXEC_ARG_KEYS = ("task_list", "thought", "task_completed")
# 携带 shell 命令的参数键。
_SHELL_CMD_ARG_KEYS = ("command", "cmd", "keystrokes")
_FILE_PATH_KEYS = ("file_path", "filePath", "path", "notebook_path", "notebookPath")

# --- Test detection ---
_TEST_RUN_RE = re.compile(
    r"\b(?:"
    r"pytest|py\.test|python3?\s+-m\s+pytest|python3?\s+-m\s+unittest"
    r"|unittest|python3?\s+test_|nosetests"
    r"|go\s+test|cargo\s+test|ctest|gtest_filter"
    # Maven / Gradle：goal 与命令之间通常隔着一串 flag 和模块选择器，不能要求紧邻
    # （`mvn -o -q test`、`./mvnw -q -pl mod -am test`、`./gradlew -q :mod:test`）
    r"|(?:mvn|mvnw)\b[^\n|&;]{0,200}?\b(?:test|verify|surefire:test|integration-test)\b"
    # tempered：`-x test` 是 Gradle 的**排除** test，不能算跑测试
    r"|(?:gradle|gradlew)\b(?:(?!-x\s)[^\n|&;]){0,200}?\b[\w:]*[Tt]est[\w:]*\b"
    r"|jest|mocha|vitest|npx\s+jest|npx\s+vitest|npx\s+mocha"
    r"|(?:npm|yarn|pnpm)\s+(?:run\s+)?test"
    r"|make\s+(?:test|check)"
    # Django 家族：./tests/runtests.py / python tests/runtests.py / manage.py test
    # （下方 `\./...(?:_test|test_)` 要求 test_/_test，"tests/runtests.py" 两者都不含）
    r"|runtests\.py|manage\.py\s+test"
    # CMake/CTest 的常见调用形式；ctest 已在上面，这里补 cmake --build --target test
    r"|cmake\s+--build\s+\S+\s+--target\s+(?:test|check)"
    # 多语言正式 test runner：Elixir(mix)/Swift/Clojure(lein)/.NET/Lua(busted)
    r"|mix\s+test|swift\s+test|lein\s+test|dotnet\s+test|busted"
    r")\b"
    r"|\.\/[^\s]*(?:_test|test_)\S*"
    # sbt 子项目测试（Scala）：sbt test / sbt "core/test" / sbt testOnly ...
    r"|\bsbt\b[^\n|&;]*?\b(?:test(?:Only|Quick)?|it:test)\b"
    # R 测试（常内联在 R -e \"...\"）：devtools::test() / testthat::test_*(...)
    r"|\b(?:devtools::test|testthat::test)"
    # Neovim Lua 测试：PlenaryBustedDirectory / PlenaryBustedFile
    r"|\bPlenaryBusted\w*",
    re.IGNORECASE,
)

# --- Bash edit path extraction ---
_WORKSPACE_PREFIXES = ("/workspace/", "/testbed/", "/repo/", "/home/swe-bench/")
# `cat > file` / `cat >> file`（含 heredoc 写文件）。不匹配 `cat foo 2>/dev/null`。
_CAT_WRITE_RE = re.compile(r"\bcat\s+(?:>>?)\s*([^\s|&;<>]+)")
# `cat inputs... > outfile`：同样排除 fd 重定向。
_CAT_STDOUT_RE = re.compile(r"\bcat\s+(?!>>?)(?:[^|&;\n]*?)(?<![0-9&])>>?\s*([^\s|&;<>]+)")
# `echo/printf ... > file`：重定向必须落在同一条简单命令内（中间不能有 |/&/;）。
_ECHO_PRINTF_WRITE_RE = re.compile(
    r"\b(?:echo|printf)\b(?:[^|&;\n]*?)(?<![0-9&])>>?\s*([^\s|&;<>]+)"
)
_TEE_RE = re.compile(r"\btee\s+(?:-[a-zA-Z]+\s+)*([^\s|&;<>]+)")
# 就地编辑：GNU/BSD sed 的各种写法，以及 macOS 习惯的 gsed、Perl 的 -i。
# 覆盖 -i / -i.bak / -pi / -pi.orig。刻意只允许纯字母的短选项簇（可带 .后缀），
# 以免 `perl -MList::Util -e ...` 这类含 i 的模块名被当成 -i 就地编辑。
_SED_INPLACE_RE = re.compile(
    r"\b(?:g?sed|perl)\s+(?:-\S+\s+)*-[A-Za-z]*i[A-Za-z]*(?:\.[\w.]+)?(?=\s|$)"
)
# `patch` 作为命令，避免命中 `foo.patch &&` 这类扩展名。
_PATCH_FILE_RE = re.compile(r"(?:^|[\n;&|]\s*)patch\s+(?:-\S+\s+)*([^\s|&;<>]+)")
_SHELL_META_TOKENS = frozenset({"&&", "||", "|", ";", "&"})
_NON_EDIT_PATH_PREFIXES = ("/dev/", "/proc/", "/sys/")

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


def _tool_call_is_execution(name_lower: str, args: dict) -> bool:
    """判断一个 tool_call 是否为“执行类”（运行 shell 命令 / 代码）。

    各数据源常把工具改名（shell_exec / run_command / exec_cmd / code_editor /
    source_editor / ...），因此**以参数特征为准**、工具名只作回退：

    - command 取值为编辑器子命令（view/create/str_replace/insert/undo_edit），或
      携带编辑器专属参数（old_str/new_str/file_text/view_range/insert_line）→ 文件
      查看/编辑工具，非执行类。
    - 携带 task_list/thought/task_completed → task_tracker/think/finish，非执行类
      （task_tracker 也有 command 参数，但取值是 plan/add 而非 shell 命令）。
    - 否则若携带非空的 shell 命令参数（command/cmd/keystrokes）→ 执行类。
    - 都不满足时回退到执行类工具名单（处理如 execute_bash 传空 command 的情况）。
    """
    if not isinstance(args, dict):
        return name_lower in _EXECUTION_TOOL_NAMES
    cmd = args.get("command")
    if isinstance(cmd, str) and cmd in _EDITOR_SUBCOMMANDS:
        return False
    if any(k in args for k in _EDITOR_ONLY_ARG_KEYS):
        return False
    if any(k in args for k in _NON_EXEC_ARG_KEYS):
        return False
    for key in _SHELL_CMD_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return name_lower in _EXECUTION_TOOL_NAMES


def _exec_cmd_text(args: dict) -> str:
    """取执行类 tool_call 的命令/代码文本（bash command / ipython code / 改名变体）。"""
    for key in ("command", "code", "cmd", "keystrokes"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _normalize_path(p: str) -> str:
    """归一化文件路径。"""
    for prefix in _WORKSPACE_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return posixpath.normpath(p)


def _is_plausible_edit_path(path: str) -> bool:
    """过滤重定向/shell 元字符等不应计入 FEC 的伪路径。"""
    if not isinstance(path, str):
        return False
    p = path.strip().strip("'\"")
    if not p or p in _SHELL_META_TOKENS:
        return False
    if p.startswith("<<") or p.startswith("&"):
        return False
    # `2>/dev/null;`、`foo;`、含重定向符的 token 都不是文件路径
    if ">" in p or "<" in p or ";" in p or "|" in p:
        return False
    if re.fullmatch(r"[0-9]+", p) or re.fullmatch(r"[&|;\d<>]+", p):
        return False
    if "/dev/" in p or "/proc/" in p or "/sys/" in p:
        return False
    if any(p == pref.rstrip("/") or p.startswith(pref) for pref in _NON_EDIT_PATH_PREFIXES):
        return False
    return True


def _extract_bash_edit_paths(cmd_str: str) -> list[str]:
    """从 bash 命令字符串中提取文件编辑目标路径。

    只保留真实写文件目标，显式忽略：
    - fd/设备重定向：`2>/dev/null`、`2>&1`、`>/dev/null`
    - 与写操作无关、仅因命令中出现 `echo` + 别处的 `>` 而被误抓的路径
    - `foo.patch &&` 这类扩展名误匹配
    - `sed -i ... file && echo` 后半段的 shell 操作符/命令词
    """
    if not isinstance(cmd_str, str) or not cmd_str:
        return []

    paths: list[str] = []

    for cre in (_CAT_WRITE_RE, _CAT_STDOUT_RE, _ECHO_PRINTF_WRITE_RE, _TEE_RE, _PATCH_FILE_RE):
        for m in cre.finditer(cmd_str):
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
            if tok in _SHELL_META_TOKENS or tok.startswith("&&") or tok.startswith("||"):
                break
            # `file 2>/dev/null` / 后续 shell 命令：停止采集，避免把重定向和命令词当路径
            if ">" in tok or "<" in tok or ";" in tok:
                break
            if tok.startswith("-"):
                if tok in ("-e", "-E", "-f"):
                    # -e/-f 的参数就是脚本表达式本身，跳过它并标记表达式已消耗，
                    # 否则后面的真实文件名会被当成表达式吃掉（perl -pi -e '...' f）
                    skip_next = True
                    in_expr = True
                continue
            if not in_expr and (
                tok.startswith("'") or tok.startswith('"') or tok.startswith("s") or tok.startswith("/")
            ):
                in_expr = True
                continue
            if in_expr:
                if not _is_plausible_edit_path(tok):
                    break
                paths.append(tok.strip("'\""))

    return [p for p in paths if _is_plausible_edit_path(p)]


def _exec_bash_edit_paths(name_lower: str, args: dict) -> list[str]:
    """执行类 tool_call 的命令文本中包含的文件编辑目标路径（未归一化）。

    单一入口，供所有需要「这个 tool_call 是否通过 shell 改了文件」的地方复用，必须走
    _tool_call_is_execution / _exec_cmd_text 而非工具名单 + args["command"]。否则改名
    shell 工具、参数键为 code/keystrokes 的工具，其 bash 编辑会被整体漏掉，使 DPI 误判
    「从未成功写入」、SCP 判 0 —— 同一条轨迹仅因工具改名就换一个分数。
    """
    if not _tool_call_is_execution(name_lower, args):
        return []
    return _extract_bash_edit_paths(_exec_cmd_text(args))


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
                    "tool_name": None,  # Terminus2 全部为终端命令输出，按执行类处理
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
            tool_name = ""
            is_execution = False
            if prev_assistant_idx is not None:
                pos = _position_counter.get(prev_assistant_idx, 0)
                _position_counter[prev_assistant_idx] = pos + 1
                tcs = messages[prev_assistant_idx].get("tool_calls", []) or []
                if pos < len(tcs):
                    tool_name, tool_args = _parse_tool_call(tcs[pos])
                    is_execution = _tool_call_is_execution(tool_name, tool_args)
            observations.append({
                "content": msg.get("content", ""),
                "msg_idx": idx,
                "prev_assistant_idx": prev_assistant_idx,
                "tool_call_position": pos,
                "tool_name": tool_name,
                "is_execution": is_execution,
            })

    return observations


# ═══════════════════════════════════════════════════════════════════════════
# Error detection
# ═══════════════════════════════════════════════════════════════════════════

def _is_error_result(content: Any, is_test_output: bool = False, is_execution: bool = True) -> bool:
    """检测 observation 是否包含错误信息。

    判定分两层：

    1. 显式工具失败标记（_TOOL_ERROR_MARKERS）：由脚手架注入，对所有工具类型生效。
    2. 命令输出错误模式（_ERROR_PATTERNS_HARD/SOFT）：仅当 is_execution=True
       （bash/terminal/ipython 等执行类工具）时才扫描。非执行类工具（文件查看、
       编辑、搜索等）返回的是源码/文件数据而非命令输出，套用这些模式会把源码里
       合法出现的 except/raise/FAILED 等 token 误判成工具调用错误。

    is_execution 默认 True 以保持旧调用点行为；逐 observation 的判定（_is_obs_error /
    _count_tool_call_errors）会按产生该 observation 的工具传入正确的值。

    文本先经 _scan_window 归一：剥离 ANSI、并同时取输出的开头与结尾（结论性信息
    通常在末尾）。
    """
    if not content:
        return False
    text = _scan_window(content)
    if not text:
        return False
    for pat in _TOOL_ERROR_MARKERS:
        if pat.search(text):
            return True
    if not is_execution:
        return False
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
            if isinstance(cmd, dict) and _is_formal_test_run_cmd(cmd.get("keystrokes", "")):
                return True
        return False

    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        # 以参数判别执行类工具（与 _extract_observations / _is_obs_error 一致）。
        # 漏判的后果是双重的：SUB 的 late-test bonus 拿不到，且该输出不被识别为
        # test output，于是正常的测试失败被计成 tool call 错误，连带压低 SUB 与 DPI。
        if _tool_call_is_execution(name_lower, args):
            if _is_formal_test_run_cmd(_exec_cmd_text(args)):
                return True
    return False


def _get_per_toolcall_test_flags(msg: dict) -> list[bool]:
    """返回 assistant 消息中每个 tool_call 是否为测试执行命令。"""
    flags: list[bool] = []
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        is_test = False
        if _tool_call_is_execution(name_lower, args):
            is_test = _is_formal_test_run_cmd(_exec_cmd_text(args))
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


def _obs_is_execution(obs: dict, scaffold: str) -> bool:
    """该 observation 是否由执行类工具（命令/代码执行）产生。

    Terminus2 的 observation 全部是终端命令输出，按执行类处理；其它脚手架在
    _extract_observations 中已按工具调用参数判定并写入 obs["is_execution"]
    （见 _tool_call_is_execution，对工具改名鲁棒）。
    """
    if scaffold == "terminus2":
        return True
    return bool(obs.get("is_execution"))


def _is_obs_error(obs: dict, messages: list[dict], scaffold: str, _test_flags_cache: dict) -> bool:
    """判断 observation 是否为错误（区分执行类/非执行类工具，排除测试输出误判）。"""
    is_test = _resolve_is_test_for_obs(obs, messages, scaffold, _test_flags_cache)
    is_exec = _obs_is_execution(obs, scaffold)
    return _is_error_result(obs.get("content", ""), is_test_output=is_test, is_execution=is_exec)


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
    # 改名的 shell/ipython 工具：按参数特征兜底判为 bash，否则 IAC/TTE/PED 会把
    # 整条轨迹的执行动作全归到 "other"。
    if _tool_call_is_execution(name_lower, args):
        return "bash"
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
                for p in _exec_bash_edit_paths(name_lower, args):
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
                if _is_write_operation(name_lower, args) or _exec_bash_edit_paths(name_lower, args):
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


def _loop_signature_for_action(
    a: dict,
    messages: list[dict],
    occurrence_in_msg: int,
) -> tuple[str, str, str] | None:
    """构造 DPI loop 检测用的 action 签名。

    返回 (tool_name, editor_subcommand, target_path)；无明确 target 时返回 None。

    对 file_editor / str_replace_editor，签名必须带上 command（view/str_replace/...），
    否则同文件上的「查看→编辑」正常迭代会被误判成死循环。
    """
    tool_name = a["tool_name"].lower()
    msg = messages[a["msg_idx"]]

    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        match_i = 0
        for tc in msg.get("tool_calls", []) or []:
            name_lower, args = _parse_tool_call(tc)
            if name_lower != tool_name:
                continue
            if match_i != occurrence_in_msg:
                match_i += 1
                continue

            target = ""
            path = _get_file_path(args)
            if path:
                target = _normalize_path(path)
            else:
                edit_paths = _exec_bash_edit_paths(name_lower, args)
                if edit_paths:
                    target = _normalize_path(edit_paths[0])
            if not target:
                return None

            subcmd = ""
            if name_lower in _MULTI_EDITOR_TOOL_NAMES:
                subcmd = str(args.get("command", "") or "")
            return (tool_name, subcmd, target)

        return None

    # Terminus2 / 无 tool_calls：退化为 target-only 签名
    files = _action_target_files(a, messages)
    if not files:
        return None
    return (tool_name, "", sorted(files)[0])


def _compute_loop_fraction(actions: list[dict], messages: list[dict], scaffold: str) -> float:
    """计算连续重复步骤占比。

    只有签名完全相同且 target 非空的连续 run 才计为 loop。
    签名为 (tool_name, editor_subcommand, target)：
      - file_editor/str_replace_editor 带上 view/str_replace 等子命令，避免把
        同文件上的查看+编辑迭代误判为循环
      - bash/terminal 无明确 target 时不参与（连续执行命令是正常行为）
    """
    if len(actions) < _LOOP_RUN_MIN_LEN:
        return 0.0

    signatures: list[tuple[str, str, str] | None] = []
    seen_counts: dict[tuple[int, str], int] = {}
    for a in actions:
        key = (a["msg_idx"], a["tool_name"].lower())
        occ = seen_counts.get(key, 0)
        seen_counts[key] = occ + 1
        signatures.append(_loop_signature_for_action(a, messages, occ))

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
            # 先剥 ANSI 再取签名：同一条错误在不同轮次可能带不同的控制序列，不归一
            # 会让重复错误看起来各不相同。_text_of 兼容 list 型 content。
            sig = _strip_ansi(_text_of(obs.get("content", "")))[:80]
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
                for p in _exec_bash_edit_paths(name_lower, args):
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
                if _is_write_operation(name_lower, args) or _exec_bash_edit_paths(name_lower, args):
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

# 文本收尾的「完成信号」：部分脚手架/模型会以一段文字总结收尾而不调 finish 工具。
# 若该文字明确宣告任务完成（强信号）或含完成类动词（弱信号），视为正常收尾而非半成品，
# 避免把「已修复 + tests passed，只是没发 finish」一刀切判 0.5。
_SUB_COMPLETION_STRONG_RE = re.compile(
    r"\b(?:"
    r"(?:fix|task|issue|bug|change|patch)\s+(?:is\s+)?(?:complete|completed|fixed|resolved|done|verified|ready)"
    r"|(?:all\s+)?tests?\s+(?:now\s+)?pass(?:ed|ing)?"
    r"|successfully\s+(?:fixed|resolved|implemented|verified|applied)"
    r"|(?:the\s+)?(?:fix|patch|implementation)\s+(?:is\s+)?(?:complete|ready|working)"
    r")\b",
    re.IGNORECASE,
)
_SUB_COMPLETION_WEAK_RE = re.compile(
    r"\b(?:complete|completed|fixed|done|resolved|verified)\b",
    re.IGNORECASE,
)


def _assistant_text_content(message: dict[str, Any]) -> str:
    """取 assistant 消息的纯文本内容（兼容 str 与 content-block 列表两种格式）。"""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text") is not None:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content or "")


def _text_summary_sub_base(content: str) -> float:
    """给「以文本总结收尾、未调 finish 工具」的末轮打 base 分。

    强完成信号=1.0, 弱完成信号=0.8, 其它文本=0.5（与旧版半成品同分）。
    """
    text = content.strip()
    if not text:
        return 0.5
    if _SUB_COMPLETION_STRONG_RE.search(text):
        return 1.0
    if _SUB_COMPLETION_WEAK_RE.search(text):
        return 0.8
    return 0.5


def _compute_sub(messages: list[dict], observations: list[dict], scaffold: str) -> float:
    """提交完整性 + 最终状态质量。

    Base: finish 工具=1.0, 文本总结(含完成信号)=0.8-1.0, 其它文本=0.5, 截断=0.0
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
                base = _text_summary_sub_base(_assistant_text_content(last_msg))
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
                else:
                    # 与 TVR / DPI / SCP / 错误检测共用 _exec_bash_edit_paths：
                    # 执行类判别以参数为准而非工具名单。
                    paths.extend(_normalize_path(p) for p in _exec_bash_edit_paths(name_lower, args))
    return paths


def _compute_fec(messages: list[dict], scaffold: str) -> float | None:
    """文件编辑集中度: 1 - clip((mean_edits - 1) / 4, 0, 1)。

    没有检测到任何编辑时返回 None（fail-soft，聚合时跳过），**不能返回 1.0** ——
    那等于把「检测盲区」和「完美的编辑集中度」判成同一件事；聚合层还要对 FEC 取
    5 次幂（3 次同文件编辑只剩 0.031），漏抓反而成了加分项。
    """
    paths = _extract_edit_paths(messages, scaffold)
    if not paths:
        return None
    file_counts = Counter(paths)
    unique_files = len(file_counts)
    if unique_files == 0:
        return None
    mean_edits = len(paths) / unique_files
    normalized = max(0.0, min((mean_edits - 1) / (_FEC_THRESHOLD - 1), 1.0))
    return 1.0 - normalized


# ═══════════════════════════════════════════════════════════════════════════
# 10. STP — 步数效率 (r=-0.21 with resolved: fewer steps = better)
# ═══════════════════════════════════════════════════════════════════════════

# 与当前 agent max_turn=200 对齐：满分落到中短轨迹，打满 turn cap 得 0。
_STP_OPTIMAL_RANGE = (5, 80)
_STP_MAX_STEPS = 200


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

# 代码文件后缀（多语言测试/复现文件判定共用）。
_CODE_EXT = (
    r"(?:py|js|jsx|ts|tsx|mjs|cjs|go|rs|java|kt|kts|scala|rb|php|cc|cpp|cxx|cu|"
    r"c|h|hpp|hxx|cs|swift|mm|pl|pm|ex|exs|lua|cljc|cljs|clj|edn|dart|sol|jl|"
    r"erl|groovy|r|sh)"
)

# 正式测试文件（多语言）。覆盖 test_x / x_test / x_spec / x.test.x、Java/Kotlin/Scala
# 的 *Test(s)/*IT、C/C++ 的 *_test.cc 等，以及 tests//__tests__//spec/ 目录下的代码文件。
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    r"conftest\.py"
    rf"|test_[^/]+\.{_CODE_EXT}"
    rf"|[^/]+_tests?\.{_CODE_EXT}"
    rf"|[^/]+_spec\.{_CODE_EXT}"
    r"|[^/]+\.(?:test|spec)\.(?:m?[jt]sx?|cjs)"
    # *Test/*Tests/*Spec 类命名（Java/Kotlin/Scala/PHP/C#/C++/Go/Swift/Dart）。
    # 必须用 (?-i:) 关掉本正则的 IGNORECASE 并要求 CamelCase —— 这类命名的判据本就是
    # 大小写；大小写不敏感时 `latest.py`/`greatest.go`/`contest.js`/`attest.go`
    # 都会因为词尾恰好是 "test" 而被误判成测试文件。
    rf"|[^/]*(?-i:[A-Z]\w*(?:Tests?|Spec))\.{_CODE_EXT}|[^/]+IT\.java"
    r")$"
    rf"|(?:^|/)(?:tests?|__tests__|specs?|testing)/(?:[^/]+/)*[^/]+\.{_CODE_EXT}$"
    # Perl 的 t/ 目录
    r"|(?:^|/)t/(?:[^/]+/)*[^/]+\.t$"
    # Java/Kotlin/Scala 的 Maven/Gradle 标准测试目录 src/test/...
    rf"|(?:^|/)src/(?:test|integrationTest|androidTest)/(?:[^/]+/)*[^/]+\.{_CODE_EXT}$",
    re.IGNORECASE,
)


# --- TVR-only verification idioms ---
# SWE-agent 轨迹常用「写 reproduce/verify 脚本 + 内联自测」来验证，而非正式 test
# runner，且这些仓库可能是任意语言（JS/TS/C++/Go/Rust/PHP/Ruby...）。这些信号只用于
# TVR，不进入错误抑制路径（_TEST_RUN_RE），以免一个真正报错的内联 traceback 被当成
# 测试输出而在 DPI/错误率中被吞掉。
_TVR_VERIFY_RE = re.compile(
    # 复现/验证脚本（任意语言、任意后缀），如 reproduce_bug.js / verify.cpp / repro.go
    rf"(?:^|[\s/])\S*(?:reproduce|repro|verify|smoke)\w*\.{_CODE_EXT}\b"
    # Python 内联自测 / 跑 test_ 脚本
    r"|\bpython3?\s+-c\b"
    r"|\bpython3?\s+(?:-\S+\s+)*(?:\S*/)?(?:test_\S+|\S+_test)\.py\b"
    # 跑普通脚本（非 test 命名）也算验证信号 —— 这是本模块**有意**的设计口径：
    # 编译/解释型语言下「跑一遍看会不会崩」就是 agent 的复现验证方式。该口径必须对
    # 所有语言一致落实（含 Python / Node），否则同一条规则会产生跨语言偏差。
    # 明显属于构建/安装的调用由下方 _TVR_NONVERIFY_RE 统一剔除。
    r"|\bpython3?\s+(?:-\S+\s+)*(?:\S*/)?\S+\.py\b"
    r"|\b(?:node|bun|deno)\s+(?:-\S+\s+)*(?:\S*/)?\S+\.(?:m?[jt]sx?|cjs)\b"
    # JS/TS：node -e 内联；跑 reproduce/verify/smoke 或 test/spec 命名脚本。
    r"|\b(?:node|deno)\s+(?:-e|--eval|eval)\b"
    r"|\b(?:node|npx|deno|bun|ts-node|tsx)\s+(?:-\S+\s+)*\S*(?:repro|verify|smoke|test|spec)\S*\.(?:m?[jt]sx?|cjs)\b"
    # 其它解释器内联与跑脚本
    r"|\b(?:ruby|perl)\s+-e\b|\bphp\s+-r\b"
    r"|\b(?:ruby|php|perl|Rscript)\s+(?:-\S+\s+)*\S+\.\w+\b"
    # R 内联自测：R -e / Rscript -e（含 devtools::test / testthat 复现）
    r"|\bR(?:script)?\b\s+(?:--\S+\s+)*-e\b"
    # 编译/解释型语言运行：go run / cargo run|test / swift run / dotnet run
    r"|\bgo\s+run\b|\bcargo\s+(?:test|run)\b|\bswift\s+run\b|\bdotnet\s+run\b"
    # Elixir：mix run / elixir|iex 跑 .ex(s) 脚本
    r"|\bmix\s+run\b|\b(?:elixir|iex)\s+(?:-\S+\s+)*\S+\.exs?\b"
    # Clojure：lein run / clj|clojure -e|-M|-X 内联
    r"|\blein\s+run\b|\b(?:clj|clojure)\s+(?:-\S+\s+)*-[eMX]\S*"
    # Java：跑 jar 或临时复现类（首字母大写的 main 类，大小写敏感避免误吞 -version）
    r"|\bjava\b[^\n|&;]*?(?:-jar\s+\S+|\b(?-i:[A-Z])\w+)\b"
    # Neovim Lua：nvim --headless 执行 dofile('test_*/repro/verify') 复现脚本
    r"|dofile\(\s*['\"][^'\"]*(?:test|repro|verify|spec)"
    # 构建系统 test runner（补 _TEST_RUN_RE 未含的）
    r"|\b(?:bazel|blaze)\s+test\b|\bdotnet\s+test\b|\bphpunit\b|\brspec\b|\btox\b"
    # 跑编译产物 / 测试脚本：./repro  ./a.out  ./run_tests.sh  ./x_test
    r"|\./\S*(?:_test|test_|repro|verify|smoke)\S*"
    r"|\./\S+\.(?:sh|out|bin|exe)\b",
    re.IGNORECASE,
)
# 明确**不是**验证行为的命令：依赖安装、打包、起开发服务器、脚手架生成。
# 这些既不是跑测试也不是「跑一遍程序看会不会崩」，不应计入 TVR 的 has_test_run。
# 注意：`./build.sh` / `go run` / `cargo run` 一类仍按既定设计口径计为验证信号。
_TVR_NONVERIFY_RE = re.compile(
    r"\b(?:pip3?|uv\s+pip|conda|apt|apt-get|yum|brew)\s+install\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:install|i|ci|add)\b"
    r"|\bpython3?\s+(?:\S*/)?setup\.py\s+(?:install|build|develop|egg_info|sdist|bdist\w*)\b"
    r"|\bmanage\.py\s+(?:runserver|migrate|makemigrations|collectstatic|shell|startapp|startproject)\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:run\s+)?(?:dev|start|build|watch|serve|lint|format)\b"
    r"|\bcargo\s+(?:build|check|fmt|clippy|install)\b"
    r"|\bgo\s+(?:build|install|mod|vet|fmt)\b"
    # `python -c` 被 agent 大量用来**改文件**（open(...,'w')/.write()），而非内联自测。
    # 这类是编辑操作，不该计入 has_test_run。
    r"|\bpython3?\s+-c\b[\s\S]*?(?:open\s*\([^)]*['\"][wa]b?\+?['\"]|\.write\s*\()",
    re.IGNORECASE,
)
_TVR_REPRO_FILE_RE = re.compile(
    r"(?:^|/)\S*(?:reproduce|repro|verify|smoke)\S*\.\w+$",
    re.IGNORECASE,
)


# late_test_success 的三档取值，见 _compute_tvr 文档。
_LATE_TEST_SCORE = {"pass": 1.0, "unknown": 0.6, "fail": 0.3}


# 只做读取/搬运、绝不构成「跑测试」的命令头。段首是这些词时整段跳过 —— 否则
# `git diff .../src/main/java/Foo.java` 会因路径里的 "java" 命中 java 运行分支，
# `rm -f vitest-tmp.config.ts` 会因文件名命中 vitest，`grep -n "go test" x` 会自命中。
_NONRUN_CMD_HEADS = frozenset({
    "rm", "ls", "cat", "grep", "egrep", "fgrep", "rg", "ag", "find", "git", "echo",
    "mkdir", "rmdir", "cp", "mv", "touch", "head", "tail", "wc", "sed", "awk",
    "diff", "which", "whereis", "chmod", "chown", "cd", "sort", "uniq", "xargs",
    "export", "printf", "tree", "stat", "du", "df", "ln", "tar", "unzip", "gzip",
    "curl", "wget", "pwd", "env", "true", "false", "sleep", "kill", "ps",
})

# 段首可跳过的包装器；timeout 还要再吃掉一个时长参数。
_CMD_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "timeout", "stdbuf"})
_ENV_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=\S*\Z")
_DURATION_RE = re.compile(r"[\d.]+[smhd]?\Z")


def _split_shell_segments(cmd: str) -> list[str]:
    """把 shell 命令链拆成简单命令片段（用于逐段判定，避免整串误判）。

    命令多为 `cd <dir> && <runner> ... | tail -40; echo ...` 这类复合形式，整串匹配会让
    `pip install && pytest` 里的 install 段和 test 段互相污染。

    分隔符必须在**引号之外**才算数，且刻意不按换行拆：多行的
    `python3 -c "p='x'; open(p,'w').write(s)"` 若按裸正则拆，脚本正文里的 `;` 会把
    一条命令切成几段，导致针对整段正文的判定（如「这个 python -c 其实是在改文件」）
    全部失效。

    已知代价：多行命令的段首若是只读命令（如 heredoc 写文件的 `cat`），其后另起一行
    的真实执行命令会被一并跳过。
    """
    segs: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(cmd)
    while i < n:
        ch = cmd[i]
        if quote is not None:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(cmd[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(cmd[i + 1])
            i += 2
            continue
        if cmd.startswith("&&", i) or cmd.startswith("||", i):
            segs.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in ";|&":
            segs.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segs.append("".join(buf))
    return [s for s in segs if s.strip()]


def _segment_head(seg: str) -> str:
    """取一个命令段的实际命令词（跳过前置环境变量赋值与 sudo/timeout 等包装器）。"""
    toks = seg.strip().split()
    i = 0
    while i < len(toks) and _ENV_ASSIGN_RE.match(toks[i]):
        i += 1
    while i < len(toks) and toks[i] in _CMD_WRAPPERS:
        wrapper = toks[i]
        i += 1
        if wrapper == "timeout":
            while i < len(toks) and toks[i].startswith("-"):
                i += 1
            if i < len(toks) and _DURATION_RE.match(toks[i]):
                i += 1
    if i >= len(toks):
        return ""
    head = toks[i]
    # 去掉路径前缀：/opt/venv/bin/python -> python，./node_modules/.bin/jest -> jest
    return head.rsplit("/", 1)[-1]


def _iter_run_segments(cmd: str):
    """逐段产出「可能真的在执行什么」的命令段（已排除纯读取/搬运类命令头）。"""
    for seg in _split_shell_segments(cmd):
        if _segment_head(seg) in _NONRUN_CMD_HEADS:
            continue
        yield seg


def _is_formal_test_run_cmd(cmd: str) -> bool:
    """严格口径：是否调用了正式 test runner（pytest/go test/cargo test/mvn test/...）。

    用于错误抑制路径（哪些 observation 属于「测试输出」）。逐段判定，避免
    `rm -f vitest-tmp.config.ts`、`grep "go test"` 这类文件名/字面量误命中。
    """
    if not cmd:
        return False
    return any(_TEST_RUN_RE.search(seg) for seg in _iter_run_segments(cmd))


def _is_test_run_cmd(cmd: str) -> bool:
    """TVR 口径的测试执行判定：正式 runner 或 reproduce/inline 自测。

    逐段判定并排除安装/打包/起服务（见 _TVR_NONVERIFY_RE），否则
    `python setup.py build`、`npm run dev` 会被当成验证。
    """
    if not cmd:
        return False
    for seg in _iter_run_segments(cmd):
        if _TVR_NONVERIFY_RE.search(seg):
            continue
        if _TEST_RUN_RE.search(seg) or _TVR_VERIFY_RE.search(seg):
            return True
    return False


def _is_test_file_path(path: str) -> bool:
    """TVR 口径的测试文件判定：正式测试文件或 reproduce/verify 脚本。"""
    if not path:
        return False
    return bool(_TEST_FILE_RE.search(path) or _TVR_REPRO_FILE_RE.search(path))


def _compute_tvr(messages: list[dict], scaffold: str) -> float:
    """测试验证: 综合测试写入、执行次数和最终测试通过信号。

    0.3 * has_test_write + 0.3 * has_test_run + 0.4 * late_test_success

    late_test_success 由 _test_run_outcome 对**最后一次测试执行**的 observation 判定：

        pass    -> 1.0   输出里有明确的通过摘要
        unknown -> 0.6   跑了测试但输出看不出结论（常见于 `| tail -5` 只截了几行）
        fail    -> 0.3   有明确失败信号

    unknown 档是必要的：把「看不出结论」等同于「通过」，会让失败识别覆盖不到的语言
    系统性地被判成通过。单列一档后，「检测不到」不再等价于「满分」。
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
                if _is_test_run_cmd(ks):
                    test_run_count += 1
                    # Find next user msg as observation。找不到时必须重置为 -1，
                    # 否则会沿用上一次测试的 observation（截断轨迹里最后一次测试
                    # 往往正是失败的那次，沿用旧值会系统性偏乐观）。
                    found = -1
                    for j in range(idx + 1, len(messages)):
                        if messages[j].get("role") == "user":
                            found = j
                            break
                    last_test_obs_idx = found
                for edited_path in _extract_bash_edit_paths(ks):
                    if _is_test_file_path(edited_path):
                        has_test_write = True
        else:
            for tc_idx, tc in enumerate(msg.get("tool_calls", []) or []):
                name_lower, args = _parse_tool_call(tc)
                # 写测试文件：编辑器写操作
                if _is_write_operation(name_lower, args):
                    if _is_test_file_path(_get_file_path(args)):
                        has_test_write = True
                # 执行类工具（bash/ipython/改名变体，以参数为准）：跑测试，或用
                # heredoc/重定向写出（多语言）reproduce 脚本。
                if _tool_call_is_execution(name_lower, args):
                    cmd = _exec_cmd_text(args)
                    for edited_path in _extract_bash_edit_paths(cmd):
                        if _is_test_file_path(edited_path):
                            has_test_write = True
                    if _is_test_run_cmd(cmd):
                        test_run_count += 1
                        # Find corresponding tool response。同样地，找不到对应
                        # observation（并行 tool_call 未全返回 / 轨迹被截断）时重置
                        # 为 -1，不沿用上一次测试的结果。
                        found = -1
                        tool_count = 0
                        for j in range(idx + 1, len(messages)):
                            if messages[j].get("role") == "tool":
                                if tool_count == tc_idx:
                                    found = j
                                    break
                                tool_count += 1
                            elif messages[j].get("role") == "assistant":
                                break
                        last_test_obs_idx = found

    has_test_run = test_run_count > 0
    # Late test success: 最后一次测试执行的结论（pass / unknown / fail）
    late_test_success = 0.0
    if last_test_obs_idx >= 0:
        outcome = _test_run_outcome(messages[last_test_obs_idx].get("content", ""))
        late_test_success = _LATE_TEST_SCORE[outcome]
    elif has_test_run:
        # 跑了测试但拿不到对应 observation：按无结论处理，而非满分
        late_test_success = _LATE_TEST_SCORE["unknown"]

    return (0.3 * float(has_test_write)
            + 0.3 * float(has_test_run)
            + 0.4 * late_test_success)


def _compute_reproduce_first(messages: list[dict], scaffold: str) -> float | None:
    """诊断: 是否在「首次编辑非测试源文件」之前先跑过测试/复现。

    对应 SWE bugfix 工作流的 Phase 4（写复现/跑测试）应先于 Phase 6（改源码）。

    返回:
        1.0  — 先跑测试/复现，再改源码 (reproduced-first，理想)
        0.0  — 改了源码但未先复现 (先改后验，或全程没跑测试)
        None — 全程没编辑任何非测试源文件 (不计入"先复现率"分母)

    因此数据集级「先复现率」= 非 None 取值的均值（自动排除 no_src_edit）。
    测试执行涵盖正式 runner、reproduce/verify 脚本、`python -c` / `node -e` / `go run`
    等多语言内联自测（见 _is_test_run_cmd），并以工具调用参数判定执行类工具
    （见 _tool_call_is_execution），对脚手架改名鲁棒。这是纯诊断字段，不计入
    composite_score。
    """
    first_fix: int | None = None   # 首次编辑非测试源文件的动作序号
    first_test: int | None = None  # 首次跑测试/复现的动作序号
    act = 0                        # 单调递增的动作序号

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
                if first_test is None and _is_test_run_cmd(ks):
                    first_test = act
                if first_fix is None:
                    for ep in _extract_bash_edit_paths(ks):
                        if ep and not _is_test_file_path(ep):
                            first_fix = act
                            break
                act += 1
            continue

        for tc in msg.get("tool_calls", []) or []:
            name_lower, args = _parse_tool_call(tc)
            if _tool_call_is_execution(name_lower, args):
                if first_test is None and _is_test_run_cmd(_exec_cmd_text(args)):
                    first_test = act
            if _is_write_operation(name_lower, args):
                path = _get_file_path(args)
                if first_fix is None and path and not _is_test_file_path(path):
                    first_fix = act
            act += 1

    if first_fix is None:
        return None
    return 1.0 if (first_test is not None and first_test < first_fix) else 0.0


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


# ═══════════════════════════════════════════════════════════════════════════
# Tool-name canonicalization
# ═══════════════════════════════════════════════════════════════════════════
# 部分数据集（如工具改名增广的 OpenHands 轨迹）把 execute_bash/str_replace_editor/
# finish/think/task_tracker 改成了任意同义名（shell_exec, edit_file, end_task,
# consider, planning ...）。这些工具的「参数 schema」与标准工具完全一致，只是名字变了。
# 打分逻辑大量依赖工具名常量，因此先按参数签名把工具名归一回标准名，scaffold 检测与
# 所有下游指标（SUB/TVR/DPI/SCP/FEC...）即可正常工作，无需改动各 helper。

# 标准名（下游常量已识别这些名字）
_CANON_BASH = "execute_bash"
_CANON_EDITOR = "str_replace_editor"
_CANON_FINISH = "finish"
_CANON_THINK = "think"
_CANON_TRACKER = "task_tracker"

# 已经是标准/已知工具的名字，无需归一
_ALREADY_CANONICAL = (
    _BASH_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES | _PURE_EDIT_TOOL_NAMES
    | {_CANON_FINISH, _CANON_THINK, _CANON_TRACKER}
)


def _infer_canonical_name(name: str, params: set[str]) -> str | None:
    """按工具声明的参数签名推断标准名；无法判定或已是标准名时返回 None。"""
    if name.lower() in _ALREADY_CANONICAL:
        return None
    # finish: 带 task_completed 的「结束任务」工具
    if "task_completed" in params:
        return _CANON_FINISH
    # think: 仅有 thought 字段的思考工具
    if "thought" in params and "command" not in params and "path" not in params:
        return _CANON_THINK
    # task_tracker: command + task_list 的待办管理工具
    if "task_list" in params:
        return _CANON_TRACKER
    # editor: str_replace_editor 风格（command + path + old_str/file_text/view_range）
    if {"old_str", "new_str", "file_text", "view_range"} & params:
        return _CANON_EDITOR
    # bash: command + is_input/timeout 的 shell 执行工具
    if "command" in params and ("is_input" in params or "timeout" in params):
        return _CANON_BASH
    # editor fallback: command + path 但无 shell 特征
    if "command" in params and "path" in params:
        return _CANON_EDITOR
    return None


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """把改名工具归一回标准名，返回新记录（不修改入参，且尽量浅拷贝避免复制大字段）。"""
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        return record

    name_map: dict[str, str] = {}
    for t in tools:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        nm = fn.get("name")
        if not nm:
            continue
        params = set(((fn.get("parameters") or {}).get("properties") or {}).keys())
        canon = _infer_canonical_name(nm, params)
        if canon and canon != nm:
            name_map[nm] = canon

    if not name_map:
        return record

    new_messages = []
    for m in record.get("messages", []):
        tcs = m.get("tool_calls")
        if m.get("role") == "assistant" and tcs:
            rebuilt = []
            changed = False
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                old = fn.get("name")
                if old in name_map:
                    new_fn = dict(fn)
                    new_fn["name"] = name_map[old]
                    new_tc = dict(tc)
                    new_tc["function"] = new_fn
                    rebuilt.append(new_tc)
                    changed = True
                else:
                    rebuilt.append(tc)
            if changed:
                m2 = dict(m)
                m2["tool_calls"] = rebuilt
                new_messages.append(m2)
                continue
        new_messages.append(m)

    new_tools = []
    for t in tools:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if fn.get("name") in name_map:
            new_fn = dict(fn)
            new_fn["name"] = name_map[fn["name"]]
            new_t = dict(t)
            new_t["function"] = new_fn
            new_tools.append(new_t)
        else:
            new_tools.append(t)

    new_record = dict(record)
    new_record["messages"] = new_messages
    new_record["tools"] = new_tools
    return new_record


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
    record = _canonicalize_record(record)
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
    # 纯诊断字段（不计入 composite）：先复现率 (Phase4 先于 Phase6)。
    reproduce_first = _compute_reproduce_first(messages, scaffold)

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
        "reproduce_first": reproduce_first,
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
        "reproduce_first",  # 诊断: mean = 改源码轨迹中的先复现率 (None=没改源码, 已排除)
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
    调用错误。此外按产生 observation 的工具区分执行类/非执行类（见
    _obs_is_execution）：文件查看/编辑/搜索类工具返回的是源码/文件数据而非命令
    输出，仅依据显式工具失败标记判定，避免源码里的 except/raise/FAILED 等 token
    被误判为工具调用错误（历史假阳性主因）。
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
            # Terminus2 的 observation 都是终端命令输出 → 执行类
            if _is_error_result(obs.get("content", ""), is_test_output=is_test, is_execution=True):
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
        is_exec = _obs_is_execution(obs, scaffold)
        total += 1
        if _is_error_result(obs.get("content", ""), is_test_output=is_test, is_execution=is_exec):
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
