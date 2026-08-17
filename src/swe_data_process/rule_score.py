#!/usr/bin/env python3
"""Trajectory quality scoring module — TQS V2 (Trajectory Quality Score).

Scores IM-format JSONL trajectories record by record for every supported
scaffold:
  Claude Code, OpenCode, OpenHands, OpenHands SDK, Terminus2

TQS V2 composite score (fail-soft weighting):
  TQS = Σ(weight_i × transformed_component_i) / Σ(weight_i)
  Only components with data and nonzero weights are included.

The five components with nonzero weights:
  SUB (0.33) — Submission completeness: clean termination, late errors, and late tests
  STP (0.27) — Step efficiency: whether assistant turns stay within a reasonable range
  TVR (0.23) — Test verification: tests written, tests run, and final test outcome
  FEC (0.10) — File-edit concentration: mean edits per file; aggregated as FEC^5
  DPI (0.07) — Undesirable-pattern penalty: truncation, no successful write,
               loops, and repeated errors; aggregated as DPI^3

OEC/IAC/PED/PSN/TTE/SCP are still computed and emitted as diagnostics. Their
current weights are zero, so they do not contribute to composite_score.

Multilanguage correctness notes:

  * Content-based observation checks first pass through _scan_window, which
    strips ANSI escapes (they can split words and break boundary anchors such
    as `\\bFAILED\\b`) and retains both the output head and tail, where the
    conclusion usually appears.
  * _test_run_outcome classifies test results as pass / unknown / fail across
    Go, Rust, Maven, Gradle, CTest, GoogleTest, Jest, Vitest, Mocha, PHPUnit,
    RSpec, pytest, unittest, and similar summary formats. It is deliberately
    separate from _is_error_result: an assertion failure is valid verification,
    not a failed tool call.
  * Test-command detection examines each shell segment via _iter_run_segments.
    Segments beginning with read-only commands such as grep, rm, git, or cat
    are skipped; otherwise a Java path in `git diff` or a Vitest filename in
    `rm -f vitest-tmp.config.ts` could look like a test run.
  * Scaffold-generated markers such as
    `[The command completed with exit code N.]` cannot indicate failure.
    Commands commonly use `... | tail -40`, so the pipeline returns tail's
    status. The agent's own `${PIPESTATUS[0]}` echo and output content are
    reliable signals.

Usage:
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
# Command observations commonly contain ANSI escapes from grep --color and
# colored Maven, Cargo, or Gradle output. Escapes can occur inside a word, as in:
#   "Tests run\x1b[m\x1b[K: 6, \x1b[01;31m\x1b[KFailures\x1b[m\x1b[K: 2"
# The K in `\x1b[K` is a word character and can break boundary anchors such as
# `\bFAILED\b`. Strip ANSI escapes and carriage returns before content matching.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|\r")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences and bare CRs that can break word matching."""
    if not text or "\x1b" not in text and "\r" not in text:
        return text
    return _ANSI_RE.sub("", text)


def _text_of(content: Any) -> str:
    """Normalize observation or message content to plain text.

    Supports both strings and content-block lists; see _assistant_text_content.
    Without normalization, list content would cause downstream regexes to raise
    TypeError.
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
# Command conclusions usually occur at the end (test summaries, BUILD FAILURE,
# or echoed exit codes), after long normal logs. Scan both the head and tail.
_ERROR_SCAN_LIMIT = 3000
_ERROR_SCAN_TAIL = 3000
# Maximum partial-line prefix to discard while aligning the tail to a line start.
# A longer prefix is treated as one exceptionally long line and retained.
_ERROR_SCAN_ALIGN = 300


def _scan_window(content: Any) -> str:
    """Return an ANSI-free head-and-tail text window for pattern matching.

    Align the tail to a line start. Because text[-N:] can split a line at any
    character, concatenating that fragment would create a false line start and
    could trigger anchors such as `^FAIL`, `^(?:FAILED|ERROR)\\s`, or
    `^\\s*\\[ERROR\\]`. Search for a newline only within the first
    _ERROR_SCAN_ALIGN characters of the tail.
    """
    text = _strip_ansi(_text_of(content))
    if len(text) <= _ERROR_SCAN_LIMIT + _ERROR_SCAN_TAIL:
        return text
    tail = text[-_ERROR_SCAN_TAIL:]
    nl = tail.find("\n")
    tail = tail[nl + 1:] if 0 <= nl < _ERROR_SCAN_ALIGN else tail
    return text[:_ERROR_SCAN_LIMIT] + "\n" + tail

# Explicit tool-failure markers are injected by the scaffold or runtime and are
# independent of tool output, so they are reliable for every tool type. Source
# returned by file-view or edit tools will not contain these wrapper strings.
_TOOL_ERROR_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"<tool_use_error>"),
    re.compile(r"The arguments provided to the tool are invalid"),
    re.compile(r"\[An error occurred during execution\.\]"),  # OpenHands SDK execution failure
    re.compile(r"Error validating args\b.*\bfor tool\b"),      # OpenHands SDK argument validation
    re.compile(r"Error executing tool\b"),                     # OpenHands SDK tool exception
    # Non-SDK OpenHands str_replace_editor failures begin with an "ERROR:" block
    # or use the specific edit-failure phrases below. Anchoring and specificity
    # prevent ordinary source content from matching.
    re.compile(r"^\s*ERROR:"),
    re.compile(r"No replacement was performed"),
    re.compile(r"parameter is required for ['\"]?\w+['\"]? command"),
    re.compile(r"Invalid `[^`]+` parameter"),
]

# Command-output error signals apply only to execution tools such as bash,
# terminal, and IPython, whose observations contain stdout/stderr. Applying
# them to file-view, edit, or search tools would mistake legitimate source
# tokens such as except, raise, AssertionError, FAILED, or exit code for tool
# failures. This was the main source of false positives in historical rates.
_ERROR_PATTERNS_HARD: list[re.Pattern[str]] = [
    re.compile(r"command not found"),
    re.compile(r"Permission denied"),
    # Allow a minus sign for scaffold timeout or signal statuses such as -1/-9.
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

# --- Multilanguage test outcome classification ---
# Used for TVR late_test_success, but deliberately excluded from
# _ERROR_PATTERNS_SOFT and tool-call error rates. A completed test run with an
# assertion failure is valid verification, not a failed tool call.
#
# Agents rarely invoke a runner directly; commands commonly look like:
#     <runner> ... 2>&1 | tail -40; echo "exit: ${PIPESTATUS[0]}"
# The pipeline makes the shell report tail's zero status, so scaffold text such as
#     [The command completed with exit code 0.]
# cannot indicate failure. Use the output content and the agent's own PIPESTATUS
# echo. Numeric failure patterns must exclude zero; otherwise normal successful
# output such as "0 failed" or "0 failures" from Rust, Go, pytest, Jest, or Maven
# would be classified as failure.
_NZ = r"(?!0+\b)\d+"

_TEST_FAIL_PATTERNS: list[re.Pattern[str]] = [
    # Nonzero PIPESTATUS echoed by the agent: exit: 1, EXIT: 2, ctest exit: 8,
    # or DOCTEST EXIT: 101. Exclude `status:` to avoid HTTP status values.
    re.compile(r"\bexit(?:\s*code)?\s*[:=]\s*-?(?!0+\b)\d{1,3}\b", re.IGNORECASE),
    # pytest / unittest
    re.compile(r"^(?:FAILED|ERROR)\s+\S+", re.MULTILINE),
    re.compile(rf"\b{_NZ}\s+failed\b", re.IGNORECASE),
    re.compile(r"^(?:FAIL|ERROR):\s", re.MULTILINE),
    # Go: a standalone FAIL line or --- FAIL:
    re.compile(r"^\s*---\s*FAIL:", re.MULTILINE),
    re.compile(r"^FAIL\b", re.MULTILINE),
    # Rust: failed test result, compilation error, or failed doctest.
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
    # CTest / GoogleTest / Catch2, commonly used for C and C++.
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
    # Generic compilation or build failure.
    re.compile(r"\b(?:compilation|build) failed\b", re.IGNORECASE),
]

# Explicit success signals distinguish a verified pass from output with no clear
# conclusion. As with failure patterns, all success counts must be nonzero.
# `0 passing`, `OK (0 tests)`, or Maven's
# `Tests run: 0, Failures: 0, Errors: 0` mean no tests ran. That outcome is
# unknown, not pass, and must not receive late_test_success=1.0.
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
    # Standard Python unittest success: `Ran 97 tests in 9.254s`, blank line, `OK`.
    re.compile(rf"Ran {_NZ} tests? in [\d.]+s\s*\n+\s*OK\b"),
    re.compile(r"^OK\s*(?:\(skipped=\d+\))?\s*$", re.MULTILINE),
    re.compile(r"\berror count:\s*0\b", re.IGNORECASE),       # Agent-authored tsc check.
]

# Scaffold-injected trailing markers. Their "exit code 0" cannot indicate
# success because commands often use `| tail -N`, making the shell return tail's
# zero status regardless of the test result. Remove these markers before
# accepting the agent's own echoed PIPESTATUS as success.
_SCAFFOLD_MARKER_RE = re.compile(
    r"^\[(?:The command completed with exit code -?\d+\.|"
    r"Command finished with exit code -?\d+|"
    r"Current working directory:[^\]]*|"
    r"Python interpreter:[^\]]*|"
    r"Below is the output of the previous command\.?)\]\s*$",
    re.MULTILINE,
)
# A zero status echoed by the agent: `exit: 0`, `exit:0`, or `=== ctest exit: 0 ===`.
_TEST_PASS_ECHO_RE = re.compile(r"\bexit(?:\s*code)?\s*[:=]\s*0+\b", re.IGNORECASE)

# Explicit markers that no tests ran. A runner may still print a successful
# ending (bare unittest `OK`, Maven `BUILD SUCCESS`, or exit code 0), but it
# verified nothing and must remain unknown. Deliberately exclude Go's
# `[no test files]`, which can describe one package while others still run tests.
_TEST_ZERO_RUN_RE = re.compile(
    r"\bRan 0+ tests?\b"
    r"|\bno tests? (?:ran|were found|to run|found)\b"
    r"|\bcollected 0+ items\b",
    re.IGNORECASE,
)


def _test_run_outcome(content: Any) -> str:
    """Classify one test run as 'fail', 'pass', or 'unknown'.

    This is separate from _is_error_result because an assertion failure is
    valid verification and must not count toward the tool-call error rate.

    Classification order is intentional:
    1. Hard tool/runtime failures (scaffold markers, command not found, nonzero
       status) mean the command never ran successfully and override its output.
    2. Explicit multilanguage failure summaries override success; for example,
       `11 passed; 1 failed` is a failure.
    3. Explicit success summaries precede soft errors. A passing test may
       legitimately print `TypeError` or `No such file or directory` while
       testing an error path, so `100% tests passed` must override those tokens.
    4. Soft error fallback.
    5. No signal means unknown, as when `| tail -5` omits the conclusion.
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
    # No-tests-run markers override OK, BUILD SUCCESS, or exit 0.
    if _TEST_ZERO_RUN_RE.search(text):
        return "unknown"
    for pat in _TEST_PASS_PATTERNS:
        if pat.search(text):
            return "pass"
    # Accept an agent-echoed zero status after removing scaffold zero markers.
    if _TEST_PASS_ECHO_RE.search(_SCAFFOLD_MARKER_RE.sub("", text)):
        return "pass"
    for pat in _ERROR_PATTERNS_SOFT:
        if pat.search(text):
            return "fail"
    return "unknown"

# --- Tool classification ---
# Pure edit tools write files whenever invoked. Include Claude Code MultiEdit
# and NotebookEdit plus OpenCode patch so FEC does not miss those edits.
_PURE_EDIT_TOOL_NAMES = frozenset({
    "edit", "write", "multiedit", "notebookedit", "patch",
})
_MULTI_EDITOR_TOOL_NAMES = frozenset({"file_editor", "str_replace_editor"})
_EDITOR_WRITE_COMMANDS = frozenset({"str_replace", "create", "insert"})
_EDIT_TOOL_NAMES = _PURE_EDIT_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES
_BASH_TOOL_NAMES = frozenset({"bash", "terminal", "execute_bash"})
# Execution-tool observations contain actual command or code output and may use
# _ERROR_PATTERNS_HARD/SOFT. Other tools (view, edit, search) use only explicit
# _TOOL_ERROR_MARKERS so returned source is not mistaken for a tool failure.
#
# SWE data sources often rename tools (shell_exec, run_command, exec_cmd,
# code_editor, and others), so execution classification primarily uses tool
# arguments; see _tool_call_is_execution. Names are only a fallback for missing
# arguments or empty commands.
_EXECUTION_TOOL_NAMES = frozenset({
    "bash", "terminal", "execute_bash", "shell", "cmd",
    "execute_ipython_cell", "run_ipython", "ipython", "python", "run_python",
})
# Editor subcommands identify non-execution file view or edit tools.
_EDITOR_SUBCOMMANDS = frozenset({"view", "create", "str_replace", "insert", "undo_edit"})
# Editor-only argument keys provide additional evidence for non-execution tools.
_EDITOR_ONLY_ARG_KEYS = ("old_str", "new_str", "file_text", "view_range", "insert_line")
# Other non-execution-only keys. task_tracker also has a command field whose
# values are plan/add, so a nonempty command alone is insufficient.
_NON_EXEC_ARG_KEYS = ("task_list", "thought", "task_completed")
# Argument keys that can carry shell commands.
_SHELL_CMD_ARG_KEYS = ("command", "cmd", "keystrokes")
_FILE_PATH_KEYS = ("file_path", "filePath", "path", "notebook_path", "notebookPath")

# --- Test detection ---
_TEST_RUN_RE = re.compile(
    r"\b(?:"
    r"pytest|py\.test|python3?\s+-m\s+pytest|python3?\s+-m\s+unittest"
    r"|unittest|python3?\s+test_|nosetests"
    r"|go\s+test|cargo\s+test|ctest|gtest_filter"
    # Maven/Gradle goals may follow several flags and module selectors.
    # Examples: `mvn -o -q test`, `./mvnw -q -pl mod -am test`,
    # and `./gradlew -q :mod:test`.
    r"|(?:mvn|mvnw)\b[^\n|&;]{0,200}?\b(?:test|verify|surefire:test|integration-test)\b"
    # Tempered match: Gradle `-x test` excludes tests and is not a test run.
    r"|(?:gradle|gradlew)\b(?:(?!-x\s)[^\n|&;]){0,200}?\b[\w:]*[Tt]est[\w:]*\b"
    r"|jest|mocha|vitest|npx\s+jest|npx\s+vitest|npx\s+mocha"
    r"|(?:npm|yarn|pnpm)\s+(?:run\s+)?test"
    r"|make\s+(?:test|check)"
    # Django: ./tests/runtests.py, python tests/runtests.py, or manage.py test.
    # The generic executable pattern below requires test_ or _test, neither of
    # which appears in "tests/runtests.py".
    r"|runtests\.py|manage\.py\s+test"
    # Common CMake/CTest form; ctest is above, so add cmake --build --target test.
    r"|cmake\s+--build\s+\S+\s+--target\s+(?:test|check)"
    # Formal runners for Elixir, Swift, Clojure, .NET, and Lua.
    r"|mix\s+test|swift\s+test|lein\s+test|dotnet\s+test|busted"
    r")\b"
    r"|\.\/[^\s]*(?:_test|test_)\S*"
    # Scala sbt subproject tests: sbt test, sbt "core/test", sbt testOnly, etc.
    r"|\bsbt\b[^\n|&;]*?\b(?:test(?:Only|Quick)?|it:test)\b"
    # R tests, often embedded in R -e: devtools::test() / testthat::test_*(...).
    r"|\b(?:devtools::test|testthat::test)"
    # Neovim Lua tests: PlenaryBustedDirectory / PlenaryBustedFile.
    r"|\bPlenaryBusted\w*",
    re.IGNORECASE,
)

# --- Bash edit path extraction ---
_WORKSPACE_PREFIXES = ("/workspace/", "/testbed/", "/repo/", "/home/swe-bench/")
# `cat > file` / `cat >> file`, including heredoc writes. Excludes
# `cat foo 2>/dev/null`.
_CAT_WRITE_RE = re.compile(r"\bcat\s+(?:>>?)\s*([^\s|&;<>]+)")
# `cat inputs... > outfile`, also excluding file-descriptor redirection.
_CAT_STDOUT_RE = re.compile(r"\bcat\s+(?!>>?)(?:[^|&;\n]*?)(?<![0-9&])>>?\s*([^\s|&;<>]+)")
# `echo/printf ... > file`; redirection must occur in the same simple command.
_ECHO_PRINTF_WRITE_RE = re.compile(
    r"\b(?:echo|printf)\b(?:[^|&;\n]*?)(?<![0-9&])>>?\s*([^\s|&;<>]+)"
)
_TEE_RE = re.compile(r"\btee\s+(?:-[a-zA-Z]+\s+)*([^\s|&;<>]+)")
# In-place edits across GNU/BSD sed, macOS gsed, and Perl -i forms. Covers -i,
# -i.bak, -pi, and -pi.orig. Restrict short-option clusters to letters with an
# optional suffix so a module name such as `perl -MList::Util -e ...` is not
# mistaken for an in-place -i edit.
_SED_INPLACE_RE = re.compile(
    r"\b(?:g?sed|perl)\s+(?:-\S+\s+)*-[A-Za-z]*i[A-Za-z]*(?:\.[\w.]+)?(?=\s|$)"
)
# Match `patch` as a command without matching extensions such as `foo.patch &&`.
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
    """Detect the scaffold type from the structure of an IM record."""
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
    """Extract a file path from tool-call arguments."""
    for key in _FILE_PATH_KEYS:
        val = args.get(key)
        if val:
            return val
    return ""


def _is_write_operation(name_lower: str, args: dict) -> bool:
    """Return whether a tool call is a file-write operation."""
    if name_lower in _PURE_EDIT_TOOL_NAMES:
        return True
    if name_lower in _MULTI_EDITOR_TOOL_NAMES:
        return args.get("command", "") in _EDITOR_WRITE_COMMANDS
    return False


def _parse_tool_call(tc: Any) -> tuple[str, dict]:
    """Extract (name_lower, parsed_args) from a tool-call dictionary."""
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
    """Return whether a tool call executes a shell command or code.

    Data sources often rename tools (shell_exec, run_command, exec_cmd,
    code_editor, source_editor, and others), so argument shape takes precedence
    and the tool name is only a fallback:

    - An editor subcommand (view/create/str_replace/insert/undo_edit), or
      editor-only arguments (old_str/new_str/file_text/view_range/insert_line),
      identifies a non-execution file view/edit tool.
    - task_list/thought/task_completed identifies a non-execution
      task_tracker/think/finish tool. task_tracker also has command values such
      as plan/add, which are not shell commands.
    - Otherwise, a nonempty shell-command argument (command/cmd/keystrokes)
      identifies an execution tool.
    - If no rule applies, fall back to known execution-tool names, including an
      execute_bash call with an empty command.
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
    """Extract command or code text from an execution tool, including renamed tools."""
    for key in ("command", "code", "cmd", "keystrokes"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _normalize_path(p: str) -> str:
    """Normalize a file path."""
    for prefix in _WORKSPACE_PREFIXES:
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    return posixpath.normpath(p)


def _is_plausible_edit_path(path: str) -> bool:
    """Reject redirections and shell metacharacters that are not FEC edit paths."""
    if not isinstance(path, str):
        return False
    p = path.strip().strip("'\"")
    if not p or p in _SHELL_META_TOKENS:
        return False
    if p.startswith("<<") or p.startswith("&"):
        return False
    # Tokens such as `2>/dev/null;`, `foo;`, or any redirection are not paths.
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
    """Extract file-edit target paths from a bash command string.

    Retain only actual write targets and explicitly ignore:
    - file-descriptor or device redirection: `2>/dev/null`, `2>&1`, `>/dev/null`
    - unrelated paths captured only because `echo` and `>` occur elsewhere
    - extension matches such as `foo.patch &&`
    - shell operators or command words after `sed -i ... file && echo`
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
            # Stop at `file 2>/dev/null` or a later command so neither is a path.
            if ">" in tok or "<" in tok or ";" in tok:
                break
            if tok.startswith("-"):
                if tok in ("-e", "-E", "-f"):
                    # The -e/-f argument is the script expression. Consume it
                    # so the following filename is not mistaken for the script
                    # in `perl -pi -e '...' f`.
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
    """Return unnormalized edit targets from an execution tool's command text.

    This shared entry point determines whether a tool call edits through the
    shell. It must use _tool_call_is_execution and _exec_cmd_text rather than a
    name allowlist plus args["command"]. Otherwise, renamed shell tools and
    tools using code/keystrokes would lose all bash edits, causing DPI to report
    no successful write and SCP to return zero solely because a tool was renamed.
    """
    if not _tool_call_is_execution(name_lower, args):
        return []
    return _extract_bash_edit_paths(_exec_cmd_text(args))


def _count_assistant_turns(messages: list[dict]) -> int:
    """Count assistant turns."""
    return sum(1 for m in messages if m.get("role") == "assistant")


# ═══════════════════════════════════════════════════════════════════════════
# Action / Observation extraction
# ═══════════════════════════════════════════════════════════════════════════

def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> tags."""
    if not content:
        return ""
    text = content.strip()
    if text.startswith("<think>") and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


def _parse_t2_assistant(msg: dict) -> dict | None:
    """Parse JSON content from a Terminus2 assistant message."""
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
    """Extract actions from a tool-call scaffold."""
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
    """Extract actions from the Terminus2 scaffold."""
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
    """Extract actions through a scaffold-independent interface."""
    if scaffold == "terminus2":
        return _extract_terminus2_actions(messages)
    return _extract_tool_call_actions(messages)


def _extract_observations(
    messages: list[dict],
    scaffold: str,
) -> list[dict[str, Any]]:
    """Extract observations."""
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
                    "tool_name": None,  # All Terminus2 observations are command output.
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
    """Return whether an observation contains an error.

    Detection has two layers:

    1. Explicit scaffold-injected tool-failure markers (_TOOL_ERROR_MARKERS),
       which apply to every tool type.
    2. Command-output error patterns (_ERROR_PATTERNS_HARD/SOFT), scanned only
       when is_execution=True for tools such as bash, terminal, or IPython.
       Non-execution tools return source or file data rather than command output;
       scanning them would mistake legitimate except, raise, or FAILED tokens
       for tool-call errors.

    is_execution defaults to True for compatibility with existing call sites.
    Per-observation paths (_is_obs_error and _count_tool_call_errors) pass the
    correct value for the tool that produced each observation.

    _scan_window first strips ANSI and retains both the head and tail, where
    conclusive output usually appears.
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
    """Return whether an assistant turn contains a test command."""
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
        # Classify execution from arguments, consistently with
        # _extract_observations and _is_obs_error. A miss both loses SUB's late
        # test bonus and treats a normal test failure as a tool-call error,
        # reducing SUB and DPI.
        if _tool_call_is_execution(name_lower, args):
            if _is_formal_test_run_cmd(_exec_cmd_text(args)):
                return True
    return False


def _get_per_toolcall_test_flags(msg: dict) -> list[bool]:
    """Return a test-command flag for each tool call in an assistant message."""
    flags: list[bool] = []
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        is_test = False
        if _tool_call_is_execution(name_lower, args):
            is_test = _is_formal_test_run_cmd(_exec_cmd_text(args))
        flags.append(is_test)
    return flags


def _resolve_is_test_for_obs(obs: dict, messages: list[dict], scaffold: str, _test_flags_cache: dict) -> bool:
    """Reuse per-tool-call test-output classification for an observation."""
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
    """Return whether an observation came from a command/code execution tool.

    Every Terminus2 observation is terminal command output. Other scaffolds set
    obs["is_execution"] in _extract_observations based on tool arguments; see
    _tool_call_is_execution for rename-robust classification.
    """
    if scaffold == "terminus2":
        return True
    return bool(obs.get("is_execution"))


def _is_obs_error(obs: dict, messages: list[dict], scaffold: str, _test_flags_cache: dict) -> bool:
    """Classify an observation error by tool type without misreading test output."""
    is_test = _resolve_is_test_for_obs(obs, messages, scaffold, _test_flags_cache)
    is_exec = _obs_is_execution(obs, scaffold)
    return _is_error_result(obs.get("content", ""), is_test_output=is_test, is_execution=is_exec)


# ═══════════════════════════════════════════════════════════════════════════
# Action classification (shared by IAC, TTE, PED)
# ═══════════════════════════════════════════════════════════════════════════

def _classify_action(name_lower: str, args: dict) -> str:
    """Map a tool call to one of seven standard action types."""
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
    # Classify renamed shell/IPython tools by argument shape so IAC/TTE/PED do
    # not classify every execution action in the trajectory as "other".
    if _tool_call_is_execution(name_lower, args):
        return "bash"
    return "other"


def _classify_action_from_action(a: dict, messages: list[dict]) -> str:
    """Get a standard action type from an extracted action."""
    msg = messages[a["msg_idx"]]
    tool_name_lower = a["tool_name"].lower()
    for tc in msg.get("tool_calls", []) or []:
        name_lower, args = _parse_tool_call(tc)
        if name_lower == tool_name_lower:
            return _classify_action(name_lower, args)
    return _classify_action(tool_name_lower, {})


def _action_target_files(a: dict, messages: list[dict]) -> set[str]:
    """Return the files targeted by an action."""
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
# 1. OEC — Observation entropy collapse
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
    # Lower min_rel means more severe collapse and a lower OEC score.
    return max(0.0, min(min_rel, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# 2. IAC — Intent-action consistency
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
# 3. DPI — Undesirable-pattern penalty
# ═══════════════════════════════════════════════════════════════════════════

def _has_successful_write(messages: list[dict], observations: list[dict], scaffold: str) -> bool:
    """Return whether the trajectory contains at least one successful write."""
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
    """Return whether a trajectory was truncated instead of ending normally."""
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
    """Build an action signature for DPI loop detection.

    Return (tool_name, editor_subcommand, target_path), or None when no target
    is available.

    file_editor and str_replace_editor signatures must include the command
    (view, str_replace, etc.) so normal view-then-edit iterations on one file
    are not mistaken for an infinite loop.
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

    # Terminus2 or no tool_calls: fall back to a target-only signature.
    files = _action_target_files(a, messages)
    if not files:
        return None
    return (tool_name, "", sorted(files)[0])


def _compute_loop_fraction(actions: list[dict], messages: list[dict], scaffold: str) -> float:
    """Compute the fraction of steps in consecutive repeated runs.

    A loop requires a consecutive run with identical signatures and nonempty
    targets. Each signature is (tool_name, editor_subcommand, target):
      - file_editor/str_replace_editor includes view/str_replace and other
        subcommands to distinguish normal view-edit iterations on one file.
      - bash/terminal calls without an explicit target are excluded because
        consecutive command executions are normal.
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
    """Compute the rate of repeated errors in adjacent observations."""
    _test_flags_cache: dict[int, list[bool]] = {}
    error_count = 0
    repeat_pairs = 0

    prev_is_error = False
    prev_signature = ""

    for obs in observations:
        is_err = _is_obs_error(obs, messages, scaffold, _test_flags_cache)
        if is_err:
            error_count += 1
            # Strip ANSI before creating the signature. The same error may use
            # different control sequences across turns, which would otherwise
            # hide the repetition. _text_of supports list content.
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
# 4. PED — Post-error strategy diversity
# ═══════════════════════════════════════════════════════════════════════════

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _action_for_observation(obs: dict, actions: list[dict], messages: list[dict], scaffold: str) -> dict | None:
    """Find the action that produced an observation."""
    prev_idx = obs.get("prev_assistant_idx")
    if prev_idx is None:
        return None
    pos = obs.get("tool_call_position") or 0
    matching = [a for a in actions if a["msg_idx"] == prev_idx]
    if pos < len(matching):
        return matching[pos]
    return matching[0] if matching else None


def _next_action_after_msg(actions: list[dict], msg_idx: int) -> dict | None:
    """Find the first action after msg_idx."""
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
# 5. PSN — Progressive scope narrowing
# ═══════════════════════════════════════════════════════════════════════════

def _per_step_target_files(messages: list[dict], scaffold: str) -> list[set[str]]:
    """Return the target-file set for each assistant turn."""
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
# 6. TTE — Tool-transition entropy
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
# 7. SCP — Timing of the first effective edit
# ═══════════════════════════════════════════════════════════════════════════

def _first_successful_write_turn(messages: list[dict], observations: list[dict], scaffold: str) -> int | None:
    """Return the 1-based assistant-turn rank of the first successful write."""
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
# 8. SUB — Submission completeness (from v5 D1, r=0.33 with resolved)
# ═══════════════════════════════════════════════════════════════════════════

# Text completion signals: some scaffolds or models end with a summary instead
# of calling finish. Treat an explicit completion statement (strong signal) or
# completion verb (weak signal) as a normal ending rather than assigning 0.5
# merely because a fixed, tested task omitted the finish call.
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
    """Return plain text from string or content-block assistant content."""
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
    """Score the SUB base for a final text summary without a finish call.

    A strong completion signal scores 1.0, a weak signal 0.8, and other text
    0.5, matching the previous incomplete-trajectory score.
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
    """Score submission completeness and final-state quality.

    Base: finish call=1.0, text summary with completion signal=0.8-1.0,
    other text=0.5, truncation=0.0.
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
# 9. FEC — File-edit concentration (from v5 E1, r=0.17 with resolved)
# ═══════════════════════════════════════════════════════════════════════════

_FEC_THRESHOLD = 5


def _extract_edit_paths(messages: list[dict], scaffold: str) -> list[str]:
    """Return normalized target paths for all file edits, including duplicates."""
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
                    # Share _exec_bash_edit_paths with TVR, DPI, SCP, and error
                    # detection; execution classification uses arguments, not names.
                    paths.extend(_normalize_path(p) for p in _exec_bash_edit_paths(name_lower, args))
    return paths


def _compute_fec(messages: list[dict], scaffold: str) -> float | None:
    """Score file-edit concentration as 1 - clip((mean_edits - 1) / 4, 0, 1).

    Return None when no edits are detected so aggregation fails soft. Returning
    1.0 would equate a detection gap with perfect concentration. Because
    aggregation also raises FEC to the fifth power (three edits to one file
    leave only 0.031), missed edits would otherwise become an advantage.
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
# 10. STP — Step efficiency (r=-0.21 with resolved: fewer steps = better)
# ═══════════════════════════════════════════════════════════════════════════

# Match the current agent max_turn=200: short-to-medium trajectories receive
# full credit, while reaching the turn cap scores zero.
_STP_OPTIMAL_RANGE = (5, 80)
_STP_MAX_STEPS = 200


def _compute_stp(messages: list[dict]) -> float:
    """Score step efficiency with full credit in the optimal range.

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
# 11. TVR — Test verification (test writing + running correlates with success)
# ═══════════════════════════════════════════════════════════════════════════

# Code-file suffixes shared by multilanguage test and reproduction detection.
_CODE_EXT = (
    r"(?:py|js|jsx|ts|tsx|mjs|cjs|go|rs|java|kt|kts|scala|rb|php|cc|cpp|cxx|cu|"
    r"c|h|hpp|hxx|cs|swift|mm|pl|pm|ex|exs|lua|cljc|cljs|clj|edn|dart|sol|jl|"
    r"erl|groovy|r|sh)"
)

# Formal multilanguage test files. Covers test_x, x_test, x_spec, x.test.x,
# Java/Kotlin/Scala *Test(s) and *IT, C/C++ *_test.cc, and code files under
# tests, __tests__, or spec directories.
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    r"conftest\.py"
    rf"|test_[^/]+\.{_CODE_EXT}"
    rf"|[^/]+_tests?\.{_CODE_EXT}"
    rf"|[^/]+_spec\.{_CODE_EXT}"
    r"|[^/]+\.(?:test|spec)\.(?:m?[jt]sx?|cjs)"
    # *Test/*Tests/*Spec names for Java, Kotlin, Scala, PHP, C#, C++, Go,
    # Swift, and Dart. Use (?-i:) to disable this regex's IGNORECASE and require
    # CamelCase. Case is intrinsic to this convention; otherwise latest.py,
    # greatest.go, contest.js, and attest.go would match merely by ending in test.
    rf"|[^/]*(?-i:[A-Z]\w*(?:Tests?|Spec))\.{_CODE_EXT}|[^/]+IT\.java"
    r")$"
    rf"|(?:^|/)(?:tests?|__tests__|specs?|testing)/(?:[^/]+/)*[^/]+\.{_CODE_EXT}$"
    # Perl t/ directory.
    r"|(?:^|/)t/(?:[^/]+/)*[^/]+\.t$"
    # Standard Maven/Gradle Java, Kotlin, and Scala test directories.
    rf"|(?:^|/)src/(?:test|integrationTest|androidTest)/(?:[^/]+/)*[^/]+\.{_CODE_EXT}$",
    re.IGNORECASE,
)


# --- TVR-only verification idioms ---
# SWE-agent trajectories often verify by writing reproduce/verify scripts and
# running inline checks rather than formal test runners, across languages such
# as JS, TS, C++, Go, Rust, PHP, and Ruby. These signals apply only to TVR and
# do not enter the _TEST_RUN_RE error-suppression path; otherwise a real inline
# traceback could be treated as test output and hidden from DPI/error rates.
_TVR_VERIFY_RE = re.compile(
    # Reproduction/verification scripts in any supported language.
    rf"(?:^|[\s/])\S*(?:reproduce|repro|verify|smoke)\w*\.{_CODE_EXT}\b"
    # Python inline checks or test_ scripts.
    r"|\bpython3?\s+-c\b"
    r"|\bpython3?\s+(?:-\S+\s+)*(?:\S*/)?(?:test_\S+|\S+_test)\.py\b"
    # Running an ordinary, non-test-named script also counts as verification by
    # design: for compiled or interpreted languages, agents often reproduce a
    # bug by checking whether a program crashes. Apply this consistently to all
    # languages, including Python and Node, to avoid language bias. Explicit
    # build and installation commands are removed by _TVR_NONVERIFY_RE below.
    r"|\bpython3?\s+(?:-\S+\s+)*(?:\S*/)?\S+\.py\b"
    r"|\b(?:node|bun|deno)\s+(?:-\S+\s+)*(?:\S*/)?\S+\.(?:m?[jt]sx?|cjs)\b"
    # JS/TS: node -e inline checks and reproduce/verify/smoke/test/spec scripts.
    r"|\b(?:node|deno)\s+(?:-e|--eval|eval)\b"
    r"|\b(?:node|npx|deno|bun|ts-node|tsx)\s+(?:-\S+\s+)*\S*(?:repro|verify|smoke|test|spec)\S*\.(?:m?[jt]sx?|cjs)\b"
    # Inline checks and scripts for other interpreters.
    r"|\b(?:ruby|perl)\s+-e\b|\bphp\s+-r\b"
    r"|\b(?:ruby|php|perl|Rscript)\s+(?:-\S+\s+)*\S+\.\w+\b"
    # R inline checks via R -e / Rscript -e, including devtools/testthat.
    r"|\bR(?:script)?\b\s+(?:--\S+\s+)*-e\b"
    # Compiled/interpreted execution: go, Cargo, Swift, and .NET.
    r"|\bgo\s+run\b|\bcargo\s+(?:test|run)\b|\bswift\s+run\b|\bdotnet\s+run\b"
    # Elixir: mix run or elixir/iex executing .ex(s) scripts.
    r"|\bmix\s+run\b|\b(?:elixir|iex)\s+(?:-\S+\s+)*\S+\.exs?\b"
    # Clojure: lein run or clj/clojure -e, -M, or -X.
    r"|\blein\s+run\b|\b(?:clj|clojure)\s+(?:-\S+\s+)*-[eMX]\S*"
    # Java: a jar or temporary reproduction class. Require an uppercase main
    # class to avoid case-insensitively matching -version.
    r"|\bjava\b[^\n|&;]*?(?:-jar\s+\S+|\b(?-i:[A-Z])\w+)\b"
    # Neovim Lua: nvim --headless running a test/repro/verify script via dofile.
    r"|dofile\(\s*['\"][^'\"]*(?:test|repro|verify|spec)"
    # Build-system test runners not already covered by _TEST_RUN_RE.
    r"|\b(?:bazel|blaze)\s+test\b|\bdotnet\s+test\b|\bphpunit\b|\brspec\b|\btox\b"
    # Compiled artifacts or test scripts: ./repro, ./a.out, ./run_tests.sh, ./x_test.
    r"|\./\S*(?:_test|test_|repro|verify|smoke)\S*"
    r"|\./\S+\.(?:sh|out|bin|exe)\b",
    re.IGNORECASE,
)
# Commands that are explicitly not verification: dependency installation,
# packaging, development-server startup, and scaffold generation. They neither
# run tests nor execute a program to check for a crash, so they do not set TVR
# has_test_run. By design, `./build.sh`, `go run`, and `cargo run` still count.
_TVR_NONVERIFY_RE = re.compile(
    r"\b(?:pip3?|uv\s+pip|conda|apt|apt-get|yum|brew)\s+install\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:install|i|ci|add)\b"
    r"|\bpython3?\s+(?:\S*/)?setup\.py\s+(?:install|build|develop|egg_info|sdist|bdist\w*)\b"
    r"|\bmanage\.py\s+(?:runserver|migrate|makemigrations|collectstatic|shell|startapp|startproject)\b"
    r"|\b(?:npm|yarn|pnpm)\s+(?:run\s+)?(?:dev|start|build|watch|serve|lint|format)\b"
    r"|\bcargo\s+(?:build|check|fmt|clippy|install)\b"
    r"|\bgo\s+(?:build|install|mod|vet|fmt)\b"
    # Agents often use `python -c` to edit files via open(..., 'w')/.write().
    # Such commands are edits, not inline verification.
    r"|\bpython3?\s+-c\b[\s\S]*?(?:open\s*\([^)]*['\"][wa]b?\+?['\"]|\.write\s*\()",
    re.IGNORECASE,
)
_TVR_REPRO_FILE_RE = re.compile(
    r"(?:^|/)\S*(?:reproduce|repro|verify|smoke)\S*\.\w+$",
    re.IGNORECASE,
)


# Three late_test_success levels; see _compute_tvr.
_LATE_TEST_SCORE = {"pass": 1.0, "unknown": 0.6, "fail": 0.3}


# Command heads that only read or move data and can never run tests. Skip the
# entire segment to prevent a Java path in `git diff`, a Vitest filename in
# `rm -f vitest-tmp.config.ts`, or a `go test` literal in grep from matching.
_NONRUN_CMD_HEADS = frozenset({
    "rm", "ls", "cat", "grep", "egrep", "fgrep", "rg", "ag", "find", "git", "echo",
    "mkdir", "rmdir", "cp", "mv", "touch", "head", "tail", "wc", "sed", "awk",
    "diff", "which", "whereis", "chmod", "chown", "cd", "sort", "uniq", "xargs",
    "export", "printf", "tree", "stat", "du", "df", "ln", "tar", "unzip", "gzip",
    "curl", "wget", "pwd", "env", "true", "false", "sleep", "kill", "ps",
})

# Wrappers that may precede the real command; timeout also consumes a duration.
_CMD_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "exec", "timeout", "stdbuf"})
_ENV_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=\S*\Z")
_DURATION_RE = re.compile(r"[\d.]+[smhd]?\Z")


def _split_shell_segments(cmd: str) -> list[str]:
    """Split a shell chain into simple segments for independent classification.

    Commands often look like `cd <dir> && <runner> ... | tail -40; echo ...`.
    Matching the whole chain would let the install and test segments in
    `pip install && pytest` contaminate each other.

    Separators count only outside quotes, and newlines deliberately do not
    split segments. A raw regex split of multiline
    `python3 -c "p='x'; open(p,'w').write(s)"` would split on the script's
    semicolon and break whole-body checks such as recognizing a file edit.

    Known tradeoff: if a multiline segment begins with a read-only command such
    as heredoc `cat`, a real command on a later line is skipped with it.
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
    """Return a segment's command word after assignments and wrappers."""
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
    # Remove path prefixes: /opt/venv/bin/python -> python, ./.bin/jest -> jest.
    return head.rsplit("/", 1)[-1]


def _iter_run_segments(cmd: str):
    """Yield segments that may execute code, excluding read/move-only heads."""
    for seg in _split_shell_segments(cmd):
        if _segment_head(seg) in _NONRUN_CMD_HEADS:
            continue
        yield seg


def _is_formal_test_run_cmd(cmd: str) -> bool:
    """Return whether a command invokes a formal test runner.

    This strict definition identifies test output for error suppression.
    Classifying each segment prevents filenames or literals in commands such as
    `rm -f vitest-tmp.config.ts` and `grep "go test"` from matching.
    """
    if not cmd:
        return False
    return any(_TEST_RUN_RE.search(seg) for seg in _iter_run_segments(cmd))


def _is_test_run_cmd(cmd: str) -> bool:
    """Apply TVR test-run detection to formal runners and reproduction checks.

    Classify each segment and exclude installation, packaging, and service
    startup via _TVR_NONVERIFY_RE so commands such as `python setup.py build`
    and `npm run dev` do not count as verification.
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
    """Return whether TVR considers a path a formal test or reproduction script."""
    if not path:
        return False
    return bool(_TEST_FILE_RE.search(path) or _TVR_REPRO_FILE_RE.search(path))


def _compute_tvr(messages: list[dict], scaffold: str) -> float:
    """Score test verification from test writes, runs, and the final outcome.

    0.3 * has_test_write + 0.3 * has_test_run + 0.4 * late_test_success

    _test_run_outcome classifies the final test run's observation:

        pass    -> 1.0   explicit success summary
        unknown -> 0.6   test ran but output has no conclusion, often after `| tail -5`
        fail    -> 0.3   explicit failure signal

    The unknown level prevents languages not covered by failure patterns from
    being systematically treated as successful. No detected conclusion is not
    equivalent to full credit.
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
                    # Find the next user message as the observation. Reset to -1
                    # when absent rather than reusing the previous test result;
                    # the truncated final run is often the failing one.
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
                # Test-file writes through editor operations.
                if _is_write_operation(name_lower, args):
                    if _is_test_file_path(_get_file_path(args)):
                        has_test_write = True
                # Execution tools (bash/IPython/renamed variants, classified
                # by arguments): run tests or write reproduction scripts via
                # heredoc/redirection.
                if _tool_call_is_execution(name_lower, args):
                    cmd = _exec_cmd_text(args)
                    for edited_path in _extract_bash_edit_paths(cmd):
                        if _is_test_file_path(edited_path):
                            has_test_write = True
                    if _is_test_run_cmd(cmd):
                        test_run_count += 1
                        # Find the corresponding tool response. Reset to -1 if
                        # parallel tool calls are incomplete or the trajectory
                        # is truncated; never reuse the previous test outcome.
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
    # Late test success is the final test run's pass / unknown / fail outcome.
    late_test_success = 0.0
    if last_test_obs_idx >= 0:
        outcome = _test_run_outcome(messages[last_test_obs_idx].get("content", ""))
        late_test_success = _LATE_TEST_SCORE[outcome]
    elif has_test_run:
        # A test run without its observation has no conclusion, not full credit.
        late_test_success = _LATE_TEST_SCORE["unknown"]

    return (0.3 * float(has_test_write)
            + 0.3 * float(has_test_run)
            + 0.4 * late_test_success)


def _compute_reproduce_first(messages: list[dict], scaffold: str) -> float | None:
    """Diagnose whether verification preceded the first non-test source edit.

    In the SWE bug-fix workflow, Phase 4 (write a reproduction or run tests)
    should precede Phase 6 (edit source).

    Returns:
        1.0  — verification before source edits (ideal reproduced-first flow)
        0.0  — source edited without prior verification
        None — no non-test source file was edited, so exclude from the rate

    The dataset-level reproduced-first rate is therefore the mean of non-None
    values. Verification includes formal runners, reproduce/verify scripts, and
    multilanguage inline checks such as `python -c`, `node -e`, and `go run`;
    see _is_test_run_cmd. Execution classification uses tool-call arguments and
    remains robust to renamed tools; see _tool_call_is_execution. This field is
    diagnostic only and does not contribute to composite_score.
    """
    first_fix: int | None = None   # Index of the first non-test source edit.
    first_test: int | None = None  # Index of the first test/reproduction run.
    act = 0                        # Monotonically increasing action index.

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
# 12. LER — Late error rate (fewer errors near the end is better)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_ler(observations: list[dict], messages: list[dict], scaffold: str) -> float | None:
    """Return the error-free fraction of observations in the final 40%.

    Resolved instances should have fewer late errors after finding a solution.
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
# Some datasets rename execute_bash, str_replace_editor, finish, think, or
# task_tracker to arbitrary synonyms such as shell_exec, edit_file, end_task,
# consider, or planning. Their parameter schemas are unchanged. Because scoring
# relies heavily on tool-name constants, canonicalize names by parameter
# signature before scaffold detection and downstream metrics such as
# SUB/TVR/DPI/SCP/FEC.

# Canonical names recognized by downstream constants.
_CANON_BASH = "execute_bash"
_CANON_EDITOR = "str_replace_editor"
_CANON_FINISH = "finish"
_CANON_THINK = "think"
_CANON_TRACKER = "task_tracker"

# Standard or already-known names require no canonicalization.
_ALREADY_CANONICAL = (
    _BASH_TOOL_NAMES | _MULTI_EDITOR_TOOL_NAMES | _PURE_EDIT_TOOL_NAMES
    | {_CANON_FINISH, _CANON_THINK, _CANON_TRACKER}
)


def _infer_canonical_name(name: str, params: set[str]) -> str | None:
    """Infer a canonical name from declared parameters, or return None."""
    if name.lower() in _ALREADY_CANONICAL:
        return None
    # finish: task-completion tool with task_completed.
    if "task_completed" in params:
        return _CANON_FINISH
    # think: reasoning tool with thought but no command or path.
    if "thought" in params and "command" not in params and "path" not in params:
        return _CANON_THINK
    # task_tracker: task management through command + task_list.
    if "task_list" in params:
        return _CANON_TRACKER
    # editor: str_replace_editor shape (command + path + edit-specific fields).
    if {"old_str", "new_str", "file_text", "view_range"} & params:
        return _CANON_EDITOR
    # bash: shell execution through command + is_input/timeout.
    if "command" in params and ("is_input" in params or "timeout" in params):
        return _CANON_BASH
    # Editor fallback: command + path without shell-specific fields.
    if "command" in params and "path" in params:
        return _CANON_EDITOR
    return None


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied record with renamed tools canonicalized."""
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
    """Score one IM record and return every TQS V2 metric.

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
    # Diagnostic only: whether reproduction in Phase 4 preceded edits in Phase 6.
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
    """Score a dataset in one pass without dataset-level statistics."""
    if not records:
        print("No records to score.")
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
            print(f"  Scored {n_scored} main-agent records")
        scored.append(scored_record)

    if not quiet:
        print(f"  Scoring complete: {n_scored} main-agent records, "
              f"{n_subagent} subagent records skipped")

    return scored


# ═══════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════

def _format_stats(values: list[float]) -> str:
    """Format N, mean, standard deviation, minimum, and maximum."""
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
    """Print statistics for one record group."""
    metrics = [
        "oec_score", "iac_score", "dpi_score",
        "ped_score", "psn_score", "tte_score", "scp_score",
        "sub_score", "fec_score", "stp_score", "tvr_score",
        "composite_score",
        # Diagnostic: mean reproduced-first rate among trajectories with source edits.
        "reproduce_first",
    ]

    print(f"\n  [{group_name}] ({len(records)} records)")
    print(f"  {'Metric':<24} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
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
    """Print score summary statistics grouped by scaffold."""
    if not scored_records:
        print("No records to summarize.")
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
    print(f"  TQS V2 trajectory quality summary — {n_scored} trajectories "
          f"({n_skipped} subagent records skipped)")
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
    """Count (total tool calls, failed tool calls) for one IM record.

    This definition matches _compute_c1:

    - Tool-call scaffolds (CC/OC/OpenHands): each role="tool" observation is one
      call, and an error in that observation is one failed call.
    - Terminus2: one user observation covers every command in the preceding
      assistant turn, so total is weighted by len(commands). An error counts as
      one failed command, matching _compute_c1's conservative estimate.

    Error detection reuses _is_error_result and C1 test-output handling. An
    observation from a test command matches only explicit execution failures
    (Tier 1), so an expected pytest failure is not a tool-call error. Detection
    also distinguishes execution and non-execution tools via _obs_is_execution.
    File-view, edit, and search tools return source or file data, so only
    explicit tool-failure markers apply; legitimate except, raise, or FAILED
    tokens in source are not tool-call failures.
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
            # Every Terminus2 observation is terminal command output.
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
    """Compute dataset-level tool-call error rates from IM records.

    - Call level: error_rate = failed tool calls / total tool calls
    - Trajectory level: trajectory_error_rate =
      trajectories with errors / trajectories with tool calls

    Error detection reuses rule_score regexes and test-output handling, keeping
    this aggregate consistent with per-trajectory c1_tool_success_rate. All
    records, including subagents, are included.
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
    """Print a dataset-level tool-call error-rate summary."""
    if not stats or not stats.get("total_tool_calls"):
        print("  Tool-call error rate: N/A (no tool calls)")
        return

    print(f"\n{'═' * 72}")
    print("  Tool-call error-rate statistics (IM records)")
    print(f"{'═' * 72}")
    print(f"  Call level: {stats['error_tool_calls']} errors / "
          f"{stats['total_tool_calls']} total "
          f"= {stats['error_rate']:.4f} ({stats['error_rate'] * 100:.2f}%)")
    print(f"  Trajectory level: {stats['trajectories_with_error']} with errors / "
          f"{stats['trajectories_with_tool_calls']} with tool calls "
          f"= {stats['trajectory_error_rate']:.4f} ({stats['trajectory_error_rate'] * 100:.2f}%)")

    by_scaffold = stats.get("by_scaffold") or {}
    if len(by_scaffold) > 1:
        print("  By scaffold:")
        for scaffold in sorted(by_scaffold):
            b = by_scaffold[scaffold]
            print(f"    [{scaffold}] call error rate {b['error_rate']:.4f} "
                  f"({b['error_tool_calls']}/{b['total_tool_calls']}), "
                  f"trajectory error rate {b['trajectory_error_rate']:.4f} "
                  f"({b['trajectories_with_error']}/{b['trajectories_with_tool_calls']})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score each trajectory in an IM-format JSONL file with TQS V2"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to the input IM JSONL file")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Path for the scored JSONL output")
    parser.add_argument("--max-instances", type=int, default=None, help="Maximum number of records to process")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output")
    parser.add_argument("--scaffold", type=str, choices=SCAFFOLD_TYPES, default=None,
                        help="Override the detected scaffold type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        sys.exit(1)

    print(f"Reading: {input_path}")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records found; exiting.")
        sys.exit(0)

    if args.max_instances is not None and args.max_instances < len(records):
        records = records[:args.max_instances]
        print(f"Limited input to {len(records)} records")

    scored_records = score_dataset(records, quiet=args.quiet, scaffold_override=args.scaffold)
    print_score_summary(scored_records)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_rule_scored.jsonl")

    save_jsonl(output_path, scored_records)
    print(f"Saved scored records to: {output_path}")


if __name__ == "__main__":
    main()
