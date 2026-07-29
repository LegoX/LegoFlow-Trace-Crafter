#!/usr/bin/env python3
"""rule_score.py 多编程语言回归测试.

覆盖多语言（Java / C / C++ / Rust / Go / TypeScript / JavaScript / Ruby / Python）
场景下 TVR 与 FEC 的跨语言口径一致性，以及测试结果判定（pass / unknown / fail）。
每个用例的命令与输出形态都取自真实轨迹中出现过的写法。

运行:
    pytest tests/test_rule_score_multilang.py
    # 未安装包时也可直接运行：python3 tests/test_rule_score_multilang.py
"""

from __future__ import annotations

import sys


def _load_rule_score():
    """优先用已安装的包；未安装时回退到源码树里的 rule_score.py。"""
    try:
        from swe_data_process import rule_score
        return rule_score
    except ImportError:
        pass

    import importlib.util
    import types
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "swe_data_process" / "rule_score.py"
    if "swe_data_process" not in sys.modules:
        pkg = types.ModuleType("swe_data_process")
        utils = types.ModuleType("swe_data_process.utils")
        utils.load_jsonl = utils.save_jsonl = lambda *a, **k: None
        sys.modules["swe_data_process"] = pkg
        sys.modules["swe_data_process.utils"] = utils
    spec = importlib.util.spec_from_file_location("_rs", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load_rule_score()

# 脚手架回填的尾部标记。命令通常经过 `| tail -N`，管道使 shell 退出码
# 恒为 tail 的 0，所以这个标记**永远是 0**，不能当作通过/失败信号。
SCAFFOLD_TAIL = (
    "\n[The command completed with exit code 0.]"
    "\n[Current working directory: /app/src]"
    "\n[Command finished with exit code 0]"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ANSI 转义不得破坏模式匹配
# ═══════════════════════════════════════════════════════════════════════════

def test_ansi_does_not_break_word_boundary():
    """`\\x1b[K` 里的 K 是 word 字符，会让 `\\bFAILED\\b` 失效。

    命令 observation 普遍含 ANSI（grep --color 与彩色 runner 的输出）。
    """
    ansi_failed = "[\x1b[01;31m\x1b[KFAILED\x1b[m\x1b[K] test_4"
    assert rs._is_error_result(ansi_failed) is True

    # Maven 的真实形态：转义序列插在 "Tests run" 和 ":" 之间
    ansi_mvn = "Tests run\x1b[m\x1b[K: 6, \x1b[01;31m\x1b[KFailures\x1b[m\x1b[K: 2"
    assert rs._test_run_outcome(ansi_mvn) == "fail"


def test_strip_ansi_is_noop_for_plain_text():
    assert rs._strip_ansi("plain text") == "plain text"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 扫描窗口必须覆盖输出结尾（测试摘要在末尾）
# ═══════════════════════════════════════════════════════════════════════════

def test_failure_at_end_of_long_output_is_detected():
    """结论在末尾、正文很长时，只扫开头会系统性漏检。"""
    long_fail = "x" * 5000 + "\nFAILED tests/test_a.py::test_x - AssertionError\n"
    assert rs._is_error_result(long_fail) is True
    assert rs._test_run_outcome(long_fail) == "fail"


# ═══════════════════════════════════════════════════════════════════════════
# 3. 多语言测试结果判定
# ═══════════════════════════════════════════════════════════════════════════

FAIL_OUTPUTS = {
    # 关键：Go 与 Jest 只打 FAIL，不是 FAILED —— 旧的 `\bFAILED\b` 抓不到
    "go":         "ok  \tgithub.com/x/y\t0.02s\nFAIL\nexit: 1",
    "go_verbose": "--- FAIL: TestPRRevert (0.00s)\n    revert_test.go:12: got 1 want 2\nFAIL",
    "jest":       "Tests:       1 failed, 36 passed, 37 total",
    "jest_suite": "Test Suites: 1 failed, 1 total",
    "maven":      "[ERROR] Tests run: 2, Failures: 0, Errors: 2, Skipped: 0\n[INFO] BUILD FAILURE",
    "gradle":     "FooTest > bar FAILED\n2 tests completed, 1 failed",
    "rust":       "test result: FAILED. 11 passed; 1 failed; 1 ignored",
    "rust_doc":   "error: doctest failed, to rerun pass `--doc`\nDOCTEST EXIT: 101",
    "rustc":      "error[E0308]: mismatched types\n --> src/lib.rs:3:5",
    "ctest":      "96% tests passed, 1 tests failed out of 24\nThe following tests FAILED:\n\t  1 - bad_inputs_test (Failed)",
    "gtest":      "[  FAILED  ] test_4\n[  FAILED  ] test_5\n=== exit: 2 ===",
    "pytest":     "FAILED tests/test_a.py::test_x - AssertionError\n1 failed, 30 passed in 3.44s",
    "phpunit":    "FAILURES!\nTests: 3, Assertions: 5, Failures: 1.",
    "rspec":      "2 examples, 1 failure",
    "mocha":      "  1 passing (130ms)\n  1 failing",
    "vitest_none": 'Error: No test files found: "/tmp/x.spec.js"\nEXIT: 1',
    "yarn":       "error Command failed with exit code 1.",
    "cc_werror":  "cc1: all warnings being treated as errors\nexit: 1",
}

PASS_OUTPUTS = {
    "go":        "ok  \tgithub.com/cli/cli/v2/pkg/cmd/pr/revert\t0.023s\ntest exit: 0",
    "go_pass":   "--- PASS: TestFoo (0.00s)\nPASS\nok  \texample.com/pkg\t0.02s",
    "rust":      "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured",
    "maven":     "[INFO] Tests run: 6, Failures: 0, Errors: 0, Skipped: 0\n[INFO] BUILD SUCCESS",
    "ctest":     "100% tests passed, 0 tests failed out of 24",
    "gtest":     "[==========] 24 tests ran.\n[  PASSED  ] 24 tests.\nexit: 0",
    "jest":      "Test Suites: 1 passed, 1 total\nTests:       13 passed, 13 total",
    "pytest":    "===== 5 passed in 0.3s =====",
    # Python unittest 的标准通过输出，旧实现完全没覆盖
    "unittest":  "----------------------------------------\nRan 97 tests in 9.254s\n\nOK",
    "mocha":     "  1 passing (66ms)",
    "ruby":      "Total: 151 tests, 459 assertions\nPass: 148, fail: 0, skip: 3.",
    "c_custom":  "repeated updates (7 bytes)       PASS\n\nAll checks PASS\n=== run exit: 0 ===",
    "tsc":       "full typecheck exit: 0\n=== error count: ===\n0",
}


def test_multilang_failures_detected():
    for name, out in FAIL_OUTPUTS.items():
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "fail", f"{name} 应判为 fail"


def test_multilang_passes_detected():
    for name, out in PASS_OUTPUTS.items():
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "pass", f"{name} 应判为 pass"


def test_zero_counts_are_not_failures():
    """`0 failed` / `0 failures` 是**通过**时的正常输出，绝不能判成失败。"""
    for out in ("1 passed; 0 failed; 0 ignored",
                "Tests run: 6, Failures: 0, Errors: 0",
                "2 examples, 0 failures",
                "0 tests failed out of 24"):
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) != "fail", out


def test_zero_passed_is_not_a_pass_signal():
    """`0 passed` 说明一个测试都没跑成，不能算通过。"""
    assert rs._test_run_outcome("collected 0 items\n0 passed, 0 failed" + SCAFFOLD_TAIL) == "unknown"
    assert rs._test_run_outcome("===== 5 passed in 0.3s =====" + SCAFFOLD_TAIL) == "pass"


def test_scaffold_exit_code_marker_is_not_a_pass_signal():
    """脚手架回填的 `exit code 0` 因管道恒为 0，不能据此判通过。"""
    assert rs._test_run_outcome("some log output" + SCAFFOLD_TAIL) == "unknown"


def test_http_status_200_is_not_an_exit_code():
    """`status: 200` 不能被当成非零退出码。"""
    assert rs._test_run_outcome("HTTP status: 200\n5 passed" + SCAFFOLD_TAIL) == "pass"


def test_failure_wins_over_partial_pass():
    assert rs._test_run_outcome("[  PASSED  ] 24 tests.\nexit: 0\n=== next ===\nTerminated\nexit: 143") == "fail"


def test_explicit_pass_beats_incidental_soft_error_token():
    """通过的测试完全可能在输出里合法打印 TypeError（正是它在测的错误路径）。"""
    out = "caught TypeError as expected\n100% tests passed, 0 tests failed out of 3"
    assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "pass"


# ═══════════════════════════════════════════════════════════════════════════
# 4. 测试执行命令识别
# ═══════════════════════════════════════════════════════════════════════════

TEST_RUN_CASES = [
    # (命令, 是否算测试执行)
    # --- Maven/Gradle 的常见形态：goal 与命令之间隔着 flag 和模块选择器 ---
    ("cd /app/src && timeout 600 ./mvnw -q -pl mod -am test -Dtest=FooTest 2>&1 | tail", True),
    ("cd /app/src/main/keychain && timeout 180 mvn -o -q test 2>&1 | tail -30", True),
    ("timeout 900 ./gradlew -q :core:test --tests '*.FooTest'", True),
    ("cd /app/src && timeout 600 mvn clean install -DskipTests", False),
    ("./gradlew build -x test", False),          # -x 是排除 test
    ("mvn -version", False),
    # --- 各语言 runner ---
    ("cd /app/src && go test ./... 2>&1 | tail -40", True),
    ("cd /app/src && cargo test -p rye --test test_sync", True),
    ("cd /app/src/build && ctest --output-on-failure 2>&1 | tail -40; echo \"exit: ${PIPESTATUS[0]}\"", True),
    ("/opt/venv/bin/python -m pytest tests/ -q", True),
    ("cd /app/src && npx vitest run", True),
    ("for f in a b; do go test $f; done", True),
    # --- Django（旧实现三种写法全漏）---
    ("./tests/runtests.py --settings=test_sqlite", True),
    ("python tests/runtests.py auth", True),
    ("python manage.py test", True),
    ("python manage.py runserver", False),
    # --- 「跑一遍程序」口径：既定设计，但需对所有语言一致 ---
    ("go run main.go", True),
    ("cargo run", True),
    ("java -cp \"$CP\" MyRepro", True),
    ("ruby app.rb", True),
    ("python app.py", True),        # 旧实现漏（Python 被排除在该口径外）
    ("node server.js", True),       # 旧实现漏（Node 被排除在该口径外）
    # --- 安装/打包/起服务不算验证 ---
    ("pip install -e . && python -m pytest", True),   # 链里有 pytest 仍算
    ("python setup.py build", False),
    ("npm run dev", False),
    ("yarn build", False),
    ("cargo build --release", False),
    ("go build ./...", False),
    # --- 假阳性：命令头是只读工具，不能因路径/文件名命中 ---
    ("cd /app/src && git diff a/src/main/java/org/x/TaskExecuteThread.java", False),
    ("cd /app/src && rm -f vitest-wiki-tmp.config.ts && git status --short", False),
    ("rm -f /tmp/verify_constraints.c /tmp/verify_constraints && echo cleaned", False),
    ("cat settings.gradle.kts | grep -i 'jupiter\\|junit' | head", False),
    ("grep -n 'go test' Makefile", False),
    ("cd /app/src && grep -rn 'EchoStrategy' src/test/", False),
    # --- python -c 被大量用来改文件，那是编辑不是自测 ---
    ("cd /app/src && python3 -c \"\np='pom.xml'; s=open(p).read()\nopen(p,'w').write(s)\n\"", False),
    ("cd /app/src && python3 -c \"import x; assert x.f()==1; print('ok')\"", True),
    # --- 纯读取 ---
    ("ls -la", False),
    ("git diff", False),
]


def test_test_run_command_detection():
    for cmd, expected in TEST_RUN_CASES:
        assert rs._is_test_run_cmd(cmd) is expected, f"{cmd!r} 期望 {expected}"


def test_formal_runner_detection_excludes_readonly_heads():
    """错误抑制路径用的严格口径同样要逐段判定。"""
    assert rs._is_formal_test_run_cmd("cd /app && go test ./...") is True
    assert rs._is_formal_test_run_cmd("rm -f vitest-tmp.config.ts") is False
    assert rs._is_formal_test_run_cmd("grep -n 'pytest' tox.ini") is False


def test_quote_aware_segment_split():
    """引号内的 ; 和 && 不是命令分隔符。"""
    segs = rs._split_shell_segments("python3 -c \"a='x'; b=1\" && echo done")
    assert len(segs) == 2, segs


def test_segment_head_strips_wrappers_and_paths():
    assert rs._segment_head("timeout 600 mvn test") == "mvn"
    assert rs._segment_head("PYTHONPATH=/app/src /opt/venv/bin/python -m pytest") == "python"
    assert rs._segment_head("./node_modules/.bin/jest") == "jest"


# ═══════════════════════════════════════════════════════════════════════════
# 5. 测试文件识别（多语言）
# ═══════════════════════════════════════════════════════════════════════════

def test_test_file_paths_multilang():
    positives = [
        "tests/test_foo.py", "conftest.py",
        "pkg/util/helper_test.go", "foo_test.cc",
        "tests/integration.rs",
        "src/test/java/com/x/FooTest.java", "FooSpec.scala",
        "src/a.test.ts", "src/__tests__/a.tsx",
        "spec/models/user_spec.rb", "test/user_test.rb",
        "src/FooTest.php",     # 旧实现漏：Tests? 后缀集只挂了 java/kt/kts/scala
        "FooTests.cs",         # 旧实现漏
        "t/basic.t",           # 旧实现漏：Perl 惯例
    ]
    for p in positives:
        assert rs._is_test_file_path(p) is True, f"{p} 应识别为测试文件"

    negatives = ["src/main.py", "src/lib.rs", "README.md", "pom.xml", "src/app/service.ts"]
    for p in negatives:
        assert rs._is_test_file_path(p) is False, f"{p} 不应识别为测试文件"


def test_words_ending_in_test_are_not_test_files():
    """*Test 后缀判据本质是 CamelCase 大小写，不能大小写不敏感地匹配。

    否则 latest / greatest / contest / attest 等普通文件名会因词尾恰好是 "test"
    被误判成测试文件（这是放宽后缀语言集时极易引入的回归）。
    """
    for p in ["latest.py", "greatest.go", "fastest.rs", "contest.js",
              "protest.ts", "attest.go", "latest.java", "request.go"]:
        assert rs._is_test_file_path(p) is False, f"{p} 不应识别为测试文件"
    # CamelCase 的真测试文件仍需命中
    for p in ["FooTest.php", "FooTests.cs", "UserSpec.kt", "MyServiceTest.java"]:
        assert rs._is_test_file_path(p) is True, f"{p} 应识别为测试文件"


# ═══════════════════════════════════════════════════════════════════════════
# 6. FEC：编辑提取不得有盲区，且盲区不能反向加分
# ═══════════════════════════════════════════════════════════════════════════

def _tc(name, args):
    import json
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_fec_catches_renamed_shell_ipython_and_multiedit():
    """旧实现用工具名单 + args['command']，这三种编辑一个都抓不到。"""
    msgs = [{"role": "assistant", "tool_calls": [
        _tc("shell_exec", {"command": "sed -i 's/a/b/' src/main.go"}),
        _tc("execute_ipython_cell", {"code": "open('x.py','w').write('1')"}),
        _tc("MultiEdit", {"file_path": "/testbed/pkg/util.go"}),
        _tc("NotebookEdit", {"notebook_path": "/testbed/nb.ipynb"}),
    ]}]
    paths = rs._extract_edit_paths(msgs, "openhands")
    assert "src/main.go" in paths
    assert "pkg/util.go" in paths
    assert "nb.ipynb" in paths


def test_fec_returns_none_when_no_edits_detected():
    """「没检测到编辑」必须走 fail-soft，而不是当成完美集中度给满分。"""
    msgs = [{"role": "assistant", "content": "just thinking"}]
    assert rs._compute_fec(msgs, "openhands") is None


def test_fec_none_is_skipped_by_aggregate():
    base = {"sub": 1.0, "stp": 1.0, "tvr": 1.0, "dpi": 1.0, "fec": None}
    assert rs._aggregate_tqs(base) == 1.0


def test_bash_edit_idioms_multilang():
    cases = [
        ("sed -i 's/a/b/' src/main.go", ["src/main.go"]),
        ("sed -i.bak 's/a/b/' src/x.c", ["src/x.c"]),
        ("gsed -i 's/a/b/' a.js", ["a.js"]),                  # 旧实现漏
        ("perl -pi -e 's/a/b/' src/x.pl", ["src/x.pl"]),      # 旧实现漏
        ("cat > pkg/foo_test.go <<'EOF'", ["pkg/foo_test.go"]),
        ("printf 'x' >> src/a.rs", ["src/a.rs"]),
        ("sed -n '1,60p' file.py", []),                       # 只读，不是编辑
        # `-MList::Util` 里含 i，不能被当成 `-i` 就地编辑
        ("perl -MList::Util -e 'print 1' file.py", []),
        ("perl -pi.orig -e 's/a/b/' x.pl", ["x.pl"]),
        ("sed -i -f script.sed src/a.go", ["src/a.go"]),      # -f 的参数是脚本不是目标
    ]
    for cmd, expected in cases:
        assert rs._extract_bash_edit_paths(cmd) == expected, cmd


# ═══════════════════════════════════════════════════════════════════════════
# 7. TVR 端到端
# ═══════════════════════════════════════════════════════════════════════════

def _traj(cmd, obs):
    return [
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": cmd})]},
        {"role": "tool", "content": obs},
        {"role": "assistant", "content": "done"},
    ]


def test_tvr_go_failure_no_longer_outscores_python_failure():
    """核心回归：同样是「跑了测试且测试挂了」，Go 不该比 Python 拿更高分。"""
    go = rs._compute_tvr(_traj("go test ./...", "ok x\nFAIL" + SCAFFOLD_TAIL), "openhands")
    py = rs._compute_tvr(_traj("python -m pytest", "FAILED t.py::x - AssertionError" + SCAFFOLD_TAIL), "openhands")
    assert go == py, f"go={go} python={py}"


def test_tvr_stale_observation_not_reused_on_truncation():
    """最后一次测试没有对应 observation 时，不能沿用上一次测试的结果。"""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": "pytest"})]},
        {"role": "tool", "content": "5 passed" + SCAFFOLD_TAIL},
        # 第二次测试被截断，没有 tool 响应
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": "pytest"})]},
    ]
    tvr = rs._compute_tvr(msgs, "openhands")
    # 0.3(run) + 0.4*0.6(unknown) = 0.54，而不是沿用上次的 pass 拿 0.7
    assert abs(tvr - 0.54) < 1e-9, tvr


# ───────────────────────────────────────────────────────────────────────────
# 第二轮 code review 发现的问题（2026-07-29）
# ───────────────────────────────────────────────────────────────────────────

def _renamed_shell_traj(tool_name):
    """两次 shell 编辑 + 文本收尾；工具名可替换，用于验证改名鲁棒性。"""
    import json
    return {
        "tools": [{"type": "function", "function": {
            "name": tool_name,
            # 只声明 command —— _infer_canonical_name 无法据此归一，
            # 因此所有依赖工具名单的路径都必须自己扛住。
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        }}],
        "messages": [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "", "tool_calls": [
                _tc(tool_name, {"command": "cat > src/a.py <<'EOF'\nx=1\nEOF"})]},
            {"role": "tool", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [
                _tc(tool_name, {"command": "sed -i 's/x=1/x=2/' src/a.py"})]},
            {"role": "tool", "content": ""},
            {"role": "assistant", "content": "done, all tests pass"},
        ],
    }


def test_renamed_shell_tool_scores_identically_to_execute_bash():
    """改名 shell 工具不得因为「名字不在 _BASH_TOOL_NAMES」而被扣分。

    修复前：DPI 1.0 -> 0.6（误判从未成功写入，-0.40），SCP 1.0 -> 0.0，
    composite 0.5857 -> 0.5309，差异纯粹来自工具名。
    """
    std = rs.score_record(_renamed_shell_traj("execute_bash"))
    ren = rs.score_record(_renamed_shell_traj("shell_exec"))
    for k in ("fec_score", "dpi_score", "scp_score", "psn_score", "composite_score"):
        assert std[k] == ren[k], f"{k}: execute_bash={std[k]} shell_exec={ren[k]}"


def test_has_successful_write_survives_tool_rename():
    msgs = _renamed_shell_traj("shell_exec")["messages"]
    obs = rs._extract_observations(msgs, "openhands")
    assert rs._has_successful_write(msgs, obs, "openhands") is True
    assert rs._first_successful_write_turn(msgs, obs, "openhands") == 1
    assert rs._per_step_target_files(msgs, "openhands")[0] == {"src/a.py"}


def test_scan_window_tail_aligns_to_line_start():
    """尾窗不得从行中间切开 —— 否则会伪造出行首，让 ^FAILED/^FAIL/^[ERROR] 误命中。"""
    filler = "x" * 8000
    body = filler + "The state was FAILED: cleanup done\n" + "y" * 2960 + "\n"
    assert len(body) > 6000
    win = rs._scan_window(body)
    assert "FAILED: cleanup done" not in win.split("\n", 1)[-1][3000:], "残行未被丢弃"
    assert rs._test_run_outcome(body) == "unknown"


def test_zero_counts_are_not_pass_signals():
    """一个测试都没跑成不是「通过」，应为 unknown。"""
    for text in [
        "  0 passing (2ms)",
        "Pass: 0, fail: 0, error: 0",
        "OK (0 tests)",
        "[INFO] Tests run: 0, Failures: 0, Errors: 0, Skipped: 0",
        "0 examples, 0 failures",
        "Ran 0 tests in 0.001s\n\nOK",
        "collected 0 items\n\nno tests ran in 0.01s",
        "No tests were found!!!\n[The command completed with exit code 0.]",
    ]:
        assert rs._test_run_outcome(text) == "unknown", text
    for text in ["1 passed in 0.5s", "5 passing (30ms)", "OK (12 tests)",
                 "[INFO] Tests run: 6, Failures: 0, Errors: 0",
                 "Ran 97 tests in 9.254s\n\nOK"]:
        assert rs._test_run_outcome(text) == "pass", text


def test_negative_exit_codes_count_as_failure():
    """超时/信号杀死时脚手架回填负退出码，缺 `-?` 会整类漏判成 unknown。"""
    assert rs._test_run_outcome("[The command completed with exit code -1.]") == "fail"
    assert rs._test_run_outcome("[The command completed with exit code -9.]") == "fail"
    assert rs._test_run_outcome("[The command completed with exit code 0.]") != "fail"


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in fns:
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
    print(f"passed={passed} failed={len(failed)}")
    for name, err in failed:
        print(f"  FAIL {name}\n       {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
