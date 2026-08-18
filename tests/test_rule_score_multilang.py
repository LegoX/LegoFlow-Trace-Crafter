#!/usr/bin/env python3
"""Multilanguage regression tests for rule_score.py.

Verify consistent TVR and FEC behavior across Java, C, C++, Rust, Go,
TypeScript, JavaScript, Ruby, and Python, including test outcome
classification (pass / unknown / fail). Command and output fixtures use forms
observed in real trajectories.

Run:
    pytest tests/test_rule_score_multilang.py
    # If the package is not installed: python3 tests/test_rule_score_multilang.py
"""

from __future__ import annotations

import sys


def _load_rule_score():
    """Load the installed package, falling back to rule_score.py in the source tree."""
    try:
        from legoflow_trace_crafter import rule_score
        return rule_score
    except ImportError:
        pass

    import importlib.util
    import types
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "legoflow_trace_crafter" / "rule_score.py"
    if "legoflow_trace_crafter" not in sys.modules:
        pkg = types.ModuleType("legoflow_trace_crafter")
        utils = types.ModuleType("legoflow_trace_crafter.utils")
        utils.load_jsonl = utils.save_jsonl = lambda *a, **k: None
        sys.modules["legoflow_trace_crafter"] = pkg
        sys.modules["legoflow_trace_crafter.utils"] = utils
    spec = importlib.util.spec_from_file_location("_rs", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load_rule_score()

# Trailing markers injected by the scaffold. Commands commonly use `| tail -N`,
# so the shell reports tail's status (always 0 here). These markers therefore
# cannot indicate whether the test passed or failed.
SCAFFOLD_TAIL = (
    "\n[The command completed with exit code 0.]"
    "\n[Current working directory: /app/src]"
    "\n[Command finished with exit code 0]"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ANSI escapes must not disrupt pattern matching
# ═══════════════════════════════════════════════════════════════════════════

def test_ansi_does_not_break_word_boundary():
    """The word character K in `\\x1b[K` can break `\\bFAILED\\b` matching.

    Command observations commonly contain ANSI escapes from colored grep and
    test-runner output.
    """
    ansi_failed = "[\x1b[01;31m\x1b[KFAILED\x1b[m\x1b[K] test_4"
    assert rs._is_error_result(ansi_failed) is True

    # Real Maven output can insert escape sequences between "Tests run" and ":".
    ansi_mvn = "Tests run\x1b[m\x1b[K: 6, \x1b[01;31m\x1b[KFailures\x1b[m\x1b[K: 2"
    assert rs._test_run_outcome(ansi_mvn) == "fail"


def test_strip_ansi_is_noop_for_plain_text():
    assert rs._strip_ansi("plain text") == "plain text"


# ═══════════════════════════════════════════════════════════════════════════
# 2. The scan window must include the output tail, where summaries appear
# ═══════════════════════════════════════════════════════════════════════════

def test_failure_at_end_of_long_output_is_detected():
    """Scanning only the head of long output would consistently miss its conclusion."""
    long_fail = "x" * 5000 + "\nFAILED tests/test_a.py::test_x - AssertionError\n"
    assert rs._is_error_result(long_fail) is True
    assert rs._test_run_outcome(long_fail) == "fail"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Multilanguage test outcome classification
# ═══════════════════════════════════════════════════════════════════════════

FAIL_OUTPUTS = {
    # Go and Jest emit FAIL rather than FAILED, which `\bFAILED\b` cannot match.
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
    # Standard successful Python unittest output was previously unsupported.
    "unittest":  "----------------------------------------\nRan 97 tests in 9.254s\n\nOK",
    "mocha":     "  1 passing (66ms)",
    "ruby":      "Total: 151 tests, 459 assertions\nPass: 148, fail: 0, skip: 3.",
    "c_custom":  "repeated updates (7 bytes)       PASS\n\nAll checks PASS\n=== run exit: 0 ===",
    "tsc":       "full typecheck exit: 0\n=== error count: ===\n0",
}


def test_multilang_failures_detected():
    for name, out in FAIL_OUTPUTS.items():
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "fail", f"{name} should be fail"


def test_multilang_passes_detected():
    for name, out in PASS_OUTPUTS.items():
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "pass", f"{name} should be pass"


def test_zero_counts_are_not_failures():
    """`0 failed` and `0 failures` are normal successful output, not failures."""
    for out in ("1 passed; 0 failed; 0 ignored",
                "Tests run: 6, Failures: 0, Errors: 0",
                "2 examples, 0 failures",
                "0 tests failed out of 24"):
        assert rs._test_run_outcome(out + SCAFFOLD_TAIL) != "fail", out


def test_zero_passed_is_not_a_pass_signal():
    """`0 passed` means no tests ran and must not indicate success."""
    assert rs._test_run_outcome("collected 0 items\n0 passed, 0 failed" + SCAFFOLD_TAIL) == "unknown"
    assert rs._test_run_outcome("===== 5 passed in 0.3s =====" + SCAFFOLD_TAIL) == "pass"


def test_scaffold_exit_code_marker_is_not_a_pass_signal():
    """A scaffold `exit code 0` marker reflects the pipeline, not test success."""
    assert rs._test_run_outcome("some log output" + SCAFFOLD_TAIL) == "unknown"


def test_http_status_200_is_not_an_exit_code():
    """`status: 200` must not be interpreted as a nonzero exit status."""
    assert rs._test_run_outcome("HTTP status: 200\n5 passed" + SCAFFOLD_TAIL) == "pass"


def test_failure_wins_over_partial_pass():
    assert rs._test_run_outcome("[  PASSED  ] 24 tests.\nexit: 0\n=== next ===\nTerminated\nexit: 143") == "fail"


def test_explicit_pass_beats_incidental_soft_error_token():
    """A passing test may legitimately print TypeError while testing an error path."""
    out = "caught TypeError as expected\n100% tests passed, 0 tests failed out of 3"
    assert rs._test_run_outcome(out + SCAFFOLD_TAIL) == "pass"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Test command detection
# ═══════════════════════════════════════════════════════════════════════════

TEST_RUN_CASES = [
    # (command, whether it counts as test execution)
    # --- Common Maven/Gradle forms with flags and module selectors before the goal ---
    ("cd /app/src && timeout 600 ./mvnw -q -pl mod -am test -Dtest=FooTest 2>&1 | tail", True),
    ("cd /app/src/main/keychain && timeout 180 mvn -o -q test 2>&1 | tail -30", True),
    ("timeout 900 ./gradlew -q :core:test --tests '*.FooTest'", True),
    ("cd /app/src && timeout 600 mvn clean install -DskipTests", False),
    ("./gradlew build -x test", False),          # -x excludes the test task.
    ("mvn -version", False),
    # --- Test runners across languages ---
    ("cd /app/src && go test ./... 2>&1 | tail -40", True),
    ("cd /app/src && cargo test -p rye --test test_sync", True),
    ("cd /app/src/build && ctest --output-on-failure 2>&1 | tail -40; echo \"exit: ${PIPESTATUS[0]}\"", True),
    ("/opt/venv/bin/python -m pytest tests/ -q", True),
    ("cd /app/src && npx vitest run", True),
    ("for f in a b; do go test $f; done", True),
    # --- Django forms that were all missed previously ---
    ("./tests/runtests.py --settings=test_sqlite", True),
    ("python tests/runtests.py auth", True),
    ("python manage.py test", True),
    ("python manage.py runserver", False),
    # --- Running a program once counts as verification, consistently across languages ---
    ("go run main.go", True),
    ("cargo run", True),
    ("java -cp \"$CP\" MyRepro", True),
    ("ruby app.rb", True),
    ("python app.py", True),        # Previously missed because Python was excluded.
    ("node server.js", True),       # Previously missed because Node was excluded.
    # --- Installation, packaging, and service startup are not verification ---
    ("pip install -e . && python -m pytest", True),   # The pytest segment still counts.
    ("python setup.py build", False),
    ("npm run dev", False),
    ("yarn build", False),
    ("cargo build --release", False),
    ("go build ./...", False),
    # --- Avoid false positives from paths or filenames in read-only commands ---
    ("cd /app/src && git diff a/src/main/java/org/x/TaskExecuteThread.java", False),
    ("cd /app/src && rm -f vitest-wiki-tmp.config.ts && git status --short", False),
    ("rm -f /tmp/verify_constraints.c /tmp/verify_constraints && echo cleaned", False),
    ("cat settings.gradle.kts | grep -i 'jupiter\\|junit' | head", False),
    ("grep -n 'go test' Makefile", False),
    ("cd /app/src && grep -rn 'EchoStrategy' src/test/", False),
    # --- python -c commonly edits files; that is editing, not inline verification ---
    ("cd /app/src && python3 -c \"\np='pom.xml'; s=open(p).read()\nopen(p,'w').write(s)\n\"", False),
    ("cd /app/src && python3 -c \"import x; assert x.f()==1; print('ok')\"", True),
    # --- Read-only commands ---
    ("ls -la", False),
    ("git diff", False),
]


def test_test_run_command_detection():
    for cmd, expected in TEST_RUN_CASES:
        assert rs._is_test_run_cmd(cmd) is expected, f"{cmd!r} expected {expected}"


def test_formal_runner_detection_excludes_readonly_heads():
    """Strict detection for error suppression must also inspect each segment."""
    assert rs._is_formal_test_run_cmd("cd /app && go test ./...") is True
    assert rs._is_formal_test_run_cmd("rm -f vitest-tmp.config.ts") is False
    assert rs._is_formal_test_run_cmd("grep -n 'pytest' tox.ini") is False


def test_quote_aware_segment_split():
    """Semicolons and && inside quotes are not command separators."""
    segs = rs._split_shell_segments("python3 -c \"a='x'; b=1\" && echo done")
    assert len(segs) == 2, segs


def test_segment_head_strips_wrappers_and_paths():
    assert rs._segment_head("timeout 600 mvn test") == "mvn"
    assert rs._segment_head("PYTHONPATH=/app/src /opt/venv/bin/python -m pytest") == "python"
    assert rs._segment_head("./node_modules/.bin/jest") == "jest"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Multilanguage test-file detection
# ═══════════════════════════════════════════════════════════════════════════

def test_test_file_paths_multilang():
    positives = [
        "tests/test_foo.py", "conftest.py",
        "pkg/util/helper_test.go", "foo_test.cc",
        "tests/integration.rs",
        "src/test/java/com/x/FooTest.java", "FooSpec.scala",
        "src/a.test.ts", "src/__tests__/a.tsx",
        "spec/models/user_spec.rb", "test/user_test.rb",
        "src/FooTest.php",     # Previously missed: Tests? suffixes covered only JVM languages.
        "FooTests.cs",         # Previously missed.
        "t/basic.t",           # Previously missed Perl convention.
    ]
    for p in positives:
        assert rs._is_test_file_path(p) is True, f"{p} should be recognized as a test file"

    negatives = ["src/main.py", "src/lib.rs", "README.md", "pom.xml", "src/app/service.ts"]
    for p in negatives:
        assert rs._is_test_file_path(p) is False, f"{p} should not be recognized as a test file"


def test_words_ending_in_test_are_not_test_files():
    """The *Test suffix rule requires CamelCase and must remain case-sensitive.

    Otherwise, ordinary names such as latest, greatest, contest, and attest
    would be misclassified merely because they end in "test".
    """
    for p in ["latest.py", "greatest.go", "fastest.rs", "contest.js",
              "protest.ts", "attest.go", "latest.java", "request.go"]:
        assert rs._is_test_file_path(p) is False, f"{p} should not be recognized as a test file"
    # Actual CamelCase test files must still match.
    for p in ["FooTest.php", "FooTests.cs", "UserSpec.kt", "MyServiceTest.java"]:
        assert rs._is_test_file_path(p) is True, f"{p} should be recognized as a test file"


# ═══════════════════════════════════════════════════════════════════════════
# 6. FEC edit extraction must not miss edits or reward detection gaps
# ═══════════════════════════════════════════════════════════════════════════

def _tc(name, args):
    import json
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_fec_catches_renamed_shell_ipython_and_multiedit():
    """The old tool-name plus args['command'] logic missed all three edit forms."""
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
    """No detected edits must fail soft, not receive a perfect concentration score."""
    msgs = [{"role": "assistant", "content": "just thinking"}]
    assert rs._compute_fec(msgs, "openhands") is None


def test_fec_none_is_skipped_by_aggregate():
    base = {"sub": 1.0, "stp": 1.0, "tvr": 1.0, "dpi": 1.0, "fec": None}
    assert rs._aggregate_tqs(base) == 1.0


def test_bash_edit_idioms_multilang():
    cases = [
        ("sed -i 's/a/b/' src/main.go", ["src/main.go"]),
        ("sed -i.bak 's/a/b/' src/x.c", ["src/x.c"]),
        ("gsed -i 's/a/b/' a.js", ["a.js"]),                  # Previously missed.
        ("perl -pi -e 's/a/b/' src/x.pl", ["src/x.pl"]),      # Previously missed.
        ("cat > pkg/foo_test.go <<'EOF'", ["pkg/foo_test.go"]),
        ("printf 'x' >> src/a.rs", ["src/a.rs"]),
        ("sed -n '1,60p' file.py", []),                       # Read-only, not an edit.
        # The i in `-MList::Util` must not be mistaken for an in-place `-i`.
        ("perl -MList::Util -e 'print 1' file.py", []),
        ("perl -pi.orig -e 's/a/b/' x.pl", ["x.pl"]),
        ("sed -i -f script.sed src/a.go", ["src/a.go"]),      # The -f argument is a script.
    ]
    for cmd, expected in cases:
        assert rs._extract_bash_edit_paths(cmd) == expected, cmd


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-end TVR behavior
# ═══════════════════════════════════════════════════════════════════════════

def _traj(cmd, obs):
    return [
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": cmd})]},
        {"role": "tool", "content": obs},
        {"role": "assistant", "content": "done"},
    ]


def test_tvr_go_failure_no_longer_outscores_python_failure():
    """Equivalent failed Go and Python test runs must receive the same score."""
    go = rs._compute_tvr(_traj("go test ./...", "ok x\nFAIL" + SCAFFOLD_TAIL), "openhands")
    py = rs._compute_tvr(_traj("python -m pytest", "FAILED t.py::x - AssertionError" + SCAFFOLD_TAIL), "openhands")
    assert go == py, f"go={go} python={py}"


def test_tvr_stale_observation_not_reused_on_truncation():
    """A truncated final test run must not reuse the previous observation."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": "pytest"})]},
        {"role": "tool", "content": "5 passed" + SCAFFOLD_TAIL},
        # The second test run is truncated before its tool response.
        {"role": "assistant", "tool_calls": [_tc("terminal", {"command": "pytest"})]},
    ]
    tvr = rs._compute_tvr(msgs, "openhands")
    # 0.3 (run) + 0.4 * 0.6 (unknown) = 0.54, not 0.7 from reusing the prior pass.
    assert abs(tvr - 0.54) < 1e-9, tvr


# ───────────────────────────────────────────────────────────────────────────
# Issues found in the second code review (2026-07-29)
# ───────────────────────────────────────────────────────────────────────────

def _renamed_shell_traj(tool_name):
    """Build two shell edits plus a text conclusion to test tool-name robustness."""
    import json
    return {
        "tools": [{"type": "function", "function": {
            "name": tool_name,
            # Declaring only command prevents _infer_canonical_name from
            # canonicalizing this tool, so downstream paths must handle it.
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
    """A renamed shell tool must not lose points because its name is unknown.

    Before the fix, a false "never successfully wrote" result changed DPI from
    1.0 to 0.6, SCP from 1.0 to 0.0, and the composite from 0.5857 to 0.5309.
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
    """The tail window must not create false line starts by splitting a line."""
    filler = "x" * 8000
    body = filler + "The state was FAILED: cleanup done\n" + "y" * 2960 + "\n"
    assert len(body) > 6000
    win = rs._scan_window(body)
    assert "FAILED: cleanup done" not in win.split("\n", 1)[-1][3000:], "partial line was not dropped"
    assert rs._test_run_outcome(body) == "unknown"


def test_zero_counts_are_not_pass_signals():
    """A run that executes no tests is unknown, not a pass."""
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
    """Negative timeout or signal exit codes must be classified as failures."""
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
