from __future__ import annotations

import json

import pytest

from swe_data_process.rule_score import (
    _compute_a1,
    _compute_a2,
    _compute_b1,
    _compute_c1,
    _compute_d1,
    _compute_d2,
    _compute_e1,
    _compute_e2,
    _count_tool_call_errors,
    _extract_actions,
    _extract_observations,
    _is_error_result,
    compute_tool_call_error_rate,
    detect_scaffold,
    score_record,
)


class TestDetectScaffold:
    def test_terminus2_no_tools(self):
        record = {"messages": [{"role": "user", "content": "hi"}]}
        assert detect_scaffold(record) == "terminus2"

    def test_terminus2_none_tools(self):
        record = {"messages": [], "tools": None}
        assert detect_scaffold(record) == "terminus2"

    def test_terminus2_empty_tools(self):
        record = {"messages": [], "tools": []}
        assert detect_scaffold(record) == "terminus2"

    def test_openhands_sdk(self):
        record = {
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "terminal"}},
                {"type": "function", "function": {"name": "file_editor"}},
                {"type": "function", "function": {"name": "finish"}},
            ],
        }
        assert detect_scaffold(record) == "openhands_sdk"

    def test_claudecode(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are Claude Code, by Anthropic."},
                {"role": "user", "content": "fix bug"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
        }
        assert detect_scaffold(record) == "claudecode"

    def test_opencode(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "fix bug"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
        }
        assert detect_scaffold(record) == "opencode"

    def test_openhands_native(self):
        record = {
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "execute_bash"}},
                {"type": "function", "function": {"name": "str_replace_editor"}},
            ],
        }
        assert detect_scaffold(record) == "openhands"


class TestIsErrorResult:
    def test_command_not_found(self):
        assert _is_error_result("bash: foo: command not found") is True

    def test_permission_denied(self):
        assert _is_error_result("Permission denied") is True

    def test_exit_code_nonzero(self):
        assert _is_error_result("exit code: 1") is True

    def test_tool_use_error(self):
        assert _is_error_result("<tool_use_error>invalid") is True

    def test_traceback_not_test(self):
        assert _is_error_result("Traceback (most recent call last)") is True

    def test_traceback_in_test_output(self):
        assert _is_error_result("Traceback (most recent call last)", is_test_output=True) is False

    def test_failed_not_test(self):
        assert _is_error_result("FAILED tests/test_foo.py") is True

    def test_failed_in_test_output(self):
        assert _is_error_result("FAILED tests/test_foo.py", is_test_output=True) is False

    def test_clean_output(self):
        assert _is_error_result("file1.py\nfile2.py\n") is False

    def test_empty_string(self):
        assert _is_error_result("") is False

    def test_hard_error_in_test_output_still_detected(self):
        assert _is_error_result("Permission denied", is_test_output=True) is True


class TestScoreRecord:
    @pytest.fixture
    def cc_record(self):
        return {
            "messages": [
                {"role": "system", "content": "You are Claude Code, by Anthropic."},
                {"role": "user", "content": "Fix the bug in utils.py"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path":"/testbed/utils.py"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "def foo():\n    pass"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "Edit",
                                "arguments": '{"file_path":"/testbed/utils.py","old_string":"pass","new_string":"return 42"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "File edited successfully."},
                {"role": "assistant", "content": "Fixed the bug."},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "Edit", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "Write", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "Grep", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}}},
            ],
        }

    def test_returns_all_expected_keys(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        expected_keys = {
            "scaffold", "assistant_turns", "total_tool_calls", "error_retry_cycles",
            "a1_error_retry", "a2_step_count_ratio",
            "b1_action_diversity", "b2_observation_utilization",
            "c1_tool_success_rate", "c2_tool_parallelism",
            "d1_submission_completeness", "d2_test_verification",
            "e1_file_edit_concentration", "e2_delete_then_modify",
            "efficiency_score", "style_score", "tool_mastery_score",
            "completion_score", "precision_score",
            "composite_score_v3", "composite_score_v4", "composite_score",
        }
        assert set(result.keys()) == expected_keys

    def test_scores_in_valid_range(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        for key, value in result.items():
            if key in ("scaffold", "assistant_turns", "total_tool_calls", "error_retry_cycles"):
                continue
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"

    def test_scaffold_detected(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        assert result["scaffold"] == "claudecode"

    def test_scaffold_override(self, cc_record):
        result = score_record(cc_record, median_steps=5.0, scaffold_override="opencode")
        assert result["scaffold"] == "opencode"

    def test_no_errors_high_c1(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        assert result["c1_tool_success_rate"] == 1.0

    def test_submission_complete(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        assert result["d1_submission_completeness"] == 1.0


class TestComputeA2:
    def test_steps_equal_median(self):
        result = _compute_a2(10, 10.0)
        assert result == 0.8

    def test_fewer_steps_higher_score(self):
        few = _compute_a2(3, 10.0)
        many = _compute_a2(20, 10.0)
        assert few > many

    def test_zero_median(self):
        result = _compute_a2(5, 0.0)
        assert 0.0 <= result <= 1.0


class TestComputeB1:
    def test_single_tool_type(self):
        actions = [{"tool_name": "Read", "msg_idx": 0}] * 5
        result = _compute_b1(actions, n_available_tools=5)
        assert result == 0.0

    def test_diverse_tools(self):
        actions = [
            {"tool_name": "Read", "msg_idx": 0},
            {"tool_name": "Edit", "msg_idx": 1},
            {"tool_name": "Bash", "msg_idx": 2},
            {"tool_name": "Grep", "msg_idx": 3},
        ]
        result = _compute_b1(actions, n_available_tools=5)
        assert result > 0.5

    def test_empty_actions(self):
        assert _compute_b1([], n_available_tools=5) == 0.0


class TestComputeC1:
    def test_no_errors(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]},
            {"role": "tool", "content": "success"},
        ]
        observations = _extract_observations(messages, "claudecode")
        result = _compute_c1(observations, messages, "claudecode")
        assert result == 1.0

    def test_all_errors(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Bash", "arguments": '{"command":"bad"}'}}]},
            {"role": "tool", "content": "bash: bad: command not found"},
        ]
        observations = _extract_observations(messages, "claudecode")
        result = _compute_c1(observations, messages, "claudecode")
        assert result == 0.0

    def test_empty_observations(self):
        assert _compute_c1([], [], "claudecode") == 1.0


class TestComputeToolCallErrorRate:
    @staticmethod
    def _cc_record(tool_contents: list[str]) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": "You are Claude Code, by Anthropic."},
            {"role": "user", "content": "fix"},
        ]
        for content in tool_contents:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": '{"command":"ls"}'}}],
            })
            messages.append({"role": "tool", "content": content})
        messages.append({"role": "assistant", "content": "done"})
        return {
            "messages": messages,
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
        }

    @staticmethod
    def _t2_record(commands: list[str], obs_content: str) -> dict:
        assistant_json = json.dumps({"commands": [{"keystrokes": c} for c in commands]})
        return {
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": assistant_json},
                {"role": "user", "content": obs_content},
            ],
            # T2 has no tools field -> detect_scaffold returns terminus2
        }

    def test_single_tool_call_scaffold_one_error(self):
        record = self._cc_record(["ok output", "bash: bad: command not found", "ok"])
        total, errors = _count_tool_call_errors(record, "claudecode")
        assert (total, errors) == (3, 1)

        stats = compute_tool_call_error_rate([record])
        assert stats["total_tool_calls"] == 3
        assert stats["error_tool_calls"] == 1
        assert stats["error_rate"] == pytest.approx(1 / 3, rel=1e-3)
        assert stats["trajectories_with_tool_calls"] == 1
        assert stats["trajectories_with_error"] == 1
        assert stats["trajectory_error_rate"] == 1.0

    def test_tool_use_error_counts(self):
        record = self._cc_record(["<tool_use_error>invalid args", "fine"])
        total, errors = _count_tool_call_errors(record, "claudecode")
        assert (total, errors) == (2, 1)

    def test_terminus2_weighted_by_command_count(self):
        # 3 commands in one assistant turn, single non-error observation
        record = self._t2_record(["ls", "cat foo", "pwd"], "file listing\n")
        total, errors = _count_tool_call_errors(record, "terminus2")
        assert total == 3
        assert errors == 0

    def test_terminus2_error_counts_one_failed_command(self):
        record = self._t2_record(["ls", "bad-cmd"], "bash: bad-cmd: command not found")
        total, errors = _count_tool_call_errors(record, "terminus2")
        # total weighted by 2 commands; error observation counts as 1 failed command
        assert total == 2
        assert errors == 1

    def test_test_run_failed_output_not_counted(self):
        # A pytest run whose output contains FAILED must NOT count as a tool error
        record = {
            "messages": [
                {"role": "system", "content": "You are Claude Code, by Anthropic."},
                {"role": "user", "content": "run tests"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": '{"command":"pytest tests/"}'}}],
                },
                {"role": "tool", "content": "FAILED tests/test_foo.py::test_x - assert 1 == 2"},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
        }
        total, errors = _count_tool_call_errors(record, "claudecode")
        assert total == 1
        assert errors == 0

    def test_test_run_hard_error_still_counted(self):
        # Even in a test-run turn, a hard execution error (Tier 1) should count
        record = {
            "messages": [
                {"role": "system", "content": "You are Claude Code, by Anthropic."},
                {"role": "user", "content": "run tests"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "Bash", "arguments": '{"command":"pytest tests/"}'}}],
                },
                {"role": "tool", "content": "pytest: command not found"},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [{"type": "function", "function": {"name": "Bash"}}, {"type": "function", "function": {"name": "Read"}}, {"type": "function", "function": {"name": "Edit"}}],
        }
        total, errors = _count_tool_call_errors(record, "claudecode")
        assert (total, errors) == (1, 1)

    def test_per_scaffold_bucketing_and_aggregate(self):
        cc = self._cc_record(["ok", "Permission denied"])
        t2 = self._t2_record(["ls", "pwd"], "ok output")
        stats = compute_tool_call_error_rate([cc, t2])

        assert stats["total_tool_calls"] == 4  # cc: 2, t2: 2 commands
        assert stats["error_tool_calls"] == 1
        assert stats["error_rate"] == pytest.approx(0.25, rel=1e-3)
        assert stats["trajectories_with_tool_calls"] == 2
        assert stats["trajectories_with_error"] == 1

        by = stats["by_scaffold"]
        assert set(by) == {"claudecode", "terminus2"}
        assert by["claudecode"]["total_tool_calls"] == 2
        assert by["claudecode"]["error_tool_calls"] == 1
        assert by["claudecode"]["error_rate"] == pytest.approx(0.5, rel=1e-3)
        assert by["terminus2"]["total_tool_calls"] == 2
        assert by["terminus2"]["error_tool_calls"] == 0
        assert by["terminus2"]["error_rate"] == 0.0

    def test_record_without_tool_calls_skipped(self):
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "tools": [{"type": "function", "function": {"name": "Bash"}}, {"type": "function", "function": {"name": "Read"}}, {"type": "function", "function": {"name": "Edit"}}],
        }
        stats = compute_tool_call_error_rate([record])
        assert stats["total_tool_calls"] == 0
        assert stats["trajectories_with_tool_calls"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["by_scaffold"] == {}

    def test_empty_dataset(self):
        stats = compute_tool_call_error_rate([])
        assert stats["total_tool_calls"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["trajectory_error_rate"] == 0.0
        assert stats["by_scaffold"] == {}


class TestComputeD1:
    def test_cc_ends_with_assistant_no_tools(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "Done."},
        ]
        assert _compute_d1(messages, "claudecode") == 1.0

    def test_cc_ends_with_tool_call(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read"}}]},
        ]
        assert _compute_d1(messages, "claudecode") == 0.0

    def test_ohsdk_with_finish(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "finish", "arguments": '{"message":"done"}'}}],
            },
            {"role": "tool", "content": "Task completed."},
        ]
        assert _compute_d1(messages, "openhands_sdk") == 1.0


class TestComputeD2:
    def test_has_test_write_and_run(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "Edit", "arguments": '{"file_path":"test_foo.py","old_string":"","new_string":"def test_x(): pass"}'}},
                ],
            },
            {"role": "tool", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "Bash", "arguments": '{"command":"pytest test_foo.py"}'}},
                ],
            },
            {"role": "tool", "content": "1 passed"},
            {"role": "assistant", "content": "Tests pass."},
        ]
        result = _compute_d2(messages, "claudecode")
        assert result == 1.0

    def test_no_test_activity(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Edit", "arguments": '{"file_path":"utils.py"}'}}],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Done."},
        ]
        result = _compute_d2(messages, "claudecode")
        assert result == 0.0


class TestComputeE1:
    def test_single_file_single_edit(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Edit", "arguments": '{"file_path":"/testbed/utils.py","old_string":"a","new_string":"b"}'}}],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Done."},
        ]
        result = _compute_e1(messages, "claudecode")
        assert result == 1.0

    def test_many_edits_same_file(self):
        messages = [{"role": "user", "content": "fix"}]
        for i in range(6):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Edit", "arguments": f'{{"file_path":"/testbed/utils.py","old_string":"a{i}","new_string":"b{i}"}}'}}],
            })
            messages.append({"role": "tool", "content": "ok"})
        messages.append({"role": "assistant", "content": "Done."})
        result = _compute_e1(messages, "claudecode")
        assert result < 0.5


class TestComputeE2:
    def test_no_deletes(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Edit", "arguments": '{"file_path":"/testbed/f.py"}'}}],
            },
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Done."},
        ]
        result = _compute_e2(messages, "claudecode")
        assert result == 1.0
