from __future__ import annotations

import json

import pytest

from swe_data_process.rule_score import (
    _compute_fec,
    _compute_stp,
    _compute_sub,
    _compute_tvr,
    _count_tool_call_errors,
    _extract_observations,
    _is_error_result,
    compute_tool_call_error_rate,
    detect_scaffold,
    score_dataset,
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


@pytest.fixture
def cc_record():
    return {
        "messages": [
            {"role": "system", "content": "You are Claude Code, by Anthropic."},
            {"role": "user", "content": "Fix the bug in utils.py"},
            {
                "role": "assistant",
                "content": "I will read the file first.",
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
                "content": "I will edit utils.py now.",
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
            {"type": "function", "function": {"name": "Bash"}},
            {"type": "function", "function": {"name": "Read"}},
            {"type": "function", "function": {"name": "Edit"}},
            {"type": "function", "function": {"name": "Write"}},
            {"type": "function", "function": {"name": "Grep"}},
        ],
    }


class TestScoreRecord:
    def test_returns_tqs_v2_keys(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        expected_keys = {
            "scaffold",
            "assistant_turns",
            "total_tool_calls",
            "oec_score",
            "iac_score",
            "dpi_score",
            "ped_score",
            "psn_score",
            "tte_score",
            "scp_score",
            "sub_score",
            "fec_score",
            "stp_score",
            "tvr_score",
            "composite_score",
            "reproduce_first",  # Diagnostic only; excluded from the composite score.
        }
        assert set(result.keys()) == expected_keys

    def test_scores_in_valid_range_or_none(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        for key, value in result.items():
            if key in ("scaffold", "assistant_turns", "total_tool_calls"):
                continue
            if value is None:
                continue
            assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"

    def test_scaffold_detected(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        assert result["scaffold"] == "claudecode"

    def test_scaffold_override(self, cc_record):
        result = score_record(cc_record, median_steps=5.0, scaffold_override="opencode")
        assert result["scaffold"] == "opencode"

    def test_old_positional_scaffold_override_still_works(self, cc_record):
        result = score_record(cc_record, "opencode")
        assert result["scaffold"] == "opencode"

    def test_basic_completion_and_edit_scores(self, cc_record):
        result = score_record(cc_record, median_steps=5.0)
        assert result["sub_score"] == 1.0
        assert result["fec_score"] == 1.0


class TestScoreDataset:
    def test_subagent_records_are_skipped(self, cc_record):
        subagent = {**cc_record, "_agent_type": "subagent"}
        scored = score_dataset([cc_record, subagent], quiet=True)
        assert isinstance(scored[0]["_score"], dict)
        assert scored[1]["_score"] is None


class TestTqsV2Components:
    def test_stp_optimal_range_gets_full_score(self):
        messages = [{"role": "assistant", "content": "step"} for _ in range(10)]
        assert _compute_stp(messages) == 1.0

    def test_stp_short_trajectory_is_scaled(self):
        messages = [{"role": "assistant", "content": "step"} for _ in range(3)]
        assert _compute_stp(messages) == pytest.approx(0.6)

    def test_fec_single_file_single_edit(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Edit", "arguments": '{"file_path":"/testbed/utils.py"}'}}]},
        ]
        assert _compute_fec(messages, "claudecode") == 1.0

    def test_fec_many_edits_same_file_is_penalized(self):
        messages = []
        for i in range(6):
            arguments = json.dumps({
                "file_path": "/testbed/utils.py",
                "old_string": str(i),
                "new_string": str(i + 1),
            })
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Edit", "arguments": arguments}}],
            })
        assert _compute_fec(messages, "claudecode") < 0.5

    def test_sub_complete_cc_trajectory(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "Done."},
        ]
        assert _compute_sub(messages, [], "claudecode") == 1.0

    def test_sub_truncated_tool_call(self):
        messages = [
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read"}}]},
        ]
        assert _compute_sub(messages, [], "claudecode") == 0.0

    def test_tvr_has_test_write_and_successful_run(self):
        messages = [
            {"role": "user", "content": "fix"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "Edit", "arguments": '{"file_path":"tests/test_foo.py","old_string":"","new_string":"def test_x(): pass"}'}},
                ],
            },
            {"role": "tool", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "Bash", "arguments": '{"command":"pytest tests/test_foo.py"}'}},
                ],
            },
            {"role": "tool", "content": "1 passed"},
        ]
        assert _compute_tvr(messages, "claudecode") == 1.0

    def test_tvr_no_test_activity(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Edit", "arguments": '{"file_path":"utils.py"}'}}]},
            {"role": "tool", "content": "ok"},
        ]
        assert _compute_tvr(messages, "claudecode") == 0.0


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
        record = self._t2_record(["ls", "cat foo", "pwd"], "file listing\n")
        total, errors = _count_tool_call_errors(record, "terminus2")
        assert total == 3
        assert errors == 0

    def test_terminus2_error_counts_one_failed_command(self):
        record = self._t2_record(["ls", "bad-cmd"], "bash: bad-cmd: command not found")
        total, errors = _count_tool_call_errors(record, "terminus2")
        assert total == 2
        assert errors == 1

    def test_test_run_failed_output_not_counted(self):
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
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
        }
        total, errors = _count_tool_call_errors(record, "claudecode")
        assert (total, errors) == (1, 1)

    def test_per_scaffold_bucketing_and_aggregate(self):
        cc = self._cc_record(["ok", "Permission denied"])
        t2 = self._t2_record(["ls", "pwd"], "ok output")
        stats = compute_tool_call_error_rate([cc, t2])

        assert stats["total_tool_calls"] == 4
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
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Edit"}},
            ],
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
