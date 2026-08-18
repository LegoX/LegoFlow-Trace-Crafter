from __future__ import annotations

import json

import pytest

from legoflow_trace_crafter.terminus2.common import (
    convert_one_record,
    extract_json_content_from_assistant,
    extract_observation_text,
    get_steps,
    iter_records,
    normalize_command,
    split_analysis_plan,
)


class TestSplitAnalysisPlan:
    def test_both_fields(self):
        msg = "Analysis: The test is failing.\nPlan: Fix the assertion."
        result = split_analysis_plan(msg)
        assert result["analysis"] == "The test is failing."
        assert result["plan"] == "Fix the assertion."

    def test_only_analysis(self):
        msg = "Analysis: Something is wrong."
        result = split_analysis_plan(msg)
        assert result["analysis"] == "Something is wrong."
        assert result["plan"] == ""

    def test_only_plan(self):
        msg = "Plan: Do the thing."
        result = split_analysis_plan(msg)
        assert result["analysis"] == ""
        assert result["plan"] == "Do the thing."

    def test_empty_string(self):
        result = split_analysis_plan("")
        assert result == {"analysis": "", "plan": ""}

    def test_none(self):
        result = split_analysis_plan(None)
        assert result == {"analysis": "", "plan": ""}

    def test_no_prefix(self):
        msg = "Just some text without prefixes."
        result = split_analysis_plan(msg)
        assert result["analysis"] == "Just some text without prefixes."
        assert result["plan"] == ""


class TestNormalizeCommand:
    def test_valid_keystrokes(self):
        tc = {"function_name": "execute_command", "arguments": {"keystrokes": "ls -la\n"}}
        result = normalize_command(tc)
        assert result == {"keystrokes": "ls -la\n"}

    def test_command_fallback_adds_newline(self):
        tc = {"function_name": "execute_command", "arguments": {"command": "ls"}}
        result = normalize_command(tc)
        assert result == {"keystrokes": "ls\n"}

    def test_command_already_has_newline(self):
        tc = {"function_name": "execute_command", "arguments": {"command": "ls\n"}}
        result = normalize_command(tc)
        assert result == {"keystrokes": "ls\n"}

    def test_mark_task_complete_returns_none(self):
        tc = {"function_name": "mark_task_complete", "arguments": {}}
        assert normalize_command(tc) is None

    def test_no_keystrokes_or_command(self):
        tc = {"function_name": "execute_command", "arguments": {}}
        assert normalize_command(tc) is None

    def test_duration_preserved(self):
        tc = {
            "function_name": "execute_command",
            "arguments": {"keystrokes": "sleep 5\n", "duration": 5000},
        }
        result = normalize_command(tc)
        assert result == {"keystrokes": "sleep 5\n", "duration": 5000}

    def test_non_dict_returns_none(self):
        assert normalize_command("not a dict") is None

    def test_non_dict_arguments_returns_none(self):
        tc = {"function_name": "x", "arguments": "bad"}
        assert normalize_command(tc) is None


class TestExtractObservationText:
    def test_string(self):
        assert extract_observation_text("output text") == "output text"

    def test_dict_with_results(self):
        obs = {"results": [{"content": "result text"}]}
        assert extract_observation_text(obs) == "result text"

    def test_dict_with_results_content_dict(self):
        obs = {"results": [{"content": {"key": "val"}}]}
        result = extract_observation_text(obs)
        assert result == json.dumps({"key": "val"}, ensure_ascii=False)

    def test_dict_with_results_string_item(self):
        obs = {"results": ["string result"]}
        assert extract_observation_text(obs) == "string result"

    def test_list_with_string(self):
        assert extract_observation_text(["first item"]) == "first item"

    def test_list_with_non_string(self):
        obs = [{"key": "val"}]
        assert extract_observation_text(obs) == json.dumps(obs, ensure_ascii=False)

    def test_empty_string(self):
        assert extract_observation_text("") == ""

    def test_none(self):
        assert extract_observation_text(None) == ""

    def test_empty_list(self):
        assert extract_observation_text([]) == ""

    def test_dict_without_results(self):
        obs = {"other": "data"}
        assert extract_observation_text(obs) == json.dumps(obs, ensure_ascii=False)


class TestExtractJsonContentFromAssistant:
    def test_strips_think_tags(self):
        content = "<think>reasoning</think>actual content"
        assert extract_json_content_from_assistant(content) == "actual content"

    def test_no_think_tags(self):
        content = '{"analysis": "test"}'
        assert extract_json_content_from_assistant(content) == '{"analysis": "test"}'

    def test_empty_string(self):
        assert extract_json_content_from_assistant("") == ""

    def test_none(self):
        assert extract_json_content_from_assistant(None) == ""


class TestGetSteps:
    def test_from_trajectory(self):
        record = {"trajectory": {"steps": [{"message": "step1"}]}}
        assert get_steps(record) == [{"message": "step1"}]

    def test_from_steps_directly(self):
        record = {"steps": [{"message": "step1"}]}
        assert get_steps(record) == [{"message": "step1"}]

    def test_non_dict(self):
        assert get_steps("not a dict") == []

    def test_empty_dict(self):
        assert get_steps({}) == []

    def test_trajectory_priority(self):
        record = {
            "trajectory": {"steps": [{"message": "from_trajectory"}]},
            "steps": [{"message": "from_steps"}],
        }
        assert get_steps(record) == [{"message": "from_trajectory"}]


class TestIterRecords:
    def test_list_of_dicts(self):
        obj = [{"steps": []}, {"steps": []}]
        result = list(iter_records(obj))
        assert len(result) == 2

    def test_single_record_with_steps(self):
        obj = {"steps": [{"message": "hi"}]}
        result = list(iter_records(obj))
        assert len(result) == 1
        assert result[0] == obj

    def test_single_record_with_trajectory(self):
        obj = {"trajectory": {"steps": []}}
        result = list(iter_records(obj))
        assert len(result) == 1

    def test_dict_with_data_list(self):
        obj = {"data": [{"steps": []}, {"steps": []}]}
        result = list(iter_records(obj))
        assert len(result) == 2

    def test_list_with_non_dicts_skipped(self):
        obj = [{"steps": []}, "not a dict", 42]
        result = list(iter_records(obj))
        assert len(result) == 1


class TestConvertOneRecord:
    def test_valid_record(self, terminus2_step_record):
        result = convert_one_record(terminus2_step_record)
        assert "messages" in result
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][0]["role"] == "user"

    def test_alternating_roles(self, terminus2_step_record):
        result = convert_one_record(terminus2_step_record)
        for i, msg in enumerate(result["messages"]):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role

    def test_assistant_content_is_valid_json(self, terminus2_step_record):
        result = convert_one_record(terminus2_step_record)
        for i in range(1, len(result["messages"]), 2):
            content = extract_json_content_from_assistant(result["messages"][i]["content"])
            parsed = json.loads(content)
            assert "analysis" in parsed or "plan" in parsed or "commands" in parsed

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="Empty messages"):
            convert_one_record({"steps": []})

    def test_think_mode_with_reasoning(self, terminus2_step_record):
        result = convert_one_record(terminus2_step_record)
        assert result["think_mode"] == "slow"

    def test_think_mode_without_reasoning(self):
        record = {
            "steps": [
                {"source": "user", "message": "Fix it"},
                {
                    "message": "Analysis: Done.\nPlan: Complete.",
                    "tool_calls": [{"function_name": "mark_task_complete", "arguments": {}}],
                    "observation": "",
                },
            ]
        }
        result = convert_one_record(record)
        assert result["think_mode"] == "fast"
