from __future__ import annotations

import pytest

from swe_data_process.utils import (
    detect_agent_type,
    get_record_think_mode,
    infer_language_code,
    infer_think_mode_from_messages,
    is_im_record,
)


class TestIsImRecord:
    def test_valid_record(self, minimal_im_record):
        assert is_im_record(minimal_im_record) is True

    def test_non_dict(self):
        assert is_im_record("not a dict") is False
        assert is_im_record(42) is False
        assert is_im_record(None) is False

    def test_missing_messages(self):
        assert is_im_record({"tools": []}) is False

    def test_empty_messages(self):
        assert is_im_record({"messages": []}) is False

    def test_messages_not_list(self):
        assert is_im_record({"messages": "not a list"}) is False

    def test_messages_without_role(self):
        assert is_im_record({"messages": [{"content": "hi"}]}) is False

    def test_messages_with_non_string_role(self):
        assert is_im_record({"messages": [{"role": 123}]}) is False

    def test_minimal_valid(self):
        assert is_im_record({"messages": [{"role": "user", "content": "hi"}]}) is True


class TestInferThinkMode:
    def test_slow_with_reasoning(self):
        messages = [
            {"role": "assistant", "content": "", "reasoning_content": "thinking"},
        ]
        assert infer_think_mode_from_messages(messages) == "slow"

    def test_fast_without_reasoning(self):
        messages = [
            {"role": "assistant", "content": "hi"},
        ]
        assert infer_think_mode_from_messages(messages) == "fast"

    def test_fast_with_empty_reasoning(self):
        messages = [
            {"role": "assistant", "content": "hi", "reasoning_content": ""},
        ]
        assert infer_think_mode_from_messages(messages) == "fast"

    def test_non_assistant_reasoning_ignored(self):
        messages = [
            {"role": "user", "content": "hi", "reasoning_content": "user thinking"},
        ]
        assert infer_think_mode_from_messages(messages) == "fast"

    def test_empty_messages(self):
        assert infer_think_mode_from_messages([]) == "fast"


class TestGetRecordThinkMode:
    def test_explicit_think_mode(self):
        record = {"messages": [], "think_mode": "slow"}
        assert get_record_think_mode(record) == "slow"

    def test_from_meta_info_unique_info(self):
        record = {
            "messages": [],
            "meta_info": {"unique_info": {"think_mode": "slow"}},
        }
        assert get_record_think_mode(record) == "slow"

    def test_fallback_to_inference(self):
        record = {
            "messages": [
                {"role": "assistant", "content": "", "reasoning_content": "yes"},
            ]
        }
        assert get_record_think_mode(record) == "slow"

    def test_fallback_fast(self):
        record = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert get_record_think_mode(record) == "fast"


class TestDetectAgentType:
    def test_main_with_edit_tool(self, minimal_im_record):
        assert detect_agent_type(minimal_im_record) == "main"

    def test_subagent_read_only(self, subagent_record):
        assert detect_agent_type(subagent_record) == "subagent"

    def test_no_tools_defaults_main(self):
        record = {"messages": [{"role": "user", "content": "hi"}]}
        assert detect_agent_type(record) == "main"

    def test_empty_tools_defaults_main(self):
        record = {"messages": [], "tools": []}
        assert detect_agent_type(record) == "main"

    def test_write_tool_is_main(self):
        record = {
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "Write", "description": "", "parameters": {}}},
            ],
        }
        assert detect_agent_type(record) == "main"


class TestInferLanguageCode:
    def test_english(self):
        messages = [{"role": "user", "content": "Fix the bug in utils.py"}]
        assert infer_language_code(messages) == "en"

    def test_chinese(self):
        messages = [{"role": "user", "content": "修复这个错误"}]
        assert infer_language_code(messages) == "zh"

    def test_empty_messages(self):
        assert infer_language_code([]) == "en"

    def test_no_user_messages(self):
        messages = [{"role": "assistant", "content": "你好"}]
        assert infer_language_code(messages) == "en"

    def test_first_user_with_empty_content(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "hello"},
        ]
        assert infer_language_code(messages) == "en"
