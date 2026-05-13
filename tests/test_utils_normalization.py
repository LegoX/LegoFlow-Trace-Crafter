from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pytest

from swe_data_process.utils import (
    _align_tool_call_ids,
    _extract_reasoning_from_content,
    _normalize_content_to_string,
    _normalize_messages_for_panguml,
    _normalize_tool_arguments,
    _normalize_tool_call,
    _normalize_tools_for_panguml,
    to_panguml_v2_record,
)


class TestNormalizeContentToString:
    def test_string_passthrough(self):
        assert _normalize_content_to_string("hello") == "hello"

    def test_none_returns_empty(self):
        assert _normalize_content_to_string(None) == ""

    def test_list_of_text_blocks(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _normalize_content_to_string(content) == "first\n\nsecond"

    def test_list_with_empty_text(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "third"},
        ]
        assert _normalize_content_to_string(content) == "first\n\nthird"

    def test_list_without_text_key(self):
        content = [{"type": "image", "url": "http://example.com"}]
        result = _normalize_content_to_string(content)
        assert result == json.dumps(content, ensure_ascii=False)

    def test_dict_serialized(self):
        content = {"key": "value"}
        assert _normalize_content_to_string(content) == json.dumps(content, ensure_ascii=False)

    def test_integer_converted(self):
        assert _normalize_content_to_string(42) == "42"


class TestExtractReasoningFromContent:
    def test_with_think_tags(self):
        content = "<think>reasoning here</think>actual content"
        reasoning, remainder = _extract_reasoning_from_content(content)
        assert reasoning == "reasoning here"
        assert remainder == "actual content"

    def test_without_think_tags(self):
        content = "just normal content"
        reasoning, remainder = _extract_reasoning_from_content(content)
        assert reasoning == ""
        assert remainder == "just normal content"

    def test_empty_string(self):
        reasoning, remainder = _extract_reasoning_from_content("")
        assert reasoning == ""
        assert remainder == ""

    def test_think_tag_with_whitespace(self):
        content = "  <think> reasoning </think> content  "
        reasoning, remainder = _extract_reasoning_from_content(content)
        assert reasoning == "reasoning"
        assert remainder == "content"

    def test_no_closing_tag(self):
        content = "<think>no closing tag"
        reasoning, remainder = _extract_reasoning_from_content(content)
        assert reasoning == ""
        assert remainder == content


class TestNormalizeToolArguments:
    def test_json_string(self):
        result = _normalize_tool_arguments('{"path": "/tmp/file.py"}')
        assert result == '{"path":"/tmp/file.py"}'

    def test_dict(self):
        result = _normalize_tool_arguments({"path": "/tmp/file.py"})
        assert result == '{"path":"/tmp/file.py"}'

    def test_none(self):
        result = _normalize_tool_arguments(None)
        assert result == "{}"

    def test_invalid_json_string(self):
        result = _normalize_tool_arguments("not json")
        assert result == '"not json"'

    def test_empty_dict(self):
        result = _normalize_tool_arguments({})
        assert result == "{}"


class TestNormalizeToolCall:
    def test_valid_tool_call(self):
        tc = {
            "type": "function",
            "function": {"name": "Read", "arguments": '{"file_path":"/tmp/f.py"}'},
            "id": "call_001",
        }
        result = _normalize_tool_call(tc)
        assert result["type"] == "function"
        assert result["function"]["name"] == "Read"
        assert result["id"] == "call_001"

    def test_missing_function(self):
        tc = {"type": "function", "id": "call_001"}
        result = _normalize_tool_call(tc)
        assert result["function"]["name"] == ""

    def test_missing_id(self):
        tc = {"type": "function", "function": {"name": "Read", "arguments": "{}"}}
        result = _normalize_tool_call(tc)
        assert "id" not in result

    def test_non_dict_returns_none(self):
        assert _normalize_tool_call("not a dict") is None
        assert _normalize_tool_call(None) is None

    def test_empty_id_excluded(self):
        tc = {"type": "function", "function": {"name": "Read"}, "id": ""}
        result = _normalize_tool_call(tc)
        assert "id" not in result


class TestAlignToolCallIds:
    def test_assigns_ids_from_following_tools(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read"}},
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "tool_123"},
        ]
        _align_tool_call_ids(messages)
        assert messages[0]["tool_calls"][0]["id"] == "tool_123"

    def test_generates_ids_when_no_tool_messages(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read"}},
                ],
            },
            {"role": "tool", "content": "result"},
        ]
        _align_tool_call_ids(messages)
        assert messages[0]["tool_calls"][0]["id"] == "call_000001"
        assert messages[1]["tool_call_id"] == "call_000001"

    def test_preserves_existing_ids(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read"}, "id": "existing"},
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "existing"},
        ]
        _align_tool_call_ids(messages)
        assert messages[0]["tool_calls"][0]["id"] == "existing"

    def test_multiple_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read"}},
                    {"type": "function", "function": {"name": "Grep"}},
                ],
            },
            {"role": "tool", "content": "r1", "tool_call_id": "id_a"},
            {"role": "tool", "content": "r2", "tool_call_id": "id_b"},
        ]
        _align_tool_call_ids(messages)
        assert messages[0]["tool_calls"][0]["id"] == "id_a"
        assert messages[0]["tool_calls"][1]["id"] == "id_b"


class TestNormalizeMessagesForPanguml:
    def test_inserts_system_if_missing(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_panguml(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == ""

    def test_preserves_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_panguml(messages)
        assert result[0]["content"] == "You are helpful."

    def test_assistant_reasoning_extracted(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think>answer"},
        ]
        result = _normalize_messages_for_panguml(messages)
        assistant = [m for m in result if m["role"] == "assistant"][0]
        assert assistant["reasoning_content"] == "reasoning"
        assert assistant["content"] == "answer"

    def test_assistant_explicit_reasoning_preserved(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "explicit reasoning",
            },
        ]
        result = _normalize_messages_for_panguml(messages)
        assistant = [m for m in result if m["role"] == "assistant"][0]
        assert assistant["reasoning_content"] == "explicit reasoning"

    def test_tool_call_id_preserved(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "Read", "arguments": "{}"}, "id": "c1"}
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
        result = _normalize_messages_for_panguml(messages)
        tool_msg = [m for m in result if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "c1"

    def test_skips_non_dict_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            "not a dict",
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_messages_for_panguml(messages)
        roles = [m["role"] for m in result]
        assert "not a dict" not in roles


class TestNormalizeToolsForPanguml:
    def test_valid_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = _normalize_tools_for_panguml(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "Read"

    def test_input_schema_fallback(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read",
                    "input_schema": {"type": "object", "properties": {}},
                },
            }
        ]
        result = _normalize_tools_for_panguml(tools)
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_non_list_returns_empty(self):
        assert _normalize_tools_for_panguml(None) == []
        assert _normalize_tools_for_panguml("bad") == []

    def test_tool_without_function_wrapper(self):
        tools = [{"name": "Read", "description": "Read", "parameters": {}}]
        result = _normalize_tools_for_panguml(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "Read"


class TestToPangumlV2Record:
    def test_structure(self, minimal_im_record):
        result = to_panguml_v2_record(minimal_im_record)
        assert result["version"] == "2.0.0"
        assert "meta_info" in result
        assert "tools" in result
        assert "messages" in result

    def test_meta_info_fields(self, minimal_im_record):
        result = to_panguml_v2_record(minimal_im_record)
        meta = result["meta_info"]
        assert "teacher" in meta
        assert "query_source" in meta
        assert "response_generate_time" in meta
        assert "language" in meta
        assert "category" in meta
        assert "rounds" in meta
        assert "unique_info" in meta

    def test_messages_normalized(self, minimal_im_record):
        result = to_panguml_v2_record(minimal_im_record)
        assert result["messages"][0]["role"] == "system"
        for msg in result["messages"]:
            if msg["role"] == "assistant":
                assert "reasoning_content" in msg

    def test_tools_normalized(self, minimal_im_record):
        result = to_panguml_v2_record(minimal_im_record)
        for tool in result["tools"]:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_instance_id_in_unique_info(self):
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "_instance_id": "owner__repo-123",
            "_agent_type": "main",
        }
        result = to_panguml_v2_record(record)
        unique_info = result["meta_info"]["unique_info"]
        assert unique_info["_instance_id"] == "owner__repo-123"
        assert unique_info["_agent_type"] == "main"
