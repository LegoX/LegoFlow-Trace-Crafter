from __future__ import annotations

import json

import pytest

from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import (
    convert_assistant_blocks,
    convert_message,
    convert_record,
    extract_system_message,
    infer_think_mode,
    normalize_tool_content,
    normalize_tool_definition,
    remove_tool_use_error_turns,
    reorder_arguments_by_properties,
    trim_trailing_tools,
)


class TestNormalizeToolContent:
    def test_list_of_blocks(self):
        content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert normalize_tool_content(content) == "hello\n\nworld"

    def test_json_string_with_list(self):
        content = json.dumps([{"type": "text", "text": "parsed"}])
        assert normalize_tool_content(content) == "parsed"

    def test_plain_string(self):
        assert normalize_tool_content("plain text") == "plain text"

    def test_none(self):
        assert normalize_tool_content(None) == ""

    def test_non_json_string(self):
        assert normalize_tool_content("not json {") == "not json {"

    def test_json_string_non_list(self):
        content = json.dumps({"key": "value"})
        assert normalize_tool_content(content) == content

    def test_empty_list(self):
        assert normalize_tool_content([]) == ""

    def test_dict_content(self):
        content = {"key": "value"}
        assert normalize_tool_content(content) == json.dumps(content, ensure_ascii=False)


class TestConvertAssistantBlocks:
    def test_thinking_block(self):
        blocks = [{"type": "thinking", "thinking": "reasoning here"}]
        result = convert_assistant_blocks(blocks)
        assert result["reasoning_content"] == "reasoning here"
        assert result["content"] == ""

    def test_text_block(self):
        blocks = [{"type": "text", "text": "answer"}]
        result = convert_assistant_blocks(blocks)
        assert result["content"] == "answer"

    def test_tool_use_block(self):
        blocks = [
            {
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/tmp/f.py"},
                "id": "call_1",
            }
        ]
        result = convert_assistant_blocks(blocks)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "Read"
        assert result["tool_calls"][0]["id"] == "call_1"

    def test_mixed_blocks(self):
        blocks = [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "here's my answer"},
            {"type": "tool_use", "name": "Grep", "input": {"pattern": "foo"}, "id": "c1"},
        ]
        result = convert_assistant_blocks(blocks)
        assert result["reasoning_content"] == "let me think"
        assert result["content"] == "here's my answer"
        assert len(result["tool_calls"]) == 1

    def test_all_empty_returns_none(self):
        blocks = [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": ""},
        ]
        assert convert_assistant_blocks(blocks) is None

    def test_empty_blocks(self):
        assert convert_assistant_blocks([]) is None


class TestRemoveToolUseErrorTurns:
    def test_removes_error_and_preceding_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "<tool_use_error>invalid args"},
            {"role": "assistant", "content": "retry"},
        ]
        result = remove_tool_use_error_turns(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "retry"

    def test_removes_invalid_arguments_error(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "The arguments provided to the tool are invalid: ..."},
            {"role": "assistant", "content": "fixed"},
        ]
        result = remove_tool_use_error_turns(messages)
        assert len(result) == 2
        assert result[-1]["content"] == "fixed"

    def test_preserves_valid_turns(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "success result"},
            {"role": "assistant", "content": "done"},
        ]
        result = remove_tool_use_error_turns(messages)
        assert len(result) == 4

    def test_consecutive_errors(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "<tool_use_error>err1"},
            {"role": "assistant", "content": "a2"},
            {"role": "tool", "content": "<tool_use_error>err2"},
            {"role": "assistant", "content": "final"},
        ]
        result = remove_tool_use_error_turns(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "final"

    def test_skips_following_tool_messages_in_error_block(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "<tool_use_error>err"},
            {"role": "tool", "content": "also part of error block"},
            {"role": "assistant", "content": "recovered"},
        ]
        result = remove_tool_use_error_turns(messages)
        assert len(result) == 2
        assert result[-1]["content"] == "recovered"


class TestTrimTrailingTools:
    def test_removes_trailing_tool(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "result"},
        ]
        result = trim_trailing_tools(messages)
        assert len(result) == 2
        assert result[-1]["role"] == "assistant"

    def test_no_op_when_last_is_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "done"},
        ]
        result = trim_trailing_tools(messages)
        assert len(result) == 2

    def test_removes_multiple_trailing_tools(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
        ]
        result = trim_trailing_tools(messages)
        assert len(result) == 2

    def test_empty_messages(self):
        assert trim_trailing_tools([]) == []


class TestNormalizeToolDefinition:
    def test_with_function_wrapper(self):
        tool = {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
        result = normalize_tool_definition(tool)
        assert result["function"]["name"] == "Read"
        assert result["type"] == "function"

    def test_without_function_wrapper(self):
        tool = {"name": "Read", "description": "Read a file", "parameters": {"type": "object"}}
        result = normalize_tool_definition(tool)
        assert result["function"]["name"] == "Read"

    def test_input_schema_fallback(self):
        tool = {
            "function": {
                "name": "Read",
                "description": "Read",
                "input_schema": {"type": "object", "properties": {}},
            }
        }
        result = normalize_tool_definition(tool)
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}


class TestReorderArgumentsByProperties:
    def test_reorders_keys(self):
        arguments = {"b": 2, "a": 1, "c": 3}
        order = ["a", "b", "c"]
        result = reorder_arguments_by_properties(arguments, order)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_extra_keys_appended(self):
        arguments = {"b": 2, "a": 1, "extra": 99}
        order = ["a", "b"]
        result = reorder_arguments_by_properties(arguments, order)
        assert list(result.keys()) == ["a", "b", "extra"]

    def test_non_dict_passthrough(self):
        assert reorder_arguments_by_properties("string", ["a"]) == "string"
        assert reorder_arguments_by_properties(42, ["a"]) == 42

    def test_none_order_passthrough(self):
        arguments = {"b": 2, "a": 1}
        assert reorder_arguments_by_properties(arguments, None) == arguments

    def test_empty_order_passthrough(self):
        arguments = {"b": 2, "a": 1}
        assert reorder_arguments_by_properties(arguments, []) == arguments


class TestExtractSystemMessage:
    def test_extracts_from_blocks(self):
        request_body = {"system": [{"type": "text", "text": "You are helpful."}]}
        result = extract_system_message(request_body)
        assert result == {"role": "system", "content": "You are helpful."}

    def test_returns_none_for_non_list(self):
        assert extract_system_message({"system": "string"}) is None

    def test_returns_none_for_empty_text(self):
        assert extract_system_message({"system": [{"type": "text", "text": ""}]}) is None

    def test_joins_multiple_blocks(self):
        request_body = {
            "system": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]
        }
        result = extract_system_message(request_body)
        assert result["content"] == "Part 1\n\nPart 2"


class TestInferThinkMode:
    def test_slow_with_reasoning(self):
        messages = [{"role": "assistant", "content": "", "reasoning_content": "thinking"}]
        assert infer_think_mode(messages) == "slow"

    def test_fast_without_reasoning(self):
        messages = [{"role": "assistant", "content": "hi"}]
        assert infer_think_mode(messages) == "fast"


class TestConvertMessage:
    def test_assistant_with_string_content(self):
        msg = {"role": "assistant", "content": "hello", "reasoning_content": "thought"}
        result = convert_message(msg)
        assert len(result) == 1
        assert result[0]["content"] == "hello"
        assert result[0]["reasoning_content"] == "thought"

    def test_assistant_with_list_content(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "thinking", "thinking": "reasoning"},
            ],
        }
        result = convert_message(msg)
        assert len(result) == 1
        assert result[0]["content"] == "answer"
        assert result[0]["reasoning_content"] == "reasoning"

    def test_user_with_string(self):
        msg = {"role": "user", "content": "hello"}
        result = convert_message(msg)
        assert result == [{"role": "user", "content": "hello"}]

    def test_user_empty_string(self):
        msg = {"role": "user", "content": ""}
        assert convert_message(msg) == []

    def test_tool_message(self):
        msg = {"role": "tool", "content": "result", "tool_call_id": "c1"}
        result = convert_message(msg)
        assert result == [{"role": "tool", "content": "result", "tool_call_id": "c1"}]

    def test_system_with_list(self):
        msg = {"role": "system", "content": [{"type": "text", "text": "sys prompt"}]}
        result = convert_message(msg)
        assert result == [{"role": "system", "content": "sys prompt"}]


class TestConvertRecord:
    def test_basic_record(self):
        record = {
            "request_body": {
                "system": [{"type": "text", "text": "You are helpful."}],
                "messages": [
                    {"role": "user", "content": "hi"},
                ],
                "tools": [],
            },
            "response_body": {
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}}
                ]
            },
        }
        result = convert_record(record)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "hi"}
        assert result["messages"][2]["role"] == "assistant"
        assert result["messages"][2]["content"] == "hello"
        assert result["think_mode"] == "fast"

    def test_with_tool_calls(self):
        record = {
            "request_body": {
                "messages": [
                    {"role": "user", "content": "fix bug"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "let me check"},
                            {"type": "tool_use", "name": "Read", "input": {"file_path": "/f.py"}, "id": "c1"},
                        ],
                    },
                    {"role": "user", "content": [{"type": "tool_result", "content": "file content", "tool_use_id": "c1"}]},
                ],
                "tools": [
                    {"type": "function", "function": {"name": "Read", "description": "Read", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}
                ],
            },
            "response_body": {
                "content": [{"type": "text", "text": "Fixed!"}]
            },
        }
        result = convert_record(record)
        assert result["think_mode"] == "slow"
        assert any(m.get("tool_calls") for m in result["messages"] if m["role"] == "assistant")
