from __future__ import annotations

import json
from pathlib import Path

import pytest

from legoflow_trace_crafter.openhands.common import (
    add_reasoning_content_to_json_data,
    extract_text,
    list_sorted_json_files,
    process_tool_call,
)


class TestExtractText:
    def test_string_passthrough(self):
        assert extract_text("hello") == "hello"

    def test_list_of_blocks(self):
        content = [{"type": "text", "text": "extracted"}]
        assert extract_text(content) == "extracted"

    def test_empty_list(self):
        assert extract_text([]) == ""

    def test_none(self):
        assert extract_text(None) == ""

    def test_integer(self):
        assert extract_text(42) == ""

    def test_list_with_non_dict(self):
        assert extract_text(["string"]) == ""

    def test_list_missing_text_key(self):
        content = [{"type": "image"}]
        assert extract_text(content) == ""


class TestProcessToolCall:
    def test_normalizes_dict_arguments(self):
        tool_calls = [
            {
                "type": "function",
                "function": {"name": "terminal", "arguments": {"command": "ls"}},
                "id": "call_1",
            }
        ]
        result = process_tool_call(tool_calls)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"
        assert result[0]["function"]["arguments"] == '{"command":"ls"}'
        assert result[0]["id"] == "call_1"

    def test_normalizes_string_arguments(self):
        tool_calls = [
            {
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
            }
        ]
        result = process_tool_call(tool_calls)
        assert result[0]["function"]["arguments"] == '{"command":"ls"}'

    def test_invalid_json_string_arguments(self):
        tool_calls = [
            {"type": "function", "function": {"name": "terminal", "arguments": "not json"}}
        ]
        result = process_tool_call(tool_calls)
        assert result[0]["function"]["arguments"] == '"not json"'

    def test_none_arguments(self):
        tool_calls = [
            {"type": "function", "function": {"name": "terminal", "arguments": None}}
        ]
        result = process_tool_call(tool_calls)
        assert result[0]["function"]["arguments"] == "{}"

    def test_missing_id(self):
        tool_calls = [
            {"type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ]
        result = process_tool_call(tool_calls)
        assert "id" not in result[0]

    def test_non_list_returns_empty(self):
        assert process_tool_call(None) == []
        assert process_tool_call("bad") == []

    def test_non_dict_items_skipped(self):
        tool_calls = ["not a dict", {"function": {"name": "x", "arguments": "{}"}}]
        result = process_tool_call(tool_calls)
        assert len(result) == 1

    def test_missing_function_key(self):
        tool_calls = [{"type": "function"}]
        result = process_tool_call(tool_calls)
        assert result[0]["function"]["name"] == ""


class TestListSortedJsonFiles:
    def test_sorts_by_timestamp(self, tmp_path):
        (tmp_path / "1000.5.json").write_text("{}")
        (tmp_path / "999.1.json").write_text("{}")
        (tmp_path / "1001.0.json").write_text("{}")
        result = list_sorted_json_files(tmp_path)
        assert result == ["999.1.json", "1000.5.json", "1001.0.json"]

    def test_handles_hash_suffix(self, tmp_path):
        (tmp_path / "1000.5-abc123.json").write_text("{}")
        (tmp_path / "999.1-def456.json").write_text("{}")
        result = list_sorted_json_files(tmp_path)
        assert result == ["999.1-def456.json", "1000.5-abc123.json"]

    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_sorted_json_files(tmp_path / "nonexistent")

    def test_raises_on_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(NotADirectoryError):
            list_sorted_json_files(f)

    def test_empty_dir(self, tmp_path):
        assert list_sorted_json_files(tmp_path) == []

    def test_ignores_non_json(self, tmp_path):
        (tmp_path / "1000.5.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("hi")
        result = list_sorted_json_files(tmp_path)
        assert result == ["1000.5.json"]


class TestAddReasoningContentToJsonData:
    def test_merges_reasoning(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "bye"},
        ]
        reasoning_contents = [
            {"turn_idx": 0, "reasoning_content": "thinking about hello"},
            {"turn_idx": 1, "reasoning_content": "thinking about bye"},
        ]
        result = add_reasoning_content_to_json_data(messages, reasoning_contents)
        assert result[1]["reasoning_content"] == "thinking about hello"
        assert result[3]["reasoning_content"] == "thinking about bye"

    def test_out_of_range_ignored(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        reasoning_contents = [
            {"turn_idx": 5, "reasoning_content": "out of range"},
        ]
        result = add_reasoning_content_to_json_data(messages, reasoning_contents)
        assert "reasoning_content" not in result[1]

    def test_empty_reasoning_list(self):
        messages = [
            {"role": "assistant", "content": "hello"},
        ]
        result = add_reasoning_content_to_json_data(messages, [])
        assert "reasoning_content" not in result[0]

    def test_none_turn_idx_skipped(self):
        messages = [
            {"role": "assistant", "content": "hello"},
        ]
        reasoning_contents = [{"reasoning_content": "no idx"}]
        result = add_reasoning_content_to_json_data(messages, reasoning_contents)
        assert "reasoning_content" not in result[0]
