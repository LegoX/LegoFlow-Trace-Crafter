"""Shared helpers for OpenHands trajectory conversion."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_text(content: Any) -> str:
    """Extract plain text from a message content field (str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return first.get('text', '')
    return ''


def process_tool_call(tool_calls: Any) -> list[dict[str, Any]]:
    """Normalize tool_calls: keep function arguments as JSON strings.

    Returns a new list — does *not* mutate the input.
    """
    if not isinstance(tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        function = tool_call.get('function')
        if not isinstance(function, dict):
            function = {}

        arguments = function.get('arguments')
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = arguments
        elif arguments is None:
            parsed_arguments = {}
        else:
            parsed_arguments = arguments

        tc = {
            'type': 'function',
            'function': {
                'name': function.get('name', ''),
                'arguments': json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }
        call_id = tool_call.get('id')
        if isinstance(call_id, str) and call_id:
            tc['id'] = call_id
        normalized.append(tc)
    return normalized


def list_sorted_json_files(folder_path: Path) -> list[str]:
    """List JSON filenames in *folder_path* sorted by embedded timestamp."""
    if not folder_path.exists():
        raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"路径不是文件夹: {folder_path}")

    files = [f.name for f in folder_path.iterdir() if f.suffix == '.json']

    def _ts(file_name: str) -> float:
        m = re.search(r"(\d+\.\d+)(?:-[0-9a-f]+)?\.json$", file_name)
        return float(m.group(1)) if m else float('-inf')

    files.sort(key=_ts)
    return files




def add_reasoning_content_to_json_data(
    messages: list[dict[str, Any]],
    reasoning_contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-turn reasoning_content into the corresponding assistant messages."""
    assistant_indices = [i for i, msg in enumerate(messages) if msg.get('role') == 'assistant']

    for d in reasoning_contents:
        turn_idx = d.get('turn_idx')
        if turn_idx is None:
            continue

        if 0 <= turn_idx < len(assistant_indices):
            msg_idx = assistant_indices[turn_idx]
            messages[msg_idx]['reasoning_content'] = d.get('reasoning_content')
    return messages
