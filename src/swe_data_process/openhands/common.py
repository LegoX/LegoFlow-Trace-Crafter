"""Shared helpers for OpenHands trajectory conversion."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from swe_data_process.utils import (
    check_reasoning_content,
    check_roles,
    get_resolved_instances,
    instance_id_matches_excluded_repos,
    load_json,
)


# ---------------------------------------------------------------------------
# OpenHands SDK tools — complete OpenAI function-calling format
# ---------------------------------------------------------------------------
# chaofan 原始 SDK 轨迹中的 tools 缺少 parameters schema，
# 因此在此处硬编码完整定义，供所有 ohsdk 转换脚本共享。

OPENHANDS_SDK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": (
                "Execute a bash command in the terminal within a persistent shell session.\n\n\n"
                "### Command Execution\n"
                "* One command at a time: You can only execute one bash command at a time. "
                "If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.\n"
                "* Persistent session: Commands execute in a persistent shell session where "
                "environment variables, virtual environments, and working directory persist between commands.\n"
                "* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, "
                "you have the option to continue or interrupt the command (see section below for details)\n"
                "* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts "
                "or commands in this environment. The runtime may not support them and can cause unusable "
                "shell sessions. If you want to run multi-line bash commands, write the commands to a file "
                "and then run it, instead.\n\n"
                "### Long-running Commands\n"
                "* For commands that may run indefinitely, run them in the background and redirect output "
                "to a file, e.g. `python3 app.py > server.log 2>&1 &`.\n"
                "* For commands that may run for a long time (e.g. installation or testing commands), "
                "or commands that run for a fixed amount of time (e.g. sleep), you should set the "
                "\"timeout\" parameter of your function call to an appropriate value.\n"
                "* If a bash command returns exit code `-1`, this means the process hit the soft timeout "
                "and is not yet finished. By setting `is_input` to `true`, you can:\n"
                "  - Send empty `command` to retrieve additional logs\n"
                "  - Send text (set `command` to the text) to STDIN of the running process\n"
                "  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) "
                "to interrupt the process\n"
                "  - If you do C-c, you can re-start the process with a longer \"timeout\" parameter "
                "to let it run to completion\n\n"
                "### Best Practices\n"
                "* Directory verification: Before creating new directories or files, first verify "
                "the parent directory exists and is the correct location.\n"
                "* Directory management: Try to maintain working directory by using absolute paths "
                "and avoiding excessive use of `cd`.\n\n"
                "### Output Handling\n"
                "* Output truncation: If the output exceeds a maximum length, it will be truncated "
                "before being returned.\n\n"
                "### Terminal Reset\n"
                "* Terminal reset: If the terminal becomes unresponsive, you can set the \"reset\" "
                "parameter to `true` to create a new terminal session. This will terminate the current "
                "session and start fresh.\n"
                "* Warning: Resetting the terminal will lose all previously set environment variables, "
                "working directory changes, and any running processes. Use this only when the terminal "
                "stops responding to commands.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The bash command to execute. Can be empty string to view additional "
                            "logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt "
                            "the currently running process. Note: You can only execute one bash command "
                            "at a time. If you need to run multiple commands sequentially, you can use "
                            "`&&` or `;` to chain them together."
                        ),
                    },
                    "is_input": {
                        "type": "boolean",
                        "description": (
                            "If True, the command is an input to the running process. "
                            "If False, the command is a bash command to be executed in the terminal. "
                            "Default is False."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Optional. Sets a maximum time limit (in seconds) for running the command. "
                            "If the command takes longer than this limit, you'll be asked whether to "
                            "continue or stop it. If you don't set a value, the command will instead "
                            "pause and ask for confirmation when it produces no new output for 30 seconds. "
                            "Use a higher value if the command is expected to take a long time (like "
                            "installation or testing), or if it has a known fixed duration (like sleep)."
                        ),
                    },
                    "reset": {
                        "type": "boolean",
                        "description": (
                            "If True, reset the terminal by creating a new session. Use this only when "
                            "the terminal becomes unresponsive. Note that all previously set environment "
                            "variables and session state will be lost after reset. "
                            "Cannot be used with is_input=True."
                        ),
                    },
                    "security_risk": {
                        "type": "string",
                        "description": (
                            "Security risk levels for actions.\n\n"
                            "Based on OpenHands security risk levels but adapted for agent-sdk.\n"
                            "Integer values allow for easy comparison and ordering."
                        ),
                        "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"],
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A concise summary (approximately 10 words) describing what this specific "
                            "action does. Focus on the key operation and target. "
                            "Example: 'List all Python files in current directory'"
                        ),
                    },
                },
                "required": ["command", "security_risk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_editor",
            "description": (
                "Custom editing tool for viewing, creating and editing files in plain-text format\n"
                "* State is persistent across command calls and discussions with the user\n"
                "* If `path` is a text file, `view` displays the result of applying `cat -n`. "
                "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
                "* The `create` command cannot be used if the specified `path` already exists as a file\n"
                "* If a `command` generates a long output, it will be truncated and marked with "
                "`<response clipped>`\n"
                "* The `undo_edit` command will revert the last edit made to the file at `path`\n"
                "* This tool can be used for creating and editing files in plain-text format.\n\n\n"
                "Before using this tool:\n"
                "1. Use the view tool to understand the file's contents and context\n"
                "2. Verify the directory path is correct (only applicable when creating new files):\n"
                "   - Use the view tool to verify the parent directory exists and is the correct location\n\n"
                "When making edits:\n"
                "   - Ensure the edit results in idiomatic, correct code\n"
                "   - Do not leave the code in a broken state\n"
                "   - Always use absolute file paths (starting with /)\n\n"
                "CRITICAL REQUIREMENTS FOR USING THIS TOOL:\n\n"
                "1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive "
                "lines from the file, including all whitespace and indentation. The tool will fail if "
                "`old_str` matches multiple locations or doesn't match exactly with the file content.\n\n"
                "2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:\n"
                "   - Include sufficient context before and after the change point (3-5 lines recommended)\n"
                "   - If not unique, the replacement will not be performed\n\n"
                "3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace "
                "the `old_str`. Both strings must be different.\n\n"
                "Remember: when making multiple file edits in a row to the same file, you should prefer "
                "to send all edits in a single message with multiple calls to this tool, rather than "
                "multiple messages with a single call each.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The commands to run. Allowed options are: "
                            "`view`, `create`, `str_replace`, `insert`, `undo_edit`."
                        ),
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": (
                            "Required parameter of `create` command, with the content "
                            "of the file to be created."
                        ),
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "Required parameter of `str_replace` command containing "
                            "the string in `path` to replace."
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": (
                            "Optional parameter of `str_replace` command containing the new string "
                            "(if not given, no string will be added). Required parameter of `insert` "
                            "command containing the string to insert."
                        ),
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": (
                            "Required parameter of `insert` command. The `new_str` will be inserted "
                            "AFTER the line `insert_line` of `path`."
                        ),
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional parameter of `view` command when `path` points to a file. "
                            "If none is given, the full file is shown. If provided, the file will be "
                            "shown in the indicated line number range, e.g. [11, 12] will show lines "
                            "11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all "
                            "lines from `start_line` to the end of the file."
                        ),
                    },
                    "security_risk": {
                        "type": "string",
                        "description": (
                            "Security risk levels for actions.\n\n"
                            "Based on OpenHands security risk levels but adapted for agent-sdk.\n"
                            "Integer values allow for easy comparison and ordering."
                        ),
                        "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"],
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A concise summary (approximately 10 words) describing what this specific "
                            "action does. Focus on the key operation and target. "
                            "Example: 'List all Python files in current directory'"
                        ),
                    },
                },
                "required": ["command", "path", "security_risk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_tracker",
            "description": (
                "This tool provides structured task management capabilities for development workflows.\n"
                "It enables systematic tracking of work items, progress monitoring, and efficient\n"
                "organization of complex development activities.\n\n"
                "The tool maintains visibility into project status and helps communicate\n"
                "progress effectively to users."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The command to execute. `view` shows the current task list. "
                            "`plan` creates or updates the task list based on provided requirements "
                            "and progress. Always `view` the current list before making changes."
                        ),
                        "enum": ["view", "plan"],
                    },
                    "task_list": {
                        "type": "array",
                        "description": "The full task list. Required parameter of `plan` command.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "A brief title for the task.",
                                },
                                "notes": {
                                    "type": "string",
                                    "description": "Additional details or notes about the task.",
                                },
                                "status": {
                                    "type": "string",
                                    "description": (
                                        "The current status of the task. "
                                        "One of 'todo', 'in_progress', or 'done'."
                                    ),
                                    "enum": ["todo", "in_progress", "done"],
                                },
                            },
                            "required": ["title"],
                        },
                    },
                    "security_risk": {
                        "type": "string",
                        "description": (
                            "Security risk levels for actions.\n\n"
                            "Based on OpenHands security risk levels but adapted for agent-sdk.\n"
                            "Integer values allow for easy comparison and ordering."
                        ),
                        "enum": ["UNKNOWN", "LOW", "MEDIUM", "HIGH"],
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A concise summary (approximately 10 words) describing what this specific "
                            "action does. Focus on the key operation and target. "
                            "Example: 'List all Python files in current directory'"
                        ),
                    },
                },
                "required": ["security_risk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signals the completion of the current task or conversation.\n\n"
                "Use this tool when:\n"
                "- You have successfully completed the user's requested task\n"
                "- You cannot proceed further due to technical limitations or missing information\n\n"
                "The message should include:\n"
                "- A clear summary of actions taken and their results\n"
                "- Any next steps for the user\n"
                "- Explanation if you're unable to complete the task\n"
                "- Any follow-up questions if more information is needed\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final message to send to the user.",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A concise summary (approximately 10 words) describing what this specific "
                            "action does. Focus on the key operation and target. "
                            "Example: 'List all Python files in current directory'"
                        ),
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Use the tool to think about something. It will not obtain new information or make "
                "any changes to the repository, but just log the thought. Use it when complex reasoning "
                "or brainstorming is needed.\n\n"
                "Common use cases:\n"
                "1. When exploring a repository and discovering the source of a bug, call this tool to "
                "brainstorm several unique ways of fixing the bug, and assess which change(s) are likely "
                "to be simplest and most effective.\n"
                "2. After receiving test results, use this tool to brainstorm ways to fix failing tests.\n"
                "3. When planning a complex refactoring, use this tool to outline different approaches "
                "and their tradeoffs.\n"
                "4. When designing a new feature, use this tool to think through architecture decisions "
                "and implementation details.\n"
                "5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.\n\n"
                "The tool simply logs your thought process for better transparency and does not execute "
                "any code or make changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The thought to log.",
                    },
                    "summary": {
                        "type": "string",
                        "description": (
                            "A concise summary (approximately 10 words) describing what this specific "
                            "action does. Focus on the key operation and target. "
                            "Example: 'List all Python files in current directory'"
                        ),
                    },
                },
                "required": ["thought"],
            },
        },
    },
]


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
    """Normalize tool_calls: parse stringified arguments into dicts.

    Returns a new list — does *not* mutate the input.
    """
    if not isinstance(tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        tc = dict(tool_call)  # shallow copy
        try:
            if 'function' in tc and 'arguments' in tc['function']:
                args = tc['function']['arguments']
                if isinstance(args, str):
                    tc = {**tc, 'function': {**tc['function'], 'arguments': json.loads(args)}}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing tool_call: {e}")
        normalized.append(tc)
    return normalized


# ---------------------------------------------------------------------------
# Shared helpers for chaofan-style completions directories
# ---------------------------------------------------------------------------


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


def find_latest_json_filename(folder_path: Path) -> str | None:
    """Return the filename with the highest timestamp, or None."""
    files = list_sorted_json_files(folder_path)
    return files[-1] if files else None


def extract_reasoning_content(folder_path: Path) -> list[dict[str, Any]]:
    """Extract reasoning_content from every completion JSON in a directory.

    Each file represents one LLM call; its
    ``response.choices[0].message.reasoning_content`` maps to one assistant
    turn.  Files are sorted by timestamp; *turn_idx* counts from 0.
    """
    files = list_sorted_json_files(folder_path)

    reasoning_contents: list[dict[str, Any]] = []
    for turn_idx, file in enumerate(files):
        file_path = folder_path / file
        try:
            json_data = load_json(file_path)
            message = json_data['response']['choices'][0]['message']
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            print(f"在处理文件 {file} 时，读取message失败，跳过。错误: {e}")
            continue

        reasoning_content = message.get('reasoning_content')
        if reasoning_content is not None:
            reasoning_contents.append({
                'turn_idx': turn_idx,
                'reasoning_content': reasoning_content,
            })
    return reasoning_contents


def process_response_of_json_data(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize messages + final response from a completion JSON into a flat list."""
    converted_messages: list[dict[str, Any]] = []
    for msg in json_data['messages']:
        if msg['role'] in ['system', 'user', 'tool']:
            converted_messages.append({
                'role': msg['role'],
                'content': extract_text(msg.get('content'))
            })
        elif msg['role'] == 'assistant':
            converted_messages.append({
                'role': 'assistant',
                'content': extract_text(msg.get('content')),
                'tool_calls': process_tool_call(msg.get('tool_calls', []))
            })

    response = json_data['response']['choices'][0]['message']
    reformat_response = {
        "role": response['role'],
        'content': response.get('content', ''),
        'tool_calls': process_tool_call(response.get('tool_calls', []))
    }

    return converted_messages + [reformat_response]


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


# ---------------------------------------------------------------------------
# Unified convert_dataset for chaofan-style completions directories
# ---------------------------------------------------------------------------

def convert_chaofan_dataset(
    folder_path: Path,
    *,
    shared_tools: list[dict[str, Any]] | None = None,
    max_instances: int | None = None,
    exclusion_patterns: list | None = None,
    label: str = "oh-chaofan",
) -> list[dict[str, Any]]:
    """Convert chaofan-style completions directories to IM records.

    When *shared_tools* is provided, every record uses that tool list.
    Otherwise tools are read from each completion JSON via
    ``json_data['kwargs']['tools']``.
    """
    from tqdm import tqdm

    im_data: list[dict[str, Any]] = []
    skipped_no_completion = 0
    skipped_invalid = 0

    subfolders = sorted(p.name for p in folder_path.iterdir() if p.is_dir())

    result_json_path = folder_path / "result.json"
    if result_json_path.exists():
        resolved_ids = set(get_resolved_instances(result_json_path))
        before = len(subfolders)
        subfolders = [sf for sf in subfolders if sf in resolved_ids]
        print(f"  [result.json] ({label}) kept {len(subfolders)}/{before} resolved instances")
    else:
        print(f"  [result.json] ({label}) not found, treating all as resolved")

    original_count = len(subfolders)
    if exclusion_patterns:
        subfolders = [
            sf for sf in subfolders
            if not instance_id_matches_excluded_repos(sf, exclusion_patterns)
        ]
        skipped_excluded_repo = original_count - len(subfolders)
        print(f"  [repo-filter] ({label}) excluded {skipped_excluded_repo}/"
              f"{original_count} subfolders, kept {len(subfolders)}")

    for subfolder in tqdm(subfolders):
        if max_instances is not None and len(im_data) >= max_instances:
            break

        subfolder_path = folder_path / subfolder
        agent_completions_path = subfolder_path / 'agent' / 'completions'
        if not (agent_completions_path.exists() and agent_completions_path.is_dir()):
            skipped_no_completion += 1
            continue

        latest_json_file = find_latest_json_filename(agent_completions_path)
        if not latest_json_file:
            skipped_no_completion += 1
            continue

        latest_json_path = agent_completions_path / latest_json_file
        try:
            json_data = load_json(latest_json_path)
            converted_messages = process_response_of_json_data(json_data)

            reasoning_contents = extract_reasoning_content(agent_completions_path)
            converted_messages = add_reasoning_content_to_json_data(converted_messages, reasoning_contents)

            if not check_roles(converted_messages):
                skipped_invalid += 1
                continue

            if not check_reasoning_content(converted_messages, think_mode='slow', pseudo_turns=None):
                skipped_invalid += 1
                print(f"Instance {subfolder} failed reasoning content check.")
                continue

            tools = shared_tools if shared_tools is not None else json_data.get('kwargs', {}).get('tools', [])
            im_data.append({
                'messages': converted_messages,
                'tools': tools,
                'pseudo_turns': None,
                'think_mode': 'slow',
                '_instance_id': subfolder,
            })
        except Exception as e:
            skipped_invalid += 1
            print(f"在处理子文件夹 {subfolder} 时出错，跳过该实例。错误: {e}")
            continue

    print(f"Total sampled instances: {len(im_data)}")
    print(f"Skipped (no completions): {skipped_no_completion}")
    print(f"Skipped (invalid/error): {skipped_invalid}")
    return im_data
