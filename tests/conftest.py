from __future__ import annotations

import json

import pytest


@pytest.fixture
def minimal_im_record():
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug in utils.py"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let me look at the file.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": '{"file_path":"/testbed/utils.py"}',
                        },
                        "id": "call_001",
                    }
                ],
            },
            {"role": "tool", "content": "def foo():\n    pass", "tool_call_id": "call_001"},
            {
                "role": "assistant",
                "content": "I found and fixed the bug.",
                "reasoning_content": "The issue is clear.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Edit",
                    "description": "Edit a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                },
            },
        ],
    }


@pytest.fixture
def minimal_im_record_fast():
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "tools": [],
    }


@pytest.fixture
def subagent_record():
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Find relevant files"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Searching...",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "Grep", "arguments": '{"pattern":"foo"}'},
                        "id": "call_001",
                    }
                ],
            },
            {"role": "tool", "content": "Found 2 files", "tool_call_id": "call_001"},
            {
                "role": "assistant",
                "content": "Found the files.",
                "reasoning_content": "Done.",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "Grep", "description": "Search", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "Read", "description": "Read", "parameters": {}},
            },
        ],
    }


@pytest.fixture
def terminus2_step_record():
    return {
        "steps": [
            {
                "source": "user",
                "message": "Fix the failing test in test_utils.py",
            },
            {
                "message": "Analysis: The test is failing due to an assertion error.\nPlan: Fix the assertion.",
                "reasoning_content": "I need to fix the test.",
                "tool_calls": [
                    {
                        "function_name": "execute_command",
                        "arguments": {"keystrokes": "sed -i 's/old/new/' test_utils.py\n"},
                    }
                ],
                "observation": "Command executed successfully.",
            },
            {
                "message": "Analysis: Fixed.\nPlan: Mark complete.",
                "reasoning_content": "Done.",
                "tool_calls": [
                    {"function_name": "mark_task_complete", "arguments": {}}
                ],
                "observation": "",
            },
        ]
    }


@pytest.fixture
def scored_records():
    return [
        {
            "_instance_id": "owner__repo-1__abc",
            "_agent_type": "main",
            "_score": {"composite_score": 0.8},
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Fix bug"},
                {"role": "assistant", "content": "Done."},
            ],
        },
        {
            "_instance_id": "owner__repo-1__abc",
            "_agent_type": "subagent",
            "_score": None,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Search"},
                {"role": "assistant", "content": "Found."},
            ],
        },
        {
            "_instance_id": "owner__repo-2__def",
            "_agent_type": "main",
            "_score": {"composite_score": 0.3},
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "Fix"},
                {"role": "assistant", "content": "Done."},
            ],
        },
    ]
