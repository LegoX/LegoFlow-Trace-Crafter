## 读取数据，将数据转成下列Openai格式

```json
[
    {
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": {
                                ...
                            }
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": ""
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": {
                                ...
                            }
                        }
                    }
                ]
            }
        ],
        "tools": [
            {   
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            ...
                        }
                    }
                }
            },
            {   
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            ...
                        }
                    }
                }
            },
        ],
        "pseudo_turns": null,
        "think_mode": ""
    }
]
```