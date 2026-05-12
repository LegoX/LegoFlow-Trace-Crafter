import json
from pathlib import Path
from typing import Any


def join_text_parts(parts: list[str]) -> str:
    cleaned = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    return "\n\n".join(cleaned)


# 部分原始轨迹用字面量 "(empty)" 表示无正文，与空字符串语义一致
_ASSISTANT_EMPTY_PLACEHOLDER = "(empty)"


def normalize_assistant_empty_content(messages: list[dict[str, Any]]) -> None:
    """将 assistant 的 content 为 '(empty)' 的占位符统一为 ''。"""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip() == _ASSISTANT_EMPTY_PLACEHOLDER:
            msg["content"] = ""


def normalize_tool_definition(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool

    return {
        "type": "function",
        "function": {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or function.get("input_schema") or {},
        },
    }


def reorder_arguments_by_properties(arguments: Any, properties_order: list[str] | None) -> Any:
    if not isinstance(arguments, dict) or not properties_order:
        return arguments

    reordered: dict[str, Any] = {}
    for key in properties_order:
        if key in arguments:
            reordered[key] = arguments[key]
    for key, value in arguments.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def parse_tool_arguments(arguments: Any, properties_order: list[str] | None = None) -> Any:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    else:
        parsed = arguments if arguments is not None else {}

    return reorder_arguments_by_properties(parsed, properties_order)


def stringify_tool_arguments(arguments: Any, properties_order: list[str] | None = None) -> str:
    parsed = parse_tool_arguments(arguments, properties_order)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def normalize_tool_content(content: Any) -> str:
    if isinstance(content, list):
        return join_text_parts(
            [block.get("text", "") for block in content if isinstance(block, dict)]
        )

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content

        if isinstance(parsed, list):
            return join_text_parts(
                [block.get("text", "") for block in parsed if isinstance(block, dict)]
            )
        return content

    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def convert_assistant_blocks(
    blocks: list[dict[str, Any]], tool_properties_order: dict[str, list[str]] | None = None
) -> dict[str, Any] | None:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                reasoning_parts.append(thinking)
        elif block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_name = block.get("name", "")
            tool_call = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": stringify_tool_arguments(
                        block.get("input"),
                        (tool_properties_order or {}).get(tool_name),
                    ),
                },
            }
            tool_call_id = block.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_call["id"] = tool_call_id
            tool_calls.append(tool_call)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": join_text_parts(text_parts),
    }
    if reasoning_parts:
        message["reasoning_content"] = join_text_parts(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    if message["content"] or message.get("reasoning_content") or message.get("tool_calls"):
        return message
    return None


def convert_user_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    text_buffer: list[str] = []

    def flush_user_buffer() -> None:
        content = join_text_parts(text_buffer)
        if content:
            messages.append({"role": "user", "content": content})
        text_buffer.clear()

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_buffer.append(text)
        elif block_type == "tool_result":
            flush_user_buffer()
            tool_content = normalize_tool_content(block.get("content", ""))
            tool_message: dict[str, Any] = {"role": "tool", "content": tool_content}
            tool_call_id = block.get("tool_use_id") or block.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_message["tool_call_id"] = tool_call_id
            messages.append(tool_message)
        else:
            fallback_text = block.get("text") or block.get("content")
            if isinstance(fallback_text, str) and fallback_text.strip():
                text_buffer.append(fallback_text)

    flush_user_buffer()
    return messages


def _normalize_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_properties_order: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize a list of tool_call dicts: stringify arguments and reorder by schema."""
    return [
        {
            **tool_call,
            "function": {
                **(tool_call.get("function") or {}),
                "arguments": stringify_tool_arguments(
                    (tool_call.get("function") or {}).get("arguments"),
                    (tool_properties_order or {}).get(
                        ((tool_call.get("function") or {}).get("name") or "")
                    ),
                ),
            },
        }
        for tool_call in tool_calls
    ]


def convert_message(
    message: dict[str, Any], tool_properties_order: dict[str, list[str]] | None = None
) -> list[dict[str, Any]]:
    role = message.get("role")
    content = message.get("content")

    if role == "assistant":
        if isinstance(content, list):
            converted = convert_assistant_blocks(content, tool_properties_order)
            return [converted] if converted else []
        if isinstance(content, str):
            converted_message: dict[str, Any] = {"role": "assistant", "content": content}
            if message.get("reasoning_content"):
                converted_message["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                converted_message["tool_calls"] = _normalize_tool_calls(
                    message["tool_calls"], tool_properties_order
                )
            return [converted_message]
        if message.get("reasoning_content") or message.get("tool_calls"):
            converted_message = {"role": "assistant", "content": ""}
            if message.get("reasoning_content"):
                converted_message["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                converted_message["tool_calls"] = _normalize_tool_calls(
                    message["tool_calls"], tool_properties_order
                )
            return [converted_message]
        return []

    if role == "user":
        if isinstance(content, list):
            return convert_user_blocks(content)
        if isinstance(content, str) and content.strip():
            return [{"role": "user", "content": content}]
        return []

    if role == "system":
        if isinstance(content, list):
            system_text = join_text_parts(
                [block.get("text", "") for block in content if isinstance(block, dict)]
            )
        else:
            system_text = content if isinstance(content, str) else ""
        return [{"role": "system", "content": system_text}]

    if role == "tool":
        tool_text = normalize_tool_content(content)
        converted_message: dict[str, Any] = {"role": "tool", "content": tool_text}
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            converted_message["tool_call_id"] = tool_call_id
        return [converted_message]

    return []


def convert_final_response(
    response_body: dict[str, Any], tool_properties_order: dict[str, list[str]] | None = None
) -> dict[str, Any] | None:
    content = response_body.get("content")
    if isinstance(content, list):
        return convert_assistant_blocks(content, tool_properties_order)

    choices = response_body.get("choices") or []
    if not choices:
        return None

    message = choices[0].get("message") or {}
    if message.get("role") != "assistant":
        return None

    if isinstance(message.get("content"), list):
        return convert_assistant_blocks(message["content"])

    converted: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content", "") if isinstance(message.get("content"), str) else "",
    }
    if message.get("reasoning_content"):
        converted["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        tool_calls = []
        for tool_call in message["tool_calls"]:
            function = tool_call.get("function", {})
            function_name = function.get("name", "")
            tool_calls.append(
                {
                    "id": tool_call.get("id"),
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": stringify_tool_arguments(
                            function.get("arguments"),
                            (tool_properties_order or {}).get(function_name),
                        ),
                    },
                }
            )
        converted["tool_calls"] = tool_calls
    return converted


def build_tool_properties_order_map(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
    order_map: dict[str, list[str]] = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        parameters = function.get("parameters") or function.get("input_schema") or {}
        if not isinstance(parameters, dict):
            continue

        properties = parameters.get("properties")
        if isinstance(properties, dict):
            order_map[name] = list(properties.keys())

    return order_map


def extract_system_message(request_body: dict[str, Any]) -> dict[str, Any] | None:
    system_blocks = request_body.get("system")
    if not isinstance(system_blocks, list):
        return None

    text_parts = []
    for block in system_blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
            text_parts.append(block["text"])
    content = join_text_parts(text_parts)
    if not content:
        return None
    return {"role": "system", "content": content}


def trim_trailing_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed = list(messages)
    while trimmed and trimmed[-1].get("role") == "tool":
        trimmed.pop()
    return trimmed


def remove_tool_use_error_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_error_prefixes = (
        "<tool_use_error>",
        "The arguments provided to the tool are invalid",
    )
    filtered: list[dict[str, Any]] = []
    skipping_tool_block = False

    for message in messages:
        content = message.get("content")
        role = message.get("role")

        if skipping_tool_block:
            if role == "tool":
                continue
            skipping_tool_block = False

        if (
            role == "tool"
            and isinstance(content, str)
            and any(content.startswith(prefix) for prefix in tool_error_prefixes)
        ):
            while filtered and filtered[-1].get("role") == "tool":
                filtered.pop()

            if filtered and filtered[-1].get("role") == "assistant":
                filtered.pop()

            skipping_tool_block = True
            continue

        filtered.append(message)

    return filtered


def infer_think_mode(messages: list[dict[str, Any]]) -> str:
    has_reasoning = any(
        message.get("role") == "assistant" and bool(message.get("reasoning_content"))
        for message in messages
    )
    return "slow" if has_reasoning else "fast"


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    request_body = record.get("request_body") or {}
    converted_messages: list[dict[str, Any]] = []
    tools = request_body.get("tools") or []
    tool_properties_order = build_tool_properties_order_map(tools)

    system_message = extract_system_message(request_body)
    if system_message:
        converted_messages.append(system_message)

    for message in request_body.get("messages") or []:
        converted_messages.extend(convert_message(message, tool_properties_order))

    final_assistant = convert_final_response(record.get("response_body") or {}, tool_properties_order)
    if final_assistant:
        converted_messages.append(final_assistant)

    converted_messages = remove_tool_use_error_turns(converted_messages)
    converted_messages = trim_trailing_tools(converted_messages)
    normalize_assistant_empty_content(converted_messages)

    return {
        "messages": converted_messages,
        "tools": [normalize_tool_definition(tool) for tool in tools],
        "pseudo_turns": None,
        "think_mode": infer_think_mode(converted_messages),
    }


def convert_file(input_path: Path) -> list[dict[str, Any]]:
    converted_records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number} is not valid JSON: {error}") from error
            converted_records.append(convert_record(record))
    return converted_records
