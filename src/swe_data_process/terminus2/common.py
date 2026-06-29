"""Shared helpers for Terminus2 trajectory conversion."""
from __future__ import annotations

import json
from typing import Any, Iterable


def split_analysis_plan(message: str) -> dict[str, str]:
    """Split 'Analysis: ...\\nPlan: ...' text into fields."""
    text = (message or "").strip()
    if not text:
        return {"analysis": "", "plan": ""}

    analysis = ""
    plan = ""

    if text.startswith("Analysis:"):
        body = text[len("Analysis:"):].strip()
        if "\nPlan:" in body:
            analysis, plan = body.split("\nPlan:", 1)
        elif "Plan:" in body:
            analysis, plan = body.split("Plan:", 1)
        else:
            analysis = body
    elif text.startswith("Plan:"):
        plan = text[len("Plan:"):].strip()
    else:
        analysis = text

    return {"analysis": analysis.strip(), "plan": plan.strip()}


def normalize_command(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None

    function_name = tool_call.get("function_name")
    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, dict):
        return None

    if function_name == "mark_task_complete":
        return None

    keystrokes = arguments.get("keystrokes")
    if not keystrokes:
        command = arguments.get("command")
        if command:
            keystrokes = command if str(command).endswith("\n") else f"{command}\n"

    if not keystrokes:
        return None

    normalized: dict[str, Any] = {
        "keystrokes": keystrokes,
    }

    duration = arguments.get("duration")
    if duration is not None:
        normalized["duration"] = duration

    return normalized


def build_assistant_content(step: dict[str, Any]) -> str:
    parsed = split_analysis_plan(step.get("message"))
    commands: list[dict[str, Any]] = []
    task_complete = False

    for tool_call in step.get("tool_calls", []) or []:
        if not isinstance(tool_call, dict):
            continue
        if tool_call.get("function_name") == "mark_task_complete":
            task_complete = True
            continue
        command = normalize_command(tool_call)
        if command is not None:
            commands.append(command)

    payload: dict[str, Any] = {
        "analysis": parsed["analysis"],
        "plan": parsed["plan"],
        "commands": commands,
    }
    if task_complete:
        payload["task_complete"] = True

    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_observation_text(observation: Any) -> str:
    if not observation:
        return ""

    if isinstance(observation, str):
        return observation

    if isinstance(observation, dict):
        results = observation.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, (dict, list)):
                    return json.dumps(content, ensure_ascii=False)
            elif isinstance(first, str):
                return first
        return json.dumps(observation, ensure_ascii=False)

    if isinstance(observation, list):
        if not observation:
            return ""
        first = observation[0]
        if isinstance(first, str):
            return first
        return json.dumps(observation, ensure_ascii=False)

    return str(observation)


def extract_json_content_from_assistant(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    think_end_tag = "</think>"
    if text.startswith("<think>") and think_end_tag in text:
        after = text.split(think_end_tag, 1)[1].strip()
        return after

    return text


def get_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []

    if isinstance(record.get("trajectory"), dict):
        steps = record["trajectory"].get("steps")
        if isinstance(steps, list):
            return steps

    steps = record.get("steps")
    if isinstance(steps, list):
        return steps

    return []


def convert_one_record(record: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    has_reasoning = False

    for step in get_steps(record):
        if not isinstance(step, dict):
            continue

        if step.get('source') == 'user':
            user_message = step.get('message')
            if isinstance(user_message, str) and user_message:
                messages.append({"role": "user", "content": user_message})
            continue

        # 1) message + tool_calls -> assistant content
        raw_message = step.get("message")
        raw_reasoning_content = step.get("raw_reasoning_content") or step.get("reasoning_content")
        if isinstance(raw_reasoning_content, str):
            raw_reasoning_content = raw_reasoning_content.strip()
        raw_tool_calls = step.get("tool_calls") or []
        if raw_message or raw_tool_calls:
            if raw_reasoning_content:
                has_reasoning = True
                messages.append(
                    {
                        "role": "assistant",
                        "content": "<think>\n" + raw_reasoning_content + "\n</think>\n\n" + build_assistant_content(step),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": build_assistant_content(step),
                    }
                )

        # 2) observation -> user content
        observation_text = extract_observation_text(step.get("observation"))
        if observation_text:
            messages.append({"role": "user", "content": observation_text})

    out: dict[str, Any] = {"messages": messages}

    out['pseudo_turns'] = None
    out['think_mode'] = "slow" if has_reasoning else "fast"

    if not out['messages']:
        raise ValueError("Empty messages after conversion")

    if out['messages'][-1]['role'] == 'user':
        out['messages'] = out['messages'][:-1]

    if not out['messages']:
        raise ValueError("No assistant turn left after trimming trailing user message")

    # 确保 user 消息和 assistant 消息交替出现，且以 assistant 消息结尾
    for i in range(len(out['messages'])):
        role = out['messages'][i]['role']
        if role not in ('user', 'assistant'):
            raise ValueError(f"Invalid role: {role}")
        if i % 2 == 0:
            if role != 'user':
                raise ValueError(f"Expected user at index {i}, got {role}")
        else:
            if role != 'assistant':
                raise ValueError(f"Expected assistant at index {i}, got {role}")

    # 确保 assistant 消息可以被 json.loads 正确解析
    for i in range(1, len(out['messages']), 2):
        try:
            json.loads(extract_json_content_from_assistant(out['messages'][i]['content']))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in assistant content at index {i}: {e}")

    return out


def to_lf_record(im_record: dict[str, Any]) -> dict[str, Any]:
    rec = {"messages": im_record["messages"]}
    if "_score" in im_record:
        rec["_score"] = im_record["_score"]
    if "_instance_id" in im_record:
        rec["_instance_id"] = im_record["_instance_id"]
    return rec


def iter_records(obj: Any) -> Iterable[dict[str, Any]]:
    """Normalize input into records."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        if "steps" in obj or "trajectory" in obj:
            yield obj
        elif isinstance(obj.get("data"), list):
            for item in obj["data"]:
                if isinstance(item, dict):
                    yield item
