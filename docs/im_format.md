# Intermediate Message (IM) Format

IM files are JSONL: each non-empty line is one trajectory record. `save_jsonl()` normalizes every IM record to version `2.0.0` before writing it.

## Record structure

Every persisted record has four top-level fields:

- `version` (`string`): currently `"2.0.0"`.
- `meta_info` (`object`): provenance, summary metadata, and source-specific fields.
- `tools` (`array`): function-tool definitions; an empty array is valid.
- `messages` (`array`): the ordered conversation.

Fields from an input record that are not part of this top-level schema are retained under `meta_info.unique_info`.

## Metadata

`meta_info` contains:

- `teacher` (`string`): source model label.
- `query_source` (`string`): source or provenance label.
- `response_generate_time` (`string`): generation date in `YYYY-MM-DD` form.
- `response_update_time` (`string`): update date in `YYYY-MM-DD` form.
- `owner` (`string`): opaque provenance value retained for schema compatibility. This repository does not impose an identity format or ownership convention.
- `language` (`string`): language code. When absent, the serializer infers `"zh"` or `"en"` from the first user message.
- `category` (`string`): task category; generated records default to `"code"`.
- `rounds` (`integer`): number of assistant messages unless supplied explicitly.
- `unique_info` (`object`): converter-specific metadata and compatibility fields.

Common `unique_info` fields are:

- `_instance_id`: source instance identifier.
- `_agent_type`: `"main"` or `"subagent"`.
- `_score`: rule and optional LLM scores; it is `null` for subagent records after rule scoring.
- `_gen_params`: source generation parameters, when available.
- `_usage`: source usage data, when available.
- `_instance_metadata`: task metadata, when available.

`load_jsonl()` exposes these common fields at the top level in memory for compatibility. `save_jsonl()` places them back under `meta_info.unique_info`.

## Tool definitions

Each item in `tools` is normalized to:

```json
{
  "type": "function",
  "function": {
    "name": "run_tests",
    "description": "Run a selected test target.",
    "parameters": {
      "type": "object",
      "properties": {
        "target": {"type": "string"}
      },
      "required": ["target"]
    }
  }
}
```

The normalizer accepts either `function.parameters` or `function.input_schema` as input and persists the schema as `function.parameters`.

## Messages

Converters and validators use the standard roles `system`, `user`, `assistant`, and `tool`. Each message has:

- `role` (`string`), required;
- `content` (`string`), always present after normalization;
- `name` (`string`), optional.

Assistant messages also have:

- `reasoning_content` (`string`), always present after normalization and empty when no separate reasoning is available;
- `tool_calls` (`array`), optional;
- `weight` (`number`), optional.

A function tool call has this form:

```json
{
  "id": "call_001",
  "type": "function",
  "function": {
    "name": "run_tests",
    "arguments": "{\"target\":\"tests/test_example.py\"}"
  }
}
```

`function.arguments` is a JSON-encoded string in persisted IM data. A tool result may include `tool_call_id` to match the corresponding call. During normalization, missing call IDs are generated and aligned with immediately following tool messages when possible.

## Example record

```json
{
  "version": "2.0.0",
  "meta_info": {
    "teacher": "example-model",
    "query_source": "generated",
    "response_generate_time": "2026-01-01",
    "response_update_time": "2026-01-01",
    "owner": "example-source",
    "language": "en",
    "category": "code",
    "rounds": 2,
    "unique_info": {
      "_instance_id": "example__project-1",
      "_agent_type": "main",
      "_score": {
        "composite_score": 0.82
      }
    }
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "run_tests",
        "description": "Run a selected test target.",
        "parameters": {
          "type": "object",
          "properties": {
            "target": {"type": "string"}
          },
          "required": ["target"]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "You are a software-engineering assistant."
    },
    {
      "role": "user",
      "content": "Fix the failing parser test."
    },
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "I will run the focused test first.",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "run_tests",
            "arguments": "{\"target\":\"tests/test_parser.py\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "1 passed"
    },
    {
      "role": "assistant",
      "content": "The focused parser test now passes.",
      "reasoning_content": ""
    }
  ]
}
```

## Normalization and validation

When IM data is saved:

- non-string content is converted to text;
- an empty system message is inserted if the first message is not `system`;
- `<think>...</think>` content is separated into `reasoning_content`;
- tool definitions and tool calls are normalized to function form;
- tool-call arguments are serialized as JSON strings.

Converters that use the shared validators additionally require:

- the first non-system message to be `user` and the final message to be `assistant`;
- `user` to be followed by `assistant`;
- `assistant` to be followed by `tool` or `user`;
- `tool` to be followed by `tool` or `assistant`;
- every assistant message except the last assistant message to contain tool calls.

Reasoning checks are source- and option-dependent; see [Conversion filtering and scoring](data_filtering_strategy.md).
