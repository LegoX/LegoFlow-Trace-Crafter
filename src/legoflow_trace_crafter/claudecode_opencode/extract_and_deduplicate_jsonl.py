#!/usr/bin/env python3
"""
Deduplicate subtrajectories in one instance's JSONL trajectory, matching the
semantics of mini-vela/convert/dedup.py:

0) Drop failed calls marked success=false by the logger (HTTP timeouts, 5xx
   responses, upstream rejections, and so on).
1) Sort by request_time in ascending order.
2) Normalize input: deep-copy it and remove cache_control, signature, and
   generation fields; remove type=thinking items from message.content lists;
   flatten the first user message's list content to plain text; serialize with
   sort_keys.
3) Compare every pair: if normalized[j].startswith(normalized[i]), set
   keep[i]=False, retaining the longer prefix extension.
4) Filter records whose input count is at most 2, then exactly deduplicate
   complete records.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from legoflow_trace_crafter.utils import save_jsonl


def get_input_data(item: dict[str, Any]) -> Any:
    """Safely read trajectory input across logger versions and CC formats."""
    request_body = item.get("request_body", {})
    if "input" in request_body:
        return request_body.get("input")
    extra_input = request_body.get("extra_params", {}).get("input")
    if extra_input:
        return extra_input
    # Some records store messages directly under request_body.
    if "messages" in request_body:
        return request_body.get("messages")
    return ""


_STRIP_KEYS = frozenset({"cache_control", "signature", "generation"})


def _strip_for_compare(obj: Any) -> Any:
    """Deep-copy data while removing comparison-irrelevant fields.

    This also removes thinking items from content lists. It is equivalent to
    `remove_keys` plus `remove_thinking_items` in mini-vela dedup.py, but
    returns new objects instead of mutating the original record.
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _STRIP_KEYS:
                continue
            if k == "content" and isinstance(v, list):
                v = [
                    item
                    for item in v
                    if not (isinstance(item, dict) and item.get("type") == "thinking")
                ]
            result[k] = _strip_for_compare(v)
        return result
    if isinstance(obj, list):
        return [_strip_for_compare(item) for item in obj]
    return obj


def _flatten_first_user_content(messages: list[Any]) -> None:
    """Flatten the first user's list content to text for prefix comparison.

    This matches the first-user flattening step in mini-vela
    `get_messages_hash`.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            msg["content"] = "".join(parts)
        break


_CCH_PATTERN = re.compile(r"cch=[0-9a-f]+;")


def normalize_input(input_value: Any) -> str:
    """Normalize input in mini-vela style for serialized prefix comparison.

    - List input: remove fields and thinking items, flatten the first user's
      content, serialize with sort_keys, and remove the outer `[` and `]`.
    - Other input: serialize directly, or return strings unchanged.

    Dynamic fields such as Claude Code's cch= hash are replaced consistently
    so they do not break prefix relationships.
    """
    if isinstance(input_value, list):
        stripped = _strip_for_compare(input_value)
        _flatten_first_user_content(stripped)
        result = json.dumps(stripped, sort_keys=True, ensure_ascii=False)[1:-1]
        return _CCH_PATTERN.sub("cch=;", result)

    if isinstance(input_value, str):
        return input_value
    return json.dumps(input_value, sort_keys=True, ensure_ascii=False)


def _record_normalize_key(record: dict[str, Any]) -> str:
    """Build a whole-record exact-dedup key with transient fields removed."""
    return json.dumps(_strip_for_compare(record), sort_keys=True, ensure_ascii=False)


def deduplicate_trajectories(input_jsonl: str | Path) -> list[dict[str, Any]]:
    input_jsonl = Path(input_jsonl)
    records: list[dict[str, Any]] = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no} is not valid JSON; skipping: {e}")

    # Explicitly drop calls the logger marked as failed (HTTP timeout, 5xx,
    # upstream rejection, and so on). Their response_body is usually an error
    # structure that prefix deduplication can mistakenly retain as the longest
    # extension, contaminating downstream convert_record output.
    records = filter_failed_records(records)

    # Match mini-vela by sorting on request_time before deduplication. Treat a
    # missing timestamp as 0 and preserve stable ordering.
    records.sort(key=lambda r: r.get("request_time", 0))

    normalized = [normalize_input(get_input_data(rec)) for rec in records]
    keep = [True] * len(records)

    # Prefix deduplication: if normalized[j] starts with normalized[i], j
    # supersedes i, so discard i.
    for i in range(len(records)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(records)):
            if not keep[j]:
                continue
            if normalized[j].startswith(normalized[i]):
                keep[i] = False
                break

    survivors = [records[i] for i in range(len(records)) if keep[i]]
    filtered = filter_short_input_records(survivors)
    return deduplicate_exact_records(filtered)


def filter_failed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop failed calls that the logger marked success=false.

    The LiteLLM logger writes ``success: false`` for HTTP failures, upstream
    rejections, timeouts, and 5xx responses. Their ``response_body`` is
    typically an error structure. A missing ``success`` field defaults to
    success for compatibility with older logger versions.
    """
    return [r for r in records if r.get("success", True) is not False]


def filter_short_input_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop records whose input contains at most two items."""
    filtered: list[dict[str, Any]] = []
    for rec in records:
        input_data = get_input_data(rec)
        input_count = len(input_data) if isinstance(input_data, list) else 0
        if input_count <= 2:
            continue
        filtered.append(rec)
    return filtered


def deduplicate_exact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate complete records while preserving their original order."""
    seen: set[str] = set()
    unique_records: list[dict[str, Any]] = []
    for rec in records:
        key = _record_normalize_key(rec)
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(rec)
    return unique_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and deduplicate JSONL trajectories using mini-vela semantics"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input JSONL file path",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file path",
        required=True,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    records = deduplicate_trajectories(input_path)
    save_jsonl(output_path, records)

    print(f"Done: input {input_path}, output {output_path}, kept {len(records)} trajectories")


if __name__ == "__main__":
    main()
