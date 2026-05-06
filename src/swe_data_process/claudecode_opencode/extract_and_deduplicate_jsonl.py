#!/usr/bin/env python3
"""
对齐 mini-vela/convert/dedup.py 的去重语义，对单个 instance 的 JSONL 轨迹做子轨迹去重：

1) 按 request_time 升序排序
2) 规范化 input：深拷贝后剥除 cache_control / signature / generation 字段；
   剔除 message.content 列表中 type=thinking 的 item；首条 user message 的
   list-content 拍平为纯文本；用 sort_keys 序列化
3) 两两比较：若 normalized[j].startswith(normalized[i]) 则 keep[i]=False
   （保留更长的那条前缀扩展）
4) 过滤 input 数 <= 2 的记录，并做一次整条记录级精确去重
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swe_data_process.utils import save_jsonl


def get_input_data(item: dict[str, Any]) -> Any:
    """安全获取轨迹输入，兼容新旧 logger 字段及 chaofan CC 格式。"""
    request_body = item.get("request_body", {})
    if "input" in request_body:
        return request_body.get("input")
    extra_input = request_body.get("extra_params", {}).get("input")
    if extra_input:
        return extra_input
    # chaofan CC 格式: messages 直接在 request_body 下
    if "messages" in request_body:
        return request_body.get("messages")
    return ""


_STRIP_KEYS = frozenset({"cache_control", "signature", "generation"})


def _strip_for_compare(obj: Any) -> Any:
    """深度拷贝并剥除比较无关字段，同时剔除 content 列表中的 thinking item。

    与 mini-vela dedup.py 中 `remove_keys` + `remove_thinking_items` 等价，
    但用纯函数式返回新对象，避免就地修改原记录。
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
    """把首条 user message 的 list-content 拍平为纯字符串，保证 prefix 可比。

    对齐 mini-vela `get_messages_hash` 的首条 user 拍平步骤。
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


def normalize_input(input_value: Any) -> str:
    """对 input 做 mini-vela 风格的规范化后序列化成可用于 prefix 比较的字符串。

    - list 输入：先剥字段再移 thinking item，再拍平首条 user content，
      最后 sort_keys 序列化并去掉外层 `[`、`]` 以便前缀判断。
    - 非 list 输入：兜底直接序列化或原样字符串。
    """
    if isinstance(input_value, list):
        stripped = _strip_for_compare(input_value)
        _flatten_first_user_content(stripped)
        return json.dumps(stripped, sort_keys=True, ensure_ascii=False)[1:-1]

    if isinstance(input_value, str):
        return input_value
    return json.dumps(input_value, sort_keys=True, ensure_ascii=False)


def _record_normalize_key(record: dict[str, Any]) -> str:
    """整条记录的规范化 key（用于 exact dedup），剥 cache_control/signature/generation。"""
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
                print(f"[WARN] line {line_no} 不是合法 JSON，已跳过: {e}")

    # 对齐 mini-vela：去重前按 request_time 升序（缺失视为 0，保持稳定序）
    records.sort(key=lambda r: r.get("request_time", 0))

    normalized = [normalize_input(get_input_data(rec)) for rec in records]
    keep = [True] * len(records)

    # 前缀去重：若 normalized[j] 以 normalized[i] 为前缀，则 i 被 j 覆盖，丢弃 i
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


def filter_short_input_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """过滤 input 数目 <= 2 的记录。"""
    filtered: list[dict[str, Any]] = []
    for rec in records:
        input_data = get_input_data(rec)
        input_count = len(input_data) if isinstance(input_data, list) else 0
        if input_count <= 2:
            continue
        filtered.append(rec)
    return filtered


def deduplicate_exact_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对最终 records 做一次整条记录级别的去重（保持原顺序）。"""
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
    parser = argparse.ArgumentParser(description="提取并去重 JSONL 轨迹（对齐 mini-vela 语义）")
    parser.add_argument(
        "-i", "--input",
        help="输入 jsonl 文件路径",
        default="/home/ywxzml3j/ywxzml3juser30/code/harbor/jobs/swerebench-filtered-oraclesolved-claude-code-2.1.62-GLM-5-FP8-2-20260321233207_trajs_via_logger/12rambau__sepal_ui-411.jsonl",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 jsonl 文件路径",
        default="/home/ywxzml3j/ywxzml3juser57/jierun_test.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    records = deduplicate_trajectories(input_path)
    save_jsonl(output_path, records)

    print(f"完成：输入 {input_path}，输出 {output_path}，保留 {len(records)} 条轨迹")


if __name__ == "__main__":
    main()
