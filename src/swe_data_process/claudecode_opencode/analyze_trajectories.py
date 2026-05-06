#!/usr/bin/env python3
"""
轨迹文件夹分析脚本：
1. 对文件夹中每个 JSONL 进行去重，合并所有去重后的轨迹，统计文件个数和轨迹个数。
2. 统计 tool 角色中 tool_use_error 和 invalid arguments 的错误率：
   - 按轮次维度：错误 tool 轮次数 / 总 tool 轮次数
   - 按轨迹维度：含错误的轨迹数 / 总轨迹数
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from swe_data_process.claudecode_opencode.extract_and_deduplicate_jsonl import deduplicate_trajectories
from swe_data_process.claudecode_opencode.convert_jsonl_to_openai import convert_record


TOOL_ERROR_PREFIXES = (
    "<tool_use_error>",
    "The arguments provided to the tool are invalid",
)


def is_tool_error(message: dict[str, Any]) -> bool:
    """判断一条 tool 消息是否为工具调用错误。"""
    if message.get("role") != "tool":
        return False
    content = message.get("content", "")
    if not isinstance(content, str):
        return False
    return any(content.startswith(prefix) for prefix in TOOL_ERROR_PREFIXES)


def deduplicate_and_merge(input_dir: Path) -> list[dict[str, Any]]:
    """对目录中每个 JSONL 去重并合并，返回 (所有去重后的记录列表)。"""
    jsonl_files = sorted(input_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"[WARN] 目录 {input_dir} 下未找到任何 .jsonl 文件")
        return []

    all_records: list[dict[str, Any]] = []
    for jsonl_file in tqdm(jsonl_files, desc="去重中", unit="file"):
        records = deduplicate_trajectories(jsonl_file)
        all_records.extend(records)

    print(f"文件总数: {len(jsonl_files)}")
    print(f"合并后轨迹总数: {len(all_records)}")
    return all_records


def analyze_tool_errors(records: list[dict[str, Any]]) -> None:
    """将每条原始记录转换为 IM 格式（保留错误轮次），统计 tool 错误率。

    注意：这里直接调用 convert_record 后的 messages 已经经过 remove_tool_use_error_turns，
    因此我们需要在错误移除之前进行统计。这里重新从原始记录解析 messages 来统计。
    """
    total_trajectories = 0
    trajectories_with_error = 0
    total_tool_turns = 0
    error_tool_turns = 0

    # 每条轨迹内的错误轮次分布（用于展示详情）
    per_trajectory_stats: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        # 从原始记录中提取所有 messages（不经过 remove_tool_use_error_turns）
        request_body = record.get("request_body") or {}
        raw_messages = request_body.get("messages") or []
        if not raw_messages:
            # 兼容 chaofan CC 格式
            raw_messages = request_body.get("input") or []
            if not isinstance(raw_messages, list):
                continue

        # 展开 content 中嵌套的 tool_result
        tool_messages = _extract_tool_messages(raw_messages)

        traj_tool_count = len(tool_messages)
        traj_error_count = sum(1 for m in tool_messages if is_tool_error(m))

        if traj_tool_count > 0:
            total_trajectories += 1
            total_tool_turns += traj_tool_count
            error_tool_turns += traj_error_count
            if traj_error_count > 0:
                trajectories_with_error += 1

            per_trajectory_stats.append({
                "index": idx,
                "tool_turns": traj_tool_count,
                "error_turns": traj_error_count,
                "error_rate": traj_error_count / traj_tool_count,
            })

    print("\n" + "=" * 60)
    print("工具调用错误统计")
    print("=" * 60)

    print(f"\n【按轮次维度】")
    print(f"  总 tool 轮次数: {total_tool_turns}")
    print(f"  错误 tool 轮次数: {error_tool_turns}")
    if total_tool_turns > 0:
        print(f"  轮次错误率: {error_tool_turns / total_tool_turns:.4f} ({error_tool_turns / total_tool_turns * 100:.2f}%)")
    else:
        print(f"  轮次错误率: N/A (无 tool 轮次)")

    print(f"\n【按轨迹维度】")
    print(f"  含 tool 轮次的轨迹总数: {total_trajectories}")
    print(f"  含错误的轨迹数: {trajectories_with_error}")
    if total_trajectories > 0:
        print(f"  轨迹错误率: {trajectories_with_error / total_trajectories:.4f} ({trajectories_with_error / total_trajectories * 100:.2f}%)")
    else:
        print(f"  轨迹错误率: N/A (无含 tool 的轨迹)")

    # 展示含错误的轨迹详情（最多前 20 条）
    error_trajs = [s for s in per_trajectory_stats if s["error_turns"] > 0]
    if error_trajs:
        print(f"\n含错误的轨迹详情 (共 {len(error_trajs)} 条，展示前 20 条):")
        for stat in error_trajs[:20]:
            print(f"  轨迹 #{stat['index']}: "
                  f"tool 轮次={stat['tool_turns']}, "
                  f"错误轮次={stat['error_turns']}, "
                  f"错误率={stat['error_rate']:.2%}")


def _extract_tool_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从原始 messages 中提取所有 tool 角色的消息。

    原始格式中 tool 结果可能以两种方式出现：
    1. role=tool 的顶层消息
    2. role=user 消息中 content 列表里 type=tool_result 的 block
    """
    tool_messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "tool":
            # 直接是 tool 角色的消息
            tool_messages.append(msg)
        elif role == "user" and isinstance(content, list):
            # content 中可能包含 tool_result block
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    # 将嵌套的 content 标准化为字符串
                    if isinstance(tool_content, list):
                        parts = []
                        for part in tool_content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                        tool_content = "\n".join(parts)
                    tool_messages.append({"role": "tool", "content": tool_content})

    return tool_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="去重、合并轨迹并统计工具调用错误率")
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="包含 JSONL 轨迹文件的目录路径",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    print(f"输入目录: {input_dir}")
    print("-" * 60)

    # Step 1: 去重并合并
    all_records = deduplicate_and_merge(input_dir)
    if not all_records:
        print("无轨迹可分析，退出。")
        return

    # Step 2: 统计错误率
    analyze_tool_errors(all_records)


if __name__ == "__main__":
    main()
