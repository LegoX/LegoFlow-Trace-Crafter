from __future__ import annotations

import json
from pathlib import Path

from swe_data_process.claudecode_opencode.extract_and_deduplicate_jsonl import (
    deduplicate_trajectories,
    filter_failed_records,
)


class TestFilterFailedRecords:
    def test_keeps_success_true(self):
        records = [{"success": True, "id": 1}, {"success": True, "id": 2}]
        assert filter_failed_records(records) == records

    def test_drops_success_false(self):
        records = [
            {"success": True, "id": 1},
            {"success": False, "id": 2},
            {"success": True, "id": 3},
        ]
        assert [r["id"] for r in filter_failed_records(records)] == [1, 3]

    def test_missing_field_kept(self):
        # 兼容旧 logger：缺 success 字段视为成功
        records = [{"id": 1}, {"success": False, "id": 2}, {"id": 3}]
        assert [r["id"] for r in filter_failed_records(records)] == [1, 3]

    def test_truthy_non_false_kept(self):
        # 只对显式 False 过滤，其它值（如 None、"false" 字符串）保守保留
        records = [
            {"success": None, "id": 1},
            {"success": "false", "id": 2},
            {"success": 0, "id": 3},
            {"success": False, "id": 4},
        ]
        assert [r["id"] for r in filter_failed_records(records)] == [1, 2, 3]


class TestDeduplicateTrajectoriesFiltersFailed:
    def test_failed_record_filtered_before_dedup(self, tmp_path: Path):
        # 模拟 3 条记录：成功的 2 条（前缀关系）+ 失败的 1 条
        traj = tmp_path / "traj.jsonl"
        rows = [
            # success=False，要被过滤
            {
                "success": False,
                "request_time": 100,
                "request_body": {"messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "BAD"},
                ]},
            },
            # 成功的短上下文
            {
                "success": True,
                "request_time": 200,
                "request_body": {"messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                ]},
            },
            # 成功的更长上下文（应作为最长扩展保留）
            {
                "success": True,
                "request_time": 300,
                "request_body": {"messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                    {"role": "user", "content": "u3"},
                ]},
            },
        ]
        traj.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        result = deduplicate_trajectories(traj)

        # 失败行直接丢；成功短的被成功长的作为前缀覆盖；只剩最长那条
        assert len(result) == 1
        assert result[0]["request_time"] == 300
        assert result[0]["success"] is True
