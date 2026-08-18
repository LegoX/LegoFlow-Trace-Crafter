from __future__ import annotations

import json
from pathlib import Path

from legoflow_trace_crafter.claudecode_opencode.extract_and_deduplicate_jsonl import (
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
        # Older logger records without a success field count as successful.
        records = [{"id": 1}, {"success": False, "id": 2}, {"id": 3}]
        assert [r["id"] for r in filter_failed_records(records)] == [1, 3]

    def test_truthy_non_false_kept(self):
        # Drop only explicit False; conservatively retain values such as None
        # and the string "false".
        records = [
            {"success": None, "id": 1},
            {"success": "false", "id": 2},
            {"success": 0, "id": 3},
            {"success": False, "id": 4},
        ]
        assert [r["id"] for r in filter_failed_records(records)] == [1, 2, 3]


class TestDeduplicateTrajectoriesFiltersFailed:
    def test_failed_record_filtered_before_dedup(self, tmp_path: Path):
        # Simulate two successful records with a prefix relationship and one
        # failed record.
        traj = tmp_path / "traj.jsonl"
        rows = [
            # success=False must be filtered.
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
            # Successful shorter context.
            {
                "success": True,
                "request_time": 200,
                "request_body": {"messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                ]},
            },
            # Successful longer context, retained as the longest extension.
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

        # The failed row is dropped, and the longer successful row supersedes
        # the shorter prefix, leaving only the longest trajectory.
        assert len(result) == 1
        assert result[0]["request_time"] == 300
        assert result[0]["success"] is True
