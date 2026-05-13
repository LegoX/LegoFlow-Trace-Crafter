from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_data_process.utils import load_jsonl, save_jsonl


class TestSaveJsonl:
    def test_writes_records(self, tmp_path):
        output = tmp_path / "out.jsonl"
        records = [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        ]
        save_jsonl(output, records)
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["version"] == "2.0.0"

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "sub" / "dir" / "out.jsonl"
        save_jsonl(output, [{"not_im": True}])
        assert output.exists()

    def test_non_im_records_passthrough(self, tmp_path):
        output = tmp_path / "out.jsonl"
        records = [{"key": "value", "number": 42}]
        save_jsonl(output, records)
        lines = output.read_text().strip().split("\n")
        parsed = json.loads(lines[0])
        assert parsed == {"key": "value", "number": 42}


class TestLoadJsonl:
    def test_reads_valid_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        records = [
            {"messages": [{"role": "user", "content": "hi"}], "key": "val"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records))
        result = load_jsonl(f)
        assert len(result) == 1

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        content = '{"messages": [{"role": "user", "content": "hi"}]}\n\n{"messages": [{"role": "user", "content": "bye"}]}\n'
        f.write_text(content)
        result = load_jsonl(f)
        assert len(result) == 2

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        content = '{"messages": [{"role": "user", "content": "hi"}]}\nnot json\n'
        f.write_text(content)
        result = load_jsonl(f)
        assert len(result) == 1

    def test_expands_panguml_compat_fields(self, tmp_path):
        f = tmp_path / "data.jsonl"
        record = {
            "version": "2.0.0",
            "meta_info": {
                "teacher": "glm-5-thinking",
                "unique_info": {
                    "_instance_id": "owner__repo-1",
                    "_agent_type": "main",
                    "_score": {"composite_score": 0.7},
                },
            },
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        f.write_text(json.dumps(record) + "\n")
        result = load_jsonl(f)
        assert result[0]["_instance_id"] == "owner__repo-1"
        assert result[0]["_agent_type"] == "main"
        assert result[0]["_score"] == {"composite_score": 0.7}
