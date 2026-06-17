from __future__ import annotations

import json
from pathlib import Path

import pytest

from swe_data_process.utils import (
    get_instances_from_job_dir,
    get_resolved_instances_from_job_dir,
    load_jsonl,
    print_lf_token_stats_from_texts,
    save_jsonl,
)


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


class TestGetInstancesFromJobDir:
    def test_selects_instances_by_status(self, tmp_path):
        result = {
            "stats": {
                "evals": {
                    "batch-1": {
                        "reward_stats": {
                            "reward": {
                                "1.0": ["resolved-1__abc"],
                                "0.0": ["unresolved-1__def"],
                            }
                        }
                    },
                    "batch-2": {
                        "reward_stats": {
                            "reward": {
                                "1.0": ["resolved-2__ghi"],
                                "0.0": ["unresolved-2__jkl"],
                            }
                        }
                    },
                }
            }
        }
        (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")

        assert get_instances_from_job_dir(tmp_path) == [
            "resolved-1__abc",
            "resolved-2__ghi",
        ]
        assert get_resolved_instances_from_job_dir(tmp_path) == [
            "resolved-1__abc",
            "resolved-2__ghi",
        ]
        assert get_instances_from_job_dir(tmp_path, "unresolved") == [
            "unresolved-1__def",
            "unresolved-2__jkl",
        ]
        assert get_instances_from_job_dir(tmp_path, "all") == [
            "resolved-1__abc",
            "unresolved-1__def",
            "resolved-2__ghi",
            "unresolved-2__jkl",
        ]


class TestTokenStats:
    def test_includes_total_tokens(self):
        class DummyTokenizer:
            def __call__(self, texts, **kwargs):
                return {"length": [len(text.split()) for text in texts]}

        stats = print_lf_token_stats_from_texts(
            ["one two", "three four five"],
            [1, 2],
            token_batch_size=1,
            tokenizer=DummyTokenizer(),
        )

        assert stats["total_tokens"] == 5
