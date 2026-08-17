from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from swe_data_process.openhands.convert_openhands_sdk_to_im import convert_dataset
from swe_data_process.terminus2.common import to_lf_record
from swe_data_process.utils import (
    convert_json_to_lf_format,
    extract_instance_id_from_config,
    load_jsonl,
    load_task_metadata_from_trial,
    save_jsonl,
    tag_instance_records,
    to_im_v2_record,
)


@pytest.fixture
def harbor_trial(tmp_path):
    job_dir = tmp_path / "job"
    folder_name = "owner__repo-123__hash"
    trial_dir = job_dir / folder_name
    task_dir = tmp_path / "tasks" / "owner__repo-123"
    trial_dir.mkdir(parents=True)
    task_dir.mkdir(parents=True)

    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": str(task_dir)}}),
        encoding="utf-8",
    )
    (task_dir / "task.toml").write_text(
        """
[metadata]
author_name = "unknown"
difficulty = "easy"
category = "debugging"
tags = ["python", "library"]
""".strip(),
        encoding="utf-8",
    )
    return job_dir, folder_name


def test_loads_metadata_from_task_referenced_by_trial(harbor_trial):
    job_dir, folder_name = harbor_trial

    metadata = load_task_metadata_from_trial(job_dir, folder_name)

    assert metadata == {
        "author_name": "unknown",
        "difficulty": "easy",
        "category": "debugging",
        "tags": ["python", "library"],
    }
    assert extract_instance_id_from_config(job_dir, folder_name) == "owner__repo-123"


def test_resolves_relative_task_path_from_trial_directory(tmp_path):
    job_dir = tmp_path / "job"
    folder_name = "owner__repo-123__hash"
    trial_dir = job_dir / folder_name
    task_dir = trial_dir / "task"
    task_dir.mkdir(parents=True)
    (trial_dir / "config.json").write_text(
        json.dumps({"task": {"path": "task"}}),
        encoding="utf-8",
    )
    (task_dir / "task.toml").write_text(
        "[metadata]\ndifficulty = \"hard\"\n",
        encoding="utf-8",
    )

    assert load_task_metadata_from_trial(job_dir, folder_name) == {
        "difficulty": "hard"
    }


def test_missing_metadata_table_raises(harbor_trial):
    job_dir, folder_name = harbor_trial
    config = json.loads((job_dir / folder_name / "config.json").read_text())
    task_toml = Path(config["task"]["path"]) / "task.toml"
    task_toml.write_text("[task]\nname = \"example\"\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Missing \[metadata\]"):
        load_task_metadata_from_trial(job_dir, folder_name)


def test_tag_instance_records_attaches_metadata_to_all_records():
    records = [{"messages": []}, {"messages": []}]
    metadata = {"difficulty": "easy"}

    tag_instance_records(records, "owner__repo-123", metadata)

    assert all(record["_instance_metadata"] == metadata for record in records)


def test_metadata_round_trips_through_im(tmp_path):
    output = tmp_path / "im.jsonl"
    metadata = {"difficulty": "easy", "tags": ["python"]}
    records = [{
        "messages": [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Done"},
        ],
        "_instance_metadata": metadata,
    }]

    save_jsonl(output, records)

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["meta_info"]["unique_info"]["_instance_metadata"] == metadata
    assert load_jsonl(output)[0]["_instance_metadata"] == metadata


def test_metadata_is_copied_to_lf_from_im():
    metadata = {"difficulty": "easy", "tags": ["python"]}
    im_record = to_im_v2_record({
        "messages": [
            {"role": "user", "content": "Fix the bug"},
            {"role": "assistant", "content": "Done"},
        ],
        "_instance_metadata": metadata,
    })

    class DummyTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return (
                "<|im_start|>user\nFix the bug<|im_end|>"
                "<|im_start|>assistant\nDone<|im_end|>"
            )

    with patch(
        "swe_data_process.utils.AutoTokenizer.from_pretrained",
        return_value=DummyTokenizer(),
    ):
        lf_records, _ = convert_json_to_lf_format(
            [im_record],
            compute_token_stats=False,
        )

    assert lf_records[0]["_instance_metadata"] == metadata


def test_openhands_job_conversion_attaches_trial_metadata(harbor_trial):
    job_dir, folder_name = harbor_trial
    (job_dir / "result.json").write_text(
        json.dumps({
            "stats": {
                "evals": {
                    "batch": {
                        "reward_stats": {"reward": {"1.0": [folder_name]}}
                    }
                }
            }
        }),
        encoding="utf-8",
    )
    agent_dir = job_dir / folder_name / "agent"
    agent_dir.mkdir()
    logger_record = {
        "success": True,
        "request_body": {
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": "Fix the bug."},
            ],
            "tools": [],
            "model": "test-model",
        },
        "response_body": {
            "choices": [
                {"message": {"role": "assistant", "content": "Done."}}
            ]
        },
    }
    (agent_dir / "litellm-trajectory.jsonl").write_text(
        json.dumps(logger_record) + "\n",
        encoding="utf-8",
    )

    records = convert_dataset(job_dir)

    assert len(records) == 1
    assert records[0]["_instance_metadata"]["difficulty"] == "easy"
    assert records[0]["_instance_metadata"]["tags"] == ["python", "library"]


def test_terminus_lf_record_copies_metadata():
    metadata = {"difficulty": "easy"}

    record = to_lf_record({
        "messages": [{"role": "user", "content": "Fix it"}],
        "_instance_metadata": metadata,
    })

    assert record["_instance_metadata"] == metadata
