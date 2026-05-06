# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo converts raw SWE-bench agent trajectory data into LF-format JSON files ready for SFT training. The pipeline reads trajectories from different agent scaffolds (Claude Code, OpenCode, OpenHands, OpenHands SDK, Terminus2) and data sources (jierun, chaofan), converts them to an intermediate OpenAI-like format (IM), applies automatic quality scoring, then produces LLaMA-Factory sharegpt-format JSON. Dataset registration and model training happen in the `sft-train` block upstream.

## Running Scripts

**Environment**: `conda activate swelf`

**First-time setup**: install the repo as an editable package so converter scripts can `import swe_data_process.*` without any `sys.path` hacks:

```bash
conda activate swelf
pip install -e repos/swe_data_process
```

All converter scripts are run as Python modules via the installed package (data conversion only — dataset registration, experiment tracking, and training are handled by the `sft-train` block):

```bash
conda activate swelf
python -m swe_data_process.<subpackage>.<script> \
    --job-dir <input> --lf-output <output> --max-instances 1000 \
    --exclude-repos-file artifacts/excluded_repos.txt  # optional
```

CLI args vary by script — see `README.md` for the full matrix of scripts and their arguments. Scripts also have hardcoded defaults that work without arguments.

The `--exclude-repos-file` flag defaults to `artifacts/excluded_repos.txt` in the repo, enabling reference-repo filtering to exclude instances from evaluation benchmark repos. Pass `--exclude-repos-file ""` to disable.

There are no tests, linting, or build steps.

## Architecture

### Data flow

```
Raw trajectories (per-scaffold format)
  -> Intermediate "IM" format (OpenAI-style messages with tool_calls, JSONL)
  -> Trajectory quality scoring (rule_score.py, auto-invoked by converters; optionally llm_score.py)
  -> LLaMA-Factory "LF" format (sharegpt-style messages, JSON array)
```

### Directory structure

- **`src/swe_data_process/`** — The Python package (installed via `pip install -e .`):
  - **`utils.py`** — Shared utilities used by all converters: JSON I/O (`load_json`), token statistics (`print_lf_token_stats`), role validation (`check_roles`), reasoning content validation (`check_reasoning_content`), IM-to-LF conversion (`convert_json_to_lf_format`), I/O helpers (`save_jsonl`, `save_lf_json`), instance filtering (`get_resolved_instances` reads Harbor result.json), **reference-repo exclusion** (`load_exclusion_patterns`, `build_repo_exclusion_patterns`, `filter_instance_ids_by_repo`, `filter_paths_by_repo`, etc.), **main/subagent detection** (`detect_agent_type`, `tag_instance_records`, `filter_by_score_bundled`), and **path constants** (`REPO_ROOT`, `EXCLUDED_REPOS_FILE`).
  - **`rule_score.py`** — Rule-based trajectory quality scoring module (v5 framework). `score_dataset()` is auto-invoked by all converter scripts after IM generation. Scores are saved as `_score` field in both IM and LF output. For CC/OC datasets, only main agent records are scored; subagent records (`_agent_type == "subagent"`) get `_score: null`.
  - **`llm_score.py`** — LLM-as-judge trajectory quality scoring module (v2). Checklist-based evaluation using an LLM (5 categories / 15 checks, letters F-J). Complements rule-based scoring; results are merged into `_score` with `llm_` prefix.
  - **`llm_client.py`** — OpenAI-compatible LLM API client with retry, concurrency control, and token/cost tracking. Used by `llm_score.py`.
  - **`claudecode_opencode/`** — Converters for Claude Code (`convert_cc_*.py`) and OpenCode (`convert_oc_*.py`) trajectories. Also contains `extract_and_deduplicate_jsonl.py` (deduplicates raw JSONL trajectories, aligned with mini-vela prefix-based dedup semantics) and `convert_jsonl_to_openai.py` (library module that converts raw records to OpenAI message format) — used as libraries by the `cc`/`oc` converter scripts — and `analyze_trajectories.py` (batch dedup + tool-call error analysis).
  - **`openhands/`** — Converters for OpenHands trajectories. `common.py` has helpers for extracting text from content blocks, normalizing tool calls, `OPENHANDS_SDK_TOOLS` constant (complete OpenAI function-calling format for SDK tools), and `convert_chaofan_dataset()` (unified converter for chaofan-style completions directories, shared by both SDK and non-SDK chaofan scripts).
  - **`terminus2/`** — Converters for Terminus2 trajectories. `common.py` handles the Terminus2-specific format: splitting `Analysis:/Plan:` messages, normalizing commands from tool calls, and converting steps to user/assistant message pairs.

- **`artifacts/`** — Generated data files:
  - `excluded_repos.txt` — Generated list of 64 `owner/repo` entries (one per line) from the reference datasets. Used by `--exclude-repos-file`.

### Naming conventions

Each converter script follows the pattern: `convert_{scaffold}_{source}_to_im.py`
- scaffold: `cc` (Claude Code), `oc` (OpenCode), `openhands`/`openhands_sdk`, `terminus2`
- source: `jierun` or `chaofan` (different data providers with different raw formats)

### Key data format

The intermediate (IM) format per record:
```json
{
  "messages": [{"role": "user/assistant/tool", "content": "...", "reasoning_content": "...", "tool_calls": [...]}],
  "tools": [...],
  "pseudo_turns": null,
  "think_mode": "slow|fast",
  "_instance_id": "owner__repo-123",
  "_agent_type": "main|subagent",
  "_score": {"composite_score": 0.72, "efficiency_score": 0.68, "style_score": 0.65, ...}
}
```

`think_mode` is `"slow"` when `reasoning_content` is present (model used chain-of-thought), `"fast"` otherwise.

`_instance_id` and `_agent_type` are set by CC/OC converters via `tag_instance_records()`. CC/OC instances may produce multiple records after deduplication: one main agent (has Edit/Write tools) and zero or more subagents (read-only context-gatherer, no Edit/Write). `detect_agent_type()` distinguishes them by checking for write-capable tools. Non-CC/OC scaffolds do not set these fields.

`_score` is auto-populated by `rule_score.py` during conversion. For main agent records it contains `composite_score` (weighted sum), 5 group scores (`efficiency_score`, `style_score`, `tool_mastery_score`, `completion_score`, `precision_score`), and 10 sub-metrics (`a1_error_retry` through `e2_delete_then_modify`). Optionally augmented by `llm_score.py` (LLM-as-judge, 5 categories / 15 checks with `llm_` prefix). For subagent records, `_score` is `null`. The LF format also carries `_score`, `_instance_id`, and `_agent_type` as top-level fields alongside `messages`.

`filter_by_score_bundled(records, min_score)` filters records by score while keeping entire instances together: if the main agent's score meets the threshold, all records (main + subagent) from that instance are kept.

### Validation rules (in `src/swe_data_process/utils.py`)

Role ordering enforced by `check_roles`: first non-system message must be `user`, last must be `assistant`. After `assistant` only `tool` or `user`; after `tool` only `tool` or `assistant`; after `user` only `assistant`.