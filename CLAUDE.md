# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a standalone repository for converting raw SWE-bench agent trajectory data into LF-format JSON files ready for SFT training. The pipeline reads trajectories from different agent scaffolds (Claude Code, OpenCode, OpenHands SDK, Terminus2), converts them to an intermediate OpenAI-like format (IM), applies automatic quality scoring, then produces LLaMA-Factory sharegpt-format JSON. Dataset registration, experiment tracking, and model training are outside this repository's scope.

## Running Scripts

**Environment**: `conda activate swelf`

**First-time setup**: install the repo as an editable package so converter scripts can `import swe_data_process.*` without any `sys.path` hacks:

```bash
cd /path/to/swe_data_process
conda activate swelf
pip install -e '.[llm]'
```

All converter scripts are run as Python modules via the installed package. Treat the current directory as the repository root; input and output paths may be absolute paths or paths relative to this repository:

```bash
conda activate swelf
python -m swe_data_process.<subpackage>.<script> \
    --job-dir <input> --lf-output <output> --max-instances 1000 \
    --exclude-repos-file artifacts/excluded_repos.txt  # optional, defaults to this repo path
```

CLI args vary by script — see `README.md` for the full matrix of scripts and their arguments. Scripts also have hardcoded defaults that work without arguments.

The `--exclude-repos-file` flag defaults to `artifacts/excluded_repos.txt` in this repo, enabling reference-repo filtering to exclude instances from evaluation benchmark repos. Pass `--exclude-repos-file ""` to disable.

There are no linting or build steps.

## Running Tests

```bash
pip install -e '.[test]'
pytest tests/ -v
```

Tests cover: utils (validation, normalization, detection, filtering, I/O), converter modules (claudecode_opencode, openhands, terminus2), and rule-based scoring. All tests are pure unit tests — no network, GPU, or LLM calls required.

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
  - **`llm_checklist_score.py`** — OctoBench-aligned dynamic checklist LLM scoring module. It generates per-instance binary checklist items from the user question, system prompt, and tool definitions, then scores the full trajectory and writes ISR/CSR-style results into `_score`.
  - **`llm_client.py`** — OpenAI-compatible LLM API client with retry, concurrency control, and token/cost tracking. Used by `llm_score.py` and `llm_checklist_score.py`.
  - **`claudecode_opencode/`** — Converters for Claude Code (`convert_cc_to_im.py`) and OpenCode (`convert_oc_to_im.py`) trajectories. Also contains `convert_cc_session_to_im.py` (session format), `convert_dataclaw_to_im.py` (DataClaw format), `extract_and_deduplicate_jsonl.py` (deduplicates raw JSONL trajectories, aligned with mini-vela prefix-based dedup semantics) and `convert_jsonl_to_openai.py` (library module that converts raw records to OpenAI message format) — used as libraries by the `cc`/`oc` converter scripts — and `analyze_trajectories.py` (batch dedup + tool-call error analysis).
  - **`openhands/`** — Converter for OpenHands SDK trajectories (`convert_openhands_sdk_to_im.py`). `common.py` has helpers for extracting text from content blocks, normalizing tool calls, and `OPENHANDS_SDK_TOOLS` constant (complete OpenAI function-calling format for SDK tools).
  - **`terminus2/`** — Converter for Terminus2 trajectories (`convert_terminus2_to_im.py`). `common.py` handles the Terminus2-specific format: splitting `Analysis:/Plan:` messages, normalizing commands from tool calls, and converting steps to user/assistant message pairs.

- **`artifacts/`** — Generated data files:
  - `excluded_repos.txt` — Generated list of 64 `owner/repo` entries (one per line) from the reference datasets. Used by `--exclude-repos-file`.
  - `cc_im.jsonl` — Scoring example: base IM output (Claude Code converter, before optional LLM scoring).
  - `cc_im_rule_scored.jsonl` — Scoring example: after `rule_score.py` (auto-invoked by converter; adds `composite_score`, sub-indicator scores).
  - `cc_im_llm_scored.jsonl` — Scoring example: after `llm_score.py` (adds `llm_composite_score`, `llm_detailed_results`, `llm_checklist_version`).
  - `cc_im_llm_checklist_scored.jsonl` — Scoring example: after `llm_checklist_score.py` (adds `llm_checklist_csr`, `llm_checklist_category_scores`, `llm_checklist_definition`).

### Naming conventions

Each converter script follows the pattern: `convert_{scaffold}_to_im.py`
- scaffold: `cc` (Claude Code), `oc` (OpenCode), `openhands_sdk`, `terminus2`

### Key data format

The intermediate (IM) format per record:
```json
{
  "version": "2.0.0",
  "meta_info": {
    "teacher": "glm-5-thinking",
    "query_source": "synthesized",
    "response_generate_time": "2026-05-11",
    "response_update_time": "2026-05-11",
    "owner": "00000000",
    "language": "en",
    "category": "code",
    "rounds": 2,
    "unique_info": {
      "_instance_id": "owner__repo-123",
      "_agent_type": "main",
      "_score": {"composite_score": 0.72}
    }
  },
  "tools": [...],
  "messages": [{"role": "user/assistant/tool", "content": "...", "reasoning_content": "...", "tool_calls": [...]}]
}
```

IM 输出统一写成 PangUML v2 顶层结构。旧内部字段 `think_mode`、`pseudo_turns` 不再持久化；`_instance_id`、`_agent_type`、`_score` 写入 `meta_info.unique_info`。

`load_jsonl()` 会在读入时自动把 `_instance_id`、`_agent_type`、`_score` 展开回旧接口，因此 `rule_score.py`、`llm_score.py`、LF 转换等下游逻辑可以继续按原方式工作。

`assistant.reasoning_content` 始终保留；无思维链时写成空字符串。工具调用参数 `function.arguments` 统一为 JSON 字符串；`tool.tool_call_id` 会在可推断时对齐到对应的 `assistant.tool_calls[*].id`。

`filter_by_score_bundled(records, min_score)` filters records by score while keeping entire instances together: if the main agent's score meets the threshold, all records (main + subagent) from that instance are kept.

### Validation rules (in `src/swe_data_process/utils.py`)

Role ordering enforced by `check_roles`: first non-system message must be `user`, last must be `assistant`. After `assistant` only `tool` or `user`; after `tool` only `tool` or `assistant`; after `user` only `assistant`.