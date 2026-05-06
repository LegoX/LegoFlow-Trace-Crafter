# swe_data_process

Convert raw SWE-bench agent trajectory data into LF-format JSON files for SFT training.

## Overview

The data processing workflow converts raw trajectories to intermediate IM format (+ auto-scored), then to LF format. Dataset registration, experiment tracking, and model training are handled by the `sft-train` block.

The core data conversion follows this pipeline:

```
Raw trajectories (per-scaffold format)
  -> Intermediate "IM" format (OpenAI-style messages with tool_calls, JSONL)
  -> Trajectory quality scoring (rule_score.py, auto-invoked by converters; optionally llm_score.py)
  -> LLaMA-Factory "LF" format (ShareGPT-style messages, JSON array)
```

The output LF JSON is used as input to the `sft-train` block.

## Supported Scaffolds

| Scaffold | Directory | Sources |
|----------|-----------|---------|
| Claude Code | `src/swe_data_process/claudecode_opencode/` | jierun, chaofan |
| OpenCode | `src/swe_data_process/claudecode_opencode/` | jierun, chaofan |
| OpenHands | `src/swe_data_process/openhands/` | chaofan |
| OpenHands SDK | `src/swe_data_process/openhands/` | jierun, chaofan |
| Terminus2 | `src/swe_data_process/terminus2/` | jierun, chaofan |

## Project Structure

```
swe_data_process/
├── src/
│   └── swe_data_process/                # Python package (pip install -e .)
│       ├── __init__.py
│       ├── utils.py                     # Shared utilities (token stats, role validation, IM->LF conversion)
│       ├── rule_score.py                # Rule-based trajectory quality scoring (v5, auto-invoked by converters)
│       ├── llm_score.py                 # LLM-as-judge trajectory quality scoring (v2, optional)
│       ├── llm_client.py                # OpenAI-compatible LLM API client with retry and concurrency
│       ├── claudecode_opencode/
│       │   ├── convert_cc_jierun_to_im.py    # Claude Code (jierun) -> IM
│       │   ├── convert_cc_chaofan_to_im.py   # Claude Code (chaofan) -> IM
│       │   ├── convert_oc_jierun_to_im.py    # OpenCode (jierun) -> IM
│       │   ├── convert_oc_chaofan_to_im.py   # OpenCode (chaofan) -> IM
│       │   ├── extract_and_deduplicate_jsonl.py  # Trajectory dedup (mini-vela prefix semantics)
│       │   ├── convert_jsonl_to_openai.py    # Raw JSONL -> OpenAI message format (library module)
│       │   └── analyze_trajectories.py       # Batch dedup + tool error analysis
│       ├── openhands/
│       │   ├── common.py                     # Content extraction & tool call normalization
│       │   ├── convert_openhands_chaofan_to_im.py
│       │   ├── convert_openhands_sdk_chaofan_to_im.py
│       │   └── convert_openhands_sdk_jierun_to_im.py
│       └── terminus2/
│           ├── common.py                     # Analysis/Plan splitting, command normalization
│           ├── convert_terminus2_chaofan_to_im.py
│           └── convert_terminus2_jierun_to_im.py
├── artifacts/
│   └── excluded_repos.txt                # Generated list of repos to exclude (from reference datasets)
├── docs/
│   ├── data_format_requirement.md        # Data format specification
│   ├── rule_score_details.md             # Rule-based scoring framework (v5) reference
│   └── llm_score_details.md              # LLM-as-judge scoring framework (v1) reference
├── pyproject.toml                        # Package metadata (pip install -e .)
```

## Usage

### Environment

```bash
conda activate swelf
pip install -e repos/swe_data_process
```

`pip install -e .` registers the repo as the `swe_data_process` package so every script can `import swe_data_process.*` regardless of working directory. Dependencies listed in `pyproject.toml` are installed automatically.

### Running Converters

Each converter is run as a Python module via the installed package. General pattern:

```bash
python -m swe_data_process.<subpackage>.convert_{scaffold}_{source}_to_im \
    --job-dir <input> --lf-output <output> --max-instances 1000
```

CLI arguments vary by script. Full script matrix:

| 来源 | 脚手架 | 脚本路径 | CLI 参数 |
|------|--------|---------|----------|
| jierun | openhands-sdk | `openhands/convert_openhands_sdk_jierun_to_im.py` | `--job-dir`, `--trajs-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| jierun | claude-code | `claudecode_opencode/convert_cc_jierun_to_im.py` | `--job-dir`, `--trajs-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| jierun | open-code | `claudecode_opencode/convert_oc_jierun_to_im.py` | `--job-dir`, `--trajs-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| jierun | terminus2 | `terminus2/convert_terminus2_jierun_to_im.py` | `--job-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| chaofan | openhands | `openhands/convert_openhands_chaofan_to_im.py` | `--source-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| chaofan | claude-code | `claudecode_opencode/convert_cc_chaofan_to_im.py` | `--source-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| chaofan | open-code | `claudecode_opencode/convert_oc_chaofan_to_im.py` | `--source-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| chaofan | terminus2 | `terminus2/convert_terminus2_chaofan_to_im.py` | `--source-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |
| chaofan | openhands-sdk | `openhands/convert_openhands_sdk_chaofan_to_im.py` | `--source-dir`, `--im-output`, `--lf-output`, `--max-instances`, `--exclude-repos-file` |

Most converters default `--exclude-repos-file` to `artifacts/excluded_repos.txt` to filter out reference benchmark repos (pass `--exclude-repos-file ""` to disable). `--max-instances` defaults to no limit; pass a positive integer to cap.

### Repo Filtering

为避免训练数据与评测数据集（SWE-bench_Verified、SWE-bench_Pro、SWE-bench_Multilingual）在 repo 维度上有交叉，所有转换脚本默认启用 repo 过滤。

生成排除列表（只需运行一次）：

```bash
conda activate swelf
python scripts/generate_excluded_repos.py
```

禁用过滤：传 `--exclude-repos-file ""` 即可。

### Auto Scoring

所有转换脚本在生成 IM 数据后会自动调用 `rule_score.py` 中的 `score_dataset()` 对每条轨迹打分（v5 quality scoring framework，5 组 10 个子指标）。

打分公式：`Score = 0.20*Efficiency + 0.15*Style + 0.25*ToolMastery + 0.25*Completion + 0.15*Precision`

如需单独对已有 IM 文件打分：`python -m swe_data_process.rule_score --input <im.jsonl>`

### Trajectory Analysis

Deduplicate a folder of JSONL trajectories and report tool call error statistics:

```bash
python -m swe_data_process.claudecode_opencode.analyze_trajectories -i /path/to/trajectory/folder
```

## Data Formats

### Intermediate (IM) Format

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "..."}
  ],
  "tools": [...],
  "pseudo_turns": null,
  "think_mode": "slow",
  "_instance_id": "owner__repo-123",
  "_agent_type": "main",
  "_score": {"composite_score": 0.72, "efficiency_score": 0.68, "style_score": 0.65, "...": "..."}
}
```

- `think_mode`: `"slow"` when `reasoning_content` is present (chain-of-thought), `"fast"` otherwise.
- `_instance_id` and `_agent_type`: set by CC/OC converters. CC/OC instances may produce multiple records: one main agent (has Edit/Write tools) and zero or more subagents (read-only context-gatherer).
- `_score`: auto-populated by `rule_score.py` during conversion (v5 framework, 5 dimensions / 10 sub-metrics). Optionally augmented by `llm_score.py` (LLM-as-judge, 5 categories / 15 checks). For subagent records, `_score` is `null`. See `docs/rule_score_details.md` and `docs/llm_score_details.md`. The LF format also carries `_score`, `_instance_id`, and `_agent_type` as top-level fields alongside `messages`.

### Validation Rules

Role ordering enforced by `check_roles` in `utils.py`:
- First non-system message must be `user`, last must be `assistant`
- After `assistant`: only `tool` or `user`
- After `tool`: only `tool` or `assistant`
- After `user`: only `assistant`