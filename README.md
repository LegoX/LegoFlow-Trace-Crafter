# swe_data_process

Standalone Python package for converting raw SWE-bench agent trajectory data into LF-format JSON files for SFT training.

## Overview

This repository is self-contained: install it from this directory, run converters as `python -m swe_data_process...`, and write IM/LF outputs wherever your downstream training workflow expects them.

The data processing workflow converts raw trajectories to intermediate IM format (+ auto-scored), then to LF format. Dataset registration, experiment tracking, and model training are outside the scope of this repository.

The core data conversion follows this pipeline:

```
Raw trajectories (per-scaffold format)
  -> Intermediate "IM" format (OpenAI-style messages with tool_calls, JSONL)
  -> Trajectory quality scoring (rule_score.py, auto-invoked by converters; optionally llm_score.py)
  -> LLaMA-Factory "LF" format (ShareGPT-style messages, JSON array)
```

The output LF JSON can be consumed by downstream SFT training systems such as LLaMA-Factory.

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
│       ├── llm_checklist_score.py       # OctoBench-aligned dynamic checklist LLM scoring (optional)
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
│   ├── data_format_requirement_panguml_v2.md  # Data format specification
│   ├── rule_score_details.md             # Rule-based scoring framework (v5) reference
│   ├── llm_score_details.md              # LLM-as-judge scoring framework reference
│   └── llm_checklist_score_details.md    # Checklist-based LLM scoring reference
├── pyproject.toml                        # Package metadata (pip install -e .)
```

## Usage

### Environment

```bash
cd /path/to/swe_data_process
conda create -n swelf python=3.12 -y
conda activate swelf

pip install -e '.[llm]'
```

`pip install -e .` registers this repository as the `swe_data_process` package so every script can `import swe_data_process.*` regardless of working directory. Dependencies listed in `pyproject.toml` are installed automatically. Use `pip install -e '.[llm]'` when running optional LLM scoring.

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

Most converters default `--exclude-repos-file` to this repo's `artifacts/excluded_repos.txt` to filter out reference benchmark repos (pass `--exclude-repos-file ""` to disable). `--max-instances` defaults to no limit; pass a positive integer to cap.

### Repo Filtering

为避免训练数据与评测数据集（SWE-bench_Verified、SWE-bench_Pro、SWE-bench_Multilingual）在 repo 维度上有交叉，所有转换脚本默认启用 repo 过滤。

默认排除列表已随仓库提交在 `artifacts/excluded_repos.txt`。如需使用自定义列表，创建一个每行一个 `owner/repo` 的文本文件并通过 `--exclude-repos-file` 指定：

```bash
conda activate swelf
python -m swe_data_process.<subpackage>.convert_<scaffold>_<source>_to_im \
    --job-dir <input> \
    --lf-output <output> \
    --exclude-repos-file /path/to/excluded_repos.txt
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
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_001", "content": "..."}
  ]
}
```

- IM 输出现在遵循 PangUML v2：顶层固定为 `version` / `meta_info` / `tools` / `messages`。
- 旧内部字段 `think_mode`、`pseudo_turns` 不再持久化到 IM JSONL；`_instance_id`、`_agent_type`、`_score` 会写入 `meta_info.unique_info`。读取 JSONL 时，`load_jsonl()` 会自动把这三个兼容字段展开回旧接口，供打分和 LF 转换继续使用。
- `assistant.reasoning_content` 会始终保留；无思维链时写成空字符串。工具调用参数 `function.arguments` 会统一序列化为 JSON 字符串。

### Validation Rules

Role ordering enforced by `check_roles` in `utils.py`:
- First non-system message must be `user`, last must be `assistant`
- After `assistant`: only `tool` or `user`
- After `tool`: only `tool` or `assistant`
- After `user`: only `assistant`