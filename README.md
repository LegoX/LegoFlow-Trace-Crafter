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

| Scaffold | Directory | Script |
|----------|-----------|--------|
| Claude Code | `src/swe_data_process/claudecode_opencode/` | `convert_cc_to_im.py` |
| OpenCode | `src/swe_data_process/claudecode_opencode/` | `convert_oc_to_im.py` |
| OpenHands SDK | `src/swe_data_process/openhands/` | `convert_openhands_sdk_to_im.py` |
| Terminus2 | `src/swe_data_process/terminus2/` | `convert_terminus2_to_im.py` |

## Project Structure

```
swe_data_process/
├── src/
│   └── swe_data_process/                # Python package (pip install -e .)
│       ├── __init__.py
│       ├── utils.py                     # Shared utilities (token stats, role validation, IM->LF conversion)
│       ├── rule_score.py                # Rule-based trajectory quality scoring (TQS V2, auto-invoked by converters)
│       ├── llm_score.py                 # LLM-as-judge trajectory quality scoring (v2, optional)
│       ├── llm_checklist_score.py       # OctoBench-aligned dynamic checklist LLM scoring (optional)
│       ├── llm_client.py                # OpenAI-compatible LLM API client with retry and concurrency
│       ├── claudecode_opencode/
│       │   ├── convert_cc_to_im.py           # Claude Code -> IM
│       │   ├── convert_cc_session_to_im.py   # Claude Code session format -> IM
│       │   ├── convert_dataclaw_to_im.py     # DataClaw format -> IM
│       │   ├── convert_oc_to_im.py           # OpenCode -> IM
│       │   ├── extract_and_deduplicate_jsonl.py  # Trajectory dedup (mini-vela prefix semantics)
│       │   ├── convert_jsonl_to_openai.py    # Raw JSONL -> OpenAI message format (library module)
│       │   └── analyze_trajectories.py       # Batch dedup + tool error analysis
│       ├── openhands/
│       │   ├── common.py                     # Content extraction & tool call normalization
│       │   └── convert_openhands_sdk_to_im.py  # OpenHands SDK -> IM (unified)
│       └── terminus2/
│           ├── common.py                     # Analysis/Plan splitting, command normalization
│           └── convert_terminus2_to_im.py    # Terminus2 -> IM (unified)
├── artifacts/
│   ├── excluded_repos.txt                # Generated list of repos to exclude (from reference datasets)
│   ├── cc_im.jsonl                       # Example IM output with auto TQS V2 rule scores
│   ├── cc_im_rule_scored.jsonl           # Example after standalone rule_score.py
│   ├── cc_im_llm_scored.jsonl            # Scoring example: after llm_score.py
│   └── cc_im_llm_checklist_scored.jsonl  # Scoring example: after llm_checklist_score.py
├── docs/
│   ├── assets/                           # Diagrams referenced by the docs
│   ├── data_format_requirement_panguml_v2.md  # Data format specification
│   ├── data_filtering_strategy.md        # End-to-end filtering / scoring / sampling funnel
│   ├── rule_score_details.md             # Rule-based scoring framework (TQS V2) reference
│   ├── llm_score_details.md              # LLM-as-judge scoring framework reference
│   └── llm_checklist_score_details.md    # Checklist-based LLM scoring reference
├── tests/                            # Unit tests (pytest)
│   ├── conftest.py                   # Shared fixtures
│   ├── test_utils_validation.py      # check_roles, check_reasoning_content
│   ├── test_utils_detection.py       # is_im_record, detect_agent_type, infer_think_mode
│   ├── test_utils_normalization.py   # Normalization pipeline functions
│   ├── test_utils_filtering.py       # Repo exclusion, score-based filtering
│   ├── test_utils_io.py             # load_jsonl, save_jsonl
│   ├── test_convert_jsonl_to_openai.py  # Claude Code/OpenCode converter
│   ├── test_openhands_common.py      # OpenHands helpers
│   ├── test_terminus2_common.py      # Terminus2 helpers
│   ├── test_rule_score.py           # Scoring functions, scaffold detection
│   └── test_rule_score_multilang.py # Multi-language TVR/FEC regression suite
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

### Running Tests

```bash
pip install -e '.[test]'
pytest tests/ -v
```

268 unit tests covering utils, converters, and scoring. No network, GPU, or LLM calls required.

### Running Converters

Each converter is run as a Python module via the installed package. General pattern:

```bash
python -m swe_data_process.<subpackage>.convert_{scaffold}_to_im \
    --job-dir <input> \
    --im-output <im.jsonl> \
    --lf-output <lf.json> \
    --tokenizer-name <model-or-tokenizer> \
    --max-instances 1000
```

CLI arguments vary by script. Full script matrix:

| 脚手架 | 脚本路径 | CLI 参数 |
|--------|---------|----------|
| claude-code | `claudecode_opencode/convert_cc_to_im.py` | `--job-dir`, `--im-output`, `--lf-output`, `--tokenizer-name`, `--max-instances`, `--exclude-repos-file`, `--reasoning-check-mode`, `--reasoning-content-ratio-threshold` |
| open-code | `claudecode_opencode/convert_oc_to_im.py` | `--job-dir`, `--im-output`, `--lf-output`, `--tokenizer-name`, `--max-instances`, `--exclude-repos-file`, `--reasoning-check-mode`, `--reasoning-content-ratio-threshold` |
| openhands-sdk | `openhands/convert_openhands_sdk_to_im.py` | `--job-dir`, `--im-output`, `--lf-output`, `--tokenizer-name`, `--max-instances`, `--exclude-repos-file`, `--reasoning-check-mode`, `--reasoning-content-ratio-threshold` |
| terminus2 | `terminus2/convert_terminus2_to_im.py` | `--job-dir`, `--im-output`, `--lf-output`, `--tokenizer-name`, `--max-instances`, `--exclude-repos-file` |

Converter input/output paths are required: pass `--job-dir` or `--source-dir` plus `--im-output` and `--lf-output` explicitly. Most converters default `--exclude-repos-file` to this repo's `artifacts/excluded_repos.txt` to filter out reference benchmark repos (pass `--exclude-repos-file ""` to disable). `--max-instances` defaults to no limit; pass a positive integer to cap.

Converters accept `--tokenizer-name` for LLaMA-Factory ShareGPT-format JSON conversion. It defaults to `Qwen/Qwen3.5-35B-A3B`; set it to the tokenizer/model name expected by the downstream training model.

For converters with reasoning checks, `--reasoning-check-mode` defaults to `adaptive`: slow trajectories are kept when the ratio of checked assistant turns with non-empty `reasoning_content` is at least `--reasoning-content-ratio-threshold` (default `0.2`). `strict` requires every checked assistant turn to contain non-empty `reasoning_content`.

### Repo Filtering

为避免训练数据与评测数据集（SWE-bench_Verified、SWE-bench_Pro、SWE-bench_Multilingual）在 repo 维度上有交叉，所有转换脚本默认启用 repo 过滤。

默认排除列表已随仓库提交在 `artifacts/excluded_repos.txt`。如需使用自定义列表，创建一个每行一个 `owner/repo` 的文本文件并通过 `--exclude-repos-file` 指定：

```bash
conda activate swelf
python -m swe_data_process.<subpackage>.convert_<scaffold>_to_im \
    --job-dir <input> \
    --im-output <im.jsonl> \
    --lf-output <lf.json> \
    --exclude-repos-file /path/to/excluded_repos.txt
```

禁用过滤：传 `--exclude-repos-file ""` 即可。

### Auto Scoring

所有转换脚本在生成 IM 数据后会自动调用 `rule_score.py` 中的 `score_dataset()` 对每条 main agent 轨迹打分。当前规则打分使用 TQS V2：以 fail-soft 加权方式聚合 `SUB`、`STP`、`TVR`、`FEC`、`DPI` 五个核心组件，并额外输出 `OEC`、`IAC`、`PED`、`PSN`、`TTE`、`SCP` 等诊断指标。

打分公式：`composite_score = Σ(weight_i × transformed(component_i)) / Σ(weight_i)`，其中非零权重为 `SUB=0.33`、`STP=0.27`、`TVR=0.23`、`FEC=0.10`、`DPI=0.07`。subagent 记录会保留在输出中，但 `_score` 为 `null`。

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

Reasoning validation is enforced by `check_reasoning_content` in `utils.py`: `fast` trajectories always pass; `slow` trajectories use either strict per-assistant-turn reasoning checks or adaptive ratio checks depending on the converter's `--reasoning-check-mode`.
