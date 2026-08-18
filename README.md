# LegoFlow-Trace-Crafter

LegoFlow-Trace-Crafter (`legoflow-trace-crafter`) converts software-engineering agent trajectories into:

- intermediate message (IM) JSONL with normalized messages, tool calls, metadata, and trajectory scores;
- LLaMA-Factory-oriented LF JSON with a statistics sidecar.

Dataset registration, training, and downstream sample selection are outside this repository.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional dependencies when needed:

```bash
pip install -e '.[test]'
pip install -e '.[llm]'
```

## Supported inputs

The package includes converters for:

- Claude Code Harbor jobs: `legoflow_trace_crafter.claudecode_opencode.convert_cc_to_im`
- OpenCode Harbor jobs: `legoflow_trace_crafter.claudecode_opencode.convert_oc_to_im`
- OpenHands SDK Harbor jobs: `legoflow_trace_crafter.openhands.convert_openhands_sdk_to_im`
- Terminus2 Harbor jobs: `legoflow_trace_crafter.terminus2.convert_terminus2_to_im`
- Claude Code session JSONL: `legoflow_trace_crafter.claudecode_opencode.convert_cc_session_to_im`
- DataClaw conversation JSONL: `legoflow_trace_crafter.claudecode_opencode.convert_dataclaw_to_im`

Each converter writes both IM and LF output. Converter-specific arguments are available through `--help`.

## Convert a Harbor job

For example, convert Claude Code trajectories with:

```bash
python -m legoflow_trace_crafter.claudecode_opencode.convert_cc_to_im \
  --job-dir <job-directory> \
  --im-output outputs/trajectories.im.jsonl \
  --lf-output outputs/trajectories.lf.json \
  --tokenizer-name <tokenizer-name>
```

The Harbor-job converters accept `--instance-status resolved|unresolved|all` and default to `resolved`. They also accept `--max-instances`.

Repository exclusion is enabled by default where supported. The maintained list is:

`artifacts/excluded_repos.txt`

Provide another one-repository-per-line file with `--exclude-repos-file <path>`, or pass `--exclude-repos-file ""` to disable this filter.

All converters apply TQS V2 rule scoring before writing output. Main-agent records receive an `_score` object; subagent records retain `_score: null`.

## Score existing IM data

Run deterministic rule scoring:

```bash
python -m legoflow_trace_crafter.rule_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.rule-scored.jsonl
```

Optional LLM scoring uses an OpenAI-compatible API and the `llm` extra. Set `OPENAI_API_KEY` and, when required, `OPENAI_BASE_URL` in the environment.

```bash
python -m legoflow_trace_crafter.llm_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.llm-scored.jsonl

python -m legoflow_trace_crafter.llm_checklist_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.checklist-scored.jsonl
```

## Example artifacts

The tracked `artifacts/` directory contains four reference snapshots:

- `cc_im.jsonl`: normalized IM output with rule scores;
- `cc_im_rule_scored.jsonl`: standalone rule-scoring output;
- `cc_im_llm_scored.jsonl`: fixed-checklist LLM-scoring output;
- `cc_im_llm_checklist_scored.jsonl`: dynamic-checklist LLM-scoring output.

These files document the output schemas. Write newly generated data to an
ignored directory such as `outputs/`.

## Documentation

- [IM format](docs/im_format.md)
- [Conversion filtering and scoring](docs/data_filtering_strategy.md)
- [TQS V2 rule scoring](docs/rule_score_details.md)
- [Fixed-checklist LLM scoring](docs/llm_score_details.md)
- [Dynamic-checklist LLM scoring](docs/llm_checklist_score_details.md)

## Tests

```bash
pip install -e '.[test]'
pytest
```
