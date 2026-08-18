# TQS V2 Rule Scoring

`legoflow_trace_crafter.rule_score` assigns deterministic trajectory-quality metrics to IM records. All converters call the same scorer automatically, and it can also be run on an existing IM JSONL file.

Scoring does not filter records. Main-agent records receive an `_score` object; records tagged with `_agent_type: "subagent"` receive `_score: null`.

## Composite score

TQS V2 uses a fail-soft weighted average:

```text
composite_score =
    sum(weight_i * transform(component_i))
    / sum(weight_i for available components)
```

Components that return `null` are omitted from both the numerator and denominator.

The weighted components are:

- `SUB` / `sub_score`, weight `0.33`: submission completeness and late-trajectory quality.
- `STP` / `stp_score`, weight `0.27`: assistant-turn efficiency.
- `TVR` / `tvr_score`, weight `0.23`: test creation, execution, and final observed outcome.
- `FEC` / `fec_score`, weight `0.10`: file-edit concentration, transformed as `FEC^5`.
- `DPI` / `dpi_score`, weight `0.07`: dirty-pattern penalty, transformed as `DPI^3`.

All component and composite values are in `[0, 1]`.

## Weighted components

### SUB: submission completeness

The base completion signal depends on the detected scaffold:

- Claude Code and OpenCode require the trajectory to end with an assistant message that has no tool calls.
- OpenHands and OpenHands SDK recognize the `finish` tool. A final text summary can also receive partial or full credit based on completion language.
- Terminus2 recognizes `task_complete: true`.

When observations are available, errors in the final 30% reduce the base score. Running a test in the final 20% of assistant turns can add up to `0.15`, capped at `1.0`.

### STP: step efficiency

STP counts assistant messages:

```text
5 to 80 turns:       1.0
fewer than 5 turns:  turns / 5
200 or more turns:   0.0
81 to 199 turns:     decays toward zero
```

The score is record-local and does not use dataset-level step statistics.

### TVR: test verification

```text
TVR =
    0.3 * has_test_write
  + 0.3 * has_test_run
  + 0.4 * final_test_outcome
```

`final_test_outcome` is:

- `1.0` for a recognized passing summary;
- `0.6` when a test ran but its result is inconclusive;
- `0.3` for a recognized failure;
- `0.0` when no test run is detected.

Test-file and test-command detection covers common Python, Go, Rust, Java, JavaScript/TypeScript, C/C++, C#, PHP, Ruby, Perl, and build-tool conventions. Focused reproduction and verification scripts are also recognized.

### FEC: file-edit concentration

FEC measures repeated editing of the same files:

```text
mean_edits_per_file = edit_count / unique_edited_files
FEC = 1 - clip((mean_edits_per_file - 1) / 4, 0, 1)
```

One edit per file scores `1.0`; an average of five or more scores `0.0`. If no edit is detected, FEC is `null` and is omitted from the composite score.

Edits are detected from editor tools and supported shell-writing forms. Paths are normalized before counting.

### DPI: dirty-pattern penalty

DPI starts at `1.0` and subtracts penalties for:

- no detected successful write;
- a truncated or incomplete ending;
- long repeated action loops;
- repeated observation errors.

The result is clipped to `[0, 1]`.

## Diagnostic metrics

These values are included in `_score` but have zero composite weight:

- `oec_score`: observation entropy collapse.
- `iac_score`: consistency between stated intent and tool actions.
- `ped_score`: action and target diversity after an observed error.
- `psn_score`: whether the active file scope narrows during the later trajectory.
- `tte_score`: entropy of transitions between action types.
- `scp_score`: position of the first successful write; the preferred range is 20% to 50% of assistant turns.
- `reproduce_first`: `1.0` when a test or reproduction runs before the first non-test source edit, `0.0` otherwise, and `null` when no non-test source edit is detected.

Insufficient evidence produces `null` for diagnostics that cannot be computed.

## Scaffold and tool handling

The scorer detects `claudecode`, `opencode`, `openhands`, `openhands_sdk`, or `terminus2` from tool definitions and message structure. `--scaffold` can override detection.

Before scoring, known renamed OpenHands-style tools are canonicalized from their parameter schemas. Execution tools are primarily recognized from their arguments rather than their names, which keeps shell, editor, and task-management actions distinct.

Error handling separates:

- explicit tool-failure markers, which apply to every tool type;
- command-output failures, which apply only to execution tools.

Test assertion failures count as verification outcomes, not tool-call failures. Output matching strips ANSI escapes and inspects both the beginning and end of long observations.

## Output

A main-agent score has this shape:

```json
{
  "scaffold": "claudecode",
  "assistant_turns": 15,
  "total_tool_calls": 42,
  "oec_score": 0.81,
  "iac_score": 0.67,
  "dpi_score": 0.92,
  "ped_score": 0.76,
  "psn_score": 0.88,
  "tte_score": 0.54,
  "scp_score": 1.0,
  "sub_score": 1.0,
  "fec_score": 0.75,
  "stp_score": 1.0,
  "tvr_score": 0.7,
  "composite_score": 0.83,
  "reproduce_first": 1.0
}
```

Persisted IM places this object at `meta_info.unique_info._score`. The JSONL loader exposes it as top-level `_score` while processing.

## Command line

```bash
python -m legoflow_trace_crafter.rule_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.rule-scored.jsonl
```

Available options include:

- `--max-instances N`
- `--scaffold claudecode|opencode|openhands|openhands_sdk|terminus2`
- `--quiet`

If `--output` is omitted, the default is `<input-stem>_rule_scored.jsonl`.
