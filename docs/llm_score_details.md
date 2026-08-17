# Fixed-Checklist LLM Scoring

`swe_data_process.llm_score` evaluates IM trajectories with one fixed 15-item checklist and an OpenAI-compatible chat API. It complements deterministic [TQS V2 rule scoring](rule_score_details.md); its fields are merged into the same `_score` object with an `llm_` prefix.

## Checklist

The checklist contains three checks in each category:

- `F`, Problem Understanding: diagnosis depth, scope precision, and plan quality.
- `G`, Solution Quality: fix elegance, change minimality, and robustness.
- `H`, Reasoning Quality: reasoning coherence, hypothesis-driven work, and adaptability.
- `I`, Verification Rigor: reproduction, fix verification, and test quality.
- `J`, Efficiency: navigation efficiency, tool proficiency, and iteration economy.

The judge assigns an integer from `1` to `5` to every check. The embedded rubric defines `1` as a serious omission or failure and `5` as complete, high-quality performance.

## Aggregation

Each category is normalized to `[0, 1]`:

```text
category_score = (sum(check_scores) - number_of_checks) / (4 * number_of_checks)
```

With three checks per category, an all-`1` category scores `0.0` and an all-`5` category scores `1.0`.

The composite is the arithmetic mean of the five category scores:

```text
llm_composite_score = mean(F, G, H, I, J)
```

Scores are rounded to four decimal places.

## Processing behavior

For each main-agent record, the scorer:

1. Copies the trajectory and truncates long prompt content.
2. Serializes tool definitions and messages into the evaluation prompt.
3. Sends one temperature-zero judge request containing the fixed checklist.
4. Parses the returned JSON and clamps each check score to `1` through `5`.
5. Merges aggregate, detailed, and raw judge results into `_score`.

Tool results are limited to 5,000 characters. Assistant content and `reasoning_content` are each limited to 50,000 characters.

Records tagged as subagents are skipped. Records that already contain `llm_composite_score` are also skipped, which supports resumable runs. When an output file already exists, prior LLM score fields are restored by record position before pending records are scored.

A failed request or unparseable response leaves `llm_composite_score: null` and records an `llm_error` instead of aborting the entire dataset.

## Output fields

The scorer adds:

```json
{
  "llm_composite_score": 0.65,
  "llm_f_problem_understanding": 0.75,
  "llm_g_solution_quality": 0.6667,
  "llm_h_reasoning_quality": 0.5833,
  "llm_i_verification_rigor": 0.5,
  "llm_j_efficiency": 0.75,
  "llm_model": "gpt-4o",
  "llm_checklist_version": "v2",
  "llm_detailed_results": {
    "total_checks": 15,
    "total_score": 54,
    "total_max": 75,
    "by_category": {}
  },
  "llm_raw_response": {}
}
```

`llm_raw_response` retains the parsed per-check reasoning and scores. `llm_detailed_results` contains unnormalized totals by category.

Persisted IM stores `_score` under `meta_info.unique_info`; `load_jsonl()` exposes it at the top level while scoring.

## Command line

Install the optional API dependency and configure credentials in the environment:

```bash
pip install -e '.[llm]'
```

Set `OPENAI_API_KEY`. Set `OPENAI_BASE_URL` only when using a compatible non-default endpoint.

```bash
python -m swe_data_process.llm_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.llm-scored.jsonl \
  --model gpt-4o \
  --concurrency 10
```

Useful options:

- `--max-instances N`
- `--dry-run` to estimate prompt tokens and cost without calling the API
- `--no-json-mode` for endpoints that do not support `response_format=json_object`
- `--quiet`

If `--output` is omitted, the default is `<input-stem>_llm_scored.jsonl`.
