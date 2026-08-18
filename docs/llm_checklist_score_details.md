# Dynamic-Checklist LLM Scoring

`legoflow_trace_crafter.llm_checklist_score` generates a checklist from each trajectory's task context and then evaluates the trajectory against that checklist. The implementation labels this format `octobench-aligned-v2`.

This scorer uses an OpenAI-compatible chat API and writes `llm_checklist_` fields into the existing `_score` object.

## Workflow

For each main-agent record:

1. Extract the task question.
2. Read the first system message unless `--no-system-prompt` is set.
3. Summarize available tool names, descriptions, and parameter names.
4. Ask the checklist model for atomic, binary checks.
5. Ask the judge model to score every generated check against the full trajectory.
6. Calculate ISR, CSR, and per-category CSR values.

The generator prompt targets 15 to 35 checks. The parser accepts simpler tasks with fewer checks and caps normalized output at 40 checks.

The prompt suggests these source and evaluation categories:

- `user_query`
- `system_prompt`
- `tool_schema`
- `repo_policy`
- `implementation`
- `verification`
- `communication`

Empty categories may be omitted. Each check includes a stable ID, a description, a check type, and the evidence required to pass.

## Question and context extraction

`--question-field` can select an explicit top-level field. Without it, the scorer tries these fields in order:

`user_question`, `question`, `prompt`, `task`, `instruction`, `user_query`

If none contains text, it uses the first user message. A record with no extractable question is marked with an error.

Context limits are:

- question: 12,000 characters;
- first system message: 8,000 characters;
- tool summary: 10,000 characters;
- each tool result sent to the judge: 5,000 characters;
- assistant content and `reasoning_content`: 50,000 characters each.

Truncation preserves the beginning and end of the value.

## Checklist caching

Generated checklists are cached by the combination of question, system prompt, and tool summary. Records with identical context reuse one checklist. Concurrent requests for the same context also share the in-flight generation task.

The checklist and judge may use the same model or separate models.

## Scoring

Every normalized check receives `0` or `1`.

Instance Success Rate is all-or-nothing:

```text
ISR = 1.0 if every check passes, otherwise 0.0
```

Check item Success Rate is:

```text
CSR = passed_checks / total_checks
```

Each category also receives its own CSR.

Records tagged as subagents are skipped. Records that already contain `llm_checklist_csr` are skipped for resumability. Existing output is restored by record position before pending records are scored.

If checklist generation or judging fails, the scorer records `llm_checklist_error` and sets ISR and CSR to `null` for that record.

## Output fields

The scorer adds fields such as:

```json
{
  "llm_checklist_isr": 0.0,
  "llm_checklist_csr": 0.7826,
  "llm_checklist_version": "octobench-aligned-v2",
  "llm_checklist_total_checks": 23,
  "llm_checklist_total_passed": 18,
  "llm_checklist_category_scores": {
    "user_query": 0.8571,
    "verification": 0.5
  },
  "llm_checklist_question_source": "messages:first_user",
  "llm_checklist_question": "Fix the parser regression.",
  "llm_checklist_definition": {},
  "llm_checklist_judgement": {},
  "llm_checklist_detailed_results": {},
  "llm_checklist_generator_model": "gpt-4o-mini",
  "llm_checklist_judge_model": "gpt-4o-mini"
}
```

The definition and judgement fields retain the normalized checklist, per-check status, reasoning, and evidence. Persisted IM stores `_score` under `meta_info.unique_info`.

## Command line

Install the optional API dependency and set `OPENAI_API_KEY`. Set `OPENAI_BASE_URL` only for a compatible non-default endpoint.

```bash
pip install -e '.[llm]'

python -m legoflow_trace_crafter.llm_checklist_score \
  --input outputs/trajectories.im.jsonl \
  --output outputs/trajectories.checklist-scored.jsonl \
  --model gpt-4o-mini \
  --concurrency 8
```

Use separate models when needed:

```bash
python -m legoflow_trace_crafter.llm_checklist_score \
  --input outputs/trajectories.im.jsonl \
  --checklist-model <generator-model> \
  --judge-model <judge-model>
```

Other options include:

- `--max-instances N`
- `--question-field <field>`
- `--no-system-prompt`
- `--no-json-mode`
- `--quiet`

If `--output` is omitted, the default is `<input-stem>_llm_checklist_scored.jsonl`.
