#!/usr/bin/env python3
"""Checklist-based LLM-as-judge trajectory quality scoring.

Uses the checklist and LLM judge approach from MiniMax-AI/mini-vela to
evaluate IM-format trajectories. Complements the rule-based scores in
rule_score.py and writes results to the _score dict with an llm_ prefix.

Usage:
  Set OPENAI_API_KEY and, if needed, OPENAI_BASE_URL in the environment.
  python -m legoflow_trace_crafter.llm_score \
      --input outputs/trajectories.im.jsonl \
      --output outputs/trajectories.llm-scored.jsonl \
      --model gpt-4o --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

from legoflow_trace_crafter.llm_client import LLMClient, MODEL_RATES, _DEFAULT_RATE
from legoflow_trace_crafter.utils import load_jsonl, save_jsonl

# ═══════════════════════════════════════════════════════════════════════════
# Checklist definition (v2) — 5 categories, 15 checks, 1-5 scoring
# Letters F-J to avoid collision with rule-based A-E metrics in rule_score.py
# ═══════════════════════════════════════════════════════════════════════════

CHECKLIST: dict[str, Any] = {
    "F": {
        "description": "Problem Understanding",
        "checks": [
            {
                "check_id": "F1_diagnosis_depth",
                "description": "Depth of the agent's diagnosis: whether it understands the root cause, impact, and boundary conditions",
                "scoring": "1=Does not identify or seriously misunderstands the problem; 2=Identifies only the surface issue; 3=Understands the core issue but misses details; 4=Accurately understands the root cause and impact; 5=Deeply understands the root cause, boundary conditions, and potential risks",
            },
            {
                "check_id": "F2_scope_precision",
                "description": "Precision of the change scope: whether it identifies exactly the files that need changes without omissions or extras",
                "scoring": "1=Misses critical files or changes substantial unrelated code; 2=Finds some relevant files; 3=Covers the main files with omissions or minor extras; 4=Precisely covers every file that needs changes; 5=Precisely identifies the files and understands their dependencies",
            },
            {
                "check_id": "F3_plan_quality",
                "description": "Quality of the agent's plan: whether it forms a sound, complete approach before making changes",
                "scoring": "1=Starts changing code without a plan; 2=Has a vague, incomplete idea; 3=Has a basic plan but misses edge cases; 4=Has a sound, complete plan; 5=Has a thorough plan covering edge cases, compatibility, and rollback",
            },
        ],
    },
    "G": {
        "description": "Solution Quality",
        "checks": [
            {
                "check_id": "G1_fix_elegance",
                "description": "Elegance of the fix: whether it follows project conventions and is concise and maintainable",
                "scoring": "1=The fix fails or introduces new bugs; 2=Works but is awkward; 3=Mostly correct but inelegant; 4=Concise, correct, and consistent with project style; 5=Elegant, concise, fully aligned with project conventions, and highly maintainable",
            },
            {
                "check_id": "G2_change_minimality",
                "description": "Minimality of the changes: whether every changed line directly supports the fix",
                "scoring": "1=Contains extensive unrelated changes; 2=Contains many unnecessary changes; 3=Mostly focused with a few extras; 4=Precise with almost no extras; 5=Every changed line is necessary, with no redundancy",
            },
            {
                "check_id": "G3_robustness",
                "description": "Robustness of the fix: whether it considers edge cases, failures, and backward compatibility",
                "scoring": "1=Fragile and clearly misses edge cases; 2=Works in basic cases but has risks; 3=Covers the main cases; 4=Considers most edge cases; 5=Thoroughly covers edge cases, error handling, and backward compatibility",
            },
        ],
    },
    "H": {
        "description": "Reasoning Quality",
        "checks": [
            {
                "check_id": "H1_reasoning_coherence",
                "description": "Coherence of the reasoning: whether each step has a clear basis and the logic is easy to follow",
                "scoring": "1=Confused or disjointed reasoning; 2=Has a basic idea but weak logic; 3=Mostly coherent with some gaps; 4=Each step has a clear basis and the chain is clear; 5=Rigorous reasoning with strong support for every step",
            },
            {
                "check_id": "H2_hypothesis_driven",
                "description": "Whether the agent uses a hypothesis-driven approach by forming and testing hypotheses instead of guessing",
                "scoring": "1=Guesses blindly without a hypothesis; 2=Has an implicit, untested hypothesis; 3=Has a hypothesis but tests it inadequately; 4=States a hypothesis and verifies it through code inspection or tests; 5=Systematically forms, tests, and eliminates hypotheses",
            },
            {
                "check_id": "H3_adaptability",
                "description": "Adaptability when blocked: whether it quickly changes strategy instead of repeating failed actions",
                "scoring": "1=Repeats the same failed action or gets stuck; 2=Adapts only after several attempts; 3=Adapts inefficiently; 4=Quickly identifies the cause and adjusts; 5=Immediately identifies the issue and switches strategy efficiently, or shows strong preventive thinking when no obstacle occurs",
            },
        ],
    },
    "I": {
        "description": "Verification Rigor",
        "checks": [
            {
                "check_id": "I1_reproduction",
                "description": "Whether the agent reproduces the issue before fixing it and confirms that it exists",
                "scoring": "1=Makes no reproduction attempt; 2=Runs tests unrelated to the target issue; 3=Attempts reproduction without enough focus; 4=Runs a targeted test that confirms the issue; 5=Systematically reproduces the issue and understands its triggers",
            },
            {
                "check_id": "I2_fix_verification",
                "description": "Whether the agent verifies the fix after making changes",
                "scoring": "1=Performs no verification after the fix; 2=Runs tests without confirming the target issue is fixed; 3=Performs basic verification; 4=Runs a targeted test that confirms the issue is resolved; 5=Thoroughly verifies the fix and checks for regressions",
            },
            {
                "check_id": "I3_test_quality",
                "description": "Test quality and coverage: whether tests cover the main scenarios and edge cases",
                "scoring": "1=Runs no tests; 2=Runs only the most basic tests; 3=Covers the main scenarios; 4=Covers the main scenarios and some edge cases; 5=Provides comprehensive coverage, including regression tests, edge cases, and failure paths",
            },
        ],
    },
    "J": {
        "description": "Efficiency",
        "checks": [
            {
                "check_id": "J1_navigation_efficiency",
                "description": "Code navigation efficiency: whether it locates relevant files and code quickly and precisely",
                "scoring": "1=Navigates aimlessly and takes a long time; 2=Finds the target after several incorrect searches; 3=Finds the target through a somewhat indirect process; 4=Locates the target efficiently; 5=Uses a direct, precise search path with almost no wasted effort",
            },
            {
                "check_id": "J2_tool_proficiency",
                "description": "Tool proficiency: whether it uses the available tools correctly and efficiently",
                "scoring": "1=Frequently misuses tools; 2=Uses tools correctly but inefficiently; 3=Mostly uses tools correctly with room to improve; 4=Uses tools correctly and efficiently; 5=Uses tools expertly and selects the best tool for each task",
            },
            {
                "check_id": "J3_iteration_economy",
                "description": "Iteration economy: whether the total number of steps is reasonable and avoids wasted work",
                "scoring": "1=Wastes many steps, such as reinstalling dependencies or repeating searches; 2=Uses many redundant steps; 3=Has some redundancy but is acceptable overall; 4=Uses concise, efficient steps; 5=Every step has a clear purpose with no waste",
            },
        ],
    },
}

CATEGORY_SCORE_KEYS: dict[str, str] = {
    "F": "llm_f_problem_understanding",
    "G": "llm_g_solution_quality",
    "H": "llm_h_reasoning_quality",
    "I": "llm_i_verification_rigor",
    "J": "llm_j_efficiency",
}

_CHECKLIST_VERSION = "v2"
_MIN_SCORE_PER_CHECK = 1
_MAX_SCORE_PER_CHECK = 5

# ═══════════════════════════════════════════════════════════════════════════
# Truncation limits
# ═══════════════════════════════════════════════════════════════════════════

_MAX_TOOL_RESULT_CHARS = 5_000
_MAX_ASSISTANT_CONTENT_CHARS = 50_000
_MAX_REASONING_CONTENT_CHARS = 50_000

# ═══════════════════════════════════════════════════════════════════════════
# Prompt template (aligned with mini-vela evaluate.py)
# ═══════════════════════════════════════════════════════════════════════════

EVAL_PROMPT_TEMPLATE = """You are a trajectory quality judge.

Evaluate an AI coding agent's performance on a software engineering task
against every item in the provided checklist.

=====INPUT CONVERSATION=====
====TOOLS===
{tools}
====TOOLS===
====MESSAGES===
{messages}
====MESSAGES===
=====INPUT CONVERSATION=====

=====CHECKLIST TO EVALUATE=====
{checklist}
=====CHECKLIST TO EVALUATE=====

--------------------------------------------------
EVALUATION RULES
--------------------------------------------------

1. **Evaluate every item**: Score each check_id using the five-level criteria
   in its scoring field.

2. **Evidence**: Review every message with `role == "assistant"`, including:
   - Natural-language output (`content`)
   - Internal reasoning (`reasoning_content`), when present
   - Tool calls (`tool_calls`)

3. **Five-level scale (1-5)**:
   - **1**: Not done or seriously flawed
   - **2**: Attempted but ineffective
   - **3**: Mostly done with clear shortcomings
   - **4**: Done well with only minor issues
   - **5**: Excellent with no clear defects
   Follow each check's specific 1-5 criteria strictly.

4. **`reasoning` field**: Explain the basis for the score in English in 1-2
   sentences, citing specific assistant behavior or message indexes.

5. **Score strictly**: Do not award a high score merely because the agent
   attempted an action. Judge actual outcomes and quality. If the trajectory
   lacks enough evidence for an item, assign 1.

--------------------------------------------------
OUTPUT FORMAT (VALID JSON REQUIRED)
--------------------------------------------------

Return one JSON object with the same structure as the input checklist, adding
"reasoning" and "score" to every check. Use English for all reasoning and
descriptive result text.

{output_schema}

--------------------------------------------------
REQUIREMENTS
--------------------------------------------------

1. Evaluate **every** check_id in the checklist without omissions.
2. `score` must be the integer 1, 2, 3, 4, or 5.
3. Return valid JSON with no text outside the JSON object.
4. Preserve the original category structure and fields.

Apply the checklist and five-level criteria strictly, and return the complete
JSON result in English."""

_OUTPUT_SCHEMA_EXAMPLE = """{
  "F": {
    "description": "Problem Understanding",
    "checks": [
      {"check_id": "F1_diagnosis_depth", "reasoning": "The agent accurately identified the root cause and impact but did not examine boundary conditions.", "score": 4},
      {"check_id": "F2_scope_precision", "reasoning": "The agent identified exactly the files that required changes, with no omissions or extras.", "score": 5},
      {"check_id": "F3_plan_quality", "reasoning": "The agent had a basic plan but did not consider edge cases.", "score": 3}
    ]
  },
  "G": {
    "description": "Solution Quality",
    "checks": [
      {"check_id": "G1_fix_elegance", "reasoning": "...", "score": 5},
      {"check_id": "G2_change_minimality", "reasoning": "...", "score": 4},
      {"check_id": "G3_robustness", "reasoning": "...", "score": 3}
    ]
  },
  "H": {
    "description": "Reasoning Quality",
    "checks": [
      {"check_id": "H1_reasoning_coherence", "reasoning": "...", "score": 4},
      {"check_id": "H2_hypothesis_driven", "reasoning": "...", "score": 3},
      {"check_id": "H3_adaptability", "reasoning": "...", "score": 2}
    ]
  },
  "I": {
    "description": "Verification Rigor",
    "checks": [
      {"check_id": "I1_reproduction", "reasoning": "...", "score": 3},
      {"check_id": "I2_fix_verification", "reasoning": "...", "score": 4},
      {"check_id": "I3_test_quality", "reasoning": "...", "score": 3}
    ]
  },
  "J": {
    "description": "Efficiency",
    "checks": [
      {"check_id": "J1_navigation_efficiency", "reasoning": "...", "score": 5},
      {"check_id": "J2_tool_proficiency", "reasoning": "...", "score": 3},
      {"check_id": "J3_iteration_economy", "reasoning": "...", "score": 2}
    ]
  }
}"""


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory formatting (mini-vela style: pass JSON directly)
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_middle(text: str, max_chars: int) -> str:
    """Truncate text from the middle, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[content too long, truncated]\n\n"
        + text[-half:]
    )


def _truncate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy and truncate messages for eval prompt (avoids mutating originals)."""
    truncated = []
    assistant_idx = 0

    for message in messages:
        msg = deepcopy(message)
        role = msg.get("role", "")

        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > _MAX_TOOL_RESULT_CHARS:
                msg["content"] = _truncate_middle(content, _MAX_TOOL_RESULT_CHARS)

        elif role == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > _MAX_ASSISTANT_CONTENT_CHARS:
                msg["content"] = _truncate_middle(content, _MAX_ASSISTANT_CONTENT_CHARS)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if isinstance(text, str) and len(text) > _MAX_ASSISTANT_CONTENT_CHARS:
                            item["text"] = _truncate_middle(text, _MAX_ASSISTANT_CONTENT_CHARS)

            reasoning = msg.get("reasoning_content", "")
            if isinstance(reasoning, str) and len(reasoning) > _MAX_REASONING_CONTENT_CHARS:
                msg["reasoning_content"] = _truncate_middle(reasoning, _MAX_REASONING_CONTENT_CHARS)

            msg["assistant_turn_index"] = assistant_idx
            assistant_idx += 1

        truncated.append(msg)

    return truncated


def format_trajectory_for_eval(record: dict[str, Any]) -> str:
    """Format an IM record into a complete eval prompt string (mini-vela style).

    Passes tools and messages as JSON to preserve structural information.
    """
    messages = record.get("messages", [])
    tools = record.get("tools", [])

    truncated = _truncate_messages(messages)

    tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    messages_str = "\n".join(json.dumps(m, ensure_ascii=False) for m in truncated)
    checklist_str = json.dumps(CHECKLIST, ensure_ascii=False, indent=2)

    return EVAL_PROMPT_TEMPLATE.format(
        tools=tools_str,
        messages=messages_str,
        checklist=checklist_str,
        output_schema=_OUTPUT_SCHEMA_EXAMPLE,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing and score calculation
# ═══════════════════════════════════════════════════════════════════════════

def parse_judge_response(response_text: str) -> dict[str, Any] | None:
    """Parse the LLM judge's JSON response into a structured dict.

    Returns None if parsing fails. Handles markdown code fences.
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def calculate_reward(parsed: dict[str, Any]) -> dict[str, Any]:
    """Calculate per-category and composite scores from parsed judge output.

    Each check is scored 1-5. Category score = (sum - n*1) / (n*(5-1)),
    mapping [all-1s → 0, all-5s → 1]. Composite = mean of 5 category scores.
    """
    if "error" in parsed:
        return {"llm_composite_score": 0.0, "llm_checklist_version": _CHECKLIST_VERSION}

    cat_scores: list[float] = []
    scores: dict[str, Any] = {}

    for cat_key, score_key in CATEGORY_SCORE_KEYS.items():
        cat_data = parsed.get(cat_key, {})
        checks = cat_data.get("checks", [])
        if not checks:
            scores[score_key] = None
            continue
        n = len(checks)
        cat_sum = sum(_clamp_score(c.get("score", _MIN_SCORE_PER_CHECK)) for c in checks)
        cat_min = n * _MIN_SCORE_PER_CHECK
        cat_range = n * (_MAX_SCORE_PER_CHECK - _MIN_SCORE_PER_CHECK)
        val = round((cat_sum - cat_min) / cat_range, 4) if cat_range else 0.0
        scores[score_key] = val
        cat_scores.append(val)

    scores["llm_composite_score"] = (
        round(sum(cat_scores) / len(cat_scores), 4) if cat_scores else 0.0
    )
    scores["llm_checklist_version"] = _CHECKLIST_VERSION
    return scores


def _clamp_score(val: Any) -> int:
    """Clamp a score value to [_MIN_SCORE_PER_CHECK, _MAX_SCORE_PER_CHECK]."""
    try:
        v = int(val)
    except (TypeError, ValueError):
        return _MIN_SCORE_PER_CHECK
    return max(_MIN_SCORE_PER_CHECK, min(v, _MAX_SCORE_PER_CHECK))


def get_detailed_results(parsed: dict[str, Any]) -> dict[str, Any]:
    """Get detailed evaluation statistics with by_category breakdowns."""
    if "error" in parsed:
        return {}

    results: dict[str, Any] = {
        "total_checks": 0,
        "total_score": 0,
        "total_max": 0,
        "by_category": {},
    }

    for category in CATEGORY_SCORE_KEYS:
        value = parsed.get(category)
        if not isinstance(value, dict):
            continue
        cat_score = 0
        cat_count = 0

        for item in value.get("checks", []):
            s = _clamp_score(item.get("score", _MIN_SCORE_PER_CHECK))
            cat_score += s
            cat_count += 1
            results["total_checks"] += 1
            results["total_score"] += s
            results["total_max"] += _MAX_SCORE_PER_CHECK

        results["by_category"][category] = {
            "score": cat_score,
            "max": cat_count * _MAX_SCORE_PER_CHECK,
            "count": cat_count,
        }

    return results


def _build_eval_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat messages for the LLM judge call."""
    prompt = format_trajectory_for_eval(record)
    return [{"role": "user", "content": prompt}]


def estimate_tokens(record: dict[str, Any]) -> int:
    """Rough token estimate for a single record's eval prompt (~4 chars/token)."""
    msgs = _build_eval_messages(record)
    total_chars = sum(len(m["content"]) for m in msgs)
    return total_chars // 4


# ═══════════════════════════════════════════════════════════════════════════
# Async batch scoring
# ═══════════════════════════════════════════════════════════════════════════

async def score_record_llm(
    record: dict[str, Any],
    client: LLMClient,
) -> dict[str, Any] | None:
    """Score a single IM record using the LLM judge.

    Returns the llm_* score dict, or None on failure.
    """
    messages = _build_eval_messages(record)
    try:
        response_text = await client.chat(messages, temperature=0.0)
    except Exception:
        return None

    parsed = parse_judge_response(response_text)
    if parsed is None:
        return None

    scores = calculate_reward(parsed)
    scores["llm_model"] = client.model
    scores["llm_raw_response"] = parsed
    scores["llm_detailed_results"] = get_detailed_results(parsed)
    return scores


def merge_llm_scores(record: dict[str, Any], llm_scores: dict[str, Any]) -> None:
    """Merge LLM scores into the record's _score dict."""
    if record.get("_score") is None:
        record["_score"] = {}
    for k, v in llm_scores.items():
        record["_score"][k] = v


async def score_dataset_llm(
    records: list[dict[str, Any]],
    client: LLMClient,
    output_path: Path | None = None,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Score all main-agent records in a dataset using the LLM judge.

    Subagent records (with _agent_type == "subagent") are skipped.
    Records already containing llm_composite_score are skipped (resume support).
    When output_path is provided, each scored record is flushed to disk immediately
    so progress survives interruptions.
    """
    def _already_scored(r: dict[str, Any]) -> bool:
        s = r.get("_score")
        return s is not None and s.get("llm_composite_score") is not None

    scorable = [
        (i, r) for i, r in enumerate(records)
        if r.get("_agent_type") != "subagent" and not _already_scored(r)
    ]
    already_scored = sum(
        1 for r in records
        if r.get("_agent_type") != "subagent" and _already_scored(r)
    )

    if not quiet:
        print(f"LLM scoring {len(scorable)} records "
              f"(skipping {len(records) - len(scorable) - already_scored} subagents, "
              f"{already_scored} already scored)")

    if output_path is not None:
        _flush_all(records, output_path)

    pbar = tqdm(total=len(scorable), desc="LLM scoring", disable=quiet)
    write_lock = asyncio.Lock()
    flush_counter = 0
    flush_interval = max(1, min(len(scorable) // 10, 50))

    async def _score_one(idx: int, record: dict[str, Any]) -> None:
        nonlocal flush_counter
        try:
            scores = await score_record_llm(record, client)
        except Exception as exc:  # noqa: BLE001
            scores = None
            error_msg = f"{type(exc).__name__}: {exc}"
        else:
            error_msg = None
        if scores is not None:
            merge_llm_scores(record, scores)
        else:
            merge_llm_scores(record, {
                "llm_composite_score": None,
                "llm_model": client.model,
                "llm_checklist_version": _CHECKLIST_VERSION,
                "llm_error": error_msg or "unknown error (score_record_llm returned None)",
            })
        if output_path is not None:
            async with write_lock:
                flush_counter += 1
                if flush_counter % flush_interval == 0:
                    _flush_all(records, output_path)
        pbar.update(1)

    tasks = [_score_one(i, r) for i, r in scorable]
    await asyncio.gather(*tasks)
    pbar.close()

    if output_path is not None:
        _flush_all(records, output_path)

    if not quiet:
        print(f"\n{client.usage.summary(client.model)}")

    return records


def _flush_all(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically write all records to a JSONL file (tmp + rename)."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(path)


# ═══════════════════════════════════════════════════════════════════════════
# Summary and dry-run
# ═══════════════════════════════════════════════════════════════════════════

def print_llm_score_summary(records: list[dict[str, Any]]) -> None:
    """Print aggregate LLM score statistics."""
    main_records = [r for r in records if r.get("_agent_type") != "subagent"]
    scores = [
        r["_score"]["llm_composite_score"]
        for r in main_records
        if r.get("_score") and r["_score"].get("llm_composite_score") is not None
    ]
    failed = sum(
        1 for r in main_records
        if r.get("_score") and r["_score"].get("llm_error")
    )
    not_scored = len(main_records) - len(scores) - failed
    if not scores and not failed:
        print("No LLM scores available.")
        return

    print(f"\n{'='*60}")
    print(f"LLM Score Summary ({len(scores)} scored, "
          f"{failed} failed, {not_scored} not scored)")
    print(f"{'='*60}")
    if scores:
        print(f"  Composite: mean={sum(scores)/len(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")

    for cat_key, score_key in CATEGORY_SCORE_KEYS.items():
        cat_scores = [
            r["_score"][score_key]
            for r in records
            if r.get("_score") and r["_score"].get(score_key) is not None
        ]
        if cat_scores:
            desc = CHECKLIST[cat_key]["description"]
            print(f"  {cat_key} ({desc}): "
                  f"mean={sum(cat_scores)/len(cat_scores):.4f}")
    print()


def dry_run(records: list[dict[str, Any]], model: str) -> None:
    """Estimate token usage and cost without calling the API."""
    scorable = [r for r in records if r.get("_agent_type") != "subagent"]
    total_tokens = 0
    for r in tqdm(scorable, desc="Estimating tokens"):
        total_tokens += estimate_tokens(r)

    output_tokens_est = len(scorable) * 2000

    in_rate, out_rate = MODEL_RATES.get(model, _DEFAULT_RATE)
    cost = total_tokens * in_rate + output_tokens_est * out_rate

    print(f"\n{'='*50}")
    print(f"Dry Run Estimate")
    print(f"{'='*50}")
    print(f"  Scorable records: {len(scorable)}")
    print(f"  Est. input tokens:  {total_tokens:,}")
    print(f"  Est. output tokens: {output_tokens_est:,}")
    print(f"  Model: {model}")
    print(f"  Est. cost: ${cost:.2f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score trajectory quality with a checklist-based LLM judge",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to the input IM JSONL file",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: <input>_llm_scored.jsonl)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM judge model (default: gpt-4o)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate token usage and cost without calling the API",
    )
    parser.add_argument(
        "--no-json-mode", action="store_true",
        help="Disable response_format=json_object for APIs that do not support it",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce log output",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    """Single async entry point — client lifecycle on one event loop."""
    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_llm_scored.jsonl")

    print(f"Reading: {input_path}")
    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records found; exiting.")
        sys.exit(0)

    if args.max_instances is not None and args.max_instances < len(records):
        records = records[:args.max_instances]
        print(f"Limited input to {len(records)} records")

    if output_path.exists() and output_path != input_path:
        prev = load_jsonl(output_path)
        prev_scores: dict[int, dict[str, Any]] = {}
        for i, r in enumerate(prev):
            s = r.get("_score")
            if s and s.get("llm_composite_score") is not None:
                prev_scores[i] = s
        if prev_scores:
            merged = 0
            for i, s in prev_scores.items():
                if i < len(records):
                    if records[i].get("_score") is None:
                        records[i]["_score"] = {}
                    for k, v in s.items():
                        if k.startswith("llm_"):
                            records[i]["_score"][k] = v
                    merged += 1
            print(f"Restored {merged} LLM scores from the existing output")

    if args.dry_run:
        dry_run(records, args.model)
        return

    client = LLMClient(
        model=args.model,
        concurrency=args.concurrency,
        json_mode=not args.no_json_mode,
    )

    try:
        scored = await score_dataset_llm(
            records, client, output_path=output_path, quiet=args.quiet,
        )
    finally:
        await client.close()

    print_llm_score_summary(scored)

    save_jsonl(output_path, scored)
    print(f"Saved LLM scoring results to: {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
