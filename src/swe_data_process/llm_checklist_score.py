#!/usr/bin/env python3
"""OctoBench-aligned dynamic checklist LLM scoring.

Workflow:
1. Extract the user question, system prompt, and tool definitions from a record.
2. Generate an atomic, binary checklist from all instruction sources (15-35 items).
3. Judge the complete trajectory against each checklist item with a binary score (0/1).
4. Compute ISR (all items pass) and CSR (item pass rate), then write them to ``_score``.

OctoBench alignment:
- Sources: user question, system prompt, and tool definitions.
- Checklist size: 15-35 items, up to 40, compared with an OctoBench average of ~33.
- Item scoring: strict binary 0/1 (passed/failed), with no partial credit.
- Weighting: all items have equal weight.
- Aggregate metrics: ISR (Instance Success Rate) and CSR (Check Item Success Rate).

References:
- OctoBench (arXiv:2601.10343) instruction-source taxonomy and ISR/CSR framework.
- Asynchronous scoring, resume support, and JSONL I/O from ``llm_score.py``.

Usage:
  Set OPENAI_API_KEY and, if needed, OPENAI_BASE_URL in the environment.
  python -m swe_data_process.llm_checklist_score \
      --input outputs/trajectories.im.jsonl \
      --output outputs/trajectories.checklist-scored.jsonl \
      --model gpt-4o-mini --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

from swe_data_process.llm_client import LLMClient
from swe_data_process.utils import load_jsonl, save_jsonl


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_CHECKLIST_VERSION = "octobench-aligned-v2"
_JUDGE_SCORE_MIN = 0
_JUDGE_SCORE_MAX = 1
_MAX_TOTAL_CHECKS = 40
_DEFAULT_QUESTION_FIELDS = (
    "user_question",
    "question",
    "prompt",
    "task",
    "instruction",
    "user_query",
)

# Truncation limits
_MAX_TOOL_RESULT_CHARS = 5_000
_MAX_ASSISTANT_CONTENT_CHARS = 50_000
_MAX_REASONING_CONTENT_CHARS = 50_000
_MAX_QUESTION_CHARS = 12_000
_MAX_SYSTEM_PROMPT_CHARS = 8_000
_MAX_TOOLS_CONTEXT_CHARS = 10_000

_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]+")


# ═══════════════════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════════════════

_CHECKLIST_SCHEMA_EXAMPLE = """{
  "question_summary": "The user wants a command validation module, failure logging, and tests.",
  "categories": [
    {
      "category_id": "user_query",
      "description": "Explicit requirements from the user message",
      "checks": [
        {
          "check_id": "user_query_1",
          "description": "Implement the core command validation logic",
          "check_type": "implementation",
          "required_evidence": "Code changes or tool output show that the feature is integrated"
        },
        {
          "check_id": "user_query_2",
          "description": "Add failure logging",
          "check_type": "implementation",
          "required_evidence": "The trajectory contains logging-related code changes"
        },
        {
          "check_id": "user_query_3",
          "description": "Add tests directly related to the change",
          "check_type": "testing",
          "required_evidence": "The trajectory contains new test files or test cases"
        }
      ]
    },
    {
      "category_id": "system_prompt",
      "description": "Instructions and constraints from the system prompt",
      "checks": [
        {
          "check_id": "system_prompt_1",
          "description": "Follow the code style requirements in the system prompt",
          "check_type": "compliance",
          "required_evidence": "Code changes follow the style rules stated in the system prompt"
        }
      ]
    },
    {
      "category_id": "tool_schema",
      "description": "Correct usage implied by tool definitions",
      "checks": [
        {
          "check_id": "tool_schema_1",
          "description": "Use the correct tool for file edits",
          "check_type": "compliance",
          "required_evidence": "The trajectory uses an appropriate editing tool instead of an inefficient substitute"
        }
      ]
    },
    {
      "category_id": "verification",
      "description": "Verification and regression coverage",
      "checks": [
        {
          "check_id": "verification_1",
          "description": "Run tests and confirm they pass",
          "check_type": "testing",
          "required_evidence": "The trajectory contains a test command and passing output"
        }
      ]
    }
  ]
}"""

_JUDGE_SCHEMA_EXAMPLE = """{
  "summary": "The trajectory completed the main implementation and tests but violated a system-prompt style constraint.",
  "categories": [
    {
      "category_id": "user_query",
      "checks": [
        {
          "check_id": "user_query_1",
          "status": "passed",
          "score": 1,
          "reasoning": "assistant_turn_index=6 changed the core implementation, and later tool output confirms successful integration.",
          "evidence": [
            "assistant_turn_index=6 changed the target file",
            "Tool output at assistant_turn_index=8 shows the logic working"
          ]
        }
      ]
    },
    {
      "category_id": "verification",
      "checks": [
        {
          "check_id": "verification_1",
          "status": "failed",
          "score": 0,
          "reasoning": "The trajectory contains no test command.",
          "evidence": []
        }
      ]
    }
  ]
}"""

_CHECKLIST_GENERATION_PROMPT = """You design checklists for software engineering tasks.

Generate an atomic, binary-decidable, and sufficiently comprehensive checklist
for later trajectory evaluation from the user question, system prompt, and
available tool definitions.

================ USER QUESTION ================
{question}
================ USER QUESTION ================

================ SYSTEM PROMPT ================
{system_prompt}
================ SYSTEM PROMPT ================

================ AVAILABLE TOOLS ================
{tools}
================ AVAILABLE TOOLS ================

Follow these rules strictly:
1. Derive verifiable requirements from every instruction source above: the
   user question, system prompt, and tool definitions.
2. Make every check atomic and objectively decidable as complete or incomplete.
   Do not use vague or subjective descriptions.
3. Split a source containing multiple independent requirements into separate checks.
4. Group checks by instruction source using these categories; omit empty categories:
   - user_query: Explicit feature, modification, and output requirements from the user.
   - system_prompt: Behavioral constraints, style rules, and safety requirements.
   - tool_schema: Correct usage implied by tool definitions, such as using an
     editing tool instead of sed.
   - repo_policy: Constraints from repository policy files when visible in the
     system prompt.
   - implementation: Correctness and completeness requirements for the implementation.
   - verification: Testing, verification, and regression requirements.
   - communication: User-requested explanations, summaries, and output formats.
5. Generate 15-35 checks. Use fewer than 15 only for a genuinely simple task;
   a complex task may use up to 40.
6. `check_type` must be one of: compliance / implementation / modification /
   understanding / testing / configuration.
7. `required_evidence` must describe the evidence needed in the later trajectory
   to mark the check as passed.
8. Return valid JSON with no text outside the JSON object. Write all summaries,
   descriptions, and evidence requirements in English.

OUTPUT FORMAT:
{schema}
"""

_JUDGE_PROMPT_TEMPLATE = """You are a strict judge of software engineering trajectories.

Use the provided checklist to determine whether the agent's trajectory
completed every requirement in the user question.

================ USER QUESTION ================
{question}
================ USER QUESTION ================

================ CHECKLIST ================
{checklist}
================ CHECKLIST ================

================ INPUT CONVERSATION ================
==== TOOLS ====
{tools}
==== TOOLS ====
==== MESSAGES ====
{messages}
==== MESSAGES ====
================ INPUT CONVERSATION ================

SCORING RULES:
1. Score every check_id in the checklist without omissions.
2. `score` must be strictly binary:
   - 0 = Incomplete, or the trajectory lacks sufficient evidence of completion.
   - 1 = Clearly complete, with direct evidence in the trajectory.
3. `status` must match `score`:
   - 0 -> failed
   - 1 -> passed
4. Explain each decision concisely in English in `reasoning`, citing
   assistant_turn_index or specific tool behavior when possible.
5. Provide 1-3 short English evidence snippets in `evidence`; use an empty
   array when no evidence exists.
6. Score strictly. Do not assign 1 for an attempt or partial completion; assign
   1 only when the requirement is clearly and fully complete.
7. Return valid JSON with no text outside the JSON object. Write the summary,
   reasoning, and evidence in English.

OUTPUT FORMAT:
{schema}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Generic helpers
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


def _coerce_text(value: Any) -> str:
    """Convert common content payloads to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, (dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _make_id(value: str, fallback: str) -> str:
    """Make a stable identifier from free text."""
    normalized = _ID_SAFE_RE.sub("_", value).strip("_").lower()
    return normalized[:64] if normalized else fallback


def _clamp_score(value: Any) -> int:
    """Clamp a judge score into the supported range."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return _JUDGE_SCORE_MIN
    return max(_JUDGE_SCORE_MIN, min(score, _JUDGE_SCORE_MAX))


def _normalize_status(value: Any, score: int) -> str:
    """Normalize textual status to binary pass/fail."""
    text = _coerce_text(value).strip().lower()
    if text in {"failed", "passed"}:
        return text
    return "passed" if score >= 1 else "failed"


def parse_json_response(response_text: str) -> dict[str, Any] | None:
    """Parse a JSON response, tolerating markdown fences."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _first_message_by_role(messages: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    """Return the first message matching a role."""
    for message in messages:
        if message.get("role") == role:
            return message
    return None


def extract_user_question(
    record: dict[str, Any],
    question_field: str | None = None,
) -> tuple[str, str]:
    """Extract the user question from explicit fields or the first user message."""
    field_candidates = [question_field] if question_field else []
    field_candidates.extend(
        candidate for candidate in _DEFAULT_QUESTION_FIELDS
        if candidate != question_field
    )

    for field_name in field_candidates:
        if not field_name:
            continue
        if field_name not in record:
            continue
        text = _coerce_text(record.get(field_name)).strip()
        if text:
            return _truncate_middle(text, _MAX_QUESTION_CHARS), field_name

    first_user = _first_message_by_role(record.get("messages") or [], "user")
    if first_user is not None:
        text = _coerce_text(first_user.get("content")).strip()
        if text:
            return _truncate_middle(text, _MAX_QUESTION_CHARS), "messages:first_user"

    return "", "not_found"


def extract_system_prompt(record: dict[str, Any]) -> str:
    """Extract the first system prompt if present."""
    first_system = _first_message_by_role(record.get("messages") or [], "system")
    if first_system is None:
        return ""
    text = _coerce_text(first_system.get("content")).strip()
    return _truncate_middle(text, _MAX_SYSTEM_PROMPT_CHARS)


def extract_tools_summary(record: dict[str, Any]) -> str:
    """Extract tool definitions as a compact summary for checklist generation."""
    tools = record.get("tools")
    if not isinstance(tools, list) or not tools:
        return ""
    parts: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function", tool)
        name = func.get("name", "")
        desc = func.get("description", "")
        if not name:
            continue
        params = func.get("parameters", {})
        param_names = sorted(params.get("properties", {}).keys()) if isinstance(params, dict) else []
        line = f"- {name}"
        if desc:
            line += f": {desc[:200]}"
        if param_names:
            line += f" (params: {', '.join(param_names)})"
        parts.append(line)
    text = "\n".join(parts)
    return _truncate_middle(text, _MAX_TOOLS_CONTEXT_CHARS)


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory formatting
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy and truncate messages for evaluation prompts."""
    truncated: list[dict[str, Any]] = []
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
                msg["reasoning_content"] = _truncate_middle(
                    reasoning,
                    _MAX_REASONING_CONTENT_CHARS,
                )

            msg["assistant_turn_index"] = assistant_idx
            assistant_idx += 1

        truncated.append(msg)

    return truncated


def format_trajectory_for_eval(record: dict[str, Any]) -> tuple[str, str]:
    """Format tools/messages into judge-ready JSON strings."""
    messages = record.get("messages") or []
    tools = record.get("tools")
    if not isinstance(tools, list):
        tools = []
    truncated = _truncate_messages(messages)

    tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    messages_str = "\n".join(json.dumps(m, ensure_ascii=False) for m in truncated)
    return tools_str, messages_str


# ═══════════════════════════════════════════════════════════════════════════
# Checklist generation and normalization
# ═══════════════════════════════════════════════════════════════════════════

def _iter_category_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return category payloads from either list or dict layout."""
    raw_categories = payload.get("categories")
    if isinstance(raw_categories, list):
        return [item for item in raw_categories if isinstance(item, dict)]
    if isinstance(raw_categories, dict):
        normalized: list[dict[str, Any]] = []
        for category_id, category_data in raw_categories.items():
            if isinstance(category_data, dict):
                normalized.append({
                    "category_id": category_id,
                    **category_data,
                })
        return normalized
    normalized = []
    for category_id, category_data in payload.items():
        if category_id in {"question_summary", "summary", "metadata", "meta"}:
            continue
        if isinstance(category_data, dict) and isinstance(category_data.get("checks"), list):
            normalized.append({
                "category_id": category_id,
                **category_data,
            })
    if normalized:
        return normalized
    return []


def normalize_generated_checklist(
    parsed: dict[str, Any],
    question: str,
) -> dict[str, Any] | None:
    """Validate and normalize the generated checklist JSON."""
    categories = _iter_category_payloads(parsed)
    if not categories:
        return None

    normalized_categories: list[dict[str, Any]] = []
    seen_check_ids: set[str] = set()
    total_checks = 0

    for category_index, category in enumerate(categories, start=1):
        category_id = _make_id(
            _coerce_text(category.get("category_id") or category.get("name") or ""),
            f"category_{category_index}",
        )
        description = _coerce_text(
            category.get("description") or category.get("title") or category_id,
        ).strip()
        raw_checks = category.get("checks", [])
        if not isinstance(raw_checks, list):
            continue

        normalized_checks: list[dict[str, Any]] = []
        for check_index, check in enumerate(raw_checks, start=1):
            if not isinstance(check, dict):
                continue
            description_text = _coerce_text(
                check.get("description") or check.get("requirement") or check.get("check"),
            ).strip()
            if not description_text:
                continue

            check_id = _make_id(
                _coerce_text(check.get("check_id") or ""),
                f"{category_id}_{check_index}",
            )
            if check_id in seen_check_ids:
                suffix = 2
                while f"{check_id}_{suffix}" in seen_check_ids:
                    suffix += 1
                check_id = f"{check_id}_{suffix}"
            seen_check_ids.add(check_id)

            normalized_checks.append({
                "check_id": check_id,
                "description": description_text,
                "check_type": _coerce_text(
                    check.get("check_type") or category_id,
                ).strip() or category_id,
                "required_evidence": _coerce_text(
                    check.get("required_evidence") or check.get("evidence_hint"),
                ).strip(),
            })
            total_checks += 1
            if total_checks >= _MAX_TOTAL_CHECKS:
                break

        if normalized_checks:
            normalized_categories.append({
                "category_id": category_id,
                "description": description or category_id,
                "checks": normalized_checks,
            })
        if total_checks >= _MAX_TOTAL_CHECKS:
            break

    if not normalized_categories or total_checks == 0:
        return None

    summary = _coerce_text(parsed.get("question_summary")).strip()
    if not summary:
        summary = _truncate_middle(question, 280)

    return {
        "question_summary": summary,
        "categories": normalized_categories,
    }


def _build_checklist_generation_messages(
    question: str,
    system_prompt: str,
    tools_context: str = "",
) -> list[dict[str, str]]:
    """Build the prompt for checklist generation."""
    prompt = _CHECKLIST_GENERATION_PROMPT.format(
        question=question,
        system_prompt=system_prompt or "(none)",
        tools=tools_context or "(none)",
        schema=_CHECKLIST_SCHEMA_EXAMPLE,
    )
    return [{"role": "user", "content": prompt}]


async def generate_checklist(
    question: str,
    client: LLMClient,
    system_prompt: str = "",
    tools_context: str = "",
) -> dict[str, Any]:
    """Generate a normalized checklist from the question and context."""
    response_text = await client.chat(
        _build_checklist_generation_messages(question, system_prompt, tools_context),
        temperature=0.0,
        max_tokens=6000,
    )
    parsed = parse_json_response(response_text)
    if parsed is None:
        raise ValueError("failed to parse generated checklist JSON")

    normalized = normalize_generated_checklist(parsed, question)
    if normalized is None:
        raise ValueError("generated checklist JSON is invalid or empty")
    return normalized


class ChecklistManager:
    """Async checklist cache keyed by question + system prompt + tools."""

    def __init__(
        self,
        client: LLMClient,
        include_system_prompt: bool = True,
    ) -> None:
        self.client = client
        self.include_system_prompt = include_system_prompt
        self._cache: dict[str, dict[str, Any]] = {}
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def get_checklist_for_record(
        self,
        record: dict[str, Any],
        question_field: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """Get or generate a checklist for one record."""
        question, source = extract_user_question(record, question_field=question_field)
        if not question:
            raise ValueError("cannot extract user question from record")

        system_prompt = extract_system_prompt(record) if self.include_system_prompt else ""
        tools_context = extract_tools_summary(record)
        cache_key = json.dumps(
            {
                "question": question,
                "system_prompt": system_prompt,
                "tools_context": tools_context,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return question, source, deepcopy(cached)

            task = self._inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(
                    self._generate_and_store(cache_key, question, system_prompt, tools_context),
                )
                self._inflight[cache_key] = task

        checklist = await task
        return question, source, deepcopy(checklist)

    async def _generate_and_store(
        self,
        cache_key: str,
        question: str,
        system_prompt: str,
        tools_context: str,
    ) -> dict[str, Any]:
        """Generate a checklist and cache it once."""
        try:
            checklist = await generate_checklist(
                question,
                self.client,
                system_prompt=system_prompt,
                tools_context=tools_context,
            )
        finally:
            async with self._lock:
                self._inflight.pop(cache_key, None)

        async with self._lock:
            self._cache[cache_key] = checklist
        return checklist


# ═══════════════════════════════════════════════════════════════════════════
# Judgment and score aggregation
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_judgement_payload(
    parsed: dict[str, Any],
) -> tuple[str, dict[str, dict[str, dict[str, Any]]]]:
    """Flatten arbitrary judge JSON into category/check lookup maps."""
    summary = _coerce_text(parsed.get("summary")).strip()
    lookup: dict[str, dict[str, dict[str, Any]]] = {}

    for category in _iter_category_payloads(parsed):
        category_id = _make_id(
            _coerce_text(category.get("category_id") or category.get("name") or ""),
            "category",
        )
        raw_checks = category.get("checks", [])
        if not isinstance(raw_checks, list):
            continue
        category_lookup: dict[str, dict[str, Any]] = {}
        for index, check in enumerate(raw_checks, start=1):
            if not isinstance(check, dict):
                continue
            check_id = _make_id(
                _coerce_text(check.get("check_id") or ""),
                f"{category_id}_{index}",
            )
            category_lookup[check_id] = check
        if category_lookup:
            lookup[category_id] = category_lookup

    return summary, lookup


def normalize_judgement(
    parsed: dict[str, Any],
    checklist: dict[str, Any],
) -> dict[str, Any]:
    """Project the judge response back onto the checklist shape."""
    summary, lookup = _flatten_judgement_payload(parsed)
    normalized_categories: list[dict[str, Any]] = []

    for category in checklist["categories"]:
        category_id = category["category_id"]
        parsed_checks = lookup.get(category_id, {})
        normalized_checks: list[dict[str, Any]] = []

        for check in category["checks"]:
            raw_check = parsed_checks.get(check["check_id"], {})
            score = _clamp_score(raw_check.get("score", _JUDGE_SCORE_MIN))
            evidence_raw = raw_check.get("evidence", [])
            if isinstance(evidence_raw, list):
                evidence = [
                    _coerce_text(item).strip()
                    for item in evidence_raw
                    if _coerce_text(item).strip()
                ][:3]
            else:
                evidence_text = _coerce_text(evidence_raw).strip()
                evidence = [evidence_text] if evidence_text else []

            normalized_checks.append({
                **check,
                "status": _normalize_status(raw_check.get("status"), score),
                "score": score,
                "reasoning": _coerce_text(raw_check.get("reasoning")).strip()
                or "Judge response omitted reasoning.",
                "evidence": evidence,
            })

        normalized_categories.append({
            "category_id": category_id,
            "description": category["description"],
            "checks": normalized_checks,
        })

    return {
        "summary": summary,
        "categories": normalized_categories,
    }


def calculate_checklist_reward(judgement: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    """Compute ISR (all-or-nothing) and CSR (mean pass rate) from binary judgement."""
    total_checks = 0
    total_passed = 0
    all_passed = True
    by_category: dict[str, Any] = {}
    category_scores: dict[str, float] = {}

    for category in judgement["categories"]:
        cat_checks = 0
        cat_passed = 0

        for check in category["checks"]:
            score = _clamp_score(check.get("score"))
            cat_checks += 1
            if score >= 1:
                cat_passed += 1
            else:
                all_passed = False

        cat_csr = round(cat_passed / cat_checks, 4) if cat_checks else 0.0
        by_category[category["category_id"]] = {
            "description": category["description"],
            "passed": cat_passed,
            "total": cat_checks,
            "csr": cat_csr,
        }
        category_scores[category["category_id"]] = cat_csr

        total_checks += cat_checks
        total_passed += cat_passed

    isr = 1.0 if all_passed and total_checks > 0 else 0.0
    csr = round(total_passed / total_checks, 4) if total_checks else 0.0
    details = {
        "total_checks": total_checks,
        "total_passed": total_passed,
        "isr": isr,
        "csr": csr,
        "by_category": by_category,
        "category_scores": category_scores,
    }
    return isr, csr, details


def _build_judge_messages(
    record: dict[str, Any],
    question: str,
    checklist: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the prompt for the trajectory judge."""
    tools_str, messages_str = format_trajectory_for_eval(record)
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        checklist=json.dumps(checklist, ensure_ascii=False, indent=2),
        tools=tools_str,
        messages=messages_str,
        schema=_JUDGE_SCHEMA_EXAMPLE,
    )
    return [{"role": "user", "content": prompt}]


async def judge_record_against_checklist(
    record: dict[str, Any],
    question: str,
    checklist: dict[str, Any],
    client: LLMClient,
) -> dict[str, Any]:
    """Run the LLM judge on one record."""
    response_text = await client.chat(
        _build_judge_messages(record, question, checklist),
        temperature=0.0,
        max_tokens=5000,
    )
    parsed = parse_json_response(response_text)
    if parsed is None:
        raise ValueError("failed to parse judge JSON")
    return normalize_judgement(parsed, checklist)


# ═══════════════════════════════════════════════════════════════════════════
# Record / dataset scoring
# ═══════════════════════════════════════════════════════════════════════════

async def score_record_llm_checklist(
    record: dict[str, Any],
    checklist_manager: ChecklistManager,
    judge_client: LLMClient,
    question_field: str | None = None,
) -> dict[str, Any]:
    """Score one record with a generated checklist and a judge model."""
    question, question_source, checklist = await checklist_manager.get_checklist_for_record(
        record,
        question_field=question_field,
    )
    judgement = await judge_record_against_checklist(
        record,
        question,
        checklist,
        judge_client,
    )
    isr, csr, detailed = calculate_checklist_reward(judgement)

    return {
        "llm_checklist_isr": isr,
        "llm_checklist_csr": csr,
        "llm_checklist_version": _CHECKLIST_VERSION,
        "llm_checklist_total_checks": detailed["total_checks"],
        "llm_checklist_total_passed": detailed["total_passed"],
        "llm_checklist_category_scores": detailed["category_scores"],
        "llm_checklist_question_source": question_source,
        "llm_checklist_question": question,
        "llm_checklist_definition": checklist,
        "llm_checklist_judgement": judgement,
        "llm_checklist_detailed_results": detailed,
        "llm_checklist_generator_model": checklist_manager.client.model,
        "llm_checklist_judge_model": judge_client.model,
    }


def merge_llm_checklist_scores(record: dict[str, Any], score_dict: dict[str, Any]) -> None:
    """Merge dynamic checklist scores into ``record['_score']``."""
    if record.get("_score") is None:
        record["_score"] = {}
    for key, value in score_dict.items():
        record["_score"][key] = value


def _flush_all(records: list[dict[str, Any]], path: Path) -> None:
    """Atomically flush the whole dataset for resumable scoring."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.rename(path)


async def score_dataset_llm_checklist(
    records: list[dict[str, Any]],
    checklist_manager: ChecklistManager,
    judge_client: LLMClient,
    output_path: Path | None = None,
    quiet: bool = False,
    question_field: str | None = None,
) -> list[dict[str, Any]]:
    """Score all main-agent records with dynamic checklist judging."""

    def _already_scored(record: dict[str, Any]) -> bool:
        score = record.get("_score")
        return score is not None and score.get("llm_checklist_csr") is not None

    scorable = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("_agent_type") != "subagent" and not _already_scored(record)
    ]
    already_scored = sum(
        1
        for record in records
        if record.get("_agent_type") != "subagent" and _already_scored(record)
    )

    if not quiet:
        skipped_subagents = len(records) - len(scorable) - already_scored
        print(
            "Dynamic checklist scoring "
            f"{len(scorable)} records "
            f"(skipping {skipped_subagents} subagents, {already_scored} already scored)",
        )

    if output_path is not None:
        _flush_all(records, output_path)

    pbar = tqdm(total=len(scorable), desc="Checklist scoring", disable=quiet)
    write_lock = asyncio.Lock()
    flush_counter = 0
    flush_interval = max(1, min(len(scorable) // 10, 50)) if scorable else 1

    async def _score_one(index: int, record: dict[str, Any]) -> None:
        nonlocal flush_counter
        try:
            scores = await score_record_llm_checklist(
                record,
                checklist_manager,
                judge_client,
                question_field=question_field,
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-record progress
            scores = {
                "llm_checklist_isr": None,
                "llm_checklist_csr": None,
                "llm_checklist_version": _CHECKLIST_VERSION,
                "llm_checklist_generator_model": checklist_manager.client.model,
                "llm_checklist_judge_model": judge_client.model,
                "llm_checklist_error": f"{type(exc).__name__}: {exc}",
            }

        merge_llm_checklist_scores(record, scores)

        if output_path is not None:
            async with write_lock:
                flush_counter += 1
                if flush_counter % flush_interval == 0:
                    _flush_all(records, output_path)
        pbar.update(1)

    tasks = [_score_one(index, record) for index, record in scorable]
    await asyncio.gather(*tasks)
    pbar.close()

    if output_path is not None:
        _flush_all(records, output_path)

    if not quiet:
        if checklist_manager.client is judge_client:
            print(f"\n{judge_client.usage.summary(judge_client.model)}")
        else:
            print(f"\n[checklist] {checklist_manager.client.usage.summary(checklist_manager.client.model)}")
            print(f"[judge] {judge_client.usage.summary(judge_client.model)}")

    return records


def print_llm_checklist_score_summary(records: list[dict[str, Any]]) -> None:
    """Print aggregate ISR and CSR statistics for dynamic checklist scoring."""
    main_records = [record for record in records if record.get("_agent_type") != "subagent"]
    isr_values: list[float] = []
    csr_values: list[float] = []
    for record in main_records:
        score = record.get("_score")
        if not isinstance(score, dict):
            continue
        if score.get("llm_checklist_csr") is not None:
            csr_values.append(float(score["llm_checklist_csr"]))
        if score.get("llm_checklist_isr") is not None:
            isr_values.append(float(score["llm_checklist_isr"]))

    failed = sum(
        1 for record in main_records
        if record.get("_score") and record["_score"].get("llm_checklist_error")
    )
    not_scored = len(main_records) - len(csr_values) - failed

    if not csr_values and not failed:
        print("No dynamic checklist scores available.")
        return

    print(f"\n{'=' * 68}")
    print(
        "Dynamic Checklist Score Summary "
        f"({len(csr_values)} scored, {failed} failed, {not_scored} not scored)",
    )
    print(f"{'=' * 68}")

    if isr_values:
        print(f"  ISR (Instance Success Rate): {sum(isr_values) / len(isr_values):.4f}")
    if csr_values:
        print(
            f"  CSR (Check item Success Rate): "
            f"mean={sum(csr_values) / len(csr_values):.4f}  "
            f"min={min(csr_values):.4f}  max={max(csr_values):.4f}",
        )

    by_category: dict[str, list[float]] = {}
    for record in main_records:
        score = record.get("_score")
        if not isinstance(score, dict):
            continue
        category_scores = score.get("llm_checklist_category_scores", {})
        if not isinstance(category_scores, dict):
            continue
        for category_id, value in category_scores.items():
            if isinstance(value, (int, float)):
                by_category.setdefault(category_id, []).append(float(value))

    for category_id in sorted(by_category):
        values = by_category[category_id]
        print(f"  {category_id}: csr={sum(values) / len(values):.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a checklist from the user question and score an IM trajectory",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input IM JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input>_llm_checklist_scored.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Default model for checklist generation and judging",
    )
    parser.add_argument(
        "--checklist-model",
        type=str,
        default=None,
        help="Checklist generation model (default: --model)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Trajectory judge model (default: --model)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent requests (default: 8)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--question-field",
        type=str,
        default=None,
        help="Field containing the user question, such as user_query",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Exclude the system prompt from checklist generation",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable response_format=json_object for APIs that do not support it",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_llm_checklist_scored.jsonl",
        )

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
        for index, record in enumerate(prev):
            score = record.get("_score")
            if score and score.get("llm_checklist_csr") is not None:
                prev_scores[index] = score

        if prev_scores:
            merged = 0
            for index, score in prev_scores.items():
                if index >= len(records):
                    continue
                if records[index].get("_score") is None:
                    records[index]["_score"] = {}
                for key, value in score.items():
                    if key.startswith("llm_checklist_"):
                        records[index]["_score"][key] = value
                merged += 1
            print(f"Restored {merged} dynamic checklist scores from the existing output")

    checklist_model = args.checklist_model or args.model
    judge_model = args.judge_model or args.model

    if checklist_model == judge_model:
        shared_client = LLMClient(
            model=judge_model,
            concurrency=args.concurrency,
            json_mode=not args.no_json_mode,
        )
        checklist_client = shared_client
        judge_client = shared_client
    else:
        checklist_client = LLMClient(
            model=checklist_model,
            concurrency=args.concurrency,
            json_mode=not args.no_json_mode,
        )
        judge_client = LLMClient(
            model=judge_model,
            concurrency=args.concurrency,
            json_mode=not args.no_json_mode,
        )

    checklist_manager = ChecklistManager(
        checklist_client,
        include_system_prompt=not args.no_system_prompt,
    )

    try:
        scored = await score_dataset_llm_checklist(
            records,
            checklist_manager,
            judge_client,
            output_path=output_path,
            quiet=args.quiet,
            question_field=args.question_field,
        )
    finally:
        if checklist_client is judge_client:
            await checklist_client.close()
        else:
            await checklist_client.close()
            await judge_client.close()

    print_llm_checklist_score_summary(scored)
    save_jsonl(output_path, scored)
    print(f"Saved dynamic checklist scoring results to: {output_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
