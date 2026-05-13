#!/usr/bin/env python3
"""OctoBench 对齐的动态 checklist LLM 打分模块。

核心流程：
1. 从记录中提取用户问题、系统提示词、工具定义（多指令来源）
2. 让 LLM 基于所有指令来源生成原子化、二值可判定的 checklist（15-35 项）
3. 再结合完整轨迹，让 LLM 按 checklist 逐项做二值判分（0/1）
4. 计算 ISR（全通过=1）和 CSR（逐项通过率），写入 ``_score`` 字段

与 OctoBench 对齐的维度：
- checklist 来源：用户问题 + 系统提示词 + 工具定义（多指令来源）
- check 数量：15-35 项（上限 40），对标 OctoBench 平均 ~33 项
- 单项评分：严格二值 0/1（passed/failed），无 partial
- 权重：所有 check 等权，无显式权重
- 聚合指标：ISR（Instance Success Rate）+ CSR（Check item Success Rate）

参考:
- OctoBench (arXiv:2601.10343) 的指令来源分类与 ISR/CSR 评估框架
- 项目现有 ``llm_score.py`` 的异步并发、断点续评与 JSONL I/O 设计

用法:
  conda activate swelf
  export OPENAI_BASE_URL="..."
  export OPENAI_API_KEY="sk-..."
  python -m swe_data_process.llm_checklist_score \
      --input artifacts/cc_jierun_im.jsonl \
      --output artifacts/cc_jierun_im_llm_checklist_scored.jsonl \
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
  "question_summary": "用户希望增加一项命令校验模块，并补充失败日志与测试。",
  "categories": [
    {
      "category_id": "user_query",
      "description": "用户消息中明确提出的要求",
      "checks": [
        {
          "check_id": "user_query_1",
          "description": "实现命令校验模块的核心逻辑",
          "check_type": "implementation",
          "required_evidence": "代码修改或工具输出应体现功能已真正接入"
        },
        {
          "check_id": "user_query_2",
          "description": "补充失败日志记录功能",
          "check_type": "implementation",
          "required_evidence": "轨迹中出现日志相关代码修改"
        },
        {
          "check_id": "user_query_3",
          "description": "补充与改动直接相关的测试",
          "check_type": "testing",
          "required_evidence": "轨迹中出现新增测试文件或测试用例"
        }
      ]
    },
    {
      "category_id": "system_prompt",
      "description": "系统提示词中的指令与约束",
      "checks": [
        {
          "check_id": "system_prompt_1",
          "description": "遵循系统提示词中关于代码风格的要求",
          "check_type": "compliance",
          "required_evidence": "代码修改符合系统提示词中声明的风格规范"
        }
      ]
    },
    {
      "category_id": "tool_schema",
      "description": "工具定义所隐含的正确使用方式",
      "checks": [
        {
          "check_id": "tool_schema_1",
          "description": "使用正确的工具完成文件编辑操作",
          "check_type": "compliance",
          "required_evidence": "轨迹中使用了合适的编辑工具而非低效替代"
        }
      ]
    },
    {
      "category_id": "verification",
      "description": "验证与回归",
      "checks": [
        {
          "check_id": "verification_1",
          "description": "运行测试并确认通过",
          "check_type": "testing",
          "required_evidence": "轨迹中出现测试执行命令及通过结果"
        }
      ]
    }
  ]
}"""

_JUDGE_SCHEMA_EXAMPLE = """{
  "summary": "轨迹完成了主要实现和测试，但未遵循系统提示词中的风格约束。",
  "categories": [
    {
      "category_id": "user_query",
      "checks": [
        {
          "check_id": "user_query_1",
          "status": "passed",
          "score": 1,
          "reasoning": "assistant_turn_index=6 中已修改核心实现，并在后续工具输出中看到代码接入成功。",
          "evidence": [
            "assistant_turn_index=6 修改了目标文件",
            "assistant_turn_index=8 的工具输出显示逻辑已生效"
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
          "reasoning": "轨迹中没有运行任何测试命令。",
          "evidence": []
        }
      ]
    }
  ]
}"""

_CHECKLIST_GENERATION_PROMPT = """你是一个软件工程任务 checklist 设计器。

你的任务是：根据用户问题、系统提示词和可用工具定义，为后续的轨迹评审生成一个”原子化、可判定、覆盖充分”的 checklist。

================ USER QUESTION ================
{question}
================ USER QUESTION ================

================ SYSTEM PROMPT ================
{system_prompt}
================ SYSTEM PROMPT ================

================ AVAILABLE TOOLS ================
{tools}
================ AVAILABLE TOOLS ================

请严格遵守以下规则：
1. checklist 必须基于上述所有指令来源（用户问题、系统提示词、工具定义）提取可验证的要求
2. 每个 check 必须是原子化、二值可判定的（完成/未完成），不允许模糊或主观的描述
3. 如果某个来源包含多个独立要求，必须拆成多个 check
4. 将 checks 按指令来源放入以下类别，空类别可以省略：
   - user_query: 用户消息中明确提出的功能、修改、输出要求
   - system_prompt: 系统提示词中的行为约束、风格规范、安全规则
   - tool_schema: 工具定义所隐含的正确使用方式（如应使用 Edit 而非 sed）
   - repo_policy: 仓库规范文件（CLAUDE.md、AGENTS.md 等）中的约束（若在系统提示词中可见）
   - implementation: 代码实现的正确性、完整性要求
   - verification: 测试、验证、回归检查要求
   - communication: 用户要求的解释、总结、输出格式
5. 总 check 数控制在 15-35 个；只在任务确实简单时可少于 15 个，复杂任务可达 40 个
6. check_type 取值：compliance / implementation / modification / understanding / testing / configuration
7. required_evidence 说明后续轨迹里应观察到什么证据才能判定为通过
8. 输出必须是合法 JSON，JSON 外不要有任何文字

输出格式：
{schema}
"""

_JUDGE_PROMPT_TEMPLATE = """你是一个严格的软件工程轨迹评审模型。

你的任务是：根据给定的 checklist，判断 agent 的轨迹是否完成了用户问题中的各项要求。

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

评分规则：
1. 必须对 checklist 中每个 check_id 逐项打分，不可遗漏
2. score 只能取 0 或 1（严格二值）：
   - 0 = 未完成，或轨迹中没有充分证据证明已完成
   - 1 = 明确完成，且轨迹中有直接证据
3. status 必须与 score 对应：
   - 0 -> failed
   - 1 -> passed
4. reasoning 用中文简洁说明判定依据，尽量引用 assistant_turn_index 或具体工具行为
5. evidence 是 1-3 条简短证据片段；若确实没有证据，给空数组
6. 严格评分：不要因为”尝试过”或”部分完成”就给 1 分；只有清楚、完整地完成才给 1 分
7. 输出必须是合法 JSON，JSON 外不要有任何文字

输出格式：
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
        description="基于用户问题动态生成 checklist，再对 IM 轨迹进行 LLM 判分",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 IM JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL 文件路径（默认: <input>_llm_checklist_scored.jsonl）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="默认模型名称（若未显式提供 checklist/judge model，则两者都用它）",
    )
    parser.add_argument(
        "--checklist-model",
        type=str,
        default=None,
        help="生成 checklist 的模型名称（默认继承 --model）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="按轨迹判分的模型名称（默认继承 --model）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key（默认从 OPENAI_API_KEY 环境变量读取）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL（默认从 OPENAI_BASE_URL 环境变量读取）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="并发请求数（默认: 8）",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="最多处理的记录数",
    )
    parser.add_argument(
        "--question-field",
        type=str,
        default=None,
        help="显式指定用户问题字段名（如 user_query）",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="生成 checklist 时不包含系统提示词（默认包含，与 OctoBench 对齐）",
    )
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="禁用 response_format=json_object（用于不支持该参数的 API）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    input_path = args.input
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_llm_checklist_scored.jsonl",
        )

    print(f"读取: {input_path}")
    records = load_jsonl(input_path)
    print(f"加载了 {len(records)} 条记录")

    if not records:
        print("没有记录，退出。")
        sys.exit(0)

    if args.max_instances is not None and args.max_instances < len(records):
        records = records[:args.max_instances]
        print(f"截断到 {len(records)} 条记录")

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
            print(f"从已有输出恢复了 {merged} 条 dynamic checklist 评分（断点续评）")

    checklist_model = args.checklist_model or args.model
    judge_model = args.judge_model or args.model

    if checklist_model == judge_model:
        shared_client = LLMClient(
            model=judge_model,
            api_key=args.api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            json_mode=not args.no_json_mode,
        )
        checklist_client = shared_client
        judge_client = shared_client
    else:
        checklist_client = LLMClient(
            model=checklist_model,
            api_key=args.api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            json_mode=not args.no_json_mode,
        )
        judge_client = LLMClient(
            model=judge_model,
            api_key=args.api_key,
            base_url=args.base_url,
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
    print(f"Dynamic checklist 打分结果已保存到: {output_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
