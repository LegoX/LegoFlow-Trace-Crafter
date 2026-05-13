#!/usr/bin/env python3
"""Checklist-based LLM-as-Judge 轨迹质量打分模块.

参考 MiniMax-AI/mini-vela 的 checklist + LLM judge 方案，对 IM 格式轨迹
进行语义质量评估。与 rule_score.py 的规则打分互补，结果以 llm_ 前缀写入 _score dict。

用法:
  conda activate swelf
  export OPENAI_BASE_URL="..."
  export OPENAI_API_KEY="sk-..."
  python -m swe_data_process.llm_score \
      --input artifacts/cc_im.jsonl \
      --output artifacts/cc_im_llm_scored.jsonl \
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

from swe_data_process.llm_client import LLMClient, MODEL_RATES, _DEFAULT_RATE
from swe_data_process.utils import load_jsonl, save_jsonl

# ═══════════════════════════════════════════════════════════════════════════
# Checklist definition (v2) — 5 categories, 15 checks, 1-5 scoring
# Letters F-J to avoid collision with rule-based A-E metrics in rule_score.py
# ═══════════════════════════════════════════════════════════════════════════

CHECKLIST: dict[str, Any] = {
    "F": {
        "description": "Problem Understanding — 问题理解",
        "checks": [
            {
                "check_id": "F1_diagnosis_depth",
                "description": "Agent 对问题的诊断深度：是否理解了根因、影响范围和边界条件",
                "scoring": "1=未识别或严重误解问题; 2=识别了表面问题; 3=理解了核心问题但遗漏细节; 4=准确理解根因和影响范围; 5=深入理解根因、边界条件和潜在风险",
            },
            {
                "check_id": "F2_scope_precision",
                "description": "Agent 对修改范围的精准度：是否精确定位了需要修改的文件，无遗漏无冗余",
                "scoring": "1=严重遗漏关键文件或大量修改无关代码; 2=找到部分相关文件; 3=覆盖了主要文件但有遗漏或少量多余; 4=精准覆盖所有需修改的文件; 5=精准定位且理解文件间依赖关系",
            },
            {
                "check_id": "F3_plan_quality",
                "description": "Agent 的修复计划质量：是否在动手前形成了合理、完整的修复方案",
                "scoring": "1=无计划直接动手; 2=有模糊想法但不完整; 3=有基本计划但缺少对边界情况的考虑; 4=计划合理完整; 5=计划周密，考虑了边界条件、兼容性和回退方案",
            },
        ],
    },
    "G": {
        "description": "Solution Quality — 解决方案质量",
        "checks": [
            {
                "check_id": "G1_fix_elegance",
                "description": "修复方案的优雅程度：是否符合项目惯例、简洁且可维护",
                "scoring": "1=修复无效或引入新 bug; 2=能工作但方式笨拙; 3=基本正确但不够优雅; 4=简洁正确，符合项目风格; 5=优雅简洁，完全融入项目惯例，可维护性强",
            },
            {
                "check_id": "G2_change_minimality",
                "description": "修改的最小化程度：每一行改动是否都直接服务于修复",
                "scoring": "1=大量无关改动; 2=较多不必要改动; 3=基本聚焦但有少量多余; 4=改动精准，几乎无多余; 5=每一行都是必要的，零冗余",
            },
            {
                "check_id": "G3_robustness",
                "description": "修复的健壮性：是否考虑了边界条件、异常情况和向后兼容",
                "scoring": "1=修复脆弱，明显遗漏边界情况; 2=基本场景可用但有隐患; 3=覆盖了主要场景; 4=考虑了大部分边界条件; 5=全面考虑边界条件、异常处理和向后兼容",
            },
        ],
    },
    "H": {
        "description": "Reasoning Quality — 推理质量",
        "checks": [
            {
                "check_id": "H1_reasoning_coherence",
                "description": "推理链路的连贯性：每步是否有明确依据，逻辑是否清晰",
                "scoring": "1=推理混乱或跳跃; 2=有基本思路但逻辑不严密; 3=大体连贯但有跳跃; 4=每步有明确依据，链路清晰; 5=推理严密，每步都有充分论证",
            },
            {
                "check_id": "H2_hypothesis_driven",
                "description": "Agent 是否采用假设驱动的方法：先形成假设再验证，而非盲目尝试",
                "scoring": "1=盲目尝试，无假设; 2=有隐含假设但未验证; 3=有假设但验证不充分; 4=明确假设并通过代码阅读或测试验证; 5=系统性地提出、验证和排除假设",
            },
            {
                "check_id": "H3_adaptability",
                "description": "遇到障碍时的适应能力：是否能快速调整策略而非重复失败操作",
                "scoring": "1=重复相同失败操作或卡住; 2=多次尝试后才调整; 3=有调整但效率低; 4=较快分析原因并调整策略; 5=立即识别问题并高效切换策略（若无障碍则评估预防性思考）",
            },
        ],
    },
    "I": {
        "description": "Verification Rigor — 验证严谨性",
        "checks": [
            {
                "check_id": "I1_reproduction",
                "description": "Agent 是否在修复前复现了问题，确认问题确实存在",
                "scoring": "1=未做任何复现; 2=运行了测试但未针对目标问题; 3=尝试复现但方式不够针对性; 4=运行了针对性测试确认问题存在; 5=系统性复现并理解了问题的触发条件",
            },
            {
                "check_id": "I2_fix_verification",
                "description": "Agent 是否在修复后验证了修复的有效性",
                "scoring": "1=修复后未做任何验证; 2=运行了测试但未确认目标问题已修复; 3=做了基本验证; 4=运行针对性测试确认问题已解决; 5=充分验证修复有效且无副作用",
            },
            {
                "check_id": "I3_test_quality",
                "description": "测试的质量和覆盖度：是否覆盖了主要场景和边界条件",
                "scoring": "1=未运行任何测试; 2=只运行了最基本的测试; 3=覆盖了主要场景; 4=覆盖了主要场景和部分边界条件; 5=全面覆盖，包括回归测试、边界条件和异常路径",
            },
        ],
    },
    "J": {
        "description": "Efficiency — 效率",
        "checks": [
            {
                "check_id": "J1_navigation_efficiency",
                "description": "代码导航效率：是否快速精准地定位到相关文件和代码",
                "scoring": "1=导航混乱，长时间找不到; 2=多次错误搜索后找到; 3=过程有些曲折但最终找到; 4=较高效地定位; 5=搜索路径直接精准，几乎无浪费",
            },
            {
                "check_id": "J2_tool_proficiency",
                "description": "工具使用的熟练度：是否正确高效地使用了可用工具",
                "scoring": "1=工具使用错误频繁; 2=能用但效率低; 3=基本正确但有改进空间; 4=工具使用正确高效; 5=熟练运用各种工具，选择最优工具完成任务",
            },
            {
                "check_id": "J3_iteration_economy",
                "description": "迭代经济性：总步骤数是否合理，有无大量浪费的步骤",
                "scoring": "1=大量浪费步骤（如反复安装依赖、重复搜索）; 2=较多冗余步骤; 3=有一些冗余但总体可接受; 4=步骤精简高效; 5=每一步都有明确目的，零浪费",
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

EVAL_PROMPT_TEMPLATE = """你是一个轨迹质量评审模型。

你的任务是：根据给定的 Checklist，逐项评估 AI coding agent 在解决软件工程任务时的表现。

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
评估规则
--------------------------------------------------

1. **逐项评估**：对 Checklist 中的每个 check_id，根据 scoring 字段中的五级标准打分

2. **评估依据**：检查所有 `role == "assistant"` 的消息，包括：
   - 自然语言输出（content）
   - 内部推理（reasoning_content，如有）
   - 工具调用（tool_calls）

3. **五级评分标准（1-5 分）**：
   - **1 分**：完全未做到 / 严重缺陷
   - **2 分**：尝试了但效果差
   - **3 分**：基本做到，但有明显不足
   - **4 分**：做得好，只有小瑕疵
   - **5 分**：表现优秀，无明显缺陷
   每个 check 的 scoring 字段给出了该项的具体 1-5 标准，请严格参照

4. **reasoning 字段**：必须说明判定依据（中文，1-2 句话），引用具体的 assistant 行为或消息索引

5. **严格评分**：不要因为 agent "尝试了"就给高分，关注实际效果和质量。如果轨迹信息不足以判断某项，默认给 1 分

--------------------------------------------------
输出格式（必须为合法 JSON）
--------------------------------------------------

输出一个 JSON 对象，结构与输入的 Checklist 相同，但每个 check 增加 "reasoning" 和 "score" 字段：

{output_schema}

--------------------------------------------------
注意事项
--------------------------------------------------

1. 必须对 Checklist 中的**每个** check_id 进行评估，不可遗漏
2. score 只能是 1、2、3、4 或 5（整数），不允许其他值
3. 输出必须是合法 JSON，不要在 JSON 外添加任何文字
4. 保持原有的 category 结构和字段

请严格按照 Checklist 和五级评分标准进行评估，输出完整的 JSON 结果。"""

_OUTPUT_SCHEMA_EXAMPLE = """{
  "F": {
    "description": "Problem Understanding — 问题理解",
    "checks": [
      {"check_id": "F1_diagnosis_depth", "reasoning": "Agent 准确理解了根因和影响范围，但未深入考虑边界条件", "score": 4},
      {"check_id": "F2_scope_precision", "reasoning": "精确定位了需要修改的文件，无遗漏无冗余", "score": 5},
      {"check_id": "F3_plan_quality", "reasoning": "有基本计划但缺少对边界情况的考虑", "score": 3}
    ]
  },
  "G": {
    "description": "Solution Quality — 解决方案质量",
    "checks": [
      {"check_id": "G1_fix_elegance", "reasoning": "...", "score": 5},
      {"check_id": "G2_change_minimality", "reasoning": "...", "score": 4},
      {"check_id": "G3_robustness", "reasoning": "...", "score": 3}
    ]
  },
  "H": {
    "description": "Reasoning Quality — 推理质量",
    "checks": [
      {"check_id": "H1_reasoning_coherence", "reasoning": "...", "score": 4},
      {"check_id": "H2_hypothesis_driven", "reasoning": "...", "score": 3},
      {"check_id": "H3_adaptability", "reasoning": "...", "score": 2}
    ]
  },
  "I": {
    "description": "Verification Rigor — 验证严谨性",
    "checks": [
      {"check_id": "I1_reproduction", "reasoning": "...", "score": 3},
      {"check_id": "I2_fix_verification", "reasoning": "...", "score": 4},
      {"check_id": "I3_test_quality", "reasoning": "...", "score": 3}
    ]
  },
  "J": {
    "description": "Efficiency — 效率",
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
        description="Checklist-based LLM-as-Judge 轨迹质量打分",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="输入 IM JSONL 文件路径",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="输出 JSONL 文件路径（默认: <input>_llm_scored.jsonl）",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="LLM judge 模型名称（默认: gpt-4o）",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key（默认从 OPENAI_API_KEY 环境变量读取）",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="OpenAI-compatible API base URL（默认从 OPENAI_BASE_URL 环境变量读取）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="并发请求数（默认: 10）",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="最多处理的记录数",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅估算 token 用量和费用，不调用 API",
    )
    parser.add_argument(
        "--no-json-mode", action="store_true",
        help="禁用 response_format=json_object（用于不支持该参数的 API）",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="减少日志输出",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    """Single async entry point — client lifecycle on one event loop."""
    input_path: Path = args.input
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_llm_scored.jsonl")

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
            print(f"从已有输出恢复了 {merged} 条 LLM 评分（断点续评）")

    if args.dry_run:
        dry_run(records, args.model)
        return

    client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
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
    print(f"LLM 打分结果已保存到: {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
