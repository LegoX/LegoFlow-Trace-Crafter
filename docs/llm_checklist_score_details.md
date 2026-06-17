# 动态 Checklist LLM 打分系统 (octobench-aligned-v2)

本文档描述 `llm_checklist_score.py` 的 OctoBench 对齐动态 checklist 打分框架。与 `llm_score.py`（固定 checklist、1-5 五级评分）不同，本模块由 LLM 根据每条记录的指令来源动态生成 checklist，再做严格二值判分，最终输出 ISR 和 CSR 两个聚合指标。

参考论文：OctoBench (arXiv:2601.10343)，MiniMax/Fudan，2026。

## 概述

核心流程分两步：

1. **Checklist 生成**：LLM 根据用户问题 + 系统提示词 + 工具定义，动态生成 15-35 个原子化、二值可判定的 check 项
2. **轨迹判分**：另一次 LLM 调用，拿完整轨迹逐项做 0/1 二值判分

最终计算两个聚合指标：
- **ISR (Instance Success Rate)**：所有 check 全部通过 = 1，否则 = 0
- **CSR (Check item Success Rate)**：逐项通过率的均值

## 与 OctoBench 的对齐

| 维度 | OctoBench | 本模块 |
|------|-----------|--------|
| checklist 来源 | 从参考轨迹提取 + 人工审核 | LLM 从用户问题 + 系统提示词 + 工具定义生成 |
| check 数量 | 平均 32.7，中位数 34 | 15-35（上限 40） |
| 单项评分 | 严格二值 0/1 | 严格二值 0/1（passed/failed） |
| 权重 | 无显式权重 | 所有 check 等权 |
| 聚合指标 | ISR + CSR | ISR + CSR |

与 OctoBench 的主要差异在于 checklist 来源：OctoBench 从 16 条参考轨迹中提取 check 项并经人工审核，本模块因无参考轨迹，改为从多指令来源（用户问题、系统提示词、工具定义）动态生成。

## 类别体系

Checklist 按指令来源分类，对标 OctoBench 的 instruction-source taxonomy：

| 类别 ID | 描述 | 来源 |
|---------|------|------|
| `user_query` | 用户消息中明确提出的功能、修改、输出要求 | 用户消息 |
| `system_prompt` | 系统提示词中的行为约束、风格规范、安全规则 | 系统提示词 |
| `tool_schema` | 工具定义所隐含的正确使用方式 | 工具定义 |
| `repo_policy` | 仓库规范文件（CLAUDE.md 等）中的约束 | 系统提示词中可见的仓库规范 |
| `implementation` | 代码实现的正确性、完整性要求 | 综合推断 |
| `verification` | 测试、验证、回归检查要求 | 综合推断 |
| `communication` | 用户要求的解释、总结、输出格式 | 用户消息 |

空类别自动省略。每个 check 还标注 `check_type`（compliance / implementation / modification / understanding / testing / configuration）。

## 评分流程

1. 加载 IM JSONL 记录
2. 跳过 subagent 记录（`_agent_type == "subagent"`）和已有 checklist 分数的记录（支持断点续评）
3. 对每条记录：
   - 提取用户问题（优先取显式字段，其次取首条 user message）
   - 提取系统提示词（首条 system message，截断至 8K 字符）
   - 提取工具定义摘要（工具名 + 描述前 200 字符 + 参数名列表，截断至 10K 字符）
   - **第一次 LLM 调用**：将上述三者喂给 checklist 生成器，生成 15-35 个 check 项（temperature=0，max_tokens=6000）
   - 深拷贝并截断轨迹消息：tool result（≤5K）、assistant content（≤50K）、reasoning_content（≤50K）
   - **第二次 LLM 调用**：将完整轨迹 + checklist 喂给 judge，逐项做 0/1 判分（temperature=0，max_tokens=5000）
4. 计算 ISR 和 CSR
5. 结果以 `llm_checklist_` 前缀写入 `_score` dict

### Checklist 缓存

`ChecklistManager` 以 `(question, system_prompt, tools_context)` 三元组为 cache key。同一数据集中相同问题的记录（如同 instance 的 main + subagent）共享同一份 checklist，避免重复生成。并发请求同一 cache key 时，后续请求 await 同一个 task 而非重复发请求。

## 分数聚合

### ISR (Instance Success Rate)

全通过判定：

```
ISR = 1.0  if all checks scored 1
ISR = 0.0  otherwise
```

### CSR (Check item Success Rate)

逐项通过率：

```
CSR = total_passed / total_checks
```

每个类别也有独立的 CSR：

```
category_csr = category_passed / category_total
```

## 输出格式

Checklist 打分结果合并到现有 `_score` dict，使用 `llm_checklist_` 前缀：

```json
{
  "composite_score": 0.72,
  "llm_composite_score": 0.65,
  "llm_checklist_isr": 0.0,
  "llm_checklist_csr": 0.7826,
  "llm_checklist_version": "octobench-aligned-v2",
  "llm_checklist_total_checks": 23,
  "llm_checklist_total_passed": 18,
  "llm_checklist_category_scores": {
    "user_query": 0.8571,
    "system_prompt": 0.6667,
    "tool_schema": 1.0,
    "verification": 0.5
  },
  "llm_checklist_question_source": "messages:first_user",
  "llm_checklist_question": "Fix the bug in ...",
  "llm_checklist_definition": { "question_summary": "...", "categories": ["..."] },
  "llm_checklist_judgement": { "summary": "...", "categories": ["..."] },
  "llm_checklist_detailed_results": {
    "total_checks": 23,
    "total_passed": 18,
    "isr": 0.0,
    "csr": 0.7826,
    "by_category": {
      "user_query": { "description": "...", "passed": 6, "total": 7, "csr": 0.8571 }
    },
    "category_scores": { "user_query": 0.8571 }
  },
  "llm_checklist_generator_model": "gpt-4o-mini",
  "llm_checklist_judge_model": "gpt-4o-mini"
}
```

关键字段说明：
- `llm_checklist_isr`：该实例是否全部通过（0.0 或 1.0）
- `llm_checklist_csr`：逐项通过率（0.0 ~ 1.0）
- `llm_checklist_definition`：LLM 生成的完整 checklist（含每项的 description、check_type、required_evidence）
- `llm_checklist_judgement`：LLM judge 的完整判分结果（含每项的 status、score、reasoning、evidence）
- `llm_checklist_question`：用于生成 checklist 的用户问题文本

## 使用方法

```bash
conda activate swelf
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://..."  # 可选

# 基本用法
python -m swe_data_process.llm_checklist_score \
    --input artifacts/cc_im.jsonl \
    --output artifacts/cc_im_checklist_scored.jsonl \
    --model gpt-4o-mini --concurrency 8

# checklist 生成和 judge 使用不同模型
python -m swe_data_process.llm_checklist_score \
    --input artifacts/cc_im.jsonl \
    --checklist-model gpt-4o-mini \
    --judge-model gpt-4o \
    --concurrency 8

# 不包含系统提示词（默认包含）
python -m swe_data_process.llm_checklist_score \
    --input artifacts/cc_im.jsonl \
    --no-system-prompt

# 用于不支持 response_format 的 API
python -m swe_data_process.llm_checklist_score --input ... --no-json-mode
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | （必需） | 输入 IM JSONL 文件路径 |
| `--output` | `<input>_llm_checklist_scored.jsonl` | 输出 JSONL 文件路径 |
| `--model` | `gpt-4o-mini` | 默认模型（checklist 生成和 judge 共用） |
| `--checklist-model` | 继承 `--model` | checklist 生成模型 |
| `--judge-model` | 继承 `--model` | 轨迹判分模型 |
| `--api-key` | `OPENAI_API_KEY` 环境变量 | OpenAI API key |
| `--base-url` | `OPENAI_BASE_URL` 环境变量 | OpenAI-compatible API base URL |
| `--concurrency` | 8 | 并发请求数 |
| `--max-instances` | 无限制 | 最多处理的记录数 |
| `--question-field` | 自动检测 | 显式指定用户问题字段名 |
| `--no-system-prompt` | 默认包含 | 生成 checklist 时不包含系统提示词 |
| `--no-json-mode` | 默认启用 | 禁用 `response_format=json_object` |
| `--quiet` | 否 | 减少日志输出 |

## 截断参数

| 常量 | 值 | 作用阶段 | 说明 |
|------|-----|---------|------|
| `_MAX_TOOL_RESULT_CHARS` | 5,000 | judge | 工具执行结果（bash 输出等） |
| `_MAX_ASSISTANT_CONTENT_CHARS` | 50,000 | judge | assistant 回复内容 |
| `_MAX_REASONING_CONTENT_CHARS` | 50,000 | judge | 模型思考链 |
| `_MAX_QUESTION_CHARS` | 12,000 | checklist 生成 | 用户问题 |
| `_MAX_SYSTEM_PROMPT_CHARS` | 8,000 | checklist 生成 | 系统提示词 |
| `_MAX_TOOLS_CONTEXT_CHARS` | 10,000 | checklist 生成 | 工具定义摘要 |

所有截断使用 `_truncate_middle`，保留首尾、中间插入 `[content too long, truncated]`。

## 与其他打分模块的关系

| 维度 | rule_score.py | llm_score.py | llm_checklist_score.py |
|------|--------------|--------------|----------------------|
| checklist | 无（TQS V2 规则组件） | 固定 15 项（F-J） | 动态生成 15-35 项 |
| 评分方式 | 确定性规则 | LLM 1-5 五级 | LLM 0/1 二值 |
| 聚合指标 | fail-soft 加权 composite | 归一化 composite | ISR + CSR |
| 指令来源感知 | 否 | 否 | 是（多来源分类） |
| 速度 | 毫秒级 | 秒级（1 次 API） | 秒级（2 次 API） |
| 成本 | 免费 | 按 token 计费 | 约 2× llm_score（两次调用） |
| 字段前缀 | 无前缀 | `llm_` | `llm_checklist_` |

三套分数独立存储在 `_score` dict 中，下游可按需选用或组合。推荐流程：先跑 `rule_score`（免费、快速），再按需跑 `llm_score` 或 `llm_checklist_score`。

