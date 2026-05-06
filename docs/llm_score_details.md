# LLM-as-Judge 打分系统 (v2)

本文档描述 `llm_score.py` 的 checklist-based LLM-as-judge 打分框架，参考 MiniMax-AI/mini-vela 方案，对 IM 格式轨迹进行语义质量评估。与 `rule_score.py` 的规则打分互补。

## 概述

打分系统使用固定通用 checklist（5 个类别、15 个检查项），由 LLM 逐项给出 1-5 五级评分，最终归一化到 [0, 1] 作为综合分。类别编号 F-J，避免与规则打分的 A-E 冲突。

v2 相比 v1 的主要变化：将二值 pass/fail 改为五级评分（1-5），每个检查项附带具体的五级评分标准（scoring 字段），提升区分度。

## Checklist 结构

| 类别 | 描述 | 检查项 |
|------|------|--------|
| F: Problem Understanding（问题理解） | Agent 对问题的理解程度 | F1 诊断深度, F2 范围精准度, F3 计划质量 |
| G: Solution Quality（解决方案质量） | 修复的质量和健壮性 | G1 修复优雅度, G2 修改最小化, G3 健壮性 |
| H: Reasoning Quality（推理质量） | 推理链路的合理性 | H1 推理连贯性, H2 假设驱动, H3 适应能力 |
| I: Verification Rigor（验证严谨性） | 验证行为的完整性 | I1 问题复现, I2 修复验证, I3 测试质量 |
| J: Efficiency（效率） | 操作的效率 | J1 导航效率, J2 工具熟练度, J3 迭代经济性 |

## 检查项详解

每项评分标准：1=完全未做到/严重缺陷, 2=尝试了但效果差, 3=基本做到但有明显不足, 4=做得好只有小瑕疵, 5=表现优秀无明显缺陷

### F: Problem Understanding（问题理解）

#### F1: Diagnosis Depth（诊断深度）
Agent 对问题的诊断深度：是否理解了根因、影响范围和边界条件。
- 1: 未识别或严重误解问题
- 2: 识别了表面问题
- 3: 理解了核心问题但遗漏细节
- 4: 准确理解根因和影响范围
- 5: 深入理解根因、边界条件和潜在风险

#### F2: Scope Precision（范围精准度）
Agent 对修改范围的精准度：是否精确定位了需要修改的文件，无遗漏无冗余。
- 1: 严重遗漏关键文件或大量修改无关代码
- 2: 找到部分相关文件
- 3: 覆盖了主要文件但有遗漏或少量多余
- 4: 精准覆盖所有需修改的文件
- 5: 精准定位且理解文件间依赖关系

#### F3: Plan Quality（计划质量）
Agent 的修复计划质量：是否在动手前形成了合理、完整的修复方案。
- 1: 无计划直接动手
- 2: 有模糊想法但不完整
- 3: 有基本计划但缺少对边界情况的考虑
- 4: 计划合理完整
- 5: 计划周密，考虑了边界条件、兼容性和回退方案

### G: Solution Quality（解决方案质量）

#### G1: Fix Elegance（修复优雅度）
修复方案的优雅程度：是否符合项目惯例、简洁且可维护。
- 1: 修复无效或引入新 bug
- 2: 能工作但方式笨拙
- 3: 基本正确但不够优雅
- 4: 简洁正确，符合项目风格
- 5: 优雅简洁，完全融入项目惯例，可维护性强

#### G2: Change Minimality（修改最小化）
修改的最小化程度：每一行改动是否都直接服务于修复。
- 1: 大量无关改动
- 2: 较多不必要改动
- 3: 基本聚焦但有少量多余
- 4: 改动精准，几乎无多余
- 5: 每一行都是必要的，零冗余

#### G3: Robustness（健壮性）
修复的健壮性：是否考虑了边界条件、异常情况和向后兼容。
- 1: 修复脆弱，明显遗漏边界情况
- 2: 基本场景可用但有隐患
- 3: 覆盖了主要场景
- 4: 考虑了大部分边界条件
- 5: 全面考虑边界条件、异常处理和向后兼容

### H: Reasoning Quality（推理质量）

#### H1: Reasoning Coherence（推理连贯性）
推理链路的连贯性：每步是否有明确依据，逻辑是否清晰。
- 1: 推理混乱或跳跃
- 2: 有基本思路但逻辑不严密
- 3: 大体连贯但有跳跃
- 4: 每步有明确依据，链路清晰
- 5: 推理严密，每步都有充分论证

#### H2: Hypothesis Driven（假设驱动）
Agent 是否采用假设驱动的方法：先形成假设再验证，而非盲目尝试。
- 1: 盲目尝试，无假设
- 2: 有隐含假设但未验证
- 3: 有假设但验证不充分
- 4: 明确假设并通过代码阅读或测试验证
- 5: 系统性地提出、验证和排除假设

#### H3: Adaptability（适应能力）
遇到障碍时的适应能力：是否能快速调整策略而非重复失败操作。
- 1: 重复相同失败操作或卡住
- 2: 多次尝试后才调整
- 3: 有调整但效率低
- 4: 较快分析原因并调整策略
- 5: 立即识别问题并高效切换策略（若无障碍则评估预防性思考）

### I: Verification Rigor（验证严谨性）

#### I1: Reproduction（问题复现）
Agent 是否在修复前复现了问题，确认问题确实存在。
- 1: 未做任何复现
- 2: 运行了测试但未针对目标问题
- 3: 尝试复现但方式不够针对性
- 4: 运行了针对性测试确认问题存在
- 5: 系统性复现并理解了问题的触发条件

#### I2: Fix Verification（修复验证）
Agent 是否在修复后验证了修复的有效性。
- 1: 修复后未做任何验证
- 2: 运行了测试但未确认目标问题已修复
- 3: 做了基本验证
- 4: 运行针对性测试确认问题已解决
- 5: 充分验证修复有效且无副作用

#### I3: Test Quality（测试质量）
测试的质量和覆盖度：是否覆盖了主要场景和边界条件。
- 1: 未运行任何测试
- 2: 只运行了最基本的测试
- 3: 覆盖了主要场景
- 4: 覆盖了主要场景和部分边界条件
- 5: 全面覆盖，包括回归测试、边界条件和异常路径

### J: Efficiency（效率）

#### J1: Navigation Efficiency（导航效率）
代码导航效率：是否快速精准地定位到相关文件和代码。
- 1: 导航混乱，长时间找不到
- 2: 多次错误搜索后找到
- 3: 过程有些曲折但最终找到
- 4: 较高效地定位
- 5: 搜索路径直接精准，几乎无浪费

#### J2: Tool Proficiency（工具熟练度）
工具使用的熟练度：是否正确高效地使用了可用工具。
- 1: 工具使用错误频繁
- 2: 能用但效率低
- 3: 基本正确但有改进空间
- 4: 工具使用正确高效
- 5: 熟练运用各种工具，选择最优工具完成任务

#### J3: Iteration Economy（迭代经济性）
迭代经济性：总步骤数是否合理，有无大量浪费的步骤。
- 1: 大量浪费步骤（如反复安装依赖、重复搜索）
- 2: 较多冗余步骤
- 3: 有一些冗余但总体可接受
- 4: 步骤精简高效
- 5: 每一步都有明确目的，零浪费

## 评分流程

1. 加载 IM JSONL 记录
2. 跳过 subagent 记录（`_agent_type == "subagent"`）和已有 LLM 分数的记录（支持断点续评）
3. 对每条记录：
   - 深拷贝并截断消息：tool result（≤5K 字符）、assistant content（≤50K，支持 string 和 list 格式）、reasoning_content（≤50K）
   - 将 tools 和 messages 序列化为 JSON 字符串（mini-vela 风格，保留结构信息）
   - 构造 LLM judge prompt，包含 tools JSON + messages JSON + checklist JSON
   - 调用 LLM（默认 gpt-4o，temperature=0），要求输出严格 JSON（保持 checklist 原有 category 结构）
   - 解析 JSON 响应，逐项提取 1-5 评分
4. 计算分数：每个类别的归一化分 + 综合分
5. 结果以 `llm_` 前缀写入 `_score` dict

## 分数聚合

每个类别的分数归一化到 [0, 1]：

```
category_score = (sum_of_check_scores - n * 1) / (n * (5 - 1))
```

其中 n=3（每类 3 个检查项），sum 范围 [3, 15]，归一化后 [0, 1]。

综合分为 5 个类别分的算术平均：

```
llm_composite_score = mean(F_score, G_score, H_score, I_score, J_score)
```

具体字段：

```
llm_composite_score          = mean of 5 category scores
llm_f_problem_understanding  = (F1+F2+F3 - 3) / 12
llm_g_solution_quality       = (G1+G2+G3 - 3) / 12
llm_h_reasoning_quality      = (H1+H2+H3 - 3) / 12
llm_i_verification_rigor     = (I1+I2+I3 - 3) / 12
llm_j_efficiency             = (J1+J2+J3 - 3) / 12
```

所有分数范围 [0, 1]，精度 4 位小数。五级评分下每个类别有 13 种可能值（sum 从 3 到 15），composite 有更细的粒度，区分度远优于 v1 的二值评分。

## 输出格式

LLM 打分结果合并到现有 `_score` dict，使用 `llm_` 前缀：

```json
{
  "composite_score": 0.72,
  "efficiency_score": 0.68,
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
    "total_score": 52,
    "total_max": 75,
    "by_category": { "F": {"score": 12, "max": 15, "count": 3}, "..." : "..." }
  },
  "llm_raw_response": { "..." : "..." }
}
```

- `llm_raw_response` 保留 LLM 的完整 JSON 输出（含每项的 reasoning 和 score），便于人工审查
- `llm_detailed_results` 提供 by_category 的细粒度统计（原始 1-5 分值，未归一化）

## 使用方法

```bash
conda activate swelf
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://..."  # 可选，用于兼容 API

# 对单个 IM JSONL 文件进行 LLM 打分
python -m swe_data_process.llm_score \
    --input artifacts/cc_jierun_im.jsonl \
    --output artifacts/cc_jierun_im_llm_scored.jsonl \
    --model gpt-4o --concurrency 10

# 预估费用（不调用 API）
python -m swe_data_process.llm_score --input artifacts/cc_jierun_im.jsonl --dry-run

# 用于不支持 response_format 的 API
python -m swe_data_process.llm_score --input ... --no-json-mode
```

参数说明：
- `--input`：输入 IM JSONL 文件路径（必需）
- `--output`：输出 JSONL 文件路径（默认 `<input_stem>_llm_scored.jsonl`）
- `--model`：LLM judge 模型名称（默认 `gpt-4o`）
- `--api-key`：OpenAI API key（默认从 `OPENAI_API_KEY` 环境变量读取）
- `--base-url`：OpenAI-compatible API base URL（默认从 `OPENAI_BASE_URL` 环境变量读取）
- `--concurrency`：并发请求数（默认 10）
- `--max-instances`：最多处理的记录数
- `--dry-run`：仅估算 token 用量和费用
- `--no-json-mode`：禁用 `response_format=json_object`
- `--quiet`：减少日志输出

## 成本估算

| 模型 | 单条估算 | 1000 条 |
|------|---------|---------|
| gpt-4o | ~$0.10 | ~$100 |
| gpt-4o-mini | ~$0.005 | ~$5 |
| claude-opus-4-6 | ~$0.20 | ~$200 |

建议先用 gpt-4o-mini 跑全量，再用 gpt-4o 抽样验证一致性。

## 与规则打分的关系

| 维度 | 规则打分 (rule_score.py) | LLM 打分 (llm_score.py) |
|------|------------------------|------------------------|
| 评估方式 | 确定性规则、正则匹配 | LLM 语义理解 |
| 评估内容 | 结构信号（错误率、步数、工具使用） | 语义质量（推理、代码正确性、策略） |
| 速度 | 毫秒级 | 秒级（API 调用） |
| 成本 | 免费 | 按 token 计费 |
| 可复现性 | 完全确定性 | temperature=0 近似确定 |
| 字段前缀 | 无前缀（`composite_score`） | `llm_` 前缀（`llm_composite_score`） |

两套分数独立存储在 `_score` dict 中，下游可按需选用或组合。

