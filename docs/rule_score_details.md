# 规则打分系统 (v5)

本文档描述 `rule_score.py` 的规则打分框架，用于对 IM 格式的 SWE-bench agent 轨迹数据进行自动化质量评估。

## 概述

打分系统从 5 个维度、10 个子指标对每条轨迹进行评分，最终加权合成一个 `composite_score`（范围 [0, 1]）。支持所有脚手架类型：Claude Code、OpenCode、OpenHands、OpenHands SDK、Terminus2。

## 综合评分公式

```
composite_score = 0.20 × Efficiency + 0.15 × Style + 0.25 × ToolMastery + 0.25 × Completion + 0.15 × Precision
```

| 维度 | 权重 | 子指标 | 子指标权重 |
|------|------|--------|-----------|
| A: Efficiency（效率） | 0.20 | A1 错误重试循环, A2 步数比率 | 0.8, 0.2 |
| B: Style（风格） | 0.15 | B1 动作多样性, B2 观察利用率 | 0.4, 0.6 |
| C: Tool Mastery（工具掌握） | 0.25 | C1 工具调用成功率, C2 工具并行度 | 0.9, 0.1 |
| D: Completion（任务完成） | 0.25 | D1 提交完整性, D2 测试验证 | 0.3, 0.7 |
| E: Precision（代码精度） | 0.15 | E1 文件编辑集中度, E2 删后再改 | 0.85, 0.15 |

v4→v5 调整理由：
- A2(步数比率) 降权：轨迹长度≠质量，长但正确的轨迹不应被惩罚
- C2(并行度) 降权：OH-SDK 91%+ 为 0，CC 32% 为 0，区分力极弱
- D1(提交完整性) 降权：97-100% 饱和在 1.0，无区分力
- E2(删后再改) 降权：97-100% 饱和在 1.0，无区分力
- D2(测试验证) 升权：写+跑测试是可学习的高质量行为
- C1(工具成功率) 升权：反映工具使用质量，对 SFT 学习信号最直接

## 子指标详解

### A: Efficiency（效率）

`efficiency_score = 0.8 × A1 + 0.2 × A2`

#### A1: Error-Retry Cycles（错误重试循环）

衡量 agent 在遇到错误后是否盲目重试相同工具。

```
A1 = 1 - clip(adjusted_cycles / 10, 0, 1)
adjusted_cycles = raw_cycles - n_error_turns × sum(p_i²)
```

- `raw_cycles`：连续两个 assistant turn 中，前一个的 observation 包含错误且下一个使用了相同工具的次数
- `sum(p_i²)`：实际工具使用频率分布的碰撞概率（概率校正），避免工具种类少的脚手架被系统性惩罚
- 错误检测分两级：
  - Tier 1（硬错误）：`command not found`、`Permission denied`、非零退出码、`<tool_use_error>` 等，任何上下文都算错误
  - Tier 2（软错误）：`Traceback`、Python 异常名、`FAILED` 等，仅在非测试输出上下文中匹配（避免 pytest 预期失败被误判）
#### A2: Step Count Ratio（步数比率）

衡量 agent 完成任务的步数是否高效（相对于同脚手架的中位数）。

```
A2 = 1 - normalize(clip(steps / median, 0.5, 3.0))
normalize: [0.5, 3.0] → [0, 1]
```

- `steps`：assistant turn 数量
- `median`：同一数据集内、同一脚手架类型的 assistant turns 中位数
- 步数在中位数 0.5 倍以下得满分，3 倍以上得 0 分

> 注意：A2 分数仅在同一数据集内有可比性，不可跨数据集直接比较。

### B: Style（风格）

`style_score = 0.4 × B1 + 0.6 × B2`

#### B1: Action Diversity（动作多样性）

衡量 agent 使用工具的多样性（Shannon 熵）。

```
B1 = H(tool_types) / log₂(n_available_tools)
H = -Σ p_i × log₂(p_i)
```

- 当提供 `n_available_tools` 时，用 `log₂(n_available_tools)` 归一化（衡量工具使用广度）
- 否则用 `log₂(n_unique_types_used)` 归一化（衡量工具使用均匀度）
- Terminus2 使用固定常量 18 作为可用工具数

#### B2: Observation Utilization（观察利用率）

衡量 agent 是否利用了工具返回的信息来指导下一步操作。

```
B2 = mean(overlap_i / |top_K_keywords_i|)  对每个 observation i
```

- 从每个 observation 中提取频率最高的 K=10 个关键词（≥4 字符的标识符，排除停用词）
- 检查这些关键词在下一个 assistant 消息（content + tool_call arguments）中的复现比例
- 关键词提取和匹配均限制在前 5000 字符内
### C: Tool Mastery（工具掌握）

`tool_mastery_score = 0.9 × C1 + 0.1 × C2`

#### C1: Tool Call Success Rate（工具调用成功率）

衡量工具调用的成功比例。

```
C1 = 非错误 observation 数 / 总 observation 数
```

- 对 Terminus2 做加权处理：每个 observation 覆盖前一个 assistant turn 的所有 commands，若报错按 `1/n_commands` 计为失败
- 错误检测同样区分测试输出和非测试输出上下文

#### C2: Tool Call Parallelism（工具并行度）

衡量 agent 是否在单个 turn 中并行调用多个工具。

```
C2 = clip((mean(calls_per_turn) - 1) / (cap - 1), 0, 1)
cap = 5
```

- 以 1 为基线（每 turn 至少 1 次调用），只衡量"额外"并行度
- Terminus2 统计每个 assistant turn 的 commands 数量

### D: Completion（任务完成）

`completion_score = 0.3 × D1 + 0.7 × D2`

#### D1: Submission Completeness（提交完整性）

衡量 agent 是否正常完成并提交了任务。

| 脚手架 | 1.0（正常提交） | 0.5 | 0.0（截断） |
|--------|----------------|-----|-------------|
| OpenHands / SDK | 最后 assistant 调用了 `finish` | 最后消息是无 tool_calls 的 assistant | 其他 |
| Terminus2 | 最后 assistant 的 `task_complete=true` | — | 其他 |
| CC / OC | 最后消息是无 tool_calls 的 assistant | — | 最后是 tool 响应或带 tool_calls 的 assistant |

#### D2: Test Verification（测试验证）

衡量 agent 是否编写并运行了测试。

```
D2 = 0.5 × has_test_write + 0.5 × has_test_run
```

- `has_test_write`：是否通过 edit/write 工具修改了测试文件（匹配多语言测试文件命名模式）
- `has_test_run`：是否通过 bash/terminal 执行了测试命令（匹配 pytest、go test、cargo test、jest 等）

支持的测试文件模式：Python (`test_*.py`, `*_test.py`)、Go (`*_test.go`)、C/C++ (`*_test.cc`)、Rust (`tests/*.rs`)、Java (`*Test.java`)、JS/TS (`*.test.js`, `*.spec.ts`) 等。
### E: Precision（代码精度）

`precision_score = 0.85 × E1 + 0.15 × E2`

#### E1: File Edit Concentration（文件编辑集中度）

衡量编辑操作是否集中（避免对同一文件反复修改）。

```
E1 = 1 - clip((mean_edits_per_file - 1) / 4, 0, 1)
```

- `mean_edits_per_file`：总编辑次数 / 被编辑的唯一文件数
- 平均每文件编辑 1 次得满分，5 次以上得 0 分
- 编辑来源：edit/write 工具调用 + bash 命令中的文件写操作（`sed -i`、`cat >`、`tee`、`patch` 等）
- 文件路径归一化：去除 `/workspace/`、`/testbed/`、`/repo/` 等前缀

#### E2: Delete-then-Modify（删后再改）

衡量 agent 是否存在"先删除文件再重新创建/修改"的反模式。

```
E2 = 1 - clip(dtm_count / 3, 0, 1)
```

- 追踪所有 `rm` 命令删除的文件，检查后续是否对这些文件进行了写操作
- 0 次得满分，3 次以上得 0 分（渐进式惩罚）

## 脚手架自动检测

`detect_scaffold()` 根据 IM 记录的结构自动判断脚手架类型：

| 条件 | 脚手架类型 |
|------|-----------|
| 无 `tools` 字段 | Terminus2 |
| 工具名是 `{terminal, file_editor, task_tracker, finish, think}` 的子集 | OpenHands SDK |
| 包含 CC 特征工具名（bash, read, edit, write, glob, grep 等） | Claude Code（system message 含 "claude code" 或 "anthropic"）或 OpenCode（默认） |
| 其他有 tools 的情况 | OpenHands |

## 使用方法

### 单文件打分

```bash
conda activate swelf
python -m swe_data_process.rule_score --input <im.jsonl> [--output <scored.jsonl>] [--max-instances N] [--scaffold TYPE]
```

参数说明：
- `--input / -i`：输入 IM JSONL 文件路径（必需）
- `--output / -o`：输出打分后的 JSONL 文件路径（默认 `<input_stem>_scored.jsonl`）
- `--max-instances`：最多处理的记录数
- `--quiet`：减少日志输出
- `--scaffold`：强制指定脚手架类型，可选值 `claudecode / opencode / openhands / openhands_sdk / terminus2`（跳过自动检测，用于调试）

## 打分流程

1. 加载 JSONL 记录
2. Pass 1：按脚手架分组统计 assistant turns，计算各组中位数（用于 A2）。仅统计 main agent 记录，subagent 记录（`_agent_type == "subagent"`）不参与中位数计算
3. Pass 2：逐条打分。subagent 记录跳过打分，`_score` 设为 `null`；main agent 记录计算 10 个子指标 → 5 个维度分 → 综合分
4. 打分结果写入每条记录的 `_score` 字段

## 输出格式

每条记录的 `_score` 字段包含：

```json
{
  "scaffold": "claudecode",
  "assistant_turns": 15,
  "total_tool_calls": 42,
  "error_retry_cycles": 2,
  "a1_error_retry": 0.85,
  "a2_step_count_ratio": 0.72,
  "b1_action_diversity": 0.65,
  "b2_observation_utilization": 0.18,
  "c1_tool_success_rate": 0.90,
  "c2_tool_parallelism": 0.12,
  "d1_submission_completeness": 1.0,
  "d2_test_verification": 0.5,
  "e1_file_edit_concentration": 0.88,
  "e2_delete_then_modify": 1.0,
  "efficiency_score": 0.785,
  "style_score": 0.415,
  "tool_mastery_score": 0.51,
  "completion_score": 0.75,
  "precision_score": 0.94,
  "composite_score_v3": 0.60,
  "composite_score": 0.68
}
```

其中 `composite_score_v3 = 0.5 × efficiency + 0.5 × style` 和 `composite_score_v4` 为向后兼容的旧版本分数。

对于 subagent 记录（`_agent_type == "subagent"`），`_score` 字段为 `null`，不参与打分。

## 版本演进

- v3：2 组 4 指标（Efficiency + Style），用于早期 HuggingFace 数据集子集筛选
- v4：5 组 10 指标，新增 Tool Mastery、Completion、Precision 三个维度，子指标等权 mean
- v5（当前）：子指标加权 + 组权重调整。降低饱和/低区分力指标（A2、C2、D1、E2），提升有信号的指标（C1、D2、E1）。解决 v4 中 top/bottom 筛选主要按轨迹长度排序、与下游 SFT 效果不相关的问题
