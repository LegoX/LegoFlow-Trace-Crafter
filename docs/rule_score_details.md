# 规则打分系统 (TQS V2)

本文档描述 `rule_score.py` 的 TQS V2 (Trajectory Quality Score) 规则打分框架，用于对 IM 格式的 SWE-bench agent 轨迹进行自动化质量评估。该模块支持 Claude Code、OpenCode、OpenHands、OpenHands SDK、Terminus2。

## 概述

TQS V2 不再使用旧 v5 的 5 组 10 子指标框架，而是直接计算一组轨迹级组件，并通过 fail-soft 加权合成 `composite_score`（范围 [0, 1]）。

核心设计：

- 有信号的组件参与 `composite_score`：`SUB`、`STP`、`TVR`、`FEC`、`DPI`
- 诊断组件仍输出到 `_score`：`OEC`、`IAC`、`PED`、`PSN`、`TTE`、`SCP`
- 某些组件没有足够数据时返回 `null`，聚合时自动跳过
- subagent 记录（`_agent_type == "subagent"`）不打分，`_score` 写为 `null`

## 综合评分公式

```text
composite_score = Σ(weight_i × transformed(component_i)) / Σ(weight_i)
```

仅对有值且权重非 0 的组件求和。

| 组件 | 权重 | 是否输出 | 说明 |
|------|------|----------|------|
| `sub_score` | 0.33 | 是 | 提交完整性，结合正常收尾、后期错误率和后期测试 |
| `stp_score` | 0.27 | 是 | 步数效率，assistant turn 数是否落在合理范围 |
| `tvr_score` | 0.23 | 是 | 测试验证，是否写测试、跑测试、最后一次测试是否成功 |
| `fec_score` | 0.10 | 是 | 文件编辑集中度，聚合前使用 `FEC^5` 降低高分饱和 |
| `dpi_score` | 0.07 | 是 | 脏模式惩罚，聚合前使用 `DPI^3` 放大低质量差异 |
| `oec_score` | 0.00 | 是 | 观察熵坍缩，仅诊断 |
| `iac_score` | 0.00 | 是 | 意图-行动一致性，仅诊断 |
| `ped_score` | 0.00 | 是 | 感知-编辑漂移，仅诊断 |
| `psn_score` | 0.00 | 是 | 路径稳定性，仅诊断 |
| `tte_score` | 0.00 | 是 | 工具类型转移熵，仅诊断 |
| `scp_score` | 0.00 | 是 | 首次有效编辑时机，仅诊断 |

## 指标详解

### SUB: Submission Completeness

衡量轨迹是否正常收尾，并根据后期执行质量做轻微调整。

基础分：

| 脚手架 | 满分条件 | 半分条件 | 0 分条件 |
|--------|----------|----------|----------|
| Claude Code / OpenCode | 最后一条消息是无 `tool_calls` 的 assistant | 无 | 最后仍停在 tool 响应或带 `tool_calls` 的 assistant |
| OpenHands / OpenHands SDK | 最后 assistant 调用了 `finish` | 最后消息是无 `tool_calls` 的 assistant | 其他 |
| Terminus2 | 最后 assistant JSON 中 `task_complete=true` | 无 | 其他 |

附加调整：

- 后 30% observation 的错误率越低，保留越多基础分
- 轨迹后 20% assistant turn 中运行过测试，最多额外加 0.15

### STP: Step Efficiency

衡量 assistant turn 数是否处在合理范围。

```text
5 <= assistant_turns <= 80: 1.0
assistant_turns < 5: assistant_turns / 5
assistant_turns >= 200: 0.0
80 < assistant_turns < 200: quadratic decay
```

区间与当前常见 agent `max_turn=200` 对齐：满分覆盖中短轨迹，打满 turn cap 视为步数效率为 0。TQS V2 不再依赖数据集内同脚手架的中位步数，因此 `score_dataset()` 是单遍扫描，`score_record()` 不需要 `median_steps`。

### TVR: Test Verification

衡量 agent 是否建立了测试验证闭环。

```text
TVR = 0.3 × has_test_write + 0.3 × has_test_run + 0.4 × late_test_success
```

- `has_test_write`：是否通过 edit/write 工具或 bash 写入测试文件
- `has_test_run`：是否运行测试命令
- `late_test_success`：最后一次测试运行的结论，三档取值

| 结论 | 取值 | 含义 |
|------|------|------|
| `pass` | 1.0 | 输出里有明确的通过摘要 |
| `unknown` | 0.6 | 跑了测试，但输出看不出结论（常见于 `\| tail -5` 只截了几行日志） |
| `fail` | 0.3 | 有明确失败信号 |

`unknown` 档是必要的：早期实现把「看不出结论」等同于「通过」，而失败识别又只覆盖 Python，导致非 Python 语言的失败被系统性判成通过。单列一档后，「检测不到」不再等价于满分。

判定由 `_test_run_outcome()` 完成，与 `_is_error_result()`（工具调用错误率口径）**分离** —— 测试断言失败是有效的验证行为，不应计入工具调用失败。判定顺序：工具层硬失败 → 显式失败摘要 → **零测试跑成** → 显式通过摘要 → agent 回显的零退出码 → soft error 兜底 → unknown。

把「显式通过」排在 soft error 之前，是因为一个通过的测试完全可能在输出里合法地打印 `TypeError` / `No such file or directory`（正是它在测的错误路径）。

「零测试跑成」（`Ran 0 tests` / `no tests ran|were found` / `collected 0 items`）必须排在通过摘要**之前**并判为 `unknown`：一个测试都没跑起来时 runner 仍会打印裸 `OK`、`BUILD SUCCESS`、退出码 0。同理，所有通过模式里的计数都要求**非零** —— `0 passing`、`OK (0 tests)`、`Tests run: 0, Failures: 0, Errors: 0`（Maven 多模块里没有测试的模块）都不是「通过」。刻意不把 Go 的 `[no test files]` 计入：那是多包运行里没有测试的单个包，同一次运行的其它包仍在正常跑。

覆盖的摘要形态取自真实轨迹，含 Go（`FAIL`，注意**不是** `FAILED`）、Rust（`test result: FAILED`、`error[E….]`）、Maven（`Tests run: N, Failures: M`、`BUILD FAILURE`）、Gradle、CTest（`The following tests FAILED`）、GoogleTest（`[  FAILED  ]`）、Jest/Vitest（`Tests: N failed`）、Mocha（`N failing`）、PHPUnit（`FAILURES!`）、RSpec、pytest、unittest（`Ran N tests` + `OK`）等。计数类模式一律排除 0（`0 failed` / `0 failures` 是**通过**时的正常输出）。

测试命令识别包括 `pytest`、`python -m unittest`、`go test`、`cargo test`、`ctest`、`jest`、`vitest`、`npm test`、`make test`、Django 的 `runtests.py` / `manage.py test` 等。Maven/Gradle 允许 goal 与命令之间隔着 flag 与模块选择器（`mvn -o -q test`、`./mvnw -q -pl mod -am test`、`./gradlew -q :core:test`），并排除 Gradle 的 `-x test`（排除测试）。

测试文件识别覆盖 Python、Go、Rust、Java/Kotlin/Scala、JS/TS、PHP、C#、C/C++、Perl 的常见命名与目录约定，例如 `test_*.py`、`*_test.go`、`tests/*.rs`、`src/test/java/**`、`*.test.ts`、`*Test.php`、`*Tests.cs`、`t/*.t`。

此外 TVR **额外**识别 SWE-agent 常用的非正式验证习语：`python -c` 内联自测、`reproduce`/`repro`/`verify`/`smoke` 命名的脚本，以及「跑一遍程序」（`go run`、`cargo run`、`java Foo`、`ruby app.rb`、`python app.py`、`node server.js`、`./x.sh`）。这是有意的设计口径——编译/解释型语言下「跑一遍看会不会崩」就是其复现验证方式——现已对所有语言一致适用（早期实现把 Python 与 Node 排除在外）。安装/打包/起开发服务器（`pip install`、`python setup.py build`、`npm run dev`、`cargo build`、`manage.py runserver`）不计入。

这些宽口径信号**仅用于 TVR**，不进入错误抑制路径——因此一个真正报错的 `python -c` traceback 不会被当成“测试输出”而在 DPI / 错误率中被吞掉。

**命令判定按 shell 段进行**：段首是 `grep`/`rm`/`git`/`cat`/`find`/`ls` 等只读命令时整段跳过。否则 `git diff .../src/main/java/Foo.java` 会因路径里的 `java`、`rm -f vitest-tmp.config.ts` 会因文件名里的 `vitest` 被误判成跑测试。段拆分是引号感知的，避免把 `python3 -c "a='x'; b=1"` 的脚本正文切碎；被用来改文件的 `python -c`（正文含 `open(...,'w')` / `.write(`）判为编辑而非自测。

### FEC: File Edit Concentration

衡量编辑操作是否过度集中在少量文件上。

```text
FEC = 1 - clip((mean_edits_per_file - 1) / 4, 0, 1)
```

- 平均每个文件编辑 1 次得满分
- 平均每个文件编辑 5 次及以上得 0 分
- 编辑来源包括 edit/write 工具（含 Claude Code 的 `MultiEdit` / `NotebookEdit`、OpenCode 的 `patch`）、OpenHands editor 工具，以及 bash 中的 `sed -i` / `gsed -i` / `perl -pi`、重定向、`tee`、`patch` 等写入操作
- 执行类工具的判别以**调用参数**为准（`_tool_call_is_execution` / `_exec_cmd_text`），与 TVR、错误检测同一套逻辑。早期实现在此处用的是工具名单 + `args["command"]`，导致改名 shell 工具、`execute_ipython_cell`（参数是 `code`）、用 `cmd`/`keystrokes` 键的工具，其 bash 编辑被整体漏掉
- **没有检测到任何编辑时返回 `null`**（fail-soft，聚合时跳过），而不是 1.0。返回 1.0 等于把「检测盲区」和「完美的编辑集中度」判成同一件事，而聚合层还要取 `FEC^5`，于是「一次编辑都没抓到」拿满分、「抓到 3 次同文件编辑」只剩 0.031，漏抓反而成了加分项
- 路径会去除 `/workspace/`、`/testbed/`、`/repo/`、`/home/swe-bench/` 等前缀后归一化

### DPI: Dirty Pattern Index

衡量明显的低质量轨迹模式。分数越高代表脏模式越少。

惩罚来源：

- 轨迹截断或非正常结束
- 从未出现成功写操作
- 工具/动作序列出现长循环（签名为 `tool + editor子命令 + target`；`file_editor view` 与 `str_replace` 视为不同动作，避免把同文件上的正常查看→编辑迭代误判为循环）
- observation 中重复出现相同错误

聚合时使用 `DPI^3`，用于放大低分轨迹和高分轨迹之间的差异。

### OEC: Observation Entropy Collapse

诊断 observation 是否出现熵坍缩，例如反复输出相同错误或重复文本。

- 至少需要 5 个 observation，否则返回 `null`
- 使用前 5 个 observation 的字符熵作为 baseline
- 对后续 observation 做 3 点平滑，取相对 baseline 的最小值作为分数

### IAC: Intent-Action Consistency

诊断 assistant 文本中的意图是否与实际工具动作一致。

- 从 assistant 的 `reasoning_content + content` 中匹配读、写、执行、搜索、提交等意图关键词
- 与同一 turn 的工具类型进行对齐
- 少于 3 个可判断 turn 时返回 `null`

### PED: Perception-Edit Drift

诊断 agent 编辑的文件是否偏离前文读取或观察到的文件。

- 追踪 observation 关联的目标文件
- 检查后续编辑文件与近期感知文件的 Jaccard 相似度
- 只在有足够读写上下文时输出有效分数

### PSN: Path Stability

诊断连续窗口内目标文件集合是否稳定。

- 默认窗口大小为 5 个 assistant turn
- 比较窗口首尾目标文件集合的 Jaccard 相似度
- 频繁大幅切换目标路径会降低分数

### TTE: Tool Transition Entropy

诊断工具类型转移是否过于单一。

- 统计相邻 action 类型的转移熵
- action 少于 3 个时返回 `null`
- 归一化到 `[0, 1]`

### SCP: Successful Change Position

诊断首次成功写操作出现的时机。

- 首次成功写操作出现在 20%-50% assistant turn 区间内得满分
- 过早或过晚都会按距离衰减
- 没有成功写操作时为 0

## 错误检测

### 0. 文本归一（前置）

所有基于 observation 内容的匹配都先经 `_scan_window()`：

- **剥离 ANSI 转义**。命令 observation 普遍含 ANSI（`grep --color` 与彩色 runner 的输出）。转义序列会插进**词内部**，例如 Maven 的 `Tests run\x1b[m\x1b[K: 6, \x1b[01;31m\x1b[KFailures\x1b[m\x1b[K: 2`，其中 `\x1b[K` 的 `K` 是 word 字符，会让 `\bFAILED\b` 这类边界锚定完全失效。
- **同时取输出的开头与结尾**（各 3000 字符）。命令输出的结论（测试摘要、`BUILD FAILURE`、退出码回显）通常在末尾，中间是大段正常日志；只扫开头会让超长输出的失败摘要落在窗口之外。
- **尾窗对齐到行首**。`text[-3000:]` 会从任意字符位置切开，与开头窗口拼接后，被截断的**行中间**片段就成了一个伪造的行首，使 `^FAIL`、`^(?:FAILED|ERROR)\s`、`^\s*\[ERROR\]` 这些刻意行首锚定的模式命中根本不在行首的文本（例如正常日志里的 `... state was FAILED: cleanup done`）。因此尾窗先向前找到第一个换行再拼接（只在开头 300 字符内找，超过则视为超长单行不丢）。
- 兼容 list 型 content（content-block 列表），避免正则抛 `TypeError`。

> **注意**：OpenHands SDK 回填的 `[The command completed with exit code N.]` **不可用作失败信号**。命令通常写成 `<runner> ... 2>&1 | tail -40`，管道使 shell 退出码恒为 `tail` 的 0，该标记因而与测试是否通过无关。真正可用的是 agent 自己回显的 `${PIPESTATUS[0]}` 与输出内容本身。

错误检测分为两个层次：**显式工具失败标记**和**命令输出错误信号**。关键区别在于后者只对“执行类”工具的 observation 生效，避免把文件查看/编辑工具返回的源码内容误判为工具调用错误（历史上工具调用错误率假阳性的主因）。

### 1. 显式工具失败标记（对所有工具类型生效）

由脚手架/运行时注入，独立于工具输出内容，因此命中即视为工具调用错误，与工具是否为执行类无关：

- `<tool_use_error>`
- `The arguments provided to the tool are invalid`
- `[An error occurred during execution.]`（OpenHands SDK 工具执行失败）
- `Error validating args ... for tool`（OpenHands SDK 参数校验失败）
- `Error executing tool`（OpenHands SDK 工具执行异常）
- 行首 `ERROR:`（OpenHands `str_replace_editor` 失败块）
- `No replacement was performed`
- `parameter is required for '<cmd>' command`
- `` Invalid `<param>` parameter ``

这些标记均为起始锚定或高度特异，源码内容不会误命中。

### 2. 命令输出错误信号（仅对执行类工具生效）

只有执行类工具（bash/terminal/ipython 等）的 observation 内容才是命令的真实 stdout/stderr，此时才扫描下列模式；对文件查看/编辑/搜索类工具一律跳过。

硬错误（执行类工具，任何上下文）：

- `command not found`
- `Permission denied`
- 非零退出码（`exit code: N`）
- `returned non-zero exit status`

软错误（执行类工具，且仅在非测试输出上下文中匹配，避免把正常的测试失败误判为工具调用失败）：

- Python traceback
- 常见异常名，例如 `TypeError`、`ImportError`、`FileNotFoundError`
- `No such file or directory`
- `FAILED`

### 执行类 / 非执行类工具判别

工具名常被各 SWE 数据源重命名（`shell_exec`/`run_command`/`code_editor`/...），因此判别**以工具调用参数为准**，工具名单仅作回退：

- `command` 取编辑器子命令（`view`/`create`/`str_replace`/`insert`/`undo_edit`），或携带编辑器专属参数（`old_str`/`new_str`/`file_text`/`view_range`/`insert_line`）→ 文件查看/编辑工具，**非执行类**
- 携带 `task_list`/`thought`/`task_completed` → `task_tracker`/`think`/`finish`，**非执行类**（`task_tracker` 也带 `command` 参数，但取值是 `plan`/`add` 而非 shell 命令；只看「command 非空」会把它判成执行类）
- 否则若携带非空 shell 命令参数（`command`/`cmd`/`keystrokes`）→ **执行类**
- 都不满足时回退到执行类工具名单（`bash`/`terminal`/`execute_bash`/`shell`/`ipython`/`python` 等）

Terminus2 的 observation 全部是终端命令输出，按执行类处理。

**这套判别必须在所有需要「该 tool_call 是否执行了命令 / 通过 shell 改了文件」的地方统一使用**，共 7 个调用点：`_extract_observations`、`_extract_edit_paths`（FEC）、`_compute_tvr`、`_has_successful_write`（DPI）、`_first_successful_write_turn`（SCP）、`_per_step_target_files`（PSN）、`_action_target_files` / `_loop_signature_for_action`（PED、DPI loop）、`_classify_action`（IAC、TTE）。shell 编辑路径统一走 `_exec_bash_edit_paths()`。

只在部分调用点使用会造成**同一条轨迹仅因工具改名就换一个分数**：把 `execute_bash` 改名成只声明 `command` 参数的 `shell_exec`（`_infer_canonical_name` 无法据此归一）后，DPI 会因误判「从未成功写入」而扣 0.40、SCP 判 0，而参数完全一致。

## 工具名归一化（预处理）

部分数据集（如工具改名增广的 OpenHands 轨迹）把标准工具改成了任意同义名：`execute_bash` → `shell_exec`、`str_replace_editor` → `edit_file`、`finish` → `end_task`、`think` → `consider`、`task_tracker` → `planning` 等。这些工具的**参数 schema 与标准工具完全一致，只是名字变了**。

`score_record()` 在打分前先调用 `_canonicalize_record()`，按工具声明的参数签名（`_infer_canonical_name`）把改名工具归一回标准名：

| 参数特征 | 归一为 |
|----------|--------|
| 含 `task_completed` | `finish` |
| 仅有 `thought`（无 `command`/`path`） | `think` |
| 含 `task_list` | `task_tracker` |
| 含 `old_str`/`new_str`/`file_text`/`view_range` | `str_replace_editor` |
| 含 `command` 且带 `is_input`/`timeout` | `execute_bash` |
| 含 `command` + `path` 但无 shell 特征 | `str_replace_editor` |

归一后 `detect_scaffold()` 与所有下游指标（SUB/TVR/DPI/SCP/FEC...）即可正常工作，无需改动各 helper。该步骤返回新记录、不修改入参，且尽量浅拷贝以避免复制大字段。

## 脚手架自动检测

`detect_scaffold()` 根据 IM 记录结构自动判断脚手架类型：

| 条件 | 脚手架类型 |
|------|------------|
| 无 `tools` 字段，或 `tools` 为空 | `terminus2` |
| 工具名是 `{terminal, file_editor, task_tracker, finish, think}` 的子集 | `openhands_sdk` |
| 包含 CC/OC 特征工具名，且 system message 开头包含 `claude code` 或 `anthropic` | `claudecode` |
| 包含 CC/OC 特征工具名，但不满足 Claude Code system message 条件 | `opencode` |
| 其他有 `tools` 的情况 | `openhands` |

## 使用方法

### 单文件打分

```bash
conda activate swelf
python -m swe_data_process.rule_score --input <im.jsonl> [--output <scored.jsonl>] [--max-instances N] [--scaffold TYPE]
```

参数说明：

- `--input / -i`：输入 IM JSONL 文件路径，必需
- `--output / -o`：输出打分后的 JSONL 文件路径，默认 `<input_stem>_rule_scored.jsonl`
- `--max-instances`：最多处理多少条记录
- `--quiet`：减少日志输出
- `--scaffold`：强制指定脚手架类型，可选值为 `claudecode`、`opencode`、`openhands`、`openhands_sdk`、`terminus2`

### 转换脚本自动打分

各转换脚本在生成 IM 数据后会调用 `score_dataset()`。该函数会：

1. 跳过 `_agent_type == "subagent"` 的记录，并写入 `_score: null`
2. 对 main agent 记录调用 `score_record()`
3. 把 TQS V2 指标写入每条记录的 `_score`
4. 打印按脚手架分组的统计摘要

## 输出格式

每条 main agent 记录的 `_score` 字段示例：

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
  "composite_score": 0.83
}
```

可能返回 `null` 的诊断字段包括 `oec_score`、`iac_score`、`ped_score`、`psn_score`、`tte_score` 等，表示当前轨迹没有足够信号计算该指标。

对于 subagent 记录：

```json
{
  "_score": null
}
```

## 数据集级工具调用错误率

`utils.save_lf_json()` 会复用 `rule_score.py` 中的错误检测逻辑，调用：

- `compute_tool_call_error_rate(records)`
- `print_tool_call_error_summary(stats)`

该统计写入 LF sidecar stats 的 `tool_call_errors` 字段，口径包括：

- 轮次维度错误率：错误 tool 调用数 / 总 tool 调用数
- 轨迹维度错误率：含错误的轨迹数 / 含 tool 调用的轨迹数
- 按脚手架分组统计

Terminus2 的一个 observation 可能对应前一个 assistant turn 的多个 commands，因此总调用数按 command 数加权；若该 observation 报错，则按 1 个失败 command 计。

## 版本演进

- v3：早期 2 组 4 指标框架，主要包含 Efficiency 和 Style
- v4：扩展为 5 组 10 指标，加入 Tool Mastery、Completion、Precision
- v5：调整组权重和子指标权重，降低步数、并行度等饱和指标的影响
- TQS V2：改为 fail-soft 组件聚合，保留诊断指标输出，综合分主要由提交完整性、步数效率、测试验证、文件编辑集中度和脏模式惩罚决定
- TQS V2 多语言校准（当前）：修正跨语言口径偏差。原失败识别只覆盖 Python 的签名，导致 Go/Java/Rust/JS 的测试失败被系统性判成通过。要点：ANSI 剥离与头尾双扫描窗口（尾部对齐行首）；`_test_run_outcome()` 三档多语言测试结果判定，通过信号要求非零计数并单列「零测试跑成」；测试命令逐 shell 段判定；「以调用参数判别执行类工具」贯彻到全部 7 个调用点，使打分对工具改名鲁棒；FEC 零编辑返回 `null` 而非满分。回归覆盖见 `tests/test_rule_score_multilang.py`。
