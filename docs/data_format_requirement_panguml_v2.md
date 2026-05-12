# panguml 2.0 格式定义

## 1. 概览

数据格式以 OpenAI 消息格式为参考，进行合理扩展。

## 2. 数据结构定义

### 2.1 顶级对象

| 属性 | 类型 | 约束 | 语义定义 |
| --- | --- | --- | --- |
| `version` | `String` | 必选 | 规范版本号，如 `"2.0.0"` |
| `meta_info` | `Object` | 必选 | 存储数据源和生产流程信息 |
| `tools` | `Array<Tool>` | 可选 | 声明当前上下文中模型可调用的工具原型 |
| `messages` | `Array<Message>` | 必选 | 严格有序的对话序列 |

### 2.2 消息对象（Message）

消息属性受 `role` 字段驱动，遵循以下逻辑分支。

#### 2.2.1 共有属性

- `role`: `Enum ["system", "user", "assistant", "tool", "developer"]`
- `name`: `String`（可选，向后兼容），用于区分同角色下的多参与者

#### 2.2.2 逻辑分支规范

| 角色（role） | 允许/必需属性 | 约束与语义 |
| --- | --- | --- |
| `system` | `content` | 必选 `String`。定义系统行为。 |
| `user` | `content` | 必选 `String` 或 `Array<TextPart>`。 |
| `assistant` | `content`, `reasoning_content`, `tool_calls`, `weight` | 1. `reasoning_content`：必选 `String`。快思考或无思维过程时必须传 `""`。<br>2. `content`：必选 `String`。若存在 `tool_calls` 则可为 `""`。<br>3. `tool_calls`：可选 `Array<ToolCall>`。<br>4. `weight`：可选 `Number`（`0` 或 `1`），控制 Loss 计算。 |
| `tool` | `content`, `tool_call_id` | 1. `tool_call_id`：建议提供，用于匹配对应 `assistant.tool_calls[*].id`。<br>2. `content`：必选 `String` 或 `Array<TextPart>`。 |
| `developer` | `content` | 必选 `String`。开发者级指令，优先级通常高于 `system`。 |

### 2.3 嵌套对象定义

#### 2.3.1 ToolCall 对象

| 属性 | 类型 | 约束 | 语义 |
| --- | --- | --- | --- |
| `id` | `String` | 建议 | 唯一标识符，用于多工具调用结果回填对齐 |
| `type` | `String` | 必选 | 固定为 `"function"` |
| `function` | `Object` | 必选 | 包含 `name`（工具名）和 `arguments`（JSON 字符串） |

#### 2.3.2 TextPart 对象（多模态/结构化文本预留）

| 属性 | 类型 | 约束 | 语义 |
| --- | --- | --- | --- |
| `type` | `String` | 必选 | 固定为 `"text"` |
| `text` | `String` | 必选 | 具体的文本内容 |

## 3. 工具调用逻辑约束（强制性）

- ID 对齐：在并行工具调用场景下，`tool.tool_call_id` 必须与 `assistant.tool_calls` 中对应的 `id` 完全一致。
- 回复依赖：若 `assistant` 发起工具调用，则紧随其后的下一条 `assistant` 消息必须参考 `tool` 角色的返回结果。
- 内容占位：当 `assistant` 消息仅包含 `tool_calls` 时，`content` 字段必须保留并赋值为 `""`。

## 4. 样例

训练文件 `xx.jsonl` 的任意一行应为如下格式：

```json
{
  "version": "2.0.0",
  "meta_info": {
    "teacher": "glm-5-thinking",
    "query_source": "synthesized",
    "response_generate_time": "2026-03-03",
    "response_update_time": "2026-03-27",
    "owner": "00123456",
    "language": "en",
    "category": "code",
    "rounds": 2,
    "unique_info": {
      "info": {}
    }
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气情况",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "城市标识，例如北京/Beijing、上海/Shanghai、深圳/Shenzhen，具体取值以工具接口约定为准"
            },
            "unit": {
              "type": "string",
              "enum": [
                "celsius",
                "fahrenheit"
              ],
              "description": "温度单位"
            }
          },
          "required": [
            "location"
          ]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "你是一个实时天气助手。"
    },
    {
      "role": "user",
      "content": "北京和深圳现在的天气怎么样？"
    },
    {
      "role": "assistant",
      "reasoning_content": "用户询问两个城市的天气，我需要并行调用两次天气工具。",
      "content": "",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Beijing\"}"
          }
        },
        {
          "id": "call_002",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\":\"Shenzhen\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_001",
      "content": "{\"location\":\"Beijing\",\"temperature\":\"22°C\",\"condition\":\"Sunny\"}"
    },
    {
      "role": "tool",
      "tool_call_id": "call_002",
      "content": "{\"location\":\"Shenzhen\",\"temperature\":\"29°C\",\"condition\":\"Sunny\"}"
    },
    {
      "role": "assistant",
      "reasoning_content": "数据已收集完全，开始回复。",
      "content": "北京现在天气晴朗，气温约 22°C；深圳天气晴朗，气温约 29°C。",
      "weight": 1
    }
  ]
}
```

说明：

- `tools` 为工具定义，无工具调用时可省略。
- `messages` 为对话消息序列。
- `system` 轮必须包含，可为空字符串。
- `reasoning_content` 为快思考数据，可为空字符串。
- 当存在 `tool_calls` 时，`content` 可为空字符串。
- `id` 可选，用于精确配对。
- `tool_call_id` 可选；若存在，应匹配上游 tool call id。
- `arguments` 需注意为字符串。
- `weight` 可选，用于向后兼容 loss 计算控制。

# SFT 数据 `meta_info` 规范

## 候选 `meta_info` 字段

| 字段名 | 字段说明 | 样例 |
| --- | --- | --- |
| `teacher` | 蒸馏数据教师模型。非蒸馏数据标注为 `"Non-distillation"`，未知标注为 `"unknown"`（尽量避免）。 | `{"teacher": "gpt-5.1"}` |
| `query_source` | `query` 来源。自制数据统一标识为 `"Synthesized"`；回流数据标识为 `"Backflow"`；开源数据若可获取来源，则标注数据集名称与链接（如有）；若无法获取来源，统一标注为 `"Untraceable"`。 | `{"query_source": "synthesized"}`<br>`{"query_source": "Dolci-Instruct-SFT， https://xxx"}`<br>`{"query_source": "untraceable"}` |
| `response_generate_time` | `query` 首次 response 生成时间或数据引入时间，时间格式统一为 `YYYY-MM-DD`。 | `{"response_generate_time": "2026-02-04"}` |
| `response_update_time` | `query` 不变、刷新 response 后的更新时间，时间格式统一为 `YYYY-MM-DD`。 | `{"response_update_time": "2026-02-04"}` |
| `owner` | 数据负责人，仅接受八位长度的工号，`string` 格式。 | `{"owner": "00xxxxxx"}` |
| `language` | `query` 的语种，使用标准 ISO 639-1 语言 code（见附录）。 | `{"language": "zh"}` |
| `category` | 数据类别，与一阶段类别对应（见附录）。 | `{"category": "遵从"}` |
| `rounds` | 对话轮次。单轮标注 `1`，真多轮标注总轮次，伪多轮标注实际轮次（如 `2/3`）。 | `{"rounds": 1}` |
| `unique_info` | 除上述公共字段外，其余各数据特有的字段（如各专项标签体系、质量评分、难度、层次、多轮轮次等）。 | `{"unique_info": {"check_list": []}}` |

## 附录

### 数据类别

`agent` | `code` | `code4math` | `general` | `logic` | `math` | `nl2sql` | `planning` | `science` | `人设` | `安全` | `多语言` | `多轮` | `幻觉` | `行业` | `遵从` | `长文本` | `deep_diver`

### 语言 code

```json
{
  "zh": "Chinese",
  "en": "English",
  "es": "Spanish",
  "pt": "Portuguese",
  "ar": "Arabic",
  "hi": "Hindi",
  "fr": "French",
  "ru": "Russian",
  "de": "German",
  "ja": "Japanese",
  "it": "Italian",
  "bn": "Bengali",
  "uk": "Ukrainian",
  "th": "Thai",
  "ur": "Urdu",
  "ta": "Tamil",
  "te": "Telugu",
  "bg": "Bulgarian",
  "ko": "Korean",
  "pl": "Polish",
  "he": "Hebrew",
  "fa": "Persian",
  "vi": "Vietnamese",
  "ne": "Nepali",
  "sw": "Swahili",
  "kn": "Kannada",
  "mr": "Marathi",
  "gu": "Gujarati",
  "pa": "Punjabi",
  "ml": "Malayalam",
  "fi": "Finnish"
}
```