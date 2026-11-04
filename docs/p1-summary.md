# P1 核心总结：Agent 核心（Day 1–10）

对应路线图：[roadmap.md](./roadmap.md) 的"阶段 1：Agent 核心（W1–W3）"。每个模块一份：先列核心要掌握的内容，再给出与开源实现的对比。

## Agent 的核心循环

```
messages（system + 历史 + 用户输入）
  → 模型返回 assistant（可能含 tool_calls）
  → 应用执行工具，把结果作为 tool 消息回传
  → 再次请求模型，直到模型不再调用工具
```

- 模型只生成参数，真正执行的是应用代码。
- `tool_calls` 必须原样放回 messages，之后的 `tool` 消息要一一对应。
- 工具错误也作为 `tool` 消息回传，模型据此调整。
- 必须设置最大步数，防止死循环。

## Day 1：调用 LLM API

**核心要掌握**
- messages 的 system / user / assistant 角色
- temperature 控制选词的随机程度；0 也不保证输出完全一致
- token 是计费和长度限制的单位，输入和输出都计费

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day01/hello.ts`，直接用 `openai` SDK 调用，打印 `usage` |
| 开源对应 | OpenAI SDK；Vercel AI SDK 的 `generateText` |
| 相同点 | 请求结构一致，都是 messages 加参数 |
| 差异与可借鉴 | 开源 SDK 封装了重试、超时和多供应商切换；我们直接调用，便于看清每个字段 |

## Day 2：流式输出与多轮对话

**核心要掌握**
- 模型不保存历史，"记得"来自每次请求发送完整历史
- `stream: true` 时逐块读取 `delta`，完整回答由所有 `delta` 拼接而成
- 历史越长，成本和延迟越差

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day02/chat.ts`，用 `history` 数组保存全部消息，逐块输出 |
| 开源对应 | OpenAI SDK 的流式接口；Vercel AI SDK 的 `streamText` |
| 相同点 | 都是逐块读取增量内容，并在结束后得到完整结果 |
| 差异与可借鉴 | 开源库通常同时提供 UI 层的流式辅助；我们只在终端输出，没有处理中断和重连 |

## Day 3：上下文管理

**核心要掌握**
- 每轮输入 token 随历史增长
- 截断会丢信息，摘要会丢细节并多一次模型调用
- `prompt_tokens` 是这次请求真实的输入大小

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day03/context.ts`，最近 2 轮保留原文，更早的对话压缩为摘要 |
| 开源对应 | LangChain 的 memory 类（较旧的 API，如 `ConversationSummaryMemory`） |
| 相同点 | 都是"近期原文 + 早期摘要"的思路 |
| 差异与可借鉴 | 开源实现一般提供多种 memory 类型可选；我们只有一种，且摘要失败时直接保留原始历史 |

## Day 4：结构化输出

**核心要掌握**
- `json_object` 只保证输出是合法 JSON，不保证字段和类型正确
- 必须做 schema 校验，失败时把错误回传让模型修正，并限制次数
- 模型不知道当前日期，需要把它放进 prompt

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day04/extract.ts`，手写 `validateTask` 逐字段校验，最多重试 3 次 |
| 开源对应 | OpenAI Structured Outputs（`json_schema`，strict 模式）；Zod + Vercel AI SDK 的 `generateObject` |
| 相同点 | 都要求输出符合 schema，并把校验放在解析之后 |
| 差异与可借鉴 | 开源方案可以让接口直接按 schema 约束输出，减少重试；Zod 可以同时生成类型和校验，我们的校验要手写两份 |

## Day 5：Function Calling

**核心要掌握**
- `tools` 用 JSON Schema 描述参数，模型只生成参数，执行由应用完成
- 模型返回 `tool_calls` 后，执行结果以 `tool` 消息回传，直到模型不再调用工具
- 参数错误要显式报错，不能静默转换

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day05/tools.ts`，两个工具（`get_current_date`、`create_task`），最多 5 步 |
| 开源对应 | OpenAI 的 `tools` 接口；Vercel AI SDK 的 `tool()` 与 `maxSteps` / `stopWhen` |
| 相同点 | 工具描述、调用、回传、循环的流程完全一致 |
| 差异与可借鉴 | 开源 SDK 通常自动完成参数解析和循环；我们的校验是手写的，遇到错误参数时更容易看清问题 |

## Day 6：最小 Agent

**核心要掌握**
- 一轮内的消息顺序：`user` → `assistant(tool_calls)` → `tool` → … → `assistant`
- 压缩要按轮切分，按消息条数切会拆散 `tool_calls` 与 `tool`
- 一轮中途失败时整轮回滚，否则历史残缺

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day06/agent.ts`，手写 `while` 循环，三个工具，按轮压缩，失败整轮回滚 |
| 开源对应 | OpenAI Agents SDK 的 Runner 循环；Vercel AI SDK 的 `generateText` 带 `stopWhen` |
| 相同点 | 都是"请求模型 → 执行工具 → 回传 → 循环"，都限制步数 |
| 差异与可借鉴 | 开源实现内置工具结果格式和错误处理；整轮回滚是我们自己加的，开源库一般没有 |

## Day 7：工具安全

**核心要掌握**
- 顺序是先校验、再确认、最后执行，被拦截的操作不打扰用户
- `exec` 把字符串交给 shell 解释，`execFile` 直接传 argv，注入风险不同
- 模型的口头拒绝不是安全边界，安全要写在代码里

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day07/agent.ts`，`run_command` 白名单 + `execFile`，写操作前 y/N 确认 |
| 开源对应 | Claude Code 的权限模式（逐次确认、允许白名单）；LangGraph 的 `interrupt` 做人工审批 |
| 相同点 | 都是"校验 → 人工确认 → 执行"，都把安全放在应用层 |
| 差异与可借鉴 | 开源实现的确认会展示完整命令并可记住选择；我们只有 y/N，信息少；路径检查不解析符号链接，这一点需要改进 |

## Day 8：System prompt 分层

**核心要掌握**
- 基础层、规则层、运行时层分开来源，每次请求重新拼接
- 运行时层不能只生成一次，否则文件列表和摘要会过时
- 规则层只是提醒模型，安全边界仍在代码里

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day08/agent.ts`，`buildMessages` 每次拼接三层；白名单列表从常量生成 |
| 开源对应 | Claude Code 的 `CLAUDE.md` 分层加载；OpenAI Agents SDK 的 `instructions` 可以传入函数动态生成 |
| 相同点 | 都是把 prompt 拆成有来源的部分，请求时组装 |
| 差异与可借鉴 | 开源实现支持按目录层级合并多个文件；我们只有一个运行时层，结构较简单 |

## Day 9：会话持久化（JSONL）

**核心要掌握**
- 追加写入：崩溃最多丢最后一行，前面的历史完整
- 恢复时跳过无法解析的半行；截掉未完成的一轮
- 压缩和回滚写成记录，不改写文件；重放必须和运行时操作完全一致

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day09/agent.ts`，每个会话一个 `.jsonl`，记录 `message` / `summary` / `rollback` |
| 开源对应 | Claude Code 的会话记录同样是 JSONL；LangGraph 的 checkpointer（如 SqliteSaver）按步骤保存状态 |
| 相同点 | 都是追加式记录，恢复时靠重放或读取最后状态 |
| 差异与可借鉴 | LangGraph 存状态快照，适合回放和分支；我们存操作记录，更直观，但重放逻辑要自己保证正确 |

## Day 10：压缩策略对比

**核心要掌握**
- 窗口（B）会直接丢掉早期事实，固定保留 N 轮
- 摘要（A）保留结论，但每次压缩多一次模型调用，并且会丢细节
- 预算触发（C）平时不动，超预算时先截断旧的工具结果，再摘要；工具结果通常最大，截断最划算
- 预算应该按请求大小设，而不是按轮数

| 维度 | 内容 |
| --- | --- |
| 我们的做法 | `src/day10/agent.ts`，`pnpm day10 A`、`B`、`C` 三种策略对同一脚本对比 |
| 开源对应 | LangChain 的 `ConversationBufferWindowMemory`（对应 B）和 `ConversationSummaryBufferMemory`（接近 A）；Claude Code 的 auto-compact |
| 相同点 | 都用窗口、摘要或两者结合控制历史长度 |
| 差异与可借鉴 | 开源摘要通常按 token 阈值触发；先截断工具结果的做法（C）在开源库里不常见，可以作为改进方向 |

## 边界说明

- 上面的开源对应是基于经验整理的，版本细节请以官方文档为准。
- Day 10 的 token 对比只是单次运行的初步观察，不能当作定论。
