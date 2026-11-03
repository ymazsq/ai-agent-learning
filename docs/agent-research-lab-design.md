# Agent Research Lab 设计文档

日期：2026-09-03
状态：已与用户确认

## 1. 目标与定位

以学习研究为主的 agent 开发实验室。通过手写最小 agent 理解原理，研究笔记与开源对比文档和代码同等重要。最终通过逐个课题（上下文存储、压缩等）把最小 agent 演进为更完整的系统。

成功标准：
- 一个从零手写、可运行的最小 agent（~100 行，零框架依赖）
- 每个课题有"问题定义 + 研究发现 + 开源对比"的完整文档
- 每个课题的结论实现回主代码，git 历史即演进史

## 2. 关键决策（用户已确认）

| 决策项 | 选择 |
|--------|------|
| 项目目标 | 学习研究为主 |
| 语言 | TypeScript + Node 20+ |
| 模型接入 | OpenAI 兼容 API（base URL + key + model 可配） |
| 对比对象 | Claude Code、opencode、pi (badlogic)、crush (charmbracelet)、Aider/OpenHands |
| 仓库组织 | 方案 A：单一演进的 agent（`src/`）+ 课题目录（`topics/`） |
| Tool schema | v0 纯手写 JSON Schema，不引 zod |

## 3. 仓库结构

```
agent-research/
├── README.md              # 仓库入口：路线图 + 课题索引表（状态标记）
├── src/                   # 唯一演进的 agent 实现
│   ├── index.ts           # CLI REPL 入口
│   ├── llm.ts             # OpenAI 兼容客户端（手写 fetch）
│   ├── loop.ts            # agent loop 核心循环
│   └── tools/             # read_file / write_file / run_command
├── topics/                # 课题目录（见 §5）
│   ├── 01-loop-and-tools/
│   ├── 02-context-storage/
│   ├── 03-context-compression/
│   ├── 04-prompt-engineering/
│   ├── 05-memory/
│   ├── 06-multi-agent/
│   ├── 07-eval-and-observability/
│   └── 08-security-and-sandbox/
├── docs/
│   └── superpowers/specs/ # 设计文档（本文件）
├── package.json
└── tsconfig.json
```

## 4. 最小 Agent（v0）设计

**目标**：~100 行内跑通完整 agent loop，零框架依赖。

组件：
- **llm.ts**：手写 `fetch` 调 `/chat/completions`，支持 `tools` 字段；不引 SDK，直接暴露原始报文结构。
- **loop.ts**：`while(true)` → 调模型 → 响应含 `tool_calls` 则执行并回填结果 → 否则输出最终回复并结束。
- **tools/**：三个内置工具验证闭环：
  - `read_file(path)`
  - `write_file(path, content)`
  - `run_command(command)`（本地直接执行，v0 不做沙箱）
  - 工具定义采用纯手写 JSON Schema。
- **index.ts**：REPL 式 stdin 交互；每轮打印 tool call 与结果，便于观察。

错误处理：工具执行异常时，将错误信息作为 tool result 回填，由模型自行决定重试或放弃。

明确不做（留给后续课题）：历史持久化、上下文压缩、流式输出、权限确认、子 agent。

配置：`.env` 提供 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`，手写解析，不引 dotenv。

## 5. 课题目录与研究流程

每个课题目录统一结构：

```
topics/NN-slug/
├── README.md        # 问题定义：为什么需要这个机制？不做会怎样？+ 结论区
├── notes.md         # 研究发现、实验记录
├── comparison.md    # 开源对比表：Claude Code / opencode / pi / crush / Aider 各家怎么做
└── experiments/     # （可选）一次性验证代码
```

研究节奏（每课题走一遍）：
1. **撞墙**：先在当前版本上复现问题（如长对话爆上下文）
2. **调研**：读开源实现对应部分，填 comparison.md
3. **设计**：结论写进 README
4. **实现**：合回 `src/`，打 tag

课题顺序（按依赖排）：

| # | 课题 | 说明 | 依赖 |
|---|------|------|------|
| 01 | loop-and-tools | v0 实现本身即本课题产出 | - |
| 02 | context-storage | 会话持久化/恢复（JSONL 存历史） | - |
| 03 | context-compression | 压缩/摘要/滑动窗口、token 预算 | 02 |
| 04 | prompt-engineering | system prompt 结构、技能/规则注入 | - |
| 05 | memory | 长期记忆、知识库、RAG | 02 |
| 06 | multi-agent | 子 agent、并行任务、结果回传 | - |
| 07 | eval-and-observability | evals、tracing、回放调试 | - |
| 08 | security-and-sandbox | 沙箱、权限、危险命令拦截（贯穿性，最后系统化） | - |

顺序可调，但 02→03、02→05 的依赖关系保留。

## 6. 工程约定

- 运行时：Node 20+，TypeScript；`tsx` 直接运行，`tsc --noEmit` 类型检查
- 依赖极简：仅 `typescript` + `tsx`（dev）；无运行时依赖
- 包管理：npm
- 测试策略：v0 不做单测；纯逻辑模块（如后续压缩器）落地时再补；研究性实验放 `topics/*/experiments/`，不进主测试链路
- README.md：仓库入口，含课题索引表（状态：占位 / 研究中 / 完成）
- 提交约定：提交信息中文；每个里程碑打 tag（`v0-minimal` 起）

## 7. 里程碑

1. 仓库脚手架 + 8 个课题目录占位（含空骨架文档）
2. `v0-minimal`：最小 agent 跑通（tag: `v0-minimal`）
3. 起逐个课题按 §5 节奏推进
