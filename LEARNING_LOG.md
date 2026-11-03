# Agent 开发学习记录

## 基本信息
- 背景：前端工程师（TS），日常使用 VS Code Copilot，用过 Claude / GPT
- 目标：转型 Agent 开发工程师，2027 年 2 月中下旬开始面试
- 节奏：工作日每天 2–3 小时，周六做项目，周日复盘
- 技术栈：TypeScript 优先（openai SDK、Vercel AI SDK、LangGraph.js、MCP SDK），第 3 周起补 Python
- 仓库：https://github.com/ymazsq/ai-agent-learning.git
- 本地路径：`/Users/zhangshaoqi/ai-agent-learning`

## 总体路线
| 阶段 | 周次 | 主题 | 项目产出 |
|---|---|---|---|
| 1. LLM 基础 | W1–W3 | API、流式输出、结构化输出、Function Calling、Agent 循环、Prompt、上下文管理 | 能调用工具的 Agent（CLI + 网页版） |
| 2. RAG | W4–W6 | Embedding、切分、pgvector、检索与 Rerank、效果调优 | 项目 1：私有知识库问答助手 |
| 3. Agent 进阶 | W7–W10 | MCP Server、LangGraph、记忆、Human-in-the-loop、多 Agent | 项目 2：自动调研报告 Agent（或代码审查 Agent） |
| 4. 工程化与求职 | W11–W16 | 评测、可观测性（Langfuse）、安全、部署、README 与博客、面试准备 | 作品集 + 简历，W14 起投递 |

## 当前进度

### Day 2（2026-11-03）已完成
- 新增 `src/day02/chat.ts`：命令行聊天，回答逐字流式输出
- 多轮对话：用 `history` 数组保存全部消息，每次请求都整体发送给模型
- 运行方式：`pnpm day02`，输入 `exit` 退出
- 验证：第一轮告诉模型"我叫小王"，第二轮问"我叫什么名字"，模型能答出，说明历史已被带上
- 注意：历史会越来越长，每次请求的输入 token 随之增加，直接关联 Day 1 的第三个问题（费用）

### Day 1（2026-11-02）已完成
- 克隆仓库到本地，初始化 TypeScript 项目骨架
- 新增文件：`package.json`、`tsconfig.json`、`.gitignore`、`.env.example`、`src/day01/hello.ts`
- 代码内容：使用 OpenAI 兼容接口调用模型，分别以 temperature 0 和 1.2 回答同一问题，并打印 token 用量
- 本地提交：`f10e774  Day1: project scaffold and first LLM API call`

### 完成状态
- [x] **推送到 GitHub**：已推送至 `ymazsq/ai-agent-learning`（使用 ymazsq 账号 Token）
- [x] **安装依赖**：已改用 pnpm（`pnpm install`），项目 `.npmrc` 指向官方源 `registry.npmjs.org`，类型检查通过
- [x] **配置 API**：使用火山引擎方舟（OpenAI 兼容接口），配置在本地 `.env`（不提交）
- [x] **运行验证**：`pnpm day01` 两次调用均成功；temperature=0 与 1.2 的回答措辞不同，两次 prompt 均为 46 token，completion 分别为 85 与 84
- [ ] **Day 1 小问题**（已讲解，留待 Day 2 多轮对话时回顾）：
  1. 为什么多轮对话时模型"记得"前面说过的话？（每次请求都会带上历史消息）
  2. temperature 设为 0 是否意味着每次输出完全一样？（基本一致，但不保证完全相同）
  3. 一次调用的费用由哪些部分决定？（输入 token 与输出 token，输出通常更贵）

**安全提醒**：`.env.example` 曾误写入真实 API Key，需在方舟控制台作废并重新生成。

## 下一步
- **Day 2**：流式输出（streaming）；多轮对话的历史管理，做命令行聊天程序

## 每周计划速览
- W1：LLM API 基础、流式、结构化输出、Function Calling、手写 Agent 循环
- W2：Prompt 工程；Next.js + Vercel AI SDK 做流式网页对话，展示工具调用过程
- W3：上下文管理（Token 计数、截断、摘要）；Python 入门，用 Python 重写 Agent
- W4–W6：RAG（Embedding、切分、pgvector、混合检索、Rerank），完成项目 1
- W7–W9：MCP Server 并接入 VS Code；LangGraph 状态机；记忆与人工确认
- W10：项目 2 综合开发
- W11–W13：评测（LLM-as-Judge）、Langfuse 可观测性、安全、部署、博客、简历
- W14–W16：投递简历，面试准备与冲刺

## 继续学习的方式
在新会话中发送 "Day N"（例如 "Day 2"），并附上本文件路径，我会据此从上次进度继续。每天结束时，把运行结果或报错发给我检查。
