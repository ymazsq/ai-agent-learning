# AGENTS.md

## 项目目的

这是一个个人学习仓库，用于从前端工程师转型为 Agent 开发工程师。目标是 2027 年 2 月中下旬开始面试。仓库按天记录学习过程：每天一个可运行的示例，并在 [LEARNING_LOG.md](./LEARNING_LOG.md) 中记录目标、掌握程度和自测题。

长期路线见 [LEARNING_LOG.md](./LEARNING_LOG.md) 的"总体路线"：LLM 基础 → RAG → Agent 进阶 → 工程化与求职。

## 技术栈

- **语言**：TypeScript（ESM，`"type": "module"`），第 3 周起补 Python
- **包管理**：pnpm 10（不要使用 npm 或 yarn，已在 `package.json` 的 `packageManager` 中声明）
- **运行**：`tsx`，通过 `node --env-file=.env --import tsx` 运行
- **LLM 接口**：`openai` SDK（v4），接入任意 OpenAI 兼容接口。当前使用火山引擎方舟（豆包）
- **类型检查**：TypeScript 5.x，`pnpm typecheck`（`tsc --noEmit`）
- **计划使用**：Vercel AI SDK、LangGraph.js、MCP SDK、pgvector、Langfuse

## 目录与命令

- `src/dayNN/`：每天的示例代码，NN 为两位数字（如 `day01`、`day03`）
- 每天在 `package.json` 的 `scripts` 中新增一个 `dayNN` 脚本
- 常用命令：
  - `pnpm install`：安装依赖
  - `pnpm typecheck`：类型检查，提交前必须通过
  - `pnpm dayNN`：运行当天示例
- `.npmrc` 指向官方源 `registry.npmjs.org`，不要改成公司内网源

## 代码约定

- 字符串使用**单引号**（模板字符串除外）
- 注释只用于解释不直观的逻辑，并保持简短。示例代码中的"DayN："注释用于说明本节重点
- API Key 只放在本地 `.env`，`.env` 已被 `.gitignore` 忽略
- `.env.example` 只能保留占位符，禁止写入真实 key
- 不要把密钥、token 或个人账号信息写入代码或提交信息

## Git 约定

- 提交信息格式：`DayN: 简短描述`，正文用于补充要点
- 提交末尾附上 `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`
- 本仓库使用个人身份提交（`ymazsq`），不要使用公司身份
- 提交前检查暂存内容，确认不含 key 或 token

## 每日学习流程

每天的学习内容完成后，需要同时产出：

1. **学习记录**：在 `LEARNING_LOG.md` 中按以下五项记录
   - 做了什么
   - 要掌握什么
   - 掌握到什么程度
   - 对 Agent 有什么用
   - 自测题（2 到 3 道）
2. **面试题**：围绕当天内容，给出 3 到 5 道常见面试题，并附上简要参考答案。题目要能检验对原理的理解，而不只是 API 用法。
3. **代码验证**：运行 `pnpm typecheck` 和当天的 `pnpm dayNN`，确认结果后再提交。

学习内容应以"能解释清楚"为标准，而不是"能跑起来"。
