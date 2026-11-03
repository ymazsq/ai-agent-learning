# Day 1 面试题：LLM API 调用与 temperature

对应代码：[src/day01/hello.ts](../src/day01/hello.ts)

**1. temperature 是什么？它如何影响输出？**
参考答案：temperature 控制采样时概率分布的平滑程度。值越低（接近 0），模型越倾向于选择概率最高的 token，输出更稳定、可复现，适合抽取、分类、代码生成。值越高，低概率 token 被选中的机会越大，输出更多样，适合创意类任务。temperature 为 0 也不保证每次完全一致，只是接近确定性。

**2. 为什么 temperature 不是越高越好？**
参考答案：高 temperature 会增加胡说和偏离事实的概率，在需要准确结果的 Agent 场景（如工具调用参数、结构化输出）中风险更大。通常根据任务选择，而不是统一设置一个值。

**3. 调用 Chat Completions 接口时，messages 里的 system、user、assistant 各自的作用是什么？**
参考答案：system 设定角色和规则，优先级较高，用于约束行为；user 是用户输入；assistant 是模型之前的回复，用于让模型知道历史上下文。模型本身无状态，多轮对话靠把 assistant 历史放回 messages 实现。

**4. 为什么 API Key 不能写进代码或提交到仓库？正确做法是什么？**
参考答案：代码会被提交到版本库并可能公开，Key 泄露后他人可以消耗额度或冒用身份。正确做法是放在环境变量或 `.env` 中，`.env` 加入 `.gitignore`，仓库只保留 `.env.example` 占位模板。泄露后应立即在控制台吊销并重新生成。
