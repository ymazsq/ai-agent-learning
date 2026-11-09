# Agent 整体流程与知识地图

对应模块：一（Day 1–10）、二（Day 11–14）、可选的长期记忆。大纲版本见 [agent-outline.md](./agent-outline.md)。

## 一、整体流程

```mermaid
flowchart TD
    A[用户输入] --> B[组装 messages<br/>system 分层 + 历史（含压缩）+ 检索结果 + 记忆（可选）]
    B --> C{调用模型}
    C -->|返回文本| D[输出结果]
    C -->|返回 tool_calls| E[校验参数]
    E -->|白名单外 / 校验失败| G[错误作为 tool 消息回传]
    E -->|写操作| H[用户确认 y/N]
    E -->|只读且安全| I[执行工具]
    H -->|y| I
    H -->|n| G
    I --> J[结果作为 tool 消息回传]
    G --> J
    J --> K{超过最大步数?}
    K -->|否| C
    K -->|是| L[停止]
    D --> P[JSONL 持久化]
    L --> P
    B -.-> R[(RAG: 切分 → BM25 / 向量 → RRF → Rerank)]
    B -.-> M[(记忆: 抽取 → 写入 → 读取注入)]
```

## 二、知识地图

```mermaid
%%{init: {'flowchart': {'useMaxWidth': false, 'nodeSpacing': 50, 'rankSpacing': 140}}}%%
flowchart LR
    R(["一次请求"])
    R --> S1["0 启动<br/>JSONL 回放恢复 messages<br/>加载记忆，过滤 supersededAt"]
    R --> S2["1 输入处理<br/>关键词拆分（中文需分词）<br/>关键词压缩（口语转检索词）"]
    R --> S3["2 组装上下文<br/>system 分层：基础 / 输入 / 工具调用<br/>滑动窗口：保留最近 N 轮<br/>阈值触发：超过则按轮压缩<br/>失败整轮回滚<br/>注入：RAG 片段 TOP_K、相关记忆<br/>token 预算：按需供给"]
    R --> S4["3 检索 RAG<br/>离线：文档切分（块大小影响召回）<br/>Embedding 向量化<br/>BM25 索引 K1 1.5 B 0.75<br/>在线：BM25 精确词召回<br/>向量语义召回<br/>RRF 融合（只用排名，不比分数）<br/>Rerank 只重排候选池 10<br/>相关性阈值（待实现）"]
    R --> S5["4 调用模型<br/>temperature：抽取低温，润色可调高<br/>流式 delta 拼接<br/>结构化输出：schema 校验"]
    R --> S6["5 分支判断<br/>返回文本：生成回答<br/>带引用 / 无依据则拒答<br/>返回 tool_calls：<br/>白名单 / 参数校验，失败重试<br/>写操作：y/N 确认<br/>execFile 执行，结果带 tool_call_id 回传<br/>最大步数：未超过回到第 4 步"]
    R --> S7["6 持久化<br/>JSONL 逐行追加<br/>中途停止也保存，可恢复"]
    R --> S8["7 长期记忆写入<br/>抽取：固定 key 词表，低温<br/>写入：同 key 覆盖，保留历史<br/>已知限制：措辞不同导致重复"]
    R --> S9["8 评估<br/>检索：Recall K<br/>生成：引用正确率<br/>拒答：误拒 / 正确拒答<br/>记忆：跨会话正确率<br/>成本：每次请求 token 数"]
    style R fill:#ffffff,stroke:#333,color:#000
    style S1 fill:#bdd7ee,stroke:#333,color:#000
    style S2 fill:#c6e0b4,stroke:#333,color:#000
    style S3 fill:#ffe699,stroke:#333,color:#000
    style S4 fill:#f8cbad,stroke:#333,color:#000
    style S5 fill:#d9d2e9,stroke:#333,color:#000
    style S6 fill:#b4e5e0,stroke:#333,color:#000
    style S7 fill:#fbd3e0,stroke:#333,color:#000
    style S8 fill:#e2efda,stroke:#333,color:#000
    style S9 fill:#fff2cc,stroke:#333,color:#000
```

## 三、学习重点

1. Agent 循环：tool_calls 必须原样回传，必须限制步数
2. 上下文：按轮压缩，失败整轮回滚
3. 结构化输出：合法 JSON 不等于字段正确
4. 工具安全：白名单、参数校验、人工确认，安全在应用层
5. RAG 质量：切分、召回、拒答需要一起评估
6. 混合检索：BM25 与向量互补，RRF 融合的是排名而不是分数
7. Rerank：无法召回候选池之外的内容

## 四、尚未闭环

- Day 14 候选池实验（CANDIDATES=20）已跳过
- 记忆去重问题记为已知限制（措辞不同导致重复或误覆盖）
- Module 1 遗留：复现"长对话爆上下文"
