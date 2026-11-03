import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

// Day 4：结构化输出。让模型返回 JSON，用 schema 校验；校验失败时把错误回传给模型重试。
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
const MAX_ATTEMPTS = 3;

type Priority = 'low' | 'medium' | 'high';

interface Task {
  title: string;
  priority: Priority;
  due: string | null;
  tags: string[];
}

const SYSTEM_PROMPT = `你是任务抽取助手。从用户的文字中抽取一个任务，只输出 JSON 对象，不要输出其他内容。
JSON 字段：
- title: string，任务标题，简短明确
- priority: "low" | "medium" | "high"，文字中没有提到时用 "medium"
- due: string | null，截止日期，格式 YYYY-MM-DD；没有提到时为 null
- tags: string[]，相关标签，没有时为 []`;

// 手写校验：模型输出不可信，必须逐字段检查。
function validateTask(value: unknown): { ok: true; task: Task } | { ok: false; error: string } {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return { ok: false, error: '顶层必须是 JSON 对象' };
  }
  const obj = value as Record<string, unknown>;

  if (typeof obj.title !== 'string' || obj.title.trim() === '') {
    return { ok: false, error: 'title 必须是非空字符串' };
  }
  if (obj.priority !== 'low' && obj.priority !== 'medium' && obj.priority !== 'high') {
    return { ok: false, error: 'priority 必须是 "low"、"medium" 或 "high"' };
  }
  if (obj.due !== null && (typeof obj.due !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(obj.due))) {
    return { ok: false, error: 'due 必须是 YYYY-MM-DD 格式的字符串或 null' };
  }
  if (!Array.isArray(obj.tags) || !obj.tags.every((t) => typeof t === 'string')) {
    return { ok: false, error: 'tags 必须是字符串数组' };
  }

  return {
    ok: true,
    task: { title: obj.title, priority: obj.priority, due: obj.due, tags: obj.tags },
  };
}

async function extractTask(text: string): Promise<Task> {
  const messages: ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: text },
  ];

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const res = await client.chat.completions.create({
      model,
      messages,
      temperature: 0,
      response_format: { type: 'json_object' },
    });
    const raw = res.choices[0]?.message?.content ?? '';

    let error: string;
    try {
      const result = validateTask(JSON.parse(raw));
      if (result.ok) {
        console.log(`  第 ${attempt} 次调用成功`);
        return result.task;
      }
      error = result.error;
    } catch {
      error = '输出不是合法的 JSON';
    }

    console.log(`  第 ${attempt} 次校验失败：${error}`);
    // 把模型的输出和错误原因放回对话，让它修正
    messages.push({ role: 'assistant', content: raw });
    messages.push({ role: 'user', content: `输出不符合要求：${error}。请只输出修正后的 JSON 对象。` });
  }

  throw new Error(`连续 ${MAX_ATTEMPTS} 次输出仍不符合 schema`);
}

const samples = [
  '明天下午三点前把登录页的 bug 修掉，比较急，跟前端和 auth 有关',
  '下周五之前写一下 Day4 的学习笔记',
  '有空整理一下读书笔记',
];

for (const text of samples) {
  console.log(`\n输入：${text}`);
  try {
    const task = await extractTask(text);
    console.log('结果：', JSON.stringify(task, null, 2));
  } catch (err) {
    console.error('失败：', err instanceof Error ? err.message : err);
  }
}
