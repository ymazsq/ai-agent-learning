import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});
const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
const MEMORY_PATH = join(process.cwd(), 'workspace', 'memory.json');

// 长期记忆的写入：同一 key 的新值覆盖旧值，旧记录保留并标记 supersededAt。
interface Fact {
  key: string;
  value: string;
}
interface MemoryRecord extends Fact {
  createdAt: string;
  supersededAt?: string;
}

// 固定词表：key 不固定时，同一类事实会被存成不同的 key，覆盖逻辑就失效了
const ALLOWED_KEYS = ['tech_background', 'preferred_language', 'avoided_language', 'career_goal', 'study_habit', 'other'];

const SYSTEM_PROMPT = `你是记忆抽取助手。从对话中提取关于用户的长期有效信息（偏好、身份、长期目标、约定），只输出 JSON 对象。
不要提取一次性的任务内容、寒暄或助手的回答。
JSON 格式：{"facts": [{"key": string, "value": string}]}
- key：只能从以下词表中选择：${ALLOWED_KEYS.join('、')}；都不合适时用 other
- value：事实的内容，一句中文
- 没有可提取的信息时返回 {"facts": []}`;

function validateFacts(value: unknown): Fact[] {
  if (typeof value !== 'object' || value === null || !Array.isArray((value as { facts?: unknown }).facts)) {
    throw new Error('顶层必须是 {"facts": [...]}');
  }
  return (value as { facts: unknown[] }).facts.map((item, i) => {
    const f = item as Partial<Fact>;
    if (typeof f?.key !== 'string' || !ALLOWED_KEYS.includes(f.key)) throw new Error(`第 ${i + 1} 条 key 不在词表中`);
    if (typeof f.value !== 'string' || f.value.trim() === '') throw new Error(`第 ${i + 1} 条 value 为空`);
    return { key: f.key, value: f.value.trim() };
  });
}

async function extractFacts(conversation: ChatCompletionMessageParam[]): Promise<Fact[]> {
  const res = await client.chat.completions.create({
    model,
    temperature: 0,
    response_format: { type: 'json_object' },
    messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...conversation],
  });
  return validateFacts(JSON.parse(res.choices[0]?.message?.content ?? ''));
}

async function loadMemory(): Promise<MemoryRecord[]> {
  try {
    return JSON.parse(await readFile(MEMORY_PATH, 'utf8')) as MemoryRecord[];
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw err;
  }
}

async function saveFacts(facts: Fact[]) {
  const records = await loadMemory();
  const now = new Date().toISOString();
  const stats = { added: 0, superseded: 0, skipped: 0 };
  for (const fact of facts) {
    const current = records.find((r) => !r.supersededAt && r.key === fact.key);
    if (current?.value === fact.value) {
      stats.skipped++;
      continue;
    }
    if (current) {
      current.supersededAt = now;
      stats.superseded++;
    }
    records.push({ ...fact, createdAt: now });
    stats.added++;
  }
  await mkdir(dirname(MEMORY_PATH), { recursive: true });
  await writeFile(MEMORY_PATH, JSON.stringify(records, null, 2), 'utf8');
  return { records, ...stats };
}

const conversation: ChatCompletionMessageParam[] = [
  { role: 'user', content: '我是前端出身，现在主要用 TypeScript 写后端，不太想学 Java。' },
  { role: 'assistant', content: '明白，后面的示例会以 TypeScript 为主。' },
  { role: 'user', content: '我的目标是 2027 年 2 月左右开始找 Agent 开发的工作。' },
  { role: 'assistant', content: '好的，学习计划会按这个时间安排。' },
  { role: 'user', content: '今天天气不错，帮我看下明天要做什么。' },
];

const rounds: ChatCompletionMessageParam[][] = [
  conversation,
  [
    { role: 'user', content: '我的求职时间推迟到 2027 年 6 月了。' },
    { role: 'assistant', content: '好的，已记录。' },
    { role: 'user', content: '我平时习惯晚上学习，白天要上班。' },
  ],
];

for (const [i, round] of rounds.entries()) {
  console.log(`\n第 ${i + 1} 轮`);
  const facts = await extractFacts(round);
  console.log('抽取结果：', JSON.stringify(facts));
  const { records, added, superseded, skipped } = await saveFacts(facts);
  console.log(`新增 ${added}，覆盖 ${superseded}，跳过 ${skipped}；文件共 ${records.length} 条`);
}
