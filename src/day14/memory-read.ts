import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';
import OpenAI from 'openai';

const MODEL = 'Xenova/bge-small-zh-v1.5';
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';
const MEMORY_PATH = join(process.cwd(), 'workspace', 'memory.json');
const TOP_K = 3;
const RRF_K = 60;
const K1 = 1.5;
const B = 0.75;

env.remoteHost = 'https://hf-mirror.com/';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});
const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

interface MemoryRecord {
  key: string;
  value: string;
  createdAt: string;
  supersededAt?: string;
}

type Item = {
  record: MemoryRecord;
  tf: Map<string, number>;
  length: number;
  vector: number[];
};

type Embedder = (
  input: string[],
  options: { pooling: 'cls'; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

const QUESTIONS = [
  '我该用什么语言写 Agent 相关的代码？',
  '我打算什么时候开始找工作？',
  '我昨天晚饭吃的什么？',
];

function tokenize(text: string): string[] {
  const tokens: string[] = [];
  for (const run of text.toLowerCase().match(/\p{Script=Han}+|[a-z0-9]+/gu) ?? []) {
    if (/\p{Script=Han}/u.test(run)) {
      for (let i = 0; i < run.length - 1; i++) tokens.push(run.slice(i, i + 2));
      if (run.length === 1) tokens.push(run);
    } else {
      tokens.push(run);
    }
  }
  return tokens;
}

function termCounts(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
  return counts;
}

function bm25Score(query: string[], item: Item, df: Map<string, number>, avgLength: number, n: number): number {
  let score = 0;
  for (const term of new Set(query)) {
    const f = item.tf.get(term) ?? 0;
    if (f === 0) continue;
    const docFreq = df.get(term) ?? 0;
    const idf = Math.log((n - docFreq + 0.5) / (docFreq + 0.5) + 1);
    const norm = K1 * (1 - B + (B * item.length) / avgLength);
    score += idf * ((f * (K1 + 1)) / (f + norm));
  }
  return score;
}

// 向量已归一化，点积即余弦相似度
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function rankBm25(query: string[], items: Item[], df: Map<string, number>, avgLength: number): Item[] {
  return items
    .map((item) => ({ item, score: bm25Score(query, item, df, avgLength, items.length) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.item);
}

function rankVector(query: number[], items: Item[]): Item[] {
  return items
    .map((item) => ({ item, score: dot(item.vector, query) }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.item);
}

function rrf(rankings: Item[][]): { item: Item; score: number }[] {
  const merged = new Map<Item, { item: Item; score: number }>();
  for (const ranking of rankings) {
    ranking.forEach((item, i) => {
      const entry = merged.get(item) ?? { item, score: 0 };
      entry.score += 1 / (RRF_K + i + 1);
      merged.set(item, entry);
    });
  }
  return [...merged.values()].sort((a, b) => b.score - a.score);
}

async function loadMemory(): Promise<MemoryRecord[]> {
  const all = JSON.parse(await readFile(MEMORY_PATH, 'utf8')) as MemoryRecord[];
  return all.filter((r) => !r.supersededAt);
}

async function buildItems(extractor: Embedder, records: MemoryRecord[]): Promise<Item[]> {
  const texts = records.map((r) => `${r.key}：${r.value}`);
  const out = await extractor(texts, { pooling: 'cls', normalize: true });
  const vectors = out.tolist();
  return records.map((record, i) => {
    const tokens = tokenize(texts[i]);
    return { record, tf: termCounts(tokens), length: tokens.length, vector: vectors[i] };
  });
}

async function retrieve(extractor: Embedder, items: Item[], question: string): Promise<{ record: MemoryRecord; score: number }[]> {
  const df = new Map<string, number>();
  let total = 0;
  for (const item of items) {
    total += item.length;
    for (const term of item.tf.keys()) df.set(term, (df.get(term) ?? 0) + 1);
  }
  const avgLength = total / items.length;

  const [qVector] = (await extractor([QUERY_PREFIX + question], { pooling: 'cls', normalize: true })).tolist();
  const byKeyword = rankBm25(tokenize(question), items, df, avgLength);
  const byVector = rankVector(qVector, items);

  return rrf([byKeyword, byVector])
    .slice(0, TOP_K)
    .map((x) => ({ record: x.item.record, score: x.score }));
}

async function answer(question: string, hits: MemoryRecord[]): Promise<string> {
  const memory = hits.map((r) => `- ${r.value}`).join('\n');
  const res = await client.chat.completions.create({
    model,
    temperature: 0,
    messages: [
      {
        role: 'system',
        content: `你是学习助手。以下是关于用户的长期记忆。只在与问题相关时使用；无关或没有依据时直接说不知道，不要编造。\n${memory}`,
      },
      { role: 'user', content: question },
    ],
  });
  return res.choices[0]?.message?.content ?? '';
}

async function main() {
  console.log(`加载模型 ${MODEL}...`);
  const extractor = (await pipeline('feature-extraction', MODEL)) as unknown as Embedder;
  const records = await loadMemory();
  const items = await buildItems(extractor, records);
  console.log(`已加载 ${records.length} 条记忆`);

  for (const question of QUESTIONS) {
    console.log(`\n问题：${question}`);
    const hits = await retrieve(extractor, items, question);
    for (const h of hits) console.log(`  检索 ${h.score.toFixed(4)} ${h.record.value}`);
    console.log(`回答：${await answer(question, hits.map((h) => h.record))}`);
  }
}

await main();
