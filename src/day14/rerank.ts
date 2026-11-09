import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';
import OpenAI from 'openai';

const ROOT = process.cwd();
const MODEL = 'Xenova/bge-small-zh-v1.5';
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';
const CHUNK_SIZE = 500;
const OVERLAP_RATIO = 0.2;
const TOP_K = 3;
const CANDIDATES = 10;
const BATCH = 32;
const K1 = 1.5;
const B = 0.75;
const RRF_K = 60;

env.remoteHost = 'https://hf-mirror.com/';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});
const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

type Gold = { file: string; phrase: string };
const QUESTIONS: { q: string; gold: Gold }[] = [
  { q: '滑动窗口为什么会答错？', gold: { file: 'LEARNING_LOG.md', phrase: '窗口会直接丢掉早期事实' } },
  { q: '会话持久化为什么用追加写入的 JSONL？', gold: { file: 'interview/day09.md', phrase: '追加写入每条记录只写一行' } },
  { q: 'P1 总结里 Day 1 的核心要点是什么？', gold: { file: 'docs/p1-summary.md', phrase: 'temperature 控制选词的随机程度' } },
  { q: '模块一的出口标准是什么？', gold: { file: 'docs/roadmap.md', phrase: '跨会话恢复' } },
  { q: '提交信息的格式规定是什么？', gold: { file: 'AGENTS.md', phrase: '提交信息格式' } },
];

type Embedder = (
  input: string[],
  options: { pooling: 'cls'; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

type Chunk = {
  key: string;
  source: string;
  text: string;
  tf: Map<string, number>;
  length: number;
  vector: number[];
};

async function collectFiles(): Promise<string[]> {
  const files = ['AGENTS.md', 'LEARNING_LOG.md'];
  for (const dir of ['docs', 'interview']) {
    for (const name of await readdir(join(ROOT, dir))) {
      if (name.endsWith('.md')) files.push(`${dir}/${name}`);
    }
  }
  return files;
}

function chunkText(text: string, size: number): string[] {
  const step = Math.max(1, Math.floor(size * (1 - OVERLAP_RATIO)));
  const chunks: string[] = [];
  for (let start = 0; start < text.length; start += step) {
    chunks.push(text.slice(start, start + size));
    if (start + size >= text.length) break;
  }
  return chunks;
}

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
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}

function bm25Score(query: string[], chunk: Chunk, df: Map<string, number>, avgLength: number, n: number): number {
  let score = 0;
  for (const term of new Set(query)) {
    const f = chunk.tf.get(term) ?? 0;
    if (f === 0) continue;
    const docFreq = df.get(term) ?? 0;
    const idf = Math.log((n - docFreq + 0.5) / (docFreq + 0.5) + 1);
    const norm = K1 * (1 - B + (B * chunk.length) / avgLength);
    score += idf * ((f * (K1 + 1)) / (f + norm));
  }
  return score;
}

async function embedAll(extractor: Embedder, texts: string[]): Promise<number[][]> {
  const vectors: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const out = await extractor(texts.slice(i, i + BATCH), { pooling: 'cls', normalize: true });
    vectors.push(...out.tolist());
  }
  return vectors;
}

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function rankBm25(query: string[], chunks: Chunk[], df: Map<string, number>, avgLength: number): Chunk[] {
  return chunks
    .map((chunk) => ({ chunk, score: bm25Score(query, chunk, df, avgLength, chunks.length) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.chunk);
}

function rankVector(query: number[], chunks: Chunk[]): Chunk[] {
  return chunks
    .map((chunk) => ({ chunk, score: dot(chunk.vector, query) }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.chunk);
}

function rrf(rankings: Chunk[][]): Chunk[] {
  const merged = new Map<string, { chunk: Chunk; score: number }>();
  for (const ranking of rankings) {
    ranking.forEach((chunk, i) => {
      const entry = merged.get(chunk.key) ?? { chunk, score: 0 };
      entry.score += 1 / (RRF_K + i + 1);
      merged.set(chunk.key, entry);
    });
  }
  return [...merged.values()].sort((a, b) => b.score - a.score).map((x) => x.chunk);
}

// 让模型给每个候选打 0–10 分；输出不完整时直接报错，不静默回退
async function llmScores(question: string, candidates: Chunk[]): Promise<number[]> {
  const list = candidates.map((c, i) => `[${i + 1}] ${c.text}`).join('\n\n');
  const response = await client.chat.completions.create({
    model,
    temperature: 0,
    response_format: { type: 'json_object' },
    messages: [
      {
        role: 'system',
        content:
          '你是检索结果排序器。对每个候选块打 0 到 10 分，分数表示该块能在多大程度上直接回答问题。只输出 JSON，格式为 {"scores":[{"id":1,"score":8}]}，每个候选都要有一项。',
      },
      { role: 'user', content: `问题：${question}\n\n候选：\n${list}` },
    ],
  });
  const raw = response.choices[0]?.message?.content ?? '';
  const parsed = JSON.parse(raw) as { scores?: { id: number; score: number }[] };
  const scores = new Array<number>(candidates.length).fill(-1);
  for (const item of parsed.scores ?? []) {
    if (Number.isInteger(item.id) && item.id >= 1 && item.id <= candidates.length && typeof item.score === 'number') {
      scores[item.id - 1] = item.score;
    }
  }
  if (scores.some((s) => s < 0)) throw new Error(`评分不完整：${raw.slice(0, 200)}`);
  return scores;
}

function isHit(top: Chunk[], gold: Gold): boolean {
  return top.some((chunk) => chunk.source === gold.file && chunk.text.includes(gold.phrase));
}

function preview(text: string): string {
  return text.replace(/\s+/g, ' ').slice(0, 40);
}

async function main() {
  const extractor = (await pipeline('feature-extraction', MODEL)) as unknown as Embedder;

  const chunks: Chunk[] = [];
  for (const file of await collectFiles()) {
    const text = await readFile(join(ROOT, file), 'utf8');
    chunkText(text, CHUNK_SIZE).forEach((part, index) => {
      const tokens = tokenize(part);
      chunks.push({
        key: `${file}#${index}`,
        source: file,
        text: part,
        tf: termCounts(tokens),
        length: tokens.length,
        vector: [],
      });
    });
  }

  const chunkVectors = await embedAll(extractor, chunks.map((c) => c.text));
  chunks.forEach((chunk, i) => {
    chunk.vector = chunkVectors[i];
  });
  const queryVectors = await embedAll(extractor, QUESTIONS.map((x) => QUERY_PREFIX + x.q));

  const df = new Map<string, number>();
  for (const chunk of chunks) {
    for (const term of chunk.tf.keys()) df.set(term, (df.get(term) ?? 0) + 1);
  }
  const avgLength = chunks.reduce((sum, c) => sum + c.length, 0) / chunks.length;

  const hits = { rrf: 0, rerank: 0, pool: 0 };
  let fallbacks = 0;
  for (const [i, { q, gold }] of QUESTIONS.entries()) {
    const fused = rrf([rankBm25(tokenize(q), chunks, df, avgLength), rankVector(queryVectors[i], chunks)]);
    const pool = fused.slice(0, CANDIDATES);
    const rrfTop = fused.slice(0, TOP_K);

    let reranked = pool;
    try {
      const scores = await llmScores(q, pool);
      reranked = pool
        .map((chunk, idx) => ({ chunk, score: scores[idx], idx }))
        .sort((a, b) => b.score - a.score || a.idx - b.idx)
        .map((x) => x.chunk);
    } catch (error) {
      fallbacks++;
      console.log(`   [Q${i + 1}] 精排失败，沿用 RRF 顺序：${(error as Error).message}`);
    }
    const rerankTop = reranked.slice(0, TOP_K);

    const rrfHit = isHit(rrfTop, gold);
    const rerankHit = isHit(rerankTop, gold);
    const poolHit = isHit(pool, gold);
    if (rrfHit) hits.rrf++;
    if (rerankHit) hits.rerank++;
    if (poolHit) hits.pool++;

    console.log(`Q${i + 1} ${q}`);
    console.log(`   候选池(前${CANDIDATES})含答案：${poolHit ? '是' : '否'}　RRF：${rrfHit ? '命中' : '未命中'}　RRF+Rerank：${rerankHit ? '命中' : '未命中'}`);
    rerankTop.forEach((chunk, rank) => {
      console.log(`   精排 ${rank + 1}. ${chunk.key} ${preview(chunk.text)}`);
    });
    console.log('');
  }

  console.log(`候选池 Recall@${CANDIDATES}：${hits.pool}/${QUESTIONS.length}`);
  console.log(`Recall@${TOP_K}：RRF ${hits.rrf}/${QUESTIONS.length}，RRF+Rerank ${hits.rerank}/${QUESTIONS.length}`);
  console.log(`精排失败次数：${fallbacks}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
