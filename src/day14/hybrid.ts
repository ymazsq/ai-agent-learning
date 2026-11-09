import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';

const ROOT = process.cwd();
const MODEL = 'Xenova/bge-small-zh-v1.5';
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';
const CHUNK_SIZE = 500;
const OVERLAP_RATIO = 0.2;
const TOP_K = 3;
const BATCH = 32;
const K1 = 1.5;
const B = 0.75;
// RRF 常数，经验值，用来压平名次之间的差距
const RRF_K = 60;

env.remoteHost = 'https://hf-mirror.com/';

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

// 与 bm25.ts 相同：中文二元组，英文数字整词
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

// 向量已归一化，点积即余弦相似度
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// 只保留分数大于 0 的块：没有任何关键词命中的块，关键词名次没有意义
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

function rrf(rankings: Chunk[][]): { chunk: Chunk; score: number }[] {
  const merged = new Map<string, { chunk: Chunk; score: number }>();
  for (const ranking of rankings) {
    ranking.forEach((chunk, i) => {
      const entry = merged.get(chunk.key) ?? { chunk, score: 0 };
      entry.score += 1 / (RRF_K + i + 1);
      merged.set(chunk.key, entry);
    });
  }
  return [...merged.values()].sort((a, b) => b.score - a.score);
}

function isHit(top: Chunk[], gold: Gold): boolean {
  return top.some((chunk) => chunk.source === gold.file && chunk.text.includes(gold.phrase));
}

function preview(text: string): string {
  return text.replace(/\s+/g, ' ').slice(0, 40);
}

async function main() {
  console.log(`加载模型 ${MODEL}...`);
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

  console.log(`切分 ${CHUNK_SIZE} 字符，共 ${chunks.length} 块，TOP-${TOP_K}\n`);
  const hits = { vector: 0, bm25: 0, rrf: 0 };
  QUESTIONS.forEach(({ q, gold }, i) => {
    const vectorTop = rankVector(queryVectors[i], chunks);
    const bm25Top = rankBm25(tokenize(q), chunks, df, avgLength);
    const fused = rrf([bm25Top, vectorTop]);

    const tops = {
      vector: vectorTop.slice(0, TOP_K),
      bm25: bm25Top.slice(0, TOP_K),
      rrf: fused.slice(0, TOP_K).map((x) => x.chunk),
    };
    const mark = (name: keyof typeof tops) => {
      const hit = isHit(tops[name], gold);
      if (hit) hits[name]++;
      return hit ? '命中' : '未命中';
    };
    console.log(`Q${i + 1} ${q}`);
    console.log(`   向量：${mark('vector')}　关键词：${mark('bm25')}　RRF：${mark('rrf')}`);
    fused.slice(0, TOP_K).forEach(({ chunk, score }, rank) => {
      console.log(`   RRF ${rank + 1}. ${score.toFixed(4)} ${chunk.key} ${preview(chunk.text)}`);
    });
    console.log('');
  });

  console.log(`Recall@${TOP_K}：向量 ${hits.vector}/${QUESTIONS.length}，关键词 ${hits.bm25}/${QUESTIONS.length}，RRF ${hits.rrf}/${QUESTIONS.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
