import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';

const ROOT = process.cwd();
const CHUNK_SIZE = 500;
const OVERLAP_RATIO = 0.2;
const TOP_K = 3;
// BM25 参数：K1 控制词频上限，B 控制长度归一化强度
const K1 = 1.5;
const B = 0.75;

type Gold = { file: string; phrase: string };
const QUESTIONS: { q: string; gold: Gold }[] = [
  { q: '滑动窗口为什么会答错？', gold: { file: 'LEARNING_LOG.md', phrase: '窗口会直接丢掉早期事实' } },
  { q: '会话持久化为什么用追加写入的 JSONL？', gold: { file: 'interview/day09.md', phrase: '追加写入每条记录只写一行' } },
  { q: 'P1 总结里 Day 1 的核心要点是什么？', gold: { file: 'docs/p1-summary.md', phrase: 'temperature 控制选词的随机程度' } },
  { q: '模块一的出口标准是什么？', gold: { file: 'docs/roadmap.md', phrase: '跨会话恢复' } },
  { q: '提交信息的格式规定是什么？', gold: { file: 'AGENTS.md', phrase: '提交信息格式' } },
];

type Chunk = { source: string; index: number; text: string; tf: Map<string, number>; length: number };

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

// 中文切成二元组（"提交信息" → "提交""交信""信息"），英文和数字保留为整词
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

function preview(text: string): string {
  return text.replace(/\s+/g, ' ').slice(0, 40);
}

async function main() {
  const chunks: Chunk[] = [];
  for (const file of await collectFiles()) {
    const text = await readFile(join(ROOT, file), 'utf8');
    chunkText(text, CHUNK_SIZE).forEach((part, index) => {
      const tokens = tokenize(part);
      chunks.push({ source: file, index, text: part, tf: termCounts(tokens), length: tokens.length });
    });
  }

  // df：每个词出现在多少个块里
  const df = new Map<string, number>();
  for (const chunk of chunks) {
    for (const term of chunk.tf.keys()) df.set(term, (df.get(term) ?? 0) + 1);
  }
  const avgLength = chunks.reduce((sum, c) => sum + c.length, 0) / chunks.length;

  console.log(`切分 ${CHUNK_SIZE} 字符，共 ${chunks.length} 块，平均长度 ${avgLength.toFixed(1)} 词`);
  let hits = 0;
  for (const { q, gold } of QUESTIONS) {
    const query = tokenize(q);
    const top = chunks
      .map((chunk) => ({ chunk, score: bm25Score(query, chunk, df, avgLength, chunks.length) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);
    const hit = top.some(({ chunk }) => chunk.source === gold.file && chunk.text.includes(gold.phrase));
    if (hit) hits++;
    console.log(`\n${hit ? '命中' : '未命中'}（期望 ${gold.file}）：${q}`);
    console.log(`   切词：${query.join(' ')}`);
    top.forEach(({ chunk, score }, rank) => {
      console.log(`   ${rank + 1}. ${score.toFixed(3)} ${chunk.source}#${chunk.index} ${preview(chunk.text)}`);
    });
  }
  console.log(`\nRecall@${TOP_K}：${hits}/${QUESTIONS.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
