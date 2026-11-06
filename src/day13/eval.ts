import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';
import OpenAI from 'openai';

const ROOT = process.cwd();
const MODEL = 'Xenova/bge-small-zh-v1.5';
const OVERLAP_RATIO = 0.2;
const BATCH = 32;
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';
const THRESHOLD = 0.45;
const REFUSAL = '无法从资料中找到依据';

// 对比的配置：切分大小与 TOP-K
const CONFIGS = [
  { size: 300, topK: 3 },
  { size: 500, topK: 3 },
  { size: 500, topK: 5 },
];

env.remoteHost = 'https://hf-mirror.com/';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});
const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

// 标注到块级：gold.phrase 必须出现在某个块里，才算该块是答案块
type Gold = { file: string; phrase: string };
type Item = { q: string; gold?: Gold };

const ANSWERABLE: Item[] = [
  { q: '滑动窗口为什么会答错？', gold: { file: 'LEARNING_LOG.md', phrase: '窗口会直接丢掉早期事实' } },
  { q: '会话持久化为什么用追加写入的 JSONL？', gold: { file: 'interview/day09.md', phrase: '追加写入每条记录只写一行' } },
  { q: 'P1 总结里 Day 1 的核心要点是什么？', gold: { file: 'docs/p1-summary.md', phrase: 'temperature 控制选词的随机程度' } },
  { q: '模块一的出口标准是什么？', gold: { file: 'docs/roadmap.md', phrase: '跨会话恢复' } },
  { q: '提交信息的格式规定是什么？', gold: { file: 'AGENTS.md', phrase: '提交信息格式' } },
];
const UNANSWERABLE: Item[] = [
  { q: '今天北京的天气怎么样？' },
  { q: '红烧肉的做法是什么？' },
  { q: '公司的年假制度是怎么规定的？' },
];
const ITEMS = [...ANSWERABLE, ...UNANSWERABLE];

type Embedder = (
  input: string[],
  options: { pooling: 'cls'; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

type Chunk = { source: string; index: number; text: string; vector: number[] };
type Doc = { file: string; text: string };

type Row = {
  q: string;
  answerable: boolean;
  valid: boolean;
  hit: boolean;
  thresholdRefused: boolean;
  modelRefused: boolean;
  citedGold: boolean;
  invalidCitations: number;
  reason: string;
};

async function collectDocs(): Promise<Doc[]> {
  const files = ['AGENTS.md', 'LEARNING_LOG.md'];
  for (const dir of ['docs', 'interview']) {
    for (const name of await readdir(join(ROOT, dir))) {
      if (name.endsWith('.md')) files.push(`${dir}/${name}`);
    }
  }
  return Promise.all(files.map(async (file) => ({ file, text: await readFile(join(ROOT, file), 'utf8') })));
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

async function buildIndex(extractor: Embedder, docs: Doc[], size: number): Promise<Chunk[]> {
  const chunks: Chunk[] = [];
  for (const { file, text } of docs) {
    const parts = chunkText(text, size);
    const vectors = await embedAll(extractor, parts);
    parts.forEach((part, index) => chunks.push({ source: file, index, text: part, vector: vectors[index] }));
  }
  return chunks;
}

async function generate(question: string, top: { chunk: Chunk }[]): Promise<string> {
  const sources = top
    .map(({ chunk }, i) => `[${i + 1}] 来源：${chunk.source}#${chunk.index}\n${chunk.text}`)
    .join('\n\n');
  const response = await client.chat.completions.create({
    model,
    temperature: 0,
    messages: [
      {
        role: 'system',
        content:
          '你是资料问答助手。只能依据用户提供的资料回答，不得使用资料之外的知识。' +
          '每个结论后用 [编号] 标注出处，例如 [1]。如果资料不足以回答，只回复"' + REFUSAL + '"。',
      },
      { role: 'user', content: `资料：\n${sources}\n\n问题：${question}` },
    ],
  });
  return response.choices[0]?.message?.content?.trim() ?? '';
}

function isGoldChunk(chunk: Chunk, gold: Gold): boolean {
  return chunk.source === gold.file && chunk.text.includes(gold.phrase);
}

// 归因顺序：标注是否有效 → 检索是否召回 → 哪一层拒答 → 引用是否指向答案块
function reasonOf(row: Omit<Row, 'reason'>): string {
  if (!row.answerable) return row.thresholdRefused || row.modelRefused ? '正确拒答' : '编造（未拒答）';
  if (!row.valid) return '标注失效（切分切断了答案短语）';
  if (!row.hit) return row.thresholdRefused ? '检索未召回（TOP1 低于阈值）' : '检索未召回';
  if (row.thresholdRefused) return '阈值误拒（答案块已召回）';
  if (row.modelRefused) return '模型拒答（答案块已召回）';
  if (row.citedGold) return '正确';
  return '引用未指向答案块';
}

async function evaluate(
  cfg: { size: number; topK: number },
  extractor: Embedder,
  docs: Doc[],
  queryVectors: number[][],
): Promise<{ rows: Row[]; blocks: number }> {
  const index = await buildIndex(extractor, docs, cfg.size);
  const rows: Row[] = [];

  for (let qi = 0; qi < ITEMS.length; qi++) {
    const item = ITEMS[qi];
    const top = index
      .map((chunk) => ({ chunk, score: dot(chunk.vector, queryVectors[qi]) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, cfg.topK);

    const thresholdRefused = top[0].score < THRESHOLD;
    const gold = item.gold;
    const valid = gold ? index.some((c) => isGoldChunk(c, gold)) : true;
    const hit = gold ? top.some(({ chunk }) => isGoldChunk(chunk, gold)) : false;

    let modelRefused = false;
    let citedGold = false;
    let invalidCitations = 0;
    if (!thresholdRefused) {
      const answer = await generate(item.q, top);
      modelRefused = answer.startsWith(REFUSAL);
      const nums = [...new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))];
      invalidCitations = nums.filter((n) => n < 1 || n > top.length).length;
      citedGold =
        gold !== undefined &&
        nums.some((n) => n >= 1 && n <= top.length && isGoldChunk(top[n - 1].chunk, gold));
    }

    const base = {
      q: item.q,
      answerable: gold !== undefined,
      valid,
      hit,
      thresholdRefused,
      modelRefused,
      citedGold,
      invalidCitations,
    };
    rows.push({ ...base, reason: reasonOf(base) });
  }
  return { rows, blocks: index.length };
}

function printConfig(cfg: { size: number; topK: number }, rows: Row[], blocks: number) {
  const ans = rows.filter((r) => r.answerable && r.valid);
  const invalid = rows.filter((r) => r.answerable && !r.valid);
  const unans = rows.filter((r) => !r.answerable);
  const answered = ans.filter((r) => !r.thresholdRefused && !r.modelRefused);
  const hits = ans.filter((r) => r.hit).length;
  const citeOk = answered.filter((r) => r.citedGold).length;
  const falseRefuse = ans.length - answered.length;
  const correctRefuse = unans.filter((r) => r.thresholdRefused || r.modelRefused).length;
  const invalidTotal = rows.reduce((s, r) => s + r.invalidCitations, 0);

  console.log(`## 切分 ${cfg.size}，TOP-${cfg.topK}（索引 ${blocks} 块）`);
  console.log(`- Recall@${cfg.topK}：${hits}/${ans.length}`);
  console.log(`- 引用正确（回答中引用到答案块）：${citeOk}/${answered.length}`);
  console.log(`- 有依据误拒：${falseRefuse}/${ans.length}`);
  console.log(`- 无依据正确拒答：${correctRefuse}/${unans.length}`);
  console.log(`- 非法引用编号：${invalidTotal}`);
  if (invalid.length) console.log(`- 标注失效（不计入上面指标）：${invalid.map((r) => r.q).join('；')}`);
  for (const r of rows) {
    const mark = r.answerable ? (r.hit ? '命中' : '未中') : '-';
    console.log(`  - ${r.q} | ${mark} | ${r.reason}`);
  }
  console.log('');
}

async function main() {
  console.log(`加载模型 ${MODEL}...`);
  const extractor = (await pipeline('feature-extraction', MODEL)) as unknown as Embedder;
  const docs = await collectDocs();
  const queryVectors = await embedAll(
    extractor,
    ITEMS.map((item) => QUERY_PREFIX + item.q),
  );

  console.log(`阈值 ${THRESHOLD}（与 Day 12 相同，未重新标定）\n`);
  for (const cfg of CONFIGS) {
    const { rows, blocks } = await evaluate(cfg, extractor, docs, queryVectors);
    printConfig(cfg, rows, blocks);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
