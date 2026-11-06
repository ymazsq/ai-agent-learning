import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';
import OpenAI from 'openai';

const ROOT = process.cwd();
const MODEL = 'Xenova/bge-small-zh-v1.5';
const CHUNK_SIZE = 500;
const OVERLAP_RATIO = 0.2;
const TOP_K = 3;
const BATCH = 32;
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';
// 校准结果：有依据的问题 TOP1 为 0.50–0.66，无依据的为 0.31–0.39，阈值取中间
const THRESHOLD = Number(process.env.THRESHOLD ?? 0.45);

env.remoteHost = 'https://hf-mirror.com/';

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});
const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

// expect 为期望依据的源文件；没有依据的问题不设 expect，应当被拒答
const ANSWERABLE = [
  { q: 'B 滑动窗口为什么会答错？', expect: 'LEARNING_LOG.md' },
  { q: '会话持久化为什么用追加写入的 JSONL？', expect: 'interview/day09.md' },
  { q: 'P1 总结里 Day 1 的核心要点是什么？', expect: 'docs/p1-summary.md' },
  { q: 'W4 阶段的出口标准是什么？', expect: 'docs/roadmap.md' },
  { q: '提交信息的格式规定是什么？', expect: 'AGENTS.md' },
];
const UNANSWERABLE = [
  { q: '今天北京的天气怎么样？' },
  { q: '红烧肉的做法是什么？' },
  { q: '公司的年假制度是怎么规定的？' },
];

const REFUSAL = '无法从资料中找到依据';

type Embedder = (
  input: string[],
  options: { pooling: 'cls'; normalize: boolean },
) => Promise<{ tolist(): number[][] }>;

type Chunk = { source: string; index: number; text: string; vector: number[] };

async function collectDocs(): Promise<string[]> {
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

function preview(text: string): string {
  return text.replace(/\s+/g, ' ').slice(0, 60);
}

// 返回回答中出现的引用编号，以及超出检索块数量的非法编号
function checkCitations(answer: string, count: number): { cited: number[]; invalid: number[] } {
  const nums = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const unique = [...new Set(nums)];
  return {
    cited: unique.filter((n) => n >= 1 && n <= count),
    invalid: unique.filter((n) => n < 1 || n > count),
  };
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

async function main() {
  console.log(`加载模型 ${MODEL}（首次运行会下载，之后走本地缓存）...`);
  const extractor = (await pipeline('feature-extraction', MODEL)) as unknown as Embedder;

  const files = await collectDocs();
  const sources = await Promise.all(
    files.map(async (file) => ({ file, text: await readFile(join(ROOT, file), 'utf8') })),
  );

  const chunks: Chunk[] = [];
  for (const { file, text } of sources) {
    const parts = chunkText(text, CHUNK_SIZE);
    const vectors = await embedAll(extractor, parts);
    parts.forEach((part, index) => chunks.push({ source: file, index, text: part, vector: vectors[index] }));
  }
  console.log(`索引完成：${chunks.length} 块，切分大小 ${CHUNK_SIZE}，阈值 ${THRESHOLD}\n`);

  const all = [
    ...ANSWERABLE.map((item) => ({ ...item, answerable: true })),
    ...UNANSWERABLE.map((item) => ({ ...item, expect: undefined, answerable: false })),
  ];
  const queryVectors = await embedAll(
    extractor,
    all.map((item) => QUERY_PREFIX + item.q),
  );

  for (let qi = 0; qi < all.length; qi++) {
    const item = all[qi];
    const ranked = chunks
      .map((chunk) => ({ chunk, score: dot(chunk.vector, queryVectors[qi]) }))
      .sort((a, b) => b.score - a.score);
    const top = ranked.slice(0, TOP_K);
    const top1 = top[0].score;
    const gap = top1 - top[1].score;
    const refuse = top1 < THRESHOLD;

    console.log(`## ${item.answerable ? '有依据' : '无依据'}：${item.q}`);
    console.log(`TOP1 ${top1.toFixed(3)}，TOP1-TOP2 差距 ${gap.toFixed(3)}`);
    top.forEach(({ chunk, score }, rank) => {
      console.log(`   [${rank + 1}] ${score.toFixed(3)} ${chunk.source}#${chunk.index} ${preview(chunk.text)}`);
    });

    if (refuse) {
      console.log('结果：拒答（TOP1 低于阈值，未调用模型）\n');
      continue;
    }

    const answer = await generate(item.q, top);
    const { cited, invalid } = checkCitations(answer, top.length);
    const citedSources = cited.map((n) => `[${n}] ${top[n - 1].chunk.source}`);
    const modelRefused = answer.startsWith(REFUSAL);
    console.log(`回答：${answer}`);
    if (modelRefused) console.log('结果：模型拒答（检索通过阈值，但模型判断资料不足）');
    console.log(`引用：${citedSources.length ? citedSources.join('，') : '无'}`);
    if (invalid.length) console.log(`非法引用编号：${invalid.join('，')}（超出检索块数量，疑似编造）`);
    if (item.answerable && item.expect && !citedSources.some((s) => s.endsWith(item.expect))) {
      console.log(`注意：引用中没有期望来源 ${item.expect}`);
    }
    console.log('');
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
