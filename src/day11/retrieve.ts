import { readFile, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { env, pipeline } from '@huggingface/transformers';

const ROOT = process.cwd();
const MODEL = 'Xenova/bge-small-zh-v1.5';
const CHUNK_SIZES = [200, 500, 1000];
const OVERLAP_RATIO = 0.2;
const TOP_K = 3;
const BATCH = 32;
// bge 中文模型的查询前缀，用于检索场景
const QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：';

// 国内网络无法直接访问 huggingface.co，这里使用公开镜像下载模型
env.remoteHost = 'https://hf-mirror.com/';

// expect：期望命中的源文件，前 TOP_K 个块中出现即算命中
const QUESTIONS = [
  { q: 'B 滑动窗口为什么会答错？', expect: 'LEARNING_LOG.md' },
  { q: '会话持久化为什么用追加写入的 JSONL？', expect: 'interview/day09.md' },
  { q: 'P1 总结里 Day 1 的核心要点是什么？', expect: 'docs/p1-summary.md' },
  { q: 'W4 阶段的出口标准是什么？', expect: 'docs/roadmap.md' },
  { q: '提交信息的格式规定是什么？', expect: 'AGENTS.md' },
];

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

// Day11：按字符滑动切分，相邻块重叠 20%，避免关键句正好落在边界上被切断
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
  return text.replace(/\s+/g, ' ').slice(0, 40);
}

async function main() {
  console.log(`加载模型 ${MODEL}（首次运行会下载，之后走本地缓存）...`);
  const extractor = (await pipeline('feature-extraction', MODEL)) as unknown as Embedder;

  const files = await collectDocs();
  const sources = await Promise.all(
    files.map(async (file) => ({ file, text: await readFile(join(ROOT, file), 'utf8') })),
  );
  const queryVectors = await embedAll(
    extractor,
    QUESTIONS.map((item) => QUERY_PREFIX + item.q),
  );

  for (const size of CHUNK_SIZES) {
    const chunks: Chunk[] = [];
    for (const { file, text } of sources) {
      const parts = chunkText(text, size);
      const vectors = await embedAll(extractor, parts);
      parts.forEach((part, index) => chunks.push({ source: file, index, text: part, vector: vectors[index] }));
    }

    console.log(`\n## 切分大小 ${size} 字符，共 ${chunks.length} 块`);
    let hits = 0;
    QUESTIONS.forEach((item, qi) => {
      const top = chunks
        .map((chunk) => ({ chunk, score: dot(chunk.vector, queryVectors[qi]) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, TOP_K);
      const hit = top.some(({ chunk }) => chunk.source === item.expect);
      if (hit) hits++;
      console.log(`Q${qi + 1} ${hit ? '命中' : '未命中'}（期望 ${item.expect}）：${item.q}`);
      top.forEach(({ chunk, score }, rank) => {
        console.log(`   ${rank + 1}. ${score.toFixed(3)} ${chunk.source}#${chunk.index} ${preview(chunk.text)}`);
      });
    });
    console.log(`Recall@${TOP_K}：${hits}/${QUESTIONS.length}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
