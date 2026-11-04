import OpenAI from 'openai';
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';
import { execFile } from 'node:child_process';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline/promises';
import path from 'node:path';
import { promisify } from 'node:util';

// Day10：压缩策略对比。同一段 10 轮对话分别用 A / B / C 三种策略运行，记录每次请求的输入 token。
// A：每轮都用模型摘要较早的历史（与 Day 8 相同）。
// B：滑动窗口，只保留最近 WINDOW_TURNS 轮，直接丢弃更早的内容，不调用模型。
// C：按 token 预算触发；先截断较早的工具结果，仍超出预算时才摘要。
const STRATEGY = process.argv[2] ?? 'A';
if (!['A', 'B', 'C'].includes(STRATEGY)) throw new Error('策略必须是 A、B 或 C，例如 pnpm day10 B');

const execFileAsync = promisify(execFile);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
const BASE_PROMPT = '你是一个会使用工具的助手，用中文回答。';

function rulesPrompt(): string {
  return `工具规则：
- 文件工具的路径相对于 workspace/ 目录，不能访问该目录之外的文件。
- run_command 只能执行白名单中的命令：${[...ALLOWED_COMMANDS].join('、')}；不支持管道、重定向等 shell 语法。
- 需要读写文件或执行命令时调用工具；工具返回错误时，根据错误信息调整后重试，或告诉用户原因。
- 用户拒绝某个操作后，不要用换汤不换药的参数重复请求。`;
}
const KEEP_TURNS = 2;
const WINDOW_TURNS = 3;
// 粗略预算：中文约 1 字符 ≈ 1 token，JSON 转义会略微增大体积，这里只用于决定是否压缩
const TOKEN_BUDGET = 2000;
const MAX_STEPS = 5;
const COMMAND_TIMEOUT_MS = 10_000;
const WORKSPACE = path.resolve(process.cwd(), 'workspace');
const ALLOWED_COMMANDS = new Set(['ls', 'cat', 'pwd', 'wc']);

let summary = '';
let history: ChatCompletionMessageParam[] = [];

// Day10：每次模型调用的输入 token，用于最后的对比
type CallRecord = { kind: 'chat' | 'summary'; promptTokens: number };
const calls: CallRecord[] = [];
let lastPromptTokens = 0;
let truncatedCount = 0;

function safePath(p: unknown): string {
  if (typeof p !== 'string' || p.trim() === '') throw new Error('path 必须是非空字符串');
  const full = path.resolve(WORKSPACE, p);
  const rel = path.relative(WORKSPACE, full);
  if (rel === '..' || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)) {
    throw new Error('路径不在 workspace/ 目录内');
  }
  return full;
}

function parseCommand(command: string): string[] {
  if (/[;&|<>`$()\\"'\n]/.test(command)) {
    throw new Error('命令不允许包含 ; & | < > ` $ ( ) \\ 引号或换行');
  }
  const argv = command.trim().split(/\s+/);
  const [bin, ...args] = argv;
  if (!ALLOWED_COMMANDS.has(bin)) {
    throw new Error(`命令 ${bin} 不在白名单中，允许的命令：${[...ALLOWED_COMMANDS].join(', ')}`);
  }
  for (const arg of args) {
    if (arg.startsWith('/') || arg.split(/[\\/]/).includes('..')) {
      throw new Error(`参数 ${arg} 指向 workspace/ 之外，已拒绝`);
    }
  }
  return argv;
}

const rl = createInterface({ input: process.stdin, output: process.stdout });
const lines = rl[Symbol.asyncIterator]();
async function confirm(question: string): Promise<boolean> {
  process.stdout.write(`  ? ${question} [y/N] `);
  const next = await lines.next();
  if (next.done) return false;
  return /^y(es)?$/i.test(next.value.trim());
}

const toolImpls: Record<string, (args: Record<string, unknown>) => Promise<unknown>> = {
  read_file: async (args) => {
    const file = safePath(args.path);
    return { content: await readFile(file, 'utf8') };
  },
  write_file: async (args) => {
    const file = safePath(args.path);
    if (typeof args.content !== 'string') throw new Error('content 必须是字符串');
    if (!(await confirm(`允许写入 ${path.relative(WORKSPACE, file)}（${args.content.length} 字符）吗？`))) {
      throw new Error('用户拒绝了本次写入');
    }
    await mkdir(path.dirname(file), { recursive: true });
    await writeFile(file, args.content, 'utf8');
    return { ok: true, path: path.relative(WORKSPACE, file) };
  },
  run_command: async (args) => {
    if (typeof args.command !== 'string' || args.command.trim() === '') {
      throw new Error('command 必须是非空字符串');
    }
    const argv = parseCommand(args.command);
    if (!(await confirm(`允许执行命令 "${argv.join(' ')}" 吗？`))) {
      throw new Error('用户拒绝了本次命令执行');
    }
    try {
      const [bin, ...rest] = argv;
      const { stdout, stderr } = await execFileAsync(bin, rest, {
        cwd: WORKSPACE,
        timeout: COMMAND_TIMEOUT_MS,
      });
      return { exitCode: 0, stdout, stderr };
    } catch (err) {
      const e = err as { code?: number; killed?: boolean; stdout?: string; stderr?: string; message: string };
      if (e.killed) return { error: `命令超过 ${COMMAND_TIMEOUT_MS}ms 被终止` };
      return { exitCode: e.code ?? null, stdout: e.stdout ?? '', stderr: e.stderr ?? e.message };
    }
  },
};

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: '读取 workspace/ 下的文本文件',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string', description: '相对于 workspace/ 的路径' } },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: '写入 workspace/ 下的文本文件，文件存在则覆盖',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: '相对于 workspace/ 的路径' },
          content: { type: 'string', description: '完整的文件内容' },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_command',
      description: '在 workspace/ 目录下执行白名单中的命令（ls、cat、pwd、wc），不支持管道和重定向，超时 10 秒',
      parameters: {
        type: 'object',
        properties: { command: { type: 'string', description: '要执行的命令，例如 "ls -la" 或 "cat notes.txt"' } },
        required: ['command'],
      },
    },
  },
];

async function runtimeContext(): Promise<string> {
  const now = new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' });
  const files = await readdir(WORKSPACE).catch(() => [] as string[]);
  const lines = [
    `当前时间：${now}`,
    `workspace/ 中的文件：${files.length ? files.join('、') : '（空）'}`,
  ];
  if (summary) lines.push(`更早对话的摘要：\n${summary}`);
  return lines.join('\n');
}

async function buildMessages(): Promise<ChatCompletionMessageParam[]> {
  const system = [BASE_PROMPT, rulesPrompt(), `运行时状态：\n${await runtimeContext()}`].join('\n\n');
  return [{ role: 'system', content: system }, ...history];
}

// 记录一次模型调用的输入 token，返回该值
function recordCall(kind: CallRecord['kind'], promptTokens: number | undefined): number {
  const n = promptTokens ?? 0;
  calls.push({ kind, promptTokens: n });
  return n;
}

// 处理一轮用户输入，返回最终回答的文本
async function runTurn(userInput: string): Promise<string> {
  const turnStart = history.length;
  history.push({ role: 'user', content: userInput });

  try {
    for (let step = 1; step <= MAX_STEPS; step++) {
      const res = await client.chat.completions.create({
        model,
        messages: await buildMessages(),
        tools,
        temperature: 0,
      });
      lastPromptTokens = recordCall('chat', res.usage?.prompt_tokens);
      console.log(`  [请求输入 token: ${lastPromptTokens}]`);

      const message = res.choices[0].message;
      history.push(message);

      if (!message.tool_calls || message.tool_calls.length === 0) {
        console.log(`AI：${message.content}`);
        return message.content ?? '';
      }

      for (const call of message.tool_calls) {
        if (call.type !== 'function') {
          history.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify({ error: '不支持的工具类型' }) });
          continue;
        }
        const name = call.function.name;
        console.log(`  [第 ${step} 步] 调用 ${name}(${call.function.arguments})`);

        let result: unknown;
        const impl = toolImpls[name];
        if (!impl) {
          result = { error: `未知工具：${name}` };
        } else {
          try {
            result = await impl(JSON.parse(call.function.arguments));
          } catch (err) {
            result = { error: err instanceof Error ? err.message : String(err) };
          }
        }
        history.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
      }
    }
    throw new Error(`超过最大步数 ${MAX_STEPS}`);
  } catch (err) {
    history.length = turnStart;
    throw err;
  }
}

// 从最近 turns 轮开始的位置；轮数不足时返回 0（表示不需要切分）
function cutIndexKeeping(turns: number): number {
  const userIdx = history.flatMap((m, i) => (m.role === 'user' ? [i] : []));
  if (userIdx.length <= turns) return 0;
  return userIdx[userIdx.length - turns];
}

function renderMessage(m: ChatCompletionMessageParam): string {
  const text = typeof m.content === 'string' ? m.content : '';
  if (m.role === 'user') return `用户：${text}`;
  if (m.role === 'tool') return `工具结果：${text}`;
  if (m.role === 'assistant') {
    const calls = (m.tool_calls ?? []).flatMap((c) =>
      c.type === 'function' ? [`调用工具 ${c.function.name}(${c.function.arguments})`] : [],
    );
    return [text ? `助手：${text}` : '', ...calls].filter(Boolean).join('\n');
  }
  return '';
}

// 策略 A 和 C 的摘要步骤：把 KEEP_TURNS 轮之前的历史压缩为摘要
async function summarizeOld(): Promise<void> {
  const cut = cutIndexKeeping(KEEP_TURNS);
  if (cut === 0) return;

  const old = history.slice(0, cut);
  const transcript = old.map(renderMessage).filter(Boolean).join('\n');

  const response = await client.chat.completions.create({
    model,
    messages: [
      { role: 'system', content: '请用简短的中文总结下面的对话要点，保留用户提供的关键信息（如文件名、已完成的操作、约定）。' },
      { role: 'user', content: `${summary ? `已有摘要：${summary}\n\n` : ''}新对话：\n${transcript}` },
    ],
  });
  recordCall('summary', response.usage?.prompt_tokens);

  summary = response.choices[0]?.message.content ?? summary;
  history = history.slice(cut);
  console.log(`  [已把 ${old.length} 条较早消息压缩为摘要]`);
}

// 策略 C 的第一步：把 KEEP_TURNS 轮之前的长工具结果替换为一行占位，用户和助手消息保留
function truncateOldToolResults(): void {
  const cut = cutIndexKeeping(KEEP_TURNS);
  for (let i = 0; i < cut; i++) {
    const m = history[i];
    if (m.role === 'tool' && typeof m.content === 'string' && m.content.length > 200) {
      const originalChars = m.content.length;
      history[i] = { ...m, content: JSON.stringify({ truncated: true, originalChars }) };
      truncatedCount++;
    }
  }
}

async function applyStrategy(): Promise<void> {
  if (STRATEGY === 'A') {
    await summarizeOld();
    return;
  }

  if (STRATEGY === 'B') {
    const cut = cutIndexKeeping(WINDOW_TURNS);
    if (cut > 0) {
      history = history.slice(cut);
      console.log(`  [窗口：丢弃 ${cut} 条较早消息，未调用模型]`);
    }
    return;
  }

  if (lastPromptTokens <= TOKEN_BUDGET) return;
  truncateOldToolResults();
  const estimate = JSON.stringify(history).length;
  console.log(`  [超出预算 ${lastPromptTokens} > ${TOKEN_BUDGET}，截断后历史粗估 ${estimate}]`);
  if (estimate > TOKEN_BUDGET) await summarizeOld();
}

// 固定的 10 轮脚本：第 1 轮告诉模型一个事实，中间 8 轮读取文件（产生较长的工具结果），最后一轮检查事实是否还在
const TITLES = ['背景', '目标', '方法', '数据', '实现', '测试', '结果', '结论'];
const LONG_TEXT = TITLES.map((t, i) => `第${i + 1}节：${t}\n${'本节内容用于测试压缩策略，'.repeat(12)}`).join('\n\n');
const SCRIPT = [
  '请记住：这次实验的项目代号是 BLUE-HERON，后面的问题都要用到它。',
  ...TITLES.map((_, i) => `读取 long.txt，告诉我第 ${i + 1} 节的标题`),
  '我们最开始约定的项目代号是什么？只回答代号。',
];

await mkdir(WORKSPACE, { recursive: true });
await writeFile(path.join(WORKSPACE, 'long.txt'), LONG_TEXT, 'utf8');
console.log(`策略：${STRATEGY}，共 ${SCRIPT.length} 轮`);

let lastAnswer = '';
for (const text of SCRIPT) {
  console.log(`\n你：${text}`);
  try {
    lastAnswer = await runTurn(text);
  } catch (err) {
    lastAnswer = '';
    console.error('本轮失败，已回滚：', err instanceof Error ? err.message : err);
    continue;
  }

  try {
    await applyStrategy();
  } catch (err) {
    console.error('压缩失败，保留原始历史：', err instanceof Error ? err.message : err);
  }
}

const chatCalls = calls.filter((c) => c.kind === 'chat');
const summaryCalls = calls.filter((c) => c.kind === 'summary');
const peak = Math.max(0, ...chatCalls.map((c) => c.promptTokens));
const total = calls.reduce((sum, c) => sum + c.promptTokens, 0);

console.log(`\n== 策略 ${STRATEGY} 汇总 ==`);
console.log(`模型调用：对话 ${chatCalls.length} 次，摘要 ${summaryCalls.length} 次`);
console.log(`输入 token：对话请求峰值 ${peak}，所有调用合计 ${total}`);
console.log(`截断的工具结果：${truncatedCount} 条`);
console.log(`最终历史消息数：${history.length}，是否有摘要：${summary ? '是' : '否'}`);
console.log(`最后一题是否答对（回答包含 BLUE-HERON）：${/BLUE-HERON/i.test(lastAnswer) ? '是' : '否'}`);
rl.close();
