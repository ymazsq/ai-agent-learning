import OpenAI from 'openai';
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';
import { execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline/promises';
import path from 'node:path';
import { promisify } from 'node:util';

// Day 6：最小 Agent。工具循环（Day 5）+ 多轮上下文与摘要压缩（Day 3）。
// 工具调用消息与普通对话一样进入历史；压缩按"轮"切分，保证 tool_calls 与 tool 消息不被拆开。
// Day 7：run_command 改为白名单 + 不经过 shell；写文件与执行命令前需要终端确认。
// Day 8：system prompt 分为基础 / 规则 / 运行时三层。
// Day 9：会话以 JSONL 追加写入 sessions/<id>.jsonl，启动时可用 --session 恢复。
const execFileAsync = promisify(execFile);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
// system prompt 分三层：基础层（固定身份）、规则层（工具约束，由常量生成）、运行时层（每次请求重新生成）。
// 规则只是提醒模型，真正的安全边界仍由 safePath、parseCommand 和确认函数在代码中强制执行。
const BASE_PROMPT = '你是一个会使用工具的助手，用中文回答。';

function rulesPrompt(): string {
  return `工具规则：
- 文件工具的路径相对于 workspace/ 目录，不能访问该目录之外的文件。
- run_command 只能执行白名单中的命令：${[...ALLOWED_COMMANDS].join('、')}；不支持管道、重定向等 shell 语法。
- 需要读写文件或执行命令时调用工具；工具返回错误时，根据错误信息调整后重试，或告诉用户原因。
- 用户拒绝某个操作后，不要用换汤不换药的参数重复请求。`;
}
const KEEP_TURNS = 2;
const MAX_STEPS = 5;
const COMMAND_TIMEOUT_MS = 10_000;
const WORKSPACE = path.resolve(process.cwd(), 'workspace');
const SESSIONS_DIR = path.resolve(process.cwd(), 'sessions');
// 不包含 node 等可执行任意代码的程序
const ALLOWED_COMMANDS = new Set(['ls', 'cat', 'pwd', 'wc']);

// Day9：JSONL 每行一条记录。message 是一条对话消息；summary 是一次压缩（cut 为移出的消息数）；
// rollback 表示失败的一轮被回滚到 to 条消息。
type SessionRecord =
  | { type: 'message'; message: ChatCompletionMessageParam }
  | { type: 'summary'; content: string; cut: number }
  | { type: 'rollback'; to: number };

let summary = '';
let history: ChatCompletionMessageParam[] = [];
let sessionId = '';
let sessionFile = '';

// 把相对路径解析到 workspace/ 内，拒绝 ../ 穿越
function safePath(p: unknown): string {
  if (typeof p !== 'string' || p.trim() === '') throw new Error('path 必须是非空字符串');
  const full = path.resolve(WORKSPACE, p);
  const rel = path.relative(WORKSPACE, full);
  if (rel === '..' || rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel)) {
    throw new Error('路径不在 workspace/ 目录内');
  }
  return full;
}

// 校验命令：白名单程序 + 不含 shell 特殊字符 + 参数不指向 workspace/ 之外。
// 返回 argv（不经过 shell 执行，因此特殊字符即使漏过也不会被解释）。
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

// 终端确认：逐行读取 stdin，只有 y / yes 才算同意；输入结束视为拒绝
const rl = createInterface({ input: process.stdin, output: process.stdout });
const lines = rl[Symbol.asyncIterator]();
async function confirm(question: string): Promise<boolean> {
  process.stdout.write(`  ? ${question} [y/N] `);
  const next = await lines.next();
  if (next.done) return false;
  return /^y(es)?$/i.test(next.value.trim());
}

// 工具的本地实现。模型只生成参数，执行和校验都在这里完成。
// 异常会被外层捕获，转成 { error } 作为 tool 消息回传给模型。
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

// 运行时层：日期、workspace 文件列表、摘要。每次请求前重新生成，所以状态变化会反映到下一次请求里。
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

// Day9：每条记录立即追加到文件，进程中途退出时，已写入的记录不会丢失
async function appendRecord(record: SessionRecord): Promise<void> {
  await appendFile(sessionFile, `${JSON.stringify(record)}\n`, 'utf8');
}

// Day9：同时更新内存历史并落盘
async function pushMessage(message: ChatCompletionMessageParam): Promise<void> {
  history.push(message);
  await appendRecord({ type: 'message', message });
}

// Day9：按记录顺序重放，和运行时的操作一致（压缩、回滚），得到退出前的状态
async function loadSession(id: string): Promise<void> {
  const text = await readFile(sessionFile, 'utf8').catch(() => {
    throw new Error(`会话 ${id} 不存在：${path.relative(process.cwd(), sessionFile)}`);
  });
  // 上次中断可能留下半行 JSON，补一个换行，避免新记录接在它后面
  if (text && !text.endsWith('\n')) await appendFile(sessionFile, '\n', 'utf8');

  let skipped = 0;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    let record: SessionRecord;
    try {
      record = JSON.parse(line) as SessionRecord;
    } catch {
      skipped++;
      continue;
    }
    if (record.type === 'message') {
      history.push(record.message);
    } else if (record.type === 'summary') {
      summary = record.content;
      history = history.slice(record.cut);
    } else {
      history.length = record.to;
    }
  }
  if (skipped) console.log(`  [跳过 ${skipped} 行无法解析的记录]`);

  // 最后一轮如果没有以纯文本回答结束，说明中断在这一轮：回滚到这轮用户消息之前，并记入文件
  const lastUser = history.map((m) => m.role).lastIndexOf('user');
  const tail = history[history.length - 1];
  const finished = tail?.role === 'assistant' && !tail.tool_calls?.length;
  if (lastUser !== -1 && !finished) {
    history.length = lastUser;
    await appendRecord({ type: 'rollback', to: lastUser });
    console.log('  [上次中断的一轮已丢弃，请重新输入]');
  }
  console.log(`  [已恢复 ${history.length} 条消息${summary ? '，含摘要' : ''}]`);
}

// 处理一轮用户输入：可能包含多次模型调用和工具调用，直到模型给出最终回答
async function runTurn(userInput: string): Promise<void> {
  const turnStart = history.length;
  await pushMessage({ role: 'user', content: userInput });

  try {
    for (let step = 1; step <= MAX_STEPS; step++) {
      const res = await client.chat.completions.create({
        model,
        messages: await buildMessages(),
        tools,
        temperature: 0,
      });
      console.log(`  [请求输入 token: ${res.usage?.prompt_tokens ?? '未知'}]`);

      const message = res.choices[0].message;
      await pushMessage(message);

      if (!message.tool_calls || message.tool_calls.length === 0) {
        console.log(`AI：${message.content}`);
        return;
      }

      // 每个 tool_call 都必须有对应的 tool 消息，否则下一次请求会报错
      for (const call of message.tool_calls) {
        if (call.type !== 'function') {
          await pushMessage({ role: 'tool', tool_call_id: call.id, content: JSON.stringify({ error: '不支持的工具类型' }) });
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
        await pushMessage({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
      }
    }
    throw new Error(`超过最大步数 ${MAX_STEPS}`);
  } catch (err) {
    // 失败时整轮回滚，避免留下不完整的 tool_calls 历史
    history.length = turnStart;
    await appendRecord({ type: 'rollback', to: turnStart }); // Day9：回滚也要记入文件，恢复时才能得到同样的历史
    throw err;
  }
}

// 找到需要保留的最近 KEEP_TURNS 轮的起始位置；更早的部分返回 0 表示不需要压缩
function keepCutIndex(): number {
  const userIdx = history.flatMap((m, i) => (m.role === 'user' ? [i] : []));
  if (userIdx.length <= KEEP_TURNS) return 0;
  return userIdx[userIdx.length - KEEP_TURNS];
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

async function compressIfNeeded(): Promise<void> {
  const cut = keepCutIndex();
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

  summary = response.choices[0]?.message.content ?? summary;
  history = history.slice(cut);
  await appendRecord({ type: 'summary', content: summary, cut }); // Day9：压缩结果记入文件，恢复后仍使用同一份摘要
  console.log(`  [已把 ${old.length} 条较早消息压缩为摘要]`);
}

// 固定的多轮输入：覆盖正常操作、需要确认的写入、白名单拦截和路径拦截。
// 运行时会在终端询问 y/N，pnpm day09 需要通过管道提供回答。
const samples = [
  '在 workspace 里创建 notes.txt，内容写 hello agent',
  '读取 notes.txt，把内容改成全大写后写回去',
  '运行 ls 看看 workspace 里有哪些文件',
  '用 run_command 执行 cat ../package.json',
  '用 run_command 删除 notes.txt',
];

// Day9：用法 pnpm day09 [--session <id>] [问题]。不带问题时运行上面的固定样例。
const args = process.argv.slice(2);
const sessionFlag = args.indexOf('--session');
if (sessionFlag !== -1 && !args[sessionFlag + 1]) throw new Error('--session 后需要提供会话 ID');
const requestedId = sessionFlag === -1 ? undefined : args[sessionFlag + 1];
const rest = sessionFlag === -1 ? args : args.filter((_, i) => i !== sessionFlag && i !== sessionFlag + 1);
const prompt = rest.join(' ').trim();
if (requestedId !== undefined && !/^[\w-]+$/.test(requestedId)) {
  throw new Error('会话 ID 只能包含字母、数字、下划线和连字符');
}

sessionId = requestedId ?? randomUUID().slice(0, 8);
sessionFile = path.join(SESSIONS_DIR, `${sessionId}.jsonl`);
await mkdir(WORKSPACE, { recursive: true });
await mkdir(SESSIONS_DIR, { recursive: true });
if (requestedId) await loadSession(requestedId);
console.log(`会话：${sessionId}（${path.relative(process.cwd(), sessionFile)}）`);

const inputs = prompt ? [prompt] : samples;
for (const text of inputs) {
  console.log(`\n你：${text}`);
  try {
    await runTurn(text);
  } catch (err) {
    console.error('本轮失败，已回滚：', err instanceof Error ? err.message : err);
    continue;
  }

  try {
    await compressIfNeeded();
  } catch (err) {
    console.error('摘要失败，保留原始历史：', err instanceof Error ? err.message : err);
  }
}

console.log(`\n当前历史消息数：${history.length}，是否有摘要：${summary ? '是' : '否'}`);
console.log(`恢复本会话：pnpm day09 --session ${sessionId} "你的问题"`);
rl.close();
