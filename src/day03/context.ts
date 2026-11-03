import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

// Day 3：上下文管理。最近 KEEP_TURNS 轮保留原文，更早的对话压缩成一段摘要，控制输入 token 增长。
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
const SYSTEM_PROMPT = '你是一个简洁的助手，用中文回答。';
const KEEP_TURNS = 2;

let summary = '';
let recent: ChatCompletionMessageParam[] = [];

function buildMessages(): ChatCompletionMessageParam[] {
  const system = summary
    ? `${SYSTEM_PROMPT}\n\n以下是更早对话的摘要：\n${summary}`
    : SYSTEM_PROMPT;
  return [{ role: 'system', content: system }, ...recent];
}

async function chat(userInput: string): Promise<void> {
  recent.push({ role: 'user', content: userInput });

  try {
    const stream = await client.chat.completions.create({
      model,
      messages: buildMessages(),
      stream: true,
      stream_options: { include_usage: true },
    });

    let answer = '';
    let promptTokens: number | undefined;
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content ?? '';
      output.write(delta);
      answer += delta;
      if (chunk.usage) promptTokens = chunk.usage.prompt_tokens;
    }
    output.write('\n');

    recent.push({ role: 'assistant', content: answer });
    if (promptTokens !== undefined) {
      console.log(`[本轮输入 token: ${promptTokens}，保留消息数: ${recent.length}]`);
    }
  } catch (err) {
    recent.pop();
    throw err;
  }
}

async function compressIfNeeded(): Promise<void> {
  const maxMessages = KEEP_TURNS * 2;
  if (recent.length <= maxMessages) return;

  const old = recent.slice(0, recent.length - maxMessages);
  const transcript = old
    .map((m) => `${m.role === 'user' ? '用户' : '助手'}：${m.content}`)
    .join('\n');

  const response = await client.chat.completions.create({
    model,
    messages: [
      { role: 'system', content: '请用简短的中文总结下面的对话要点，保留用户提供的关键信息（如姓名、偏好、约定）。' },
      { role: 'user', content: `${summary ? `已有摘要：${summary}\n\n` : ''}新对话：\n${transcript}` },
    ],
  });

  summary = response.choices[0]?.message.content ?? summary;
  recent = recent.slice(-maxMessages);
  console.log(`[已把 ${old.length} 条较早消息压缩为摘要]`);
}

async function main(): Promise<void> {
  const rl = readline.createInterface({ input, output });
  let closed = false;
  rl.on('close', () => {
    closed = true;
  });
  console.log("输入内容后回车发送，输入 'exit' 退出。\n");

  while (!closed) {
    const userInput = (await rl.question('你: ')).trim();
    if (!userInput) continue;
    if (userInput === 'exit') break;

    output.write('AI: ');
    try {
      await chat(userInput);
    } catch (err) {
      console.error('\n调用失败：', err instanceof Error ? err.message : err);
      continue;
    }

    try {
      await compressIfNeeded();
    } catch (err) {
      console.error('摘要失败，保留原始历史：', err instanceof Error ? err.message : err);
    }
  }

  rl.close();
}

main().catch((err) => {
  console.error('程序异常：', err instanceof Error ? err.message : err);
  process.exit(1);
});
