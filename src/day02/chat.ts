import OpenAI from 'openai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

// Day 2：流式输出 + 多轮对话历史。模型本身不保存历史，每次都要把 history 整体发过去。
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';

const history: ChatCompletionMessageParam[] = [
  { role: 'system', content: '你是一个简洁的助手，用中文回答。' },
];

async function chat(userInput: string): Promise<void> {
  history.push({ role: 'user', content: userInput });

  const stream = await client.chat.completions.create({
    model,
    messages: history,
    stream: true,
  });

  let answer = '';
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? '';
    output.write(delta);
    answer += delta;
  }
  output.write('\n');

  history.push({ role: 'assistant', content: answer });
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
      history.pop();
      console.error('\n调用失败：', err instanceof Error ? err.message : err);
    }
  }

  rl.close();
}

main().catch((err) => {
  console.error('程序异常：', err instanceof Error ? err.message : err);
  process.exit(1);
});
