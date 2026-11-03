import OpenAI from 'openai';
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';

// Day 5：Function Calling。模型决定调用哪个工具，程序执行后把结果回传，直到模型给出最终回答。
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
const MAX_STEPS = 5;

const SYSTEM_PROMPT = `你是任务管理助手。用户描述任务时，先用 get_current_date 确认今天的日期，再用 create_task 创建任务。
相对日期（明天、下周五）要根据今天的日期换算成 YYYY-MM-DD。最后用中文简短告诉用户结果。`;

// 工具的本地实现。模型只会生成参数，真正执行的是这里的代码，所以参数同样要校验。
const tasks: Array<{ title: string; priority: string; due: string | null; tags: string[] }> = [];

const toolImpls: Record<string, (args: Record<string, unknown>) => unknown> = {
  get_current_date: () => {
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    return { date: `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` };
  },
  create_task: (args) => {
    if (typeof args.title !== 'string' || args.title.trim() === '') {
      return { error: 'title 必须是非空字符串' };
    }
    if (args.priority !== 'low' && args.priority !== 'medium' && args.priority !== 'high') {
      return { error: 'priority 必须是 "low"、"medium" 或 "high"' };
    }
    if (args.due !== null && (typeof args.due !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(args.due))) {
      return { error: 'due 必须是 YYYY-MM-DD 格式的字符串或 null' };
    }
    if (!Array.isArray(args.tags) || !args.tags.every((t) => typeof t === 'string')) {
      return { error: 'tags 必须是字符串数组' };
    }
    const task = { title: args.title, priority: args.priority, due: args.due, tags: args.tags };
    tasks.push(task);
    return { ok: true, id: tasks.length, task };
  },
};


// 工具的 JSON Schema 描述：模型根据这里的 name、description、parameters 决定是否调用、怎么传参
const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'get_current_date',
      description: '获取今天的日期，格式 YYYY-MM-DD',
      parameters: { type: 'object', properties: {} },
    },
  },
  {
    type: 'function',
    function: {
      name: 'create_task',
      description: '创建一个任务',
      parameters: {
        type: 'object',
        properties: {
          title: { type: 'string', description: '任务标题' },
          priority: { type: 'string', enum: ['low', 'medium', 'high'] },
          due: { type: ['string', 'null'], description: '截止日期 YYYY-MM-DD，没有则为 null' },
          tags: { type: 'array', items: { type: 'string' } },
        },
        required: ['title', 'priority', 'due', 'tags'],
      },
    },
  },
];

async function run(userInput: string): Promise<void> {
  const messages: ChatCompletionMessageParam[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: userInput },
  ];

  for (let step = 1; step <= MAX_STEPS; step++) {
    const res = await client.chat.completions.create({ model, messages, tools, temperature: 0 });
    const message = res.choices[0].message;
    messages.push(message);

    // 没有 tool_calls 说明模型已经给出最终回答，循环结束
    if (!message.tool_calls || message.tool_calls.length === 0) {
      console.log(`回答：${message.content}`);
      return;
    }

    // 每个 tool_call 都要有一条 tool 消息，并用 tool_call_id 对应
    for (const call of message.tool_calls) {
      if (call.type !== 'function') continue;
      const name = call.function.name;
      console.log(`  [第 ${step} 步] 调用工具 ${name}(${call.function.arguments})`);

      let result: unknown;
      const impl = toolImpls[name];
      if (!impl) {
        result = { error: `未知工具：${name}` };
      } else {
        try {
          result = impl(JSON.parse(call.function.arguments));
        } catch (err) {
          result = { error: err instanceof Error ? err.message : String(err) };
        }
      }

      messages.push({ role: 'tool', tool_call_id: call.id, content: JSON.stringify(result) });
    }
  }

  console.log(`超过最大步数 ${MAX_STEPS}，停止`);
}

const samples = [
  '明天下午三点前把登录页的 bug 修掉，比较急，跟前端和 auth 有关',
  '帮我记一个任务：整理读书笔记，优先级低',
];

for (const text of samples) {
  console.log(`\n输入：${text}`);
  try {
    await run(text);
  } catch (err) {
    console.error('失败：', err instanceof Error ? err.message : err);
  }
}

console.log('\n已创建的任务：', JSON.stringify(tasks, null, 2));
