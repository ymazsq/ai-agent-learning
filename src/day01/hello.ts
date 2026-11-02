import OpenAI from "openai";

// Day 1：第一次调用 LLM API，观察 messages、temperature 和 token 用量
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const model = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

async function ask(question: string, temperature: number): Promise<void> {
  const response = await client.chat.completions.create({
    model,
    temperature,
    messages: [
      { role: "system", content: "你是一个简洁的助手，用中文回答。" },
      { role: "user", content: question },
    ],
  });

  const answer = response.choices[0]?.message.content ?? "";
  const usage = response.usage;

  console.log(`\n[temperature=${temperature}] ${answer}`);
  console.log(
    `用量: prompt=${usage?.prompt_tokens} completion=${usage?.completion_tokens} total=${usage?.total_tokens}`,
  );
}

async function main(): Promise<void> {
  const question = "用一句话解释什么是 Token。";

  // 同一个问题，分别用低温和高温调用，对比输出的差异
  await ask(question, 0);
  await ask(question, 1.2);
}

main().catch((err) => {
  console.error("调用失败：", err instanceof Error ? err.message : err);
  process.exit(1);
});
