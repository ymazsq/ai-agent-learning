import { createServer } from 'node:http';

const PORT = Number(process.env.PORT ?? 3016);

// 模拟模型逐 token 输出，不依赖 API Key，便于观察 SSE 协议本身
async function* fakeTokens(question: string): AsyncGenerator<string> {
  const reply = `你问的是：${question}。SSE 让服务端可以持续推送多条消息。`;
  for (const ch of reply) {
    await new Promise((resolve) => setTimeout(resolve, 80));
    yield ch;
  }
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', 'http://localhost');

  if (req.method !== 'GET' || url.pathname !== '/chat') {
    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
    return res.end(JSON.stringify({ error: 'not found' }));
  }

  const question = url.searchParams.get('q') ?? '你好';

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });

  // 客户端断开后 res 不再可写，据此停止生成
  let aborted = false;
  res.on('close', () => {
    aborted = true;
  });

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  for await (const token of fakeTokens(question)) {
    if (aborted) break;
    send('token', { text: token });
  }

  if (!aborted) send('done', { ok: true });
  res.end();
});

server.listen(PORT, () => {
  console.log(`Day16 SSE server: http://localhost:${PORT}`);
});
