import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';

const PORT = Number(process.env.PORT ?? 3000);
const MAX_SLOW_MS = 10_000;

function sendJson(res: ServerResponse, status: number, body: unknown) {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(text),
  });
  res.end(text);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk: string) => {
      data += chunk;
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

async function route(req: IncomingMessage, res: ServerResponse) {
  const url = new URL(req.url ?? '/', 'http://localhost');

  if (req.method === 'GET' && url.pathname === '/health') {
    return sendJson(res, 200, { ok: true });
  }

  if (req.method === 'GET' && url.pathname === '/slow') {
    // Day15：模拟长耗时 I/O，观察同时发起多个请求时服务是否被阻塞
    const ms = Number(url.searchParams.get('ms') ?? 1000);
    if (!Number.isFinite(ms) || ms < 0 || ms > MAX_SLOW_MS) {
      return sendJson(res, 400, { error: `ms must be 0-${MAX_SLOW_MS}` });
    }
    await new Promise((resolve) => setTimeout(resolve, ms));
    return sendJson(res, 200, { waitedMs: ms });
  }

  if (req.method === 'POST' && url.pathname === '/echo') {
    if (!req.headers['content-type']?.startsWith('application/json')) {
      return sendJson(res, 415, { error: 'Content-Type must be application/json' });
    }
    const raw = await readBody(req);
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return sendJson(res, 400, { error: 'invalid JSON' });
    }
    return sendJson(res, 200, { received: parsed });
  }

  sendJson(res, 404, { error: 'not found' });
}

const server = createServer(async (req, res) => {
  try {
    await route(req, res);
  } catch (err) {
    console.error(err);
    if (!res.headersSent) sendJson(res, 500, { error: 'internal error' });
  }
});

server.listen(PORT, () => {
  console.log(`Day15 server: http://localhost:${PORT}`);
});
