import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomBytes, scrypt, timingSafeEqual } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import Database from 'better-sqlite3';

// Day17：数据落盘到 SQLite 文件，进程重启后数据仍在；默认放在 workspace/（已被 .gitignore 忽略）
const dbPath = process.env.DB_PATH ?? 'workspace/day17.db';
mkdirSync(dirname(dbPath), { recursive: true });
const db = new Database(dbPath);
db.pragma('journal_mode = WAL');
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
`);

const insertUser = db.prepare('INSERT INTO users (username, password_hash) VALUES (?, ?)');
const findUser = db.prepare('SELECT id, password_hash FROM users WHERE username = ?');
const insertToken = db.prepare('INSERT INTO tokens (token, user_id) VALUES (?, ?)');
const userByToken = db.prepare(
  'SELECT u.id, u.username FROM tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ?',
);
const insertMessage = db.prepare('INSERT INTO messages (user_id, content) VALUES (?, ?)');
const listMessages = db.prepare(
  'SELECT id, content, created_at FROM messages WHERE user_id = ? ORDER BY id',
);

type AuthUser = { id: number; username: string };

function scryptAsync(password: string, salt: Buffer): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    scrypt(password, salt, 64, (err, key) => (err ? reject(err) : resolve(key)));
  });
}

// 存储格式 salt:hash，明文密码不落盘
async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  const key = await scryptAsync(password, salt);
  return `${salt.toString('hex')}:${key.toString('hex')}`;
}

async function verifyPassword(password: string, stored: string): Promise<boolean> {
  const [saltHex, keyHex] = stored.split(':');
  const expected = Buffer.from(keyHex, 'hex');
  const actual = await scryptAsync(password, Buffer.from(saltHex, 'hex'));
  return expected.length === actual.length && timingSafeEqual(expected, actual);
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(body));
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
}

// 鉴权：Authorization: Bearer <token>，token 存在 tokens 表中，服务重启后仍然有效
function authUser(req: IncomingMessage): AuthUser | undefined {
  const match = /^Bearer (.+)$/.exec(req.headers.authorization ?? '');
  if (!match) return undefined;
  return userByToken.get(match[1]) as AuthUser | undefined;
}

async function route(req: IncomingMessage, res: ServerResponse) {
  const url = new URL(req.url ?? '/', 'http://localhost');

  if (req.method === 'POST' && url.pathname === '/register') {
    const body = await readJson(req);
    const { username, password } = body;
    if (typeof username !== 'string' || !username || typeof password !== 'string' || password.length < 6) {
      return sendJson(res, 400, { error: 'username 不能为空，password 至少 6 位' });
    }
    if (findUser.get(username)) return sendJson(res, 409, { error: '用户名已存在' });
    const info = insertUser.run(username, await hashPassword(password));
    return sendJson(res, 201, { id: Number(info.lastInsertRowid), username });
  }

  if (req.method === 'POST' && url.pathname === '/login') {
    const { username, password } = await readJson(req);
    const row = typeof username === 'string' ? (findUser.get(username) as { id: number; password_hash: string } | undefined) : undefined;
    if (!row || typeof password !== 'string' || !(await verifyPassword(password, row.password_hash))) {
      return sendJson(res, 401, { error: '用户名或密码错误' });
    }
    const token = randomBytes(32).toString('hex');
    insertToken.run(token, row.id);
    return sendJson(res, 200, { token });
  }

  if (url.pathname === '/messages') {
    const user = authUser(req);
    if (!user) return sendJson(res, 401, { error: '缺少或无效的 token' });

    if (req.method === 'GET') {
      return sendJson(res, 200, { user: user.username, messages: listMessages.all(user.id) });
    }
    if (req.method === 'POST') {
      const { content } = await readJson(req);
      if (typeof content !== 'string' || !content) return sendJson(res, 400, { error: 'content 不能为空' });
      const info = insertMessage.run(user.id, content);
      return sendJson(res, 201, { id: Number(info.lastInsertRowid), content });
    }
  }

  return sendJson(res, 404, { error: 'Not Found' });
}

const port = Number(process.env.PORT ?? 3017);
const server = createServer(async (req, res) => {
  try {
    await route(req, res);
  } catch (err) {
    if (err instanceof SyntaxError) return sendJson(res, 400, { error: 'JSON 格式错误' });
    console.error(err);
    if (!res.headersSent) sendJson(res, 500, { error: '服务器错误' });
  }
});

server.listen(port, () => {
  console.log(`Day17 server: http://localhost:${port}  db=${dbPath}`);
});

process.on('SIGINT', () => {
  server.close();
  db.close();
  process.exit(0);
});
