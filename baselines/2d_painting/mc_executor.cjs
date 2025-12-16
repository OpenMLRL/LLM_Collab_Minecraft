/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const mineflayer = require('mineflayer');
const { Vec3 } = require('vec3');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseArgs(argv) {
  /** @type {Record<string, string | boolean>} */
  const out = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      out[key] = true;
      continue;
    }
    out[key] = next;
    i += 1;
  }
  return out;
}

function toInt(value, fallback) {
  const n = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(n) ? n : fallback;
}

function readStdinJson() {
  try {
    const raw = fs.readFileSync(0, 'utf8');
    const trimmed = raw.trim();
    if (!trimmed) return {};
    return JSON.parse(trimmed);
  } catch {
    return {};
  }
}

function normalizeCmd(line) {
  const s = String(line ?? '').trim();
  if (!s) return null;
  return s.startsWith('/') ? s : `/${s}`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const host = String(args.host ?? '127.0.0.1');
  const port = toInt(args.port, 25565);
  const username = String(args.username ?? 'executor_bot');
  const version = typeof args.version === 'string' ? args.version : undefined;
  const stepDelayMs = toInt(args.stepDelayMs ?? args['step-delay-ms'], 200);
  const scanDelayMs = toInt(args.scanDelayMs ?? args['scan-delay-ms'], 250);
  const timeoutMs = toInt(args.timeoutMs ?? args['timeout-ms'], 120_000);

  const payload = readStdinJson();
  const preCommands = Array.isArray(payload.pre_commands) ? payload.pre_commands : [];
  const commands = Array.isArray(payload.commands) ? payload.commands : [];
  const postCommands = Array.isArray(payload.post_commands) ? payload.post_commands : [];
  const scan = payload.scan && typeof payload.scan === 'object' ? payload.scan : null;

  const startMs = Date.now();
  const result = {
    ok: false,
    host,
    port,
    username,
    version: version ?? null,
    spawn: null,
    spawn_floored: null,
    executed: [],
    scan: null,
    chat: [],
    errors: [],
    duration_ms: null,
  };

  const bot = mineflayer.createBot({
    host,
    port,
    username,
    auth: 'offline',
    ...(version ? { version } : {}),
  });

  let finished = false;
  function finish(exitCode) {
    if (finished) return;
    finished = true;
    result.duration_ms = Date.now() - startMs;
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = exitCode;
    try {
      bot.quit();
    } catch {
      // ignore
    }
  }

  const timeout = setTimeout(() => {
    result.errors.push(`timeout after ${timeoutMs}ms (server not responding?)`);
    finish(2);
  }, timeoutMs);

  bot.on('kicked', (reason) => {
    result.errors.push(`kicked: ${String(reason)}`);
    clearTimeout(timeout);
    finish(3);
  });

  bot.on('error', (err) => {
    result.errors.push(`error: ${err?.message ?? String(err)}`);
    clearTimeout(timeout);
    finish(4);
  });

  bot.on('messagestr', (msg) => {
    if (result.chat.length < 200) result.chat.push(String(msg));
  });

  bot.once('spawn', async () => {
    try {
      const spawn = bot.entity.position;
      const spawnFloored = bot.entity.position.floored();
      result.spawn = [spawn.x, spawn.y, spawn.z];
      result.spawn_floored = [spawnFloored.x, spawnFloored.y, spawnFloored.z];

      const all = [...preCommands, ...commands, ...postCommands];
      for (let i = 0; i < all.length; i += 1) {
        const cmd = normalizeCmd(all[i]);
        if (!cmd) continue;
        bot.chat(cmd);
        result.executed.push({ index: i, cmd });
        await sleep(stepDelayMs);
      }

      if (scan && Array.isArray(scan.from) && Array.isArray(scan.to) && scan.from.length === 3 && scan.to.length === 3) {
        await sleep(scanDelayMs);
        const [x1, y1, z1] = scan.from.map((v) => Number(v));
        const [x2, y2, z2] = scan.to.map((v) => Number(v));
        const minX = Math.min(x1, x2);
        const maxX = Math.max(x1, x2);
        const minY = Math.min(y1, y2);
        const maxY = Math.max(y1, y2);
        const minZ = Math.min(z1, z2);
        const maxZ = Math.max(z1, z2);

        /** @type {{pos:[number,number,number], name:string|null}[]} */
        const blocks = [];
        for (let y = minY; y <= maxY; y += 1) {
          for (let x = minX; x <= maxX; x += 1) {
            for (let z = minZ; z <= maxZ; z += 1) {
              const b = bot.blockAt(new Vec3(x, y, z));
              blocks.push({ pos: [x, y, z], name: b?.name ?? null });
            }
          }
        }
        result.scan = { from: [minX, minY, minZ], to: [maxX, maxY, maxZ], blocks };
      }

      result.ok = true;
      clearTimeout(timeout);
      finish(0);
    } catch (e) {
      result.errors.push(`exception: ${e?.stack ?? e?.message ?? String(e)}`);
      clearTimeout(timeout);
      finish(5);
    }
  });
}

main().catch((e) => {
  process.stderr.write(`${e?.stack ?? e?.message ?? String(e)}\n`);
  process.exitCode = 10;
});
