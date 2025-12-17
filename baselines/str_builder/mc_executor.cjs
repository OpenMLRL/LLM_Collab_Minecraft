/* eslint-disable no-console */
'use strict';

const fs = require('fs');
const mineflayer = require('mineflayer');
const { Vec3 } = require('vec3');

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function asCmdList(value) {
  return Array.isArray(value) ? value : [];
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

function waitForSpawnOrFail(bot, botResult, result, finish) {
  return new Promise((resolve, reject) => {
    const onSpawn = () => {
      try {
        const spawn = bot.entity.position;
        const spawnFloored = bot.entity.position.floored();
        botResult.spawn = [spawn.x, spawn.y, spawn.z];
        botResult.spawn_floored = [spawnFloored.x, spawnFloored.y, spawnFloored.z];
      } catch {
        // ignore
      }
      cleanup();
      resolve();
    };
    const onKicked = (reason) => {
      const msg = `kicked: ${String(reason)}`;
      botResult.errors.push(msg);
      result.errors.push(`${botResult.username} ${msg}`);
      cleanup();
      finish(3);
      reject(new Error(msg));
    };
    const onError = (err) => {
      const msg = `error: ${err?.message ?? String(err)}`;
      botResult.errors.push(msg);
      result.errors.push(`${botResult.username} ${msg}`);
      cleanup();
      finish(4);
      reject(new Error(msg));
    };

    function cleanup() {
      bot.removeListener('spawn', onSpawn);
      bot.removeListener('kicked', onKicked);
      bot.removeListener('error', onError);
    }

    bot.once('spawn', onSpawn);
    bot.once('kicked', onKicked);
    bot.once('error', onError);
  });
}

async function executeAll(bot, items, executedArr, stepDelayMs, botIndex) {
  for (let i = 0; i < items.length; i += 1) {
    const cmd = normalizeCmd(items[i]);
    if (!cmd) continue;
    bot.chat(cmd);
    executedArr.push({ index: i, cmd, bot: botIndex });
    await sleep(stepDelayMs);
  }
}

async function scanRegion(bot, scan, scanDelayMs) {
  if (!scan || !Array.isArray(scan.from) || !Array.isArray(scan.to) || scan.from.length !== 3 || scan.to.length !== 3) return null;
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
  return { from: [minX, minY, minZ], to: [maxX, maxY, maxZ], blocks };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const host = String(args.host ?? '127.0.0.1');
  const port = toInt(args.port, 25565);
  const username = String(args.username ?? 'executor_bot');
  const username2 = typeof args.username2 === 'string' ? String(args.username2) : null;
  const version = typeof args.version === 'string' ? args.version : undefined;
  const stepDelayMs = toInt(args.stepDelayMs ?? args['step-delay-ms'], 200);
  const scanDelayMs = toInt(args.scanDelayMs ?? args['scan-delay-ms'], 250);
  const timeoutMs = toInt(args.timeoutMs ?? args['timeout-ms'], 120_000);

  const payload = readStdinJson();
  const scan = payload.scan && typeof payload.scan === 'object' ? payload.scan : null;

  const startMs = Date.now();
  const result = {
    ok: false,
    host,
    port,
    username,
    username2: username2 ?? null,
    version: version ?? null,
    spawn: null,
    spawn_floored: null,
    executed: [],
    bots: null,
    scan: null,
    chat: [],
    errors: [],
    duration_ms: null,
  };

  /** @type {import('mineflayer').Bot | null} */
  let bot1 = null;
  /** @type {import('mineflayer').Bot | null} */
  let bot2 = null;

  let finished = false;
  function finish(exitCode) {
    if (finished) return;
    finished = true;
    result.duration_ms = Date.now() - startMs;
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = exitCode;
    for (const b of [bot1, bot2]) {
      if (!b) continue;
      try {
        b.quit();
      } catch {
        // ignore
      }
    }
  }

  const timeout = setTimeout(() => {
    result.errors.push(`timeout after ${timeoutMs}ms (server not responding?)`);
    finish(2);
  }, timeoutMs);

  try {
    if (!username2) {
      const preCommands = asCmdList(payload.pre_commands);
      const commands = asCmdList(payload.commands);
      const postCommands = asCmdList(payload.post_commands);

      bot1 = mineflayer.createBot({
        host,
        port,
        username,
        auth: 'offline',
        ...(version ? { version } : {}),
      });

      bot1.on('messagestr', (msg) => {
        if (result.chat.length < 200) result.chat.push(String(msg));
      });
      bot1.on('kicked', (reason) => {
        result.errors.push(`kicked: ${String(reason)}`);
        clearTimeout(timeout);
        finish(3);
      });
      bot1.on('error', (err) => {
        result.errors.push(`error: ${err?.message ?? String(err)}`);
        clearTimeout(timeout);
        finish(4);
      });

      bot1.once('spawn', async () => {
        try {
          const spawn = bot1.entity.position;
          const spawnFloored = bot1.entity.position.floored();
          result.spawn = [spawn.x, spawn.y, spawn.z];
          result.spawn_floored = [spawnFloored.x, spawnFloored.y, spawnFloored.z];

          const all = [...preCommands, ...commands, ...postCommands];
          await executeAll(bot1, all, result.executed, stepDelayMs, 1);

          result.scan = await scanRegion(bot1, scan, scanDelayMs);
          result.ok = true;
          clearTimeout(timeout);
          finish(0);
        } catch (e) {
          result.errors.push(`exception: ${e?.stack ?? e?.message ?? String(e)}`);
          clearTimeout(timeout);
          finish(5);
        }
      });
      return;
    }

    const botsPayload = Array.isArray(payload.bots) ? payload.bots : [];
    const p1 = botsPayload.length >= 1 && botsPayload[0] && typeof botsPayload[0] === 'object' ? botsPayload[0] : {};
    const p2 = botsPayload.length >= 2 && botsPayload[1] && typeof botsPayload[1] === 'object' ? botsPayload[1] : {};

    const pre1 = asCmdList(p1.pre_commands);
    const cmd1 = asCmdList(p1.commands);
    const post1 = asCmdList(p1.post_commands);
    const pre2 = asCmdList(p2.pre_commands);
    const cmd2 = asCmdList(p2.commands);
    const post2 = asCmdList(p2.post_commands);

    const botResults = [
      { username, spawn: null, spawn_floored: null, executed: [], chat: [], errors: [] },
      { username: username2, spawn: null, spawn_floored: null, executed: [], chat: [], errors: [] },
    ];
    result.bots = botResults;

    bot1 = mineflayer.createBot({
      host,
      port,
      username,
      auth: 'offline',
      ...(version ? { version } : {}),
    });
    bot2 = mineflayer.createBot({
      host,
      port,
      username: username2,
      auth: 'offline',
      ...(version ? { version } : {}),
    });

    bot1.on('messagestr', (msg) => {
      if (botResults[0].chat.length < 200) botResults[0].chat.push(String(msg));
      if (result.chat.length < 200) result.chat.push(`[${username}] ${String(msg)}`);
    });
    bot2.on('messagestr', (msg) => {
      if (botResults[1].chat.length < 200) botResults[1].chat.push(String(msg));
      if (result.chat.length < 200) result.chat.push(`[${username2}] ${String(msg)}`);
    });
    bot1.on('kicked', (reason) => {
      const msg = `kicked: ${String(reason)}`;
      botResults[0].errors.push(msg);
      result.errors.push(`${username} ${msg}`);
      clearTimeout(timeout);
      finish(3);
    });
    bot2.on('kicked', (reason) => {
      const msg = `kicked: ${String(reason)}`;
      botResults[1].errors.push(msg);
      result.errors.push(`${username2} ${msg}`);
      clearTimeout(timeout);
      finish(3);
    });
    bot1.on('error', (err) => {
      const msg = `error: ${err?.message ?? String(err)}`;
      botResults[0].errors.push(msg);
      result.errors.push(`${username} ${msg}`);
      clearTimeout(timeout);
      finish(4);
    });
    bot2.on('error', (err) => {
      const msg = `error: ${err?.message ?? String(err)}`;
      botResults[1].errors.push(msg);
      result.errors.push(`${username2} ${msg}`);
      clearTimeout(timeout);
      finish(4);
    });

    await Promise.all([
      waitForSpawnOrFail(bot1, botResults[0], result, (code) => {
        clearTimeout(timeout);
        finish(code);
      }),
      waitForSpawnOrFail(bot2, botResults[1], result, (code) => {
        clearTimeout(timeout);
        finish(code);
      }),
    ]);

    // Keep backwards-compatible spawn fields for bot1.
    result.spawn = botResults[0].spawn;
    result.spawn_floored = botResults[0].spawn_floored;

    const all1 = [...pre1, ...cmd1, ...post1];
    const all2 = [...pre2, ...cmd2, ...post2];

    await executeAll(bot1, all1, botResults[0].executed, stepDelayMs, 1);
    await executeAll(bot2, all2, botResults[1].executed, stepDelayMs, 2);
    result.executed = [...botResults[0].executed, ...botResults[1].executed];

    result.scan = await scanRegion(bot1, scan, scanDelayMs);
    result.ok = true;
    clearTimeout(timeout);
    finish(0);
  } catch (e) {
    result.errors.push(`exception: ${e?.stack ?? e?.message ?? String(e)}`);
    clearTimeout(timeout);
    finish(5);
  }
}

main().catch((e) => {
  process.stderr.write(`${e?.stack ?? e?.message ?? String(e)}\n`);
  process.exitCode = 10;
});
