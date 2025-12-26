# Box Builder Baseline

`baselines/box_builder/main.py` reads tasks from `../../dataset/box_builder/one.json`, prompts an LLM once per task to output **Minecraft command lines** (one command per line), then scores the final build result and writes an eval JSON/JSONL file.

The task: build a 3D structure from y-axis slices. Each layer is a grid of characters (x columns, z rows) with a legend that maps chars to block IDs.

## Prereqs

- Python env with: `pyyaml`, `torch`, `transformers`, `accelerate`
- Node.js >= 18 + repo node deps: run once from repo root: `cd LLM_Collab_MC && npm install`

Notes:

- Default (`minecraft.enabled=false`) uses an **offline simulator** for `/fill` and `/setblock` to compute metrics (no server needed).
- If you enable real Minecraft execution (`minecraft.enabled=true`), you also need a running server and the bot username must be OP (`op executor_bot`).

## Run locally

From repo root:

- Dry-run (no model load, prints first prompts):
  - `python3 baselines/box_builder/main.py --config baselines/box_builder/config.yaml --dry-run --limit 1`
- Real run:
  - `python3 baselines/box_builder/main.py --config baselines/box_builder/config.yaml`

Useful flags:

- `--limit N`: only run first N tasks
- `--dry-run`: don’t load model, just show prompts

## Output

This baseline writes **two** JSONL files per run:

- Main output: `output.path` from `baselines/box_builder/config.yaml`
- Simplified output: `output.simple_path` (optional); defaults to `{output.path.stem}.simple.jsonl`

The simplified JSONL keeps only: `task_id`, `model_id`, and `score_match`.

## Multi-agent (num_agents=2)

Set `agents.num_agents: 2` in `baselines/box_builder/config.yaml` to run two agents per task:

- Agent1 can only use `task.block_agent1` (list of allowed blocks).
- Agent2 can only use `task.block_agent2` (list of allowed blocks).
- Their validated commands are merged and executed in the same world, then scored with the same metrics.

## Metrics (offline + optional MC scan)

Each record writes:

- `metrics.match_correct`
- `metrics.match_total`
- `metrics.score_match`: `match_correct / match_total`

If Minecraft execution is enabled and scan succeeds, the same metrics are also written with `mc_` prefixes.
