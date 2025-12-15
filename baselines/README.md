# Baselines

`baselines/main.py` reads tasks from `../dataset`, prompts an LLM once per task to output **Minecraft command lines** (one command per line), then scores the final build result and writes an eval JSON/JSONL file.

## Prereqs

- Python env with: `pyyaml`, `torch`, `transformers`, `accelerate`
- Node.js >= 18 + repo node deps: run once from repo root: `cd LLM_Collab_MC && npm install`

Notes:

- Default (`minecraft.enabled=false`) uses an **offline simulator** for `/fill` and `/setblock` to compute `correct/total` (no server needed).
- If you enable real Minecraft execution (`minecraft.enabled=true`), you also need:
  - A Minecraft server is running and reachable (`minecraft.host`/`minecraft.port`)
  - The bot username is OP (in server console: `op executor_bot`)
  - Server allows offline players (since the bot uses offline auth): set `online-mode=false` in `server.properties`

## Run locally

From repo root:

- Dry-run (no model load, prints first prompts):
  - `python3 baselines/main.py --config baselines/config.yaml --dry-run --limit 1`
- Real run:
  - `python3 baselines/main.py --config baselines/config.yaml`

Useful flags:

- `--limit N`: only run first N tasks
- `--dry-run`: don’t load model, just show prompts

## Output

`output.path` in `baselines/config.yaml` controls the output format:

- If it ends with `.jsonl`: JSON Lines (one record per line)
- Otherwise: a single JSON array file

Each record includes at least:

- `commands`: the validated command sequence (what will be executed)
- `rejected_commands`: lines rejected by the validator (unsupported command / out-of-bbox / not-in-palette, etc.)
- `correct`, `total`, `correct_over_total`: accuracy vs the target voxel grid (offline simulation)
- If `minecraft.enabled=true` and the server is reachable, it also adds `mc_correct/mc_total` in `metrics`.

## Optional: run with a real Minecraft server

This mode will actually connect to Minecraft and execute the command sequence via mineflayer before scoring.

1) Start a Paper server (1.19.2 example)

In a terminal:

- `mkdir -p ~/mc-server && cd ~/mc-server`
- `curl -fsSL https://api.papermc.io/v2/projects/paper/versions/1.19.2/builds/307/downloads/paper-1.19.2-307.jar -o paper-1.19.2-307.jar`
- `echo "eula=true" > eula.txt`

First start (generates configs):

- `cd ~/mc-server && java -Xms1G -Xmx1G -jar paper-1.19.2-307.jar --nogui`

Then edit `~/mc-server/server.properties` and ensure:

- `online-mode=false` (required for the offline-auth bot)
- `gamemode=creative` (recommended)

Restart the server after editing.

2) OP the bot username

In the **server console** (the terminal running the server):

- `op executor_bot`

Or in **in-game chat** (as an admin):

- `/op executor_bot`

3) Enable Minecraft execution in `baselines/config.yaml`

Set:

- `minecraft.enabled: true`
- `minecraft.host: 127.0.0.1`
- `minecraft.port: 25565`
- `minecraft.username: executor_bot`

4) Run

- `python3 baselines/main.py --config baselines/config.yaml --limit 1`

## Slurm

### Offline eval (no Minecraft server)

Copy the template and submit:

- `cp baselines/run.example.sh baselines/run.sh`
- `ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./baselines/run.sh baselines/config.yaml`

Notes:

- Recommended on Slurm: keep `minecraft.enabled=false` (no MC server needed; eval uses offline simulation).
- If `minecraft.enabled=true`, the Minecraft server must be reachable from the Slurm compute node (so `127.0.0.1` only works if the server is started on the same node).

### Minecraft execution (server + baseline in the same job, recommended)

This avoids all “server reachable from compute nodes” issues by starting the Paper server inside the same Slurm job, so `minecraft.host=127.0.0.1` works.

1) One-time setup (on a shared filesystem, e.g. `$HOME`)

- `mkdir -p ~/mc-server && cd ~/mc-server`
- `curl -fsSL https://api.papermc.io/v2/projects/paper/versions/1.19.2/builds/307/downloads/paper-1.19.2-307.jar -o paper-1.19.2-307.jar`
- `echo "eula=true" > eula.txt`
- (Recommended) start once to generate configs: `java -Xms1G -Xmx1G -jar paper-1.19.2-307.jar --nogui`
- Ensure `online-mode=false` in `~/mc-server/server.properties` (required for the offline-auth bot)

2) Copy the template (this keeps your Slurm settings out of git)

- `cp baselines/run_sbatch.example.sh baselines/run_sbatch.sh`
- `chmod +x baselines/run_sbatch.sh`

3) Submit (examples)

- Minimal: `ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION EXTRA_ARGS="--limit 1" ./baselines/run_sbatch.sh baselines/config.yaml`
- If your jar lives elsewhere: `MC_DIR=/path/to/mc-server MC_JAR=paper-1.19.2-307.jar ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./baselines/run_sbatch.sh baselines/config.yaml`

Where to look:

- Slurm stdout: `baselines/slurm_logs/%x-%j.out`
- Minecraft server log: `${MC_DIR}/logs/server_${SLURM_JOB_ID}.log` (printed in Slurm stdout)
- Eval output: `baselines/outputs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.{jsonl|json}` (printed in Slurm stdout)

This script copies your base config to `$SLURM_TMPDIR` and only overwrites:

- `dataset.root` → absolute `${REPO_DIR}/dataset`
- `output.path` → absolute `baselines/outputs/...`
- `output.overwrite` → `true`
- `minecraft.enabled/host/port/username` → force-enable MC execution on localhost

So you can still edit `generation.max_new_tokens`, sampling params, prompts, etc. in `baselines/config.yaml` normally.

Output format:

- By default, `run_sbatch.sh` follows your base config `output.path` extension:
  - ends with `.jsonl` → JSONL (one record per line, easiest for large runs)
  - otherwise → JSON array file
- You can also force it with: `OUTPUT_FORMAT=jsonl` or `OUTPUT_FORMAT=json`

### Make a Minecraft server reachable from compute nodes (alternative)

Key point: in a Slurm job, `127.0.0.1` refers to the **compute node running the job**, not the login node.

Recommended (simplest): run the server and the baseline on the **same compute node**.

1) Get an interactive allocation:

- `salloc --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=64 --mem=100g --time=02:00:00`
- `srun --pty bash`

2) On that compute node, start the server:

- `cd ~/mc-server && java -Xms1G -Xmx1G -jar paper-1.19.2-307.jar --nogui`
- In the server console: `op executor_bot`

3) In a second shell on the **same allocation** (use `tmux`, or run another `srun --pty bash`), run the baseline with:

- `minecraft.enabled: true`
- `minecraft.host: 127.0.0.1`

Alternative: server runs on a different machine (login node / another server).

- Set `minecraft.host` to that machine’s hostname/IP that compute nodes can reach (NOT `127.0.0.1`).
- Ensure the server listens on all interfaces: keep `server.properties` with `server-ip=` empty (or `0.0.0.0`).
- Ensure firewall/policy allows TCP to `minecraft.port` (default 25565). Many clusters block random ports on login nodes; if so, use the “same compute node” approach above.
