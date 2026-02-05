# View Building Demo

This demo connects a mineflayer bot to a local Minecraft server and builds a small house with furniture.
It also starts a browser viewer (prismarine-viewer) so you can watch the result.

## Prereqs

- A running Minecraft server (Paper 1.19.2 works). Offline mode must be enabled.
- The bot username must be OP (example: `op executor_bot`).
- Node.js >= 18 and repo deps installed.
- Superflat world: set `level-type=flat` and `level-name=view_building_demo_flat` in `~/mc-server/server.properties`, then restart the server.
- Python deps for HF inference: `transformers` + `torch` (CPU-only torch is OK).

## Install deps

From repo root:

```bash
cd LLM_Collab_Minecraft
npm install
```

## Run demo

From repo root:

```bash
node test/view_building_demo/build_house_demo.cjs \
  --host 127.0.0.1 \
  --port 25565 \
  --username executor_bot \
  --viewer-port 3000
```

## Slurm (GPU) run

If CPU inference is too slow, run on a GPU node with an interactive allocation (same idea as `baselines/2d_painting`):

```bash
salloc --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=16 --mem=64g --time=02:00:00
srun --pty bash
```

On that compute node, start the Minecraft server:

```bash
cd ~/mc-server && java -Xms1G -Xmx1G -jar paper-1.19.2-307.jar --nogui
```

In a second shell on the same allocation (use `tmux`, or another `srun --pty bash`), run the demo:

```bash
cd ~/LLM_Collab_Minecraft
node test/view_building_demo/build_house_demo.cjs --llm-device cuda --viewer-port 3000
```

Note: in Slurm jobs, `127.0.0.1` points to the compute node, so keep server + demo on the same node.

HF Qwen3 (default) options:

```bash
node test/view_building_demo/build_house_demo.cjs --llm-model Qwen/Qwen3-8B
```

Force CPU (slower but avoids CUDA libs):

```bash
node test/view_building_demo/build_house_demo.cjs --llm-device cpu
```

OpenAI-compatible API (optional):

```bash
export LLM_API_KEY=...
export LLM_MODEL=gpt-4o-mini
export LLM_BASE_URL=https://api.openai.com/v1

node test/view_building_demo/build_house_demo.cjs --llm-mode openai
```

Open the viewer in a browser:

- http://127.0.0.1:3000/

## Notes

- The "LLM" prompt is in `test/view_building_demo/llm_prompt.txt`.
- Default LLM backend is `hf` (Hugging Face). Use `--llm-mode openai` for API mode.
- Use `--llm-mode stub` to run the deterministic offline stub.
- For local OpenAI-compatible servers without a key, pass `--llm-allow-no-key true` or set `LLM_ALLOW_NO_KEY=true`.
- The demo writes the last prompt + command list to `test/view_building_demo/llm_output.txt`.
- Use `--keep-alive false` if you want the process to exit after building.
