# LLM_Collab_Minecraft

## Smoke test (mineflayer executor bot)

Prereqs: Node.js >= 18, Python 3.

0) (Optional) Start a local Paper server (1.19.2 example)

In a terminal:

`mkdir -p ~/mc-server && cd ~/mc-server`

`curl -fsSL https://api.papermc.io/v2/projects/paper/versions/1.19.2/builds/307/downloads/paper-1.19.2-307.jar -o paper-1.19.2-307.jar`

`echo "eula=true" > eula.txt`

Start the server:

`cd ~/mc-server && java -Xms1G -Xmx1G -jar paper-1.19.2-307.jar --nogui`

1) OP the bot username

Important: `op executor_bot` is a Minecraft server/admin command (not a bash command).

- If you have access to the Minecraft server console (the terminal where the server is running), type:

`op executor_bot`

- If you are typing inside Minecraft in-game chat (as an admin), use:

`/op executor_bot`

2) In another terminal, enter this repo and install node deps (once):

`cd LLM_Collab_Minecraft && npm install`

3) Run the smoke test from `LLM_Collab_Minecraft/`:

`python3 test/test_env.py --host 127.0.0.1 --port 25565 --username executor_bot`

## Baselines (one-shot commands + optional MC eval)

- Edit config: `baselines/2d_painting/config.yaml`
- Edit config: `baselines/str_builder/config.yaml`
- If `minecraft.enabled=true`, keep a Minecraft server running and OP the bot username (see smoke test steps above).
- Run locally (writes `.jsonl` by default): `python3 baselines/2d_painting/main.py --config baselines/2d_painting/config.yaml`
- Run locally (writes `.jsonl` by default): `python3 baselines/str_builder/main.py --config baselines/str_builder/config.yaml`
- Slurm: copy `baselines/2d_painting/run.example.sh` to `baselines/2d_painting/run.sh` (ignored by git), then `bash baselines/2d_painting/run.sh baselines/2d_painting/config.yaml`
- Slurm: copy `baselines/str_builder/run.example.sh` to `baselines/str_builder/run.sh` (ignored by git), then `bash baselines/str_builder/run.sh baselines/str_builder/config.yaml`
- If you need `minecraft.enabled=true` on Slurm: use `baselines/2d_painting/run_sbatch.example.sh` (see `baselines/2d_painting/README.md`)
- If you need `minecraft.enabled=true` on Slurm: use `baselines/str_builder/run_sbatch.example.sh` (see `baselines/str_builder/README.md`)

## Train (str_builder, GRPO)

- Config: `str_builder/configs/str_builder_config.yaml`
- Local (requires GPU + `comlrl` env): `python3 str_builder/train/train.py --config str_builder/configs/str_builder_config.yaml`
- Slurm: copy `str_builder/scripts/train_str_builder_grpo_sbatch.example.sh` to `str_builder/scripts/train_str_builder_grpo_sbatch.sh` (ignored by git), then run it.
- Multi-turn: set `trainer.num_turns > 1` (uses `external.mode=draw_feedback` for ASCII target/progress feedback).

## Train (str_painter, GRPO)

- Config: `str_painter/configs/str_painter_config.yaml`
- Local (requires GPU + `comlrl` env): `python3 str_painter/train/train.py --config str_painter/configs/str_painter_config.yaml`
- Slurm: copy `str_painter/scripts/train_str_painter_grpo_sbatch.example.sh` to `str_painter/scripts/train_str_painter_grpo_sbatch.sh` (ignored by git), then run it.
- Multi-turn: set `trainer.num_turns > 1` (uses `external.mode=draw_feedback` for coordinate feedback).
