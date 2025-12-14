# LLM_Collab_MC

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

`cd LLM_Collab_MC && npm install`

3) Run the smoke test from `LLM_Collab_MC/`:

`python3 test/test_env.py --host 127.0.0.1 --port 25565 --username executor_bot`

## Baselines (offline inference)

- Edit config: `baselines/config.yaml`
- Run locally (writes a `.jsonl`): `python3 baselines/main.py --config baselines/config.yaml`
- Slurm: copy `baselines/run.example.sh` to `baselines/run.sh` (ignored by git), then `bash baselines/run.sh baselines/config.yaml`
