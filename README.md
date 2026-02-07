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

## Train (StrBuild)

Configs: `str_build/configs/str_build_magrpo_config.yaml`, `str_build/configs/str_build_iac_config.yaml`, `str_build/configs/str_build_maac_config.yaml`.

Local (requires GPU + `comlrl` env):
1. `python3 str_build/train/train_magrpo.py --config str_build/configs/str_build_magrpo_config.yaml`
2. `python3 str_build/train/train_iac.py --config str_build/configs/str_build_iac_config.yaml`
3. `python3 str_build/train/train_maac.py --config str_build/configs/str_build_maac_config.yaml`

Multi-turn: set `magrpo.num_turns` / `iac.num_turns` / `maac.num_turns` > 1 and choose `external.mode` from `perfect_feedback`, `position_feedback`, or `score_feedback` (see `str_build/external/__init__.py`).

## Train (HouseBuild)

Configs: `house_build/configs/house_build_magrpo_config.yaml`, `house_build/configs/house_build_iac_config.yaml`, `house_build/configs/house_build_maac_config.yaml`.

Local (requires GPU + `comlrl` env):
1. `python3 house_build/train/train_magrpo.py --config house_build/configs/house_build_magrpo_config.yaml`
2. `python3 house_build/train/train_iac.py --config house_build/configs/house_build_iac_config.yaml`
3. `python3 house_build/train/train_maac.py --config house_build/configs/house_build_maac_config.yaml`

Multi-turn: set `magrpo.num_turns` / `iac.num_turns` / `maac.num_turns` > 1 and choose `external.mode` from `perfect_feedback`, `position_feedback`, `position_modification`, `rect_modification`, `resource_schedule`, or `score_feedback` (see `house_build/external/__init__.py`).
