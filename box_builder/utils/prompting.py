from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft building agent.\n"
    "Output must be Minecraft commands only (no markdown, no code fences, no extra text).\n"
    "This is strict: any non-command text is invalid."
)

DEFAULT_USER_TEMPLATE = """Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent1_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, ABS_Y, z1, x2, ABS_Y, z2 block_id), (x1, ABS_Y, z1, x2, ABS_Y, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Threat check: There are {spider_num} spiders nearby; you have {player_hp} HP. Spider attack list: {spider_atk}, total damage dmg={spider_dmg}. If you want to kill them, output exactly one line: /kill. All other commands should be normal building commands.

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill and /kill
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_USER_TEMPLATE_AGENT1 = """You are Agent 1 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.

Task: Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent1_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, ABS_Y, z1, x2, ABS_Y, z2 block_id), (x1, ABS_Y, z1, x2, ABS_Y, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Threat check: There are {spider_num} spiders nearby; you have {player_hp} HP. Spider attack list: {spider_atk}, total damage dmg={spider_dmg}. If you want to kill them, output exactly one line: /kill. All other commands should be normal building commands.

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill and /kill
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_USER_TEMPLATE_AGENT2 = """You are Agent 2 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.

Task: Build the 3D structure from the provided y-axis slices.

Available blocks (use ONLY these):
{block_agent2_lines}

Layers (ascending WORLD y). Each layer is a set of rectangles in WORLD (x,z) coords:
- Format: y=ABS_Y: {{(x1, ABS_Y, z1, x2, ABS_Y, z2 block_id), (x1, ABS_Y, z1, x2, ABS_Y, z2 block_id)}}
{layers_text}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Threat check: There are {spider_num} spiders nearby; you have {player_hp} HP. Spider attack list: {spider_atk}, total damage dmg={spider_dmg}. If you want to kill them, output exactly one line: /kill. All other commands should be normal building commands.

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /fill and /kill
- Fill format: /fill x1 y1 z1 x2 y2 z2 block
- Use absolute integer coordinates only (no ~).
- Use ONLY blocks from the legend.
- Every coordinate must be within the bbox.
"""

DEFAULT_PROMPT_CONFIG = {
    "use_chat_template": False,
    "include_air_rects": False,
    "system": DEFAULT_SYSTEM_PROMPT,
    "user_template": DEFAULT_USER_TEMPLATE,
    "user_template_agent1": DEFAULT_USER_TEMPLATE_AGENT1,
    "user_template_agent2": DEFAULT_USER_TEMPLATE_AGENT2,
}


def apply_prompt_defaults(cfg: Dict[str, Any]) -> None:
    prompt_cfg = cfg.get("prompt")
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
        cfg["prompt"] = prompt_cfg
    for key, value in DEFAULT_PROMPT_CONFIG.items():
        if key not in prompt_cfg:
            prompt_cfg[key] = value
