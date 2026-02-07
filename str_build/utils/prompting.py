from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft building agent.\n"
    "Output must be Minecraft commands only (no markdown, no code fences, no extra text).\n"
    "This is strict: any non-command text is invalid."
)

DEFAULT_USER_TEMPLATE = """The target is a character grid made of '#' and '.'. Its size matches the bbox and target rows.
Output /setblock commands for the '#' positions.

Target grid (top-down rows):
{target_ascii}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Coordinate mapping (absolute coords):
- Let (min_x, min_y, min_z) be bbox.from and (max_x, max_y, max_z) be bbox.to.
- x increases from min_x to max_x (left to right).
- y increases from min_y to max_y (bottom to top).
- z is constant (min_z == max_z).

Available blocks (use ONLY these):
{block_agent1_lines}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /setblock only.
- Use absolute integer coordinates only (no ~).
- Place blocks ONLY at '#' positions; leave '.' as air.
- Every coordinate must be within the bbox.
Format: /setblock <x> <y> <z> <block>
"""

DEFAULT_USER_TEMPLATE_AGENT1 = """You are Agent 1 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.

The target is a character grid made of '#' and '.'. Its size matches the bbox and target rows.
Output /setblock commands for a subset of '#' positions.

Target grid (top-down rows):
{target_ascii}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Coordinate mapping (absolute coords):
- Let (min_x, min_y, min_z) be bbox.from and (max_x, max_y, max_z) be bbox.to.
- x increases from min_x to max_x (left to right).
- y increases from min_y to max_y (bottom to top).
- z is constant (min_z == max_z).

Available blocks (use ONLY these):
{block_agent1_lines}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /setblock only.
- Use absolute integer coordinates only (no ~).
- Place blocks ONLY at '#' positions; leave '.' as air.
- Adjacent blocks (sharing a side) must NOT be the same texture.
- Every coordinate must be within the bbox.
Format: /setblock <x> <y> <z> <block>
"""

DEFAULT_USER_TEMPLATE_AGENT2 = """You are Agent 2 in a 2-person Minecraft building team. You will place SOME of the blocks for the final build.
You write after Agent 1, so your /setblock commands overwrite Agent 1's results.
Choose a reasonable subset of '#' positions to write.

The target is a character grid made of '#' and '.'. Its size matches the bbox and target rows.
Output /setblock commands for a subset of '#' positions.

Target grid (top-down rows):
{target_ascii}

WORLD bbox (inclusive):
- from: {world_bbox_from}
- to:   {world_bbox_to}

Coordinate mapping (absolute coords):
- Let (min_x, min_y, min_z) be bbox.from and (max_x, max_y, max_z) be bbox.to.
- x increases from min_x to max_x (left to right).
- y increases from min_y to max_y (bottom to top).
- z is constant (min_z == max_z).

Available blocks (use ONLY these):
{block_agent2_lines}

Constraints:
- Output ONLY Minecraft commands, one per line.
- Allowed commands: /setblock only.
- Use absolute integer coordinates only (no ~).
- Place blocks ONLY at '#' positions; leave '.' as air.
- Adjacent blocks (sharing a side) must NOT be the same texture.
- Every coordinate must be within the bbox.
Format: /setblock <x> <y> <z> <block>
"""

DEFAULT_PROMPT_CONFIG = {
    "provide_graph": True,
    "use_chat_template": False,
    "system": DEFAULT_SYSTEM_PROMPT,
    "user_template": DEFAULT_USER_TEMPLATE,
    "user_template_agent1": DEFAULT_USER_TEMPLATE_AGENT1,
    "user_template_agent2": DEFAULT_USER_TEMPLATE_AGENT2,
}


def apply_graph_setting(template: str, *, provide_graph: bool) -> str:
    if provide_graph:
        return template
    lines = (template or "").splitlines()
    out = []
    drop_substrings = ("#'", "'.'")
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            if "{target_ascii}" in line:
                continue
            out.append(line)
            continue
        if any(sub in line for sub in drop_substrings):
            continue
        if line.strip() == "Target grid (top-down rows):":
            skip_next = True
            continue
        out.append(line)
    return "\n".join(out).rstrip()


def apply_prompt_defaults(cfg: Dict[str, Any]) -> None:
    prompt_cfg = cfg.get("prompt")
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
        cfg["prompt"] = prompt_cfg
    for key, value in DEFAULT_PROMPT_CONFIG.items():
        if key not in prompt_cfg:
            prompt_cfg[key] = value
