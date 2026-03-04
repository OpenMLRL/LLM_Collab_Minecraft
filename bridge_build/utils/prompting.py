from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft bridge-building agent in a 2-agent collaboration task. "
    "You must output exactly one JSON object, with no markdown, code fences, or extra text."
)

DEFAULT_USER_TEMPLATE = """Task setup:
- You are worker {agent_name}, collaborating with your teammate to build a bridge in a 2D world (default y=0).
- The world top-left coordinate is {origin}, and map size is {map_width}x{map_height}.
- Current turn: {turn_idx}/{max_turns}.
- Goal: use your /fill blocks plus pillars to make S and T 4-connected while using as few blocks as possible.
- Pillar candidates include true pillars (Y) and fake pillars (N); pillar type is unknown until probed.
- Even if /fill covers a pillar coordinate, the pillar itself remains unchanged.

Global anchor coordinates:
- S coordinates: {s_coords}
- T coordinates: {t_coords}

Your current state:
- Current position: {current_pos}
- View radius: {view} (visible area is the union of all (2*view+1)x(2*view+1) windows centered on points in your path history).
- Currently visible land coordinates: {visible_land_coords}
- Currently visible pillar candidate set P (visible subset of N union Y): {visible_p_candidates}
- Your known probe results: {known_probe_results}
- Teammate broadcast messages you received via comm: {received_messages}

Available blocks (you may only use these):
{available_blocks}

Action format and budgets:
- You must output strict JSON:
  {{
    "comm": {{ ... }},
    "probe": [[x,z], ...],
    "cmds": ["/fill x1 z1 x2 z2 block", ...],
    "path": [[x,z], ...]
  }}
- `comm`: broadcast to teammate; incurs token cost.
- `probe`: up to {max_probe} coordinates this turn; used to identify pillar type.
- `cmds`: Minecraft /fill commands (without y-axis), up to {max_commands} commands this turn.
- `path`: movement path from current position; first point must equal current position; consecutive points must be 8-connected; invalid path causes no movement.

Reward/penalty rules (resolved each turn):
- Any N that is 4-neighbor adjacent to filled blocks: -1 (counted once per N).
- Any Y that is NOT 4-neighbor adjacent to filled blocks: -2 (counted once per Y).
- If S and T are not 4-connected: -5.
- Each probe: -0.3.
- Each token in comm: -0.001.

Execution order:
- Execute A's cmds first, then B's cmds.
- Termination condition: turn limit reached or S/T already connected.

Feedback:
{feedback}
"""

DEFAULT_USER_TEMPLATE_AGENT1 = DEFAULT_USER_TEMPLATE
DEFAULT_USER_TEMPLATE_AGENT2 = DEFAULT_USER_TEMPLATE

DEFAULT_PROMPT_CONFIG = {
    "use_chat_template": False,
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
