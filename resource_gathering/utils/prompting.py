from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft resource-gathering agent in a 2-agent collaboration task. "
    "Return exactly one JSON object and then stop. "
    "Do not output markdown, reasoning, code fences, or any extra text."
)

DEFAULT_USER_TEMPLATE = """Output contract:
- Reply with exactly one JSON object and nothing else.
- If you do not want to broadcast, set "comm" to {{}}.
- Set "probe" to [].
- Set "cmds" to [].
- If you do not want to move, set "path" to [{current_pos}].
- Minimal valid no-op example:
  {{"comm":{{}},"probe":[],"cmds":[],"path":[{current_pos}]}}

Task setup:
- You are worker {agent_name} in a 2-agent Minecraft resource gathering task.
- Worker A specializes in wood. Worker B specializes in stone and iron.
- World top-left coordinate is {origin}, map size is {map_width}x{map_height}.
- Current turn: {turn_idx}/{max_turns}.
- Goal totals: wood={goal_wood}, stone={goal_stone}, iron={goal_iron}.
- Team inventory so far: wood={collected_wood}, stone={collected_stone}, iron={collected_iron}.
- Remaining goals: wood={remaining_wood}, stone={remaining_stone}, iron={remaining_iron}.

Mechanics:
- View radius: {view}.
- After movement, you automatically harvest all compatible resources within Manhattan distance <= {extraction_range}.
- You do NOT choose extraction cells manually in this version.
- A valid path must start at your current position and move one 8-connected step at a time.
- Maximum suggested path length this turn: {max_path_len} moves.

Your current state:
- Current position: {current_pos}
- Visible resources: {visible_resources}
- Teammate position if visible: {visible_teammate_pos}
- Received teammate messages: {received_messages}
- Your reachable auto-harvest zone this turn if you stay: {harvest_zone}

Action format:
{{
  "comm": {{
    "resource_facts": [
      {{"x": 1, "z": 2, "type": "wood", "count": 3}}
    ]
  }},
  "probe": [],
  "cmds": [],
  "path": [[x,z], ...]
}}

Action guidance:
- `comm`: broadcast structured resource facts visible to you right now. Use {{}} for no message.
- `probe`: always [] in this task.
- `cmds`: always [] in this task.
- `path`: first point must equal current position; use [{current_pos}] for no movement.
- Prefer telling your teammate about resources they can harvest, not the ones you harvest yourself.

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
