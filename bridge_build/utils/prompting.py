from __future__ import annotations

from typing import Any, Dict


DEFAULT_SYSTEM_PROMPT = (
    "You are a Minecraft bridge-building agent in a 2-agent collaboration task. "
    "Return exactly one JSON object and then stop. "
    "Do not repeat or quote the task, state, feedback, or your reasoning. "
    "Do not output markdown, code fences, headings, analysis, or any text before or after the JSON object. "
    "Any token outside the JSON object is invalid."
)

DEFAULT_USER_TEMPLATE = """Output contract:
- Reply with exactly one JSON object and nothing else.
- Do NOT restate the task, current state, feedback, or reasoning.
- Do NOT write labels like Analysis, Plan, Action, or Final action.
- If you do not want to broadcast, set "comm" to {{}}.
- If you do not want to probe, set "probe" to [].
- If you do not want to issue fill commands, set "cmds" to [].
- If you do not want to move, set "path" to [{current_pos}].
- Minimal valid no-op example:
  {{"comm":{{}},"probe":[],"cmds":[],"path":[{current_pos}]}}

Task setup:
- You are worker {agent_name}, collaborating with your teammate to build a bridge in a 2D world (default y=0).
- The world top-left coordinate is {origin}, and map size is {map_width}x{map_height}.
- Current turn: {turn_idx}/{max_turns}.
- Symbol meanings in this task: `#` = static land, `S` = start anchor, `T` = target anchor, `Y` = true pillar, `N` = fake pillar, `*` = filled block placed by /fill.
- Goal: discover anchor locations through visibility, then use your /fill blocks plus pillars to make S and T 4-connected while using as few blocks as possible. Static land `#` does not count toward connectivity.
- Pillar candidates include true pillars (Y) and fake pillars (N); pillar type is unknown until probed.
- Even if /fill covers a pillar coordinate, the pillar itself remains unchanged.
- Land/anchor cells (`#`, `S`, `T`) are static terrain and cannot be overwritten by /fill.

Your current state:
- Current position: {current_pos}
- View radius: {view} (visible area is the union of all (2*view+1)x(2*view+1) windows centered on points in your path history).
- Currently visible anchors (each item includes `"kind": "S"` or `"kind": "T"`): {visible_anchors}
- Currently visible land coordinates: {visible_land_coords}
- Currently visible filled block coordinates (`*`): {visible_filled_coords}
- Currently visible pillar candidate set P (visible subset of N union Y): {visible_p_candidates}
- Your known probe results: {known_probe_results}
- Teammate broadcast messages you received via comm (JSON objects): {received_messages}

Available blocks (you may only use these):
{available_blocks}

Action format and budgets:
- You must output strict JSON:
  {{
    "comm": {{
      "contour_summary": {{
        "true_pillar_pattern": "short regional summary or empty string",
        "fake_pillar_pattern": "short regional summary or empty string"
      }},
      "discovered_pillars": [
        {{"x": 2, "z": 2, "type": "Y", "source": "self_probe"}}
      ]
    }},
    "probe": [[x,z], ...],
    "cmds": ["/fill x1 z1 x2 z2 block", ...],
    "path": [[x,z], ...]
  }}
- `comm`: JSON object broadcast to teammate. Use `{{}}` for no message. Prefer structured facts over free-form text.
- `probe`: up to {max_probe} coordinates this turn; used to identify pillar type.
- `cmds`: Minecraft /fill commands (without y-axis), up to {max_commands} commands this turn.
- `path`: movement path from current position; first point must equal current position; consecutive points must be 8-connected; invalid path causes no movement.
  Example: if current position is `[2,3]`, then `[[2,3],[2,4],[3,5]]` is valid, but `[[2,3],[1,1],[3,1]]` is invalid because it jumps.

Reward/penalty rules (resolved each turn):
- Gap closeness reward: `5 * (1 - gap_ST / max_gap_ST)`, where `gap_ST` is the minimum number of additional filled blocks needed to make S and T 4-connected under the current map, and `max_gap_ST` is the same quantity on the initial empty map for this task.
- Reward for newly connected Y pillars: `(new_connected_Y / total_Y) * 5`, where a Y counts as connected when it is 4-connected to S or T through filled blocks `*` and/or other Y pillars.
- Penalty for newly adjacent N pillars: `(new_adjacent_N / total_N) * 8`, where an N is adjacent if any filled block `*` is 4-neighbor adjacent to it.
- Block placement cost: `(newly_placed_blocks / total_placeable_cells) * 5`.
- Terminal connect reward: `+10` when S and T become 4-connected through filled blocks `*` and/or pillars (`Y`/`N`), not through static land `#`.

Execution rules:
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
