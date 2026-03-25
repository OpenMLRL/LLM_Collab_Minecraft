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
- Symbol meanings in this task: `#` = static land, `S` = anchor U, `T` = anchor V, `Y` = true pillar, `N` = fake pillar, `*` = filled block placed by /fill.
- Goal: discover anchor locations through visibility, then use your /fill blocks plus pillars to make anchor U and anchor V 4-connected while using as few blocks as possible. Static land `#` does not count toward connectivity.
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
      "discovered_pillars": [
        {{"x": 2, "z": 2, "type": "Y", "source": "self_probe"}}
      ],
      "candidate_coords": [[x, z], ...]
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
- Gap improvement reward: `5 * (prev_gap_ST - gap_ST) / max_gap_ST`, where `gap_ST` is the minimum number of additional filled blocks needed to make anchor U and anchor V 4-connected under the current map, `prev_gap_ST` is the same quantity before this turn, and `max_gap_ST` is the same quantity on the initial empty map for this task.
- CC merge reward: `1.5 * new_merge / (initial_cc_components - 1)`, where CC components are the 4-connected components formed by treating anchor U, anchor V, and all true Y pillars as special nodes, with filled blocks `*` allowed to connect them.
- Reward for newly connected Y pillars remains meaningful throughout training. A Y counts as connected when it is 4-connected to anchor U or anchor V through filled blocks `*` and/or other Y pillars.
- Penalty for newly adjacent N pillars is strong and may be amplified further by shaping. Avoid creating any new N adjacency.
- Block placement cost: `(newly_placed_blocks / total_placeable_cells) * 5`.
- Movement shaping bonus: `2.5 * (prev_target_distance_total - target_distance_total) / movement_distance_norm`, where worker A is rewarded for moving along the `S -> T` direction and worker B along the `T -> S` direction.
- Terminal connect reward: `+2` when anchor U and anchor V become 4-connected through filled blocks `*` and/or pillars (`Y`/`N`), not through static land `#`.
- In late turns, a small extra quality bonus may be awarded for cleaner, more compact, closer-to-finish bridge states.

Execution rules:
- Termination condition: turn limit reached or anchor U and anchor V are already connected.

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
