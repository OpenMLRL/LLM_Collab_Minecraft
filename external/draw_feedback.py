from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_MC.utils.str_builder import (
    TaskSpec,
    extract_command_lines,
    normalize_block_id,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


def format_followup_prompts(
    *,
    ctx: Dict[str, Any],
    agent_completions: List[str],
    num_agents: int = 2,
    original_prompt_flag: bool = True,
    previous_response_flag: bool = False,
    prompt_history_per_agent: Optional[List[List[str]]] = None,  # Unused (kept for API parity)
    response_history_per_agent: Optional[List[List[str]]] = None,  # Unused (kept for API parity)
) -> List[str]:
    n = int(num_agents)
    if n <= 0:
        raise ValueError("num_agents must be >= 1")
    if len(agent_completions) != n:
        raise ValueError(f"Expected {n} agent completions, got {len(agent_completions)}")

    turn_number = 1
    if prompt_history_per_agent and prompt_history_per_agent[0] is not None:
        try:
            turn_number = len(prompt_history_per_agent[0]) + 1
        except Exception:
            turn_number = 1

    system_prompt = str(ctx.get("system_prompt") or "").rstrip()
    user_prompt_single = str(ctx.get("user_prompt_single") or "").rstrip()
    user_prompt_agent1 = str(ctx.get("user_prompt_agent1") or user_prompt_single).rstrip()
    user_prompt_agent2 = str(ctx.get("user_prompt_agent2") or user_prompt_single).rstrip()

    task_id = str(ctx.get("task_id") or "")
    text = str(ctx.get("text") or "")
    difficulty = int(ctx.get("difficulty") or 0)
    local_bbox_from = [int(v) for v in (ctx.get("local_bbox_from") or [0, 0, 0])]
    local_bbox_to = [int(v) for v in (ctx.get("local_bbox_to") or [0, 0, 0])]
    target_rows_topdown = [str(r) for r in (ctx.get("target_rows_topdown") or [])]

    allowed_blocks_agent1 = [str(b) for b in (ctx.get("allowed_blocks_agent1") or []) if str(b).strip()]
    allowed_blocks_agent2 = [str(b) for b in (ctx.get("allowed_blocks_agent2") or []) if str(b).strip()]
    max_commands_total = int(ctx.get("max_commands_total") or 600)

    max_per = max(1, max_commands_total // n)
    extra = max_commands_total % n
    max_limits = [max_per] * n
    max_limits[0] += extra

    accepted_all: List[str] = []
    for agent_idx in range(n):
        completion = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
        allowed = allowed_blocks_agent1 if agent_idx == 0 else (allowed_blocks_agent2 or allowed_blocks_agent1)
        lines = extract_command_lines(completion)
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed,
            world_bbox_from=local_bbox_from,
            world_bbox_to=local_bbox_to,
            max_commands=max_limits[agent_idx],
        )
        accepted_all.extend(accepted)

    blocks = simulate_commands_to_scan_blocks(commands=accepted_all, world_bbox_from=local_bbox_from, world_bbox_to=local_bbox_to)
    task = TaskSpec(
        task_id=task_id,
        csv_row_index=0,
        text=text,
        difficulty=difficulty,
        local_bbox_from=local_bbox_from,
        local_bbox_to=local_bbox_to,
        target_rows_topdown=target_rows_topdown,
    )

    height = len(target_rows_topdown)
    width = len(target_rows_topdown[0]) if height else 0

    target_set = set()
    for r, row in enumerate(target_rows_topdown):
        if len(row) != width:
            raise ValueError("target_rows_topdown has inconsistent row widths")
        for x, ch in enumerate(row):
            if ch != "#":
                continue
            wx = local_bbox_from[0] + x
            wy = local_bbox_from[1] + (height - 1 - r)
            wz = local_bbox_from[2]
            target_set.add((int(wx), int(wy), int(wz)))

    def _is_air(name: str | None) -> bool:
        if name is None:
            return True
        b = normalize_block_id(str(name))
        return b in {"air", "cave_air", "void_air"}

    placed_set = set()
    for b in blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        if _is_air(None if name is None else str(name)):
            continue
        placed_set.add((int(pos[0]), int(pos[1]), int(pos[2])))

    to_place = sorted(target_set - placed_set, key=lambda p: (-p[1], p[0], p[2]))
    to_remove = sorted(placed_set - target_set, key=lambda p: (-p[1], p[0], p[2]))

    def _format_positions(points: list[tuple[int, int, int]]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- {x} {y} {z}" for x, y, z in points)

    feedback = "\n".join(
        [
            "Feedback (coordinate edits):",
            f"- Turn: {turn_number}",
            "- Coordinates are absolute (x y z).",
            "- Deletion is allowed: use /setblock ... air or /fill ... air to remove blocks.",
            "- Do not output ASCII; output commands only.",
            "",
            "Place a block (any allowed type) at:",
            _format_positions(to_place),
            "",
            "Delete (place air) at:",
            _format_positions(to_remove),
        ]
    ).rstrip()

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        parts = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")
        parts.append(feedback)
        if original_prompt_flag and base_user:
            parts.append("")
            parts.append(base_user)
        if previous_response_flag:
            # Optional: include previous commands verbatim (off by default).
            prev = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
            if prev.strip():
                parts.append("")
                parts.append("Your previous commands:")
                parts.append(prev.strip())
        prompts[agent_idx] = "\n".join(parts).strip()

    return prompts
