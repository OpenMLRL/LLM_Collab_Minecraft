from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_MC.str_painter.utils.str_painter import (
    TaskSpec,
    blocks_to_map,
    extract_command_lines,
    get_background_coords,
    get_letter_coords,
    normalize_block_id,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


def _sort_coords(coords: List[tuple[int, int, int]]) -> List[tuple[int, int, int]]:
    return sorted(coords, key=lambda p: (-p[1], p[0], p[2]))


def _format_positions(points: List[tuple[int, int, int]]) -> str:
    if not points:
        return "- (none)"
    return "\n".join(f"- {x} {y} {z}" for x, y, z in points)


def format_followup_prompts(
    *,
    ctx: Dict[str, Any],
    agent_completions: List[str],
    num_agents: int = 2,
    original_prompt_flag: bool = True,
    previous_response_flag: bool = False,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
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

    expected_letter_block = normalize_block_id(allowed_blocks_agent1[0] if allowed_blocks_agent1 else "black_concrete")
    expected_bg_block = None
    if n >= 2:
        expected_bg_block = normalize_block_id(allowed_blocks_agent2[0] if allowed_blocks_agent2 else "white_concrete")

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
    obs_map = blocks_to_map(blocks)

    task = TaskSpec(
        task_id=task_id,
        csv_row_index=0,
        text=text,
        difficulty=difficulty,
        local_bbox_from=local_bbox_from,
        local_bbox_to=local_bbox_to,
        target_rows_topdown=target_rows_topdown,
    )

    letter_coords = set(get_letter_coords(task))
    background_coords = set(get_background_coords(task))

    missing_letters = _sort_coords([p for p in letter_coords if normalize_block_id(obs_map.get(p, "air")) != expected_letter_block])
    extra_black = _sort_coords([p for p in background_coords if normalize_block_id(obs_map.get(p, "air")) == expected_letter_block])

    missing_bg: List[tuple[int, int, int]] = []
    if expected_bg_block is not None:
        missing_bg = _sort_coords([p for p in background_coords if normalize_block_id(obs_map.get(p, "air")) != expected_bg_block])

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        parts: List[str] = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")

        if agent_idx == 0:
            feedback = "\n".join(
                [
                    "Feedback (coordinate edits):",
                    f"- Turn: {turn_number}",
                    "- Coordinates are absolute (x y z).",
                    "- Use /setblock or /fill.",
                    f"- Place {expected_letter_block} at:",
                    _format_positions(missing_letters),
                    f"- Remove {expected_letter_block} from:",
                    _format_positions(extra_black),
                ]
            ).rstrip()
        else:
            feedback = "\n".join(
                [
                    "Feedback (coordinate edits):",
                    f"- Turn: {turn_number}",
                    "- Coordinates are absolute (x y z).",
                    "- Use /setblock or /fill.",
                    f"- Place {expected_bg_block} at:",
                    _format_positions(missing_bg),
                ]
            ).rstrip()

        parts.append(feedback)

        if original_prompt_flag and base_user:
            parts.append("")
            parts.append(base_user)
        if previous_response_flag:
            prev = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
            if prev.strip():
                parts.append("")
                parts.append("Your previous commands:")
                parts.append(prev.strip())

        prompts[agent_idx] = "\n".join(parts).strip()

    return prompts
