from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.str_builder.utils.str_builder import TaskSpec, get_target_positions


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

    task = TaskSpec(
        task_id=task_id,
        csv_row_index=0,
        text=text,
        difficulty=difficulty,
        local_bbox_from=local_bbox_from,
        local_bbox_to=local_bbox_to,
        target_rows_topdown=target_rows_topdown,
    )

    positions = get_target_positions(task)
    positions = sorted(positions, key=lambda p: (-p[1], p[0], p[2]))

    total = len(positions)
    target_count = max(1, total // 2) if total else 0

    def _format_positions(points: List[tuple[int, int, int]]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- ({x}, {y}, {z})" for x, y, z in points)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        allowed_blocks = allowed_blocks_agent1 if agent_idx == 0 else (allowed_blocks_agent2 or allowed_blocks_agent1)
        allowed_block_lines = "\n".join(f"- {b}" for b in allowed_blocks) if allowed_blocks else "- (see original prompt)"

        parts = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")
        parts.extend(
            [
                "Position feedback (subset selection):",
                f"- Turn: {turn_number}",
                "- Coordinates are absolute (x, y, z).",
                "- Output /setblock commands only (no markdown).",
                f"- Choose a subset of about half the positions (target size ~ {target_count}).",
                "- Try to avoid selecting any 4-connected adjacent pairs (sharing a side) within your subset.",
                f"- Max commands allowed: {max_limits[agent_idx]}",
                "",
                "All letter positions:",
                _format_positions(positions),
                "",
                "Available blocks (use ONLY these):",
                allowed_block_lines,
                "",
                "Format: /setblock <x> <y> <z> <block>",
            ]
        )
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
