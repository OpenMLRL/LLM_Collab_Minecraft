from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_MC.utils.str_builder import (
    TaskSpec,
    extract_command_lines,
    render_progress_overlay_ascii,
    render_target_ascii,
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

    target_ascii = render_target_ascii(task)
    progress_ascii = render_progress_overlay_ascii(task, blocks, empty_char=".", missing_target_char="#")

    feedback = "\n".join(
        [
            "Feedback (ASCII maps):",
            "- Target mask: '.' empty, '#' should place a block (any allowed type).",
            "- Current progress: '.' empty, '#' missing target, letters = first letter of the placed block color.",
            "- Deletion is allowed: use /setblock ... air or /fill ... air to remove blocks.",
            "",
            "Target mask:",
            target_ascii,
            "",
            "Current progress:",
            progress_ascii,
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
