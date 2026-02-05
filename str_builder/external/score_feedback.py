from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.str_builder.utils.str_builder import (
    TaskSpec,
    build_target_color_map,
    extract_command_lines,
    score_str_builder,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


def _as_int_list(value: Any, default: List[int]) -> List[int]:
    if isinstance(value, list) and len(value) == len(default):
        try:
            return [int(v) for v in value]
        except Exception:
            return list(default)
    return list(default)


def _as_block_list(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, (list, tuple)):
        blocks = [str(v).strip() for v in value if str(v).strip()]
        if blocks:
            return blocks
    if value is not None:
        s = str(value).strip()
        if s:
            return [s]
    return list(fallback)


def _task_from_ctx(ctx: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=str(ctx.get("task_id") or ""),
        csv_row_index=0,
        text=str(ctx.get("text") or ""),
        difficulty=int(ctx.get("difficulty") or 0),
        local_bbox_from=_as_int_list(ctx.get("local_bbox_from"), [0, 0, 0]),
        local_bbox_to=_as_int_list(ctx.get("local_bbox_to"), [0, 0, 0]),
        target_rows_topdown=[str(r) for r in (ctx.get("target_rows_topdown") or [])],
    )


def _split_limits(total: int, num_agents: int) -> List[int]:
    n = max(1, int(num_agents))
    per = max(1, int(total) // n)
    extra = int(total) % n
    limits = [per] * n
    limits[0] += extra
    return limits


def _compute_reward(ctx: Dict[str, Any], agent_completions: List[str], num_agents: int) -> float:
    n = max(1, int(num_agents))
    task = _task_from_ctx(ctx)
    max_commands_total = int(ctx.get("max_commands_total") or 600)
    max_limits = _split_limits(max_commands_total, n)

    allowed_blocks_agent1 = _as_block_list(ctx.get("allowed_blocks_agent1"), ["black_concrete", "white_concrete"])
    allowed_blocks_agent2 = _as_block_list(ctx.get("allowed_blocks_agent2"), ["red_concrete"])
    allowed_blocks_per_agent = [allowed_blocks_agent1]
    if n >= 2:
        allowed_blocks_per_agent.append(allowed_blocks_agent2)

    accepted_all: List[str] = []
    for agent_idx in range(n):
        completion = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
        lines = extract_command_lines(completion)
        allowed = allowed_blocks_agent1 if agent_idx == 0 else allowed_blocks_agent2
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_limits[agent_idx],
        )
        accepted_all.extend(accepted)

    blocks = simulate_commands_to_scan_blocks(
        commands=accepted_all,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )
    expected_map, _owners = build_target_color_map(
        task=task,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
        num_agents=n,
    )
    metrics = score_str_builder(
        task=task,
        world_scan_blocks=blocks,
        expected_map=expected_map,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
    )
    return float(metrics.get("score_mean", 0.0))


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

    reward_val: float | None
    try:
        reward_val = _compute_reward(ctx=ctx, agent_completions=agent_completions, num_agents=n)
    except Exception:
        reward_val = None
    reward_text = f"{reward_val:.4f}" if reward_val is not None else "unavailable"

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)

        parts = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")
        parts.append("Score feedback:")
        parts.append(f"- Turn: {turn_number}")
        parts.append(f"- Previous reward: {reward_text}")
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
