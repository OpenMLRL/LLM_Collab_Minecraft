from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    score_house_builder,
    simulate_commands_to_scan_blocks,
    unique_block_list,
    validate_and_normalize_mc_commands,
)


def _as_int_list(value: Any, default: List[int]) -> List[int]:
    if isinstance(value, list) and len(value) == len(default):
        try:
            return [int(v) for v in value]
        except Exception:
            return list(default)
    return list(default)


def _task_from_ctx(ctx: Dict[str, Any]) -> TaskSpec:
    palette_raw = ctx.get("palette") or {}
    layers_raw = ctx.get("layers_by_y") or {}
    if not isinstance(palette_raw, dict):
        palette_raw = {}
    if not isinstance(layers_raw, dict):
        layers_raw = {}

    layers_by_y = {int(k): [str(r) for r in v] for k, v in layers_raw.items()}

    return TaskSpec(
        task_id=str(ctx.get("task_id") or ""),
        local_bbox_from=_as_int_list(ctx.get("local_bbox_from"), [0, 0, 0]),
        local_bbox_to=_as_int_list(ctx.get("local_bbox_to"), [0, 0, 0]),
        palette={str(k): str(v) for k, v in palette_raw.items()},
        layers_by_y=layers_by_y,
    )


def _allowed_blocks(ctx: Dict[str, Any], agent_idx: int, palette: Dict[str, str]) -> List[str]:
    key = "allowed_blocks_agent1" if agent_idx == 0 else "allowed_blocks_agent2"
    raw = ctx.get(key) or []
    if isinstance(raw, (list, tuple)):
        blocks = unique_block_list([str(v) for v in raw])
    else:
        blocks = []
    if not blocks:
        blocks = unique_block_list(palette.values())
    return blocks


def _split_limits(total: int, num_agents: int) -> List[int]:
    n = max(1, int(num_agents))
    per = max(1, int(total) // n)
    extra = int(total) % n
    limits = [per] * n
    limits[0] += extra
    return limits


def _extract_rpg_numbers(ctx: Dict[str, Any]) -> tuple[float, float]:
    candidates = [ctx.get("_rpg_state"), ctx.get("rpg_state"), ctx.get("rpg"), ctx.get("RPG")]
    player_hp = 0.0
    spider_dmg = 0.0
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        if "player_hp" in cand:
            try:
                player_hp = float(cand.get("player_hp", 0.0) or 0.0)
            except Exception:
                player_hp = 0.0
        player_cfg = cand.get("player") if isinstance(cand.get("player"), dict) else None
        if player_cfg and player_hp <= 0:
            try:
                player_hp = float(player_cfg.get("hp", 0.0) or 0.0)
            except Exception:
                player_hp = 0.0
        if "spider_total_dmg" in cand:
            try:
                spider_dmg = float(cand.get("spider_total_dmg", 0.0) or 0.0)
            except Exception:
                spider_dmg = 0.0
        spider_cfg = cand.get("spider") if isinstance(cand.get("spider"), dict) else None
        if spider_cfg and spider_dmg <= 0:
            try:
                spider_dmg = float(spider_cfg.get("dmg", spider_cfg.get("atk_high", 0.0)) or 0.0)
            except Exception:
                spider_dmg = 0.0
        if player_hp > 0 or spider_dmg > 0:
            break
    return player_hp, spider_dmg


def _has_kill(commands: List[str]) -> bool:
    for cmd in commands:
        stripped = (cmd or "").strip()
        if stripped.startswith("/"):
            stripped = stripped[1:].lstrip()
        if stripped.lower().startswith("kill"):
            return True
    return False


def _compute_reward(ctx: Dict[str, Any], agent_completions: List[str], num_agents: int) -> float:
    n = max(1, int(num_agents))
    task = _task_from_ctx(ctx)
    limited_resource = bool(ctx.get("limited_resource", False))
    max_commands_total = int(ctx.get("max_commands_total") or 600)
    max_limits = _split_limits(max_commands_total, n)
    resource_limits = compute_resource_limits(task, num_agents=n) if limited_resource else None

    accepted_all: List[str] = []
    accepted_by_agent: List[List[str]] = []
    for agent_idx in range(n):
        allowed = _allowed_blocks(ctx, agent_idx, task.palette)
        completion = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
        lines = extract_command_lines(completion)
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_limits[agent_idx],
            resource_limits=resource_limits,
        )
        accepted_by_agent.append(accepted)
        accepted_all.extend(accepted)

    blocks = simulate_commands_to_scan_blocks(
        commands=accepted_all,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )
    metrics = score_house_builder(task=task, world_scan_blocks=blocks)
    reward = float(metrics.get("score_mean", 0.0))

    player_hp, spider_dmg = _extract_rpg_numbers(ctx)
    if player_hp > 0 and spider_dmg > 0 and not any(_has_kill(cmds) for cmds in accepted_by_agent):
        reward -= min(1.0, spider_dmg / player_hp) * 0.1

    return reward


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
