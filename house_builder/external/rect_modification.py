from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    build_expected_map,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    rows_to_rects,
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


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


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


def format_followup_prompts(
    *,
    ctx: Dict[str, Any],
    agent_completions: List[str],
    num_agents: int = 2,
    limit: int | None = None,
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

    system_prompt = str(ctx.get("system_prompt") or "").rstrip()
    user_prompt_single = str(ctx.get("user_prompt_single") or "").rstrip()
    user_prompt_agent1 = str(ctx.get("user_prompt_agent1") or user_prompt_single).rstrip()
    user_prompt_agent2 = str(ctx.get("user_prompt_agent2") or user_prompt_single).rstrip()
    resource_limits_text = str(ctx.get("resource_limits_text") or "").rstrip()

    max_commands_total = _as_int(ctx.get("max_commands_total") or 600, 600)
    limited_resource = bool(ctx.get("limited_resource", False))
    max_limits = _split_limits(max_commands_total, n)

    task = _task_from_ctx(ctx)
    resource_limits = compute_resource_limits(task, num_agents=n) if limited_resource else None

    accepted_by_agent: List[List[str]] = []
    accepted_all: List[str] = []
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

    expected_map = build_expected_map(task)

    observed: Dict[Tuple[int, int, int], str] = {}
    for b in blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        observed[(int(pos[0]), int(pos[1]), int(pos[2]))] = normalize_block_id(str(name or "air"))

    wrong_positions: List[Tuple[Tuple[int, int, int], str]] = []
    for pos, expected in expected_map.items():
        expected_norm = normalize_block_id(expected)
        observed_norm = normalize_block_id(observed.get(pos, "air"))
        if observed_norm != expected_norm:
            wrong_positions.append((pos, expected_norm))

    min_x = min(task.local_bbox_from[0], task.local_bbox_to[0])
    min_z = min(task.local_bbox_from[2], task.local_bbox_to[2])
    rects_by_y: Dict[int, List[Tuple[int, int, int, int, str]]] = {}
    for y, rows in task.layers_by_y.items():
        rects = rows_to_rects(rows=rows, palette=task.palette, min_x=min_x, min_z=min_z)
        rects_by_y[int(y)] = rects

    rect_set: Dict[Tuple[int, int, int, int, int, str], None] = {}
    for (x, y, z), expected in wrong_positions:
        rects = rects_by_y.get(int(y)) or []
        for x1, z1, x2, z2, block in rects:
            if normalize_block_id(block) != expected:
                continue
            if x1 <= x <= x2 and z1 <= z <= z2:
                rect_set[(int(y), int(x1), int(z1), int(x2), int(z2), expected)] = None
                break

    rects_all = sorted(rect_set.keys(), key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))

    lim = limit
    if lim is None:
        lim = _as_int(ctx.get("rect_limit") or ctx.get("lim") or 0, 0)

    def _filter_rects(rects: List[Tuple[int, int, int, int, int, str]], agent_idx: int) -> List[Tuple[int, int, int, int, int, str]]:
        if n == 1:
            return rects
        if agent_idx == 0:
            return [r for r in rects if r[0] <= 2]
        return [r for r in rects if r[0] >= 2]

    def _format_accepted(lines: List[str]) -> str:
        if not lines:
            return "- (none)"
        return "\n".join(f"- {line}" for line in lines)

    def _format_rects(rects: List[Tuple[int, int, int, int, int, str]]) -> str:
        if not rects:
            return "(none)"
        lines: List[str] = []
        for y, x1, z1, x2, z2, block in rects:
            lines.append(f"y={y}: ({x1}, {z1}, {x2}, {z2} {block})")
        return "\n".join(lines)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        agent_rects = _filter_rects(rects_all, agent_idx)
        if lim is not None and lim > 0:
            agent_rects = agent_rects[: int(lim)]

        parts: List[str] = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")
        if resource_limits_text and not original_prompt_flag:
            parts.append(resource_limits_text)
            parts.append("")
        if original_prompt_flag and base_user:
            parts.append(base_user)
            parts.append("")
        parts.append("Reminder: each turn starts from an empty (all-air) world.")
        parts.append("Accepted commands from your previous turn:")
        parts.append(_format_accepted(accepted_by_agent[agent_idx]))
        parts.append("")
        parts.append("Suggested rectangles to fix mismatches:")
        parts.append(_format_rects(agent_rects))
        if previous_response_flag:
            prev = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
            if prev.strip():
                parts.append("")
                parts.append("Your previous raw output:")
                parts.append(prev.strip())
        prompts[agent_idx] = "\n".join(parts).strip()

    return prompts
