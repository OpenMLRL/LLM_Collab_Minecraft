from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    rows_to_rects,
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


def _count_blocks_from_commands(commands: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cmd in commands:
        stripped = (cmd or "").strip()
        if stripped.startswith("/"):
            stripped = stripped[1:].lstrip()
        parts = stripped.split()
        if len(parts) < 8 or parts[0].lower() != "fill":
            continue
        try:
            x1, y1, z1 = int(parts[1]), int(parts[2]), int(parts[3])
            x2, y2, z2 = int(parts[4]), int(parts[5]), int(parts[6])
        except Exception:
            continue
        block = normalize_block_id(parts[7])
        dx = abs(x2 - x1) + 1
        dy = abs(y2 - y1) + 1
        dz = abs(z2 - z1) + 1
        volume = int(dx * dy * dz)
        counts[block] = counts.get(block, 0) + volume
    return counts


def _best_subrect(
    *,
    x1: int,
    z1: int,
    x2: int,
    z2: int,
    max_area: int,
) -> Tuple[int, int, int, int, int]:
    width = x2 - x1 + 1
    depth = z2 - z1 + 1
    best_area = 0
    best_w = 0
    best_d = 0
    for w in range(width, 0, -1):
        d = min(depth, max_area // w)
        area = w * d
        if area > best_area:
            best_area = area
            best_w = w
            best_d = d
        if best_area == max_area:
            break
    if best_area <= 0:
        return x1, z1, x1, z1, 0
    return x1, z1, x1 + best_w - 1, z1 + best_d - 1, best_area


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

    system_prompt = str(ctx.get("system_prompt") or "").rstrip()
    user_prompt_single = str(ctx.get("user_prompt_single") or "").rstrip()
    user_prompt_agent1 = str(ctx.get("user_prompt_agent1") or user_prompt_single).rstrip()
    user_prompt_agent2 = str(ctx.get("user_prompt_agent2") or user_prompt_single).rstrip()
    resource_limits_text = str(ctx.get("resource_limits_text") or "").rstrip()

    max_commands_total = _as_int(ctx.get("max_commands_total") or 600, 600)
    max_limits = _split_limits(max_commands_total, n)

    task = _task_from_ctx(ctx)
    limited_resource = bool(ctx.get("limited_resource", False))
    if not limited_resource:
        raise ValueError("resource_schedule requires limited_resource=true in task config")
    resource_limits = compute_resource_limits(task, num_agents=n)

    rects_all: List[Tuple[int, int, int, int, int, str, int]] = []
    min_x = min(task.local_bbox_from[0], task.local_bbox_to[0])
    min_z = min(task.local_bbox_from[2], task.local_bbox_to[2])
    for y, rows in task.layers_by_y.items():
        rects = rows_to_rects(rows=rows, palette=task.palette, min_x=min_x, min_z=min_z)
        for x1, z1, x2, z2, block in rects:
            block_norm = normalize_block_id(block)
            if block_norm in ("air", "cave_air", "void_air"):
                continue
            area = (x2 - x1 + 1) * (z2 - z1 + 1)
            rects_all.append((int(y), int(x1), int(z1), int(x2), int(z2), block_norm, int(area)))

    accepted_by_agent: List[List[str]] = []
    used_by_agent: List[Dict[str, int]] = []
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
            resource_limits=resource_limits if limited_resource else None,
        )
        accepted_by_agent.append(accepted)
        used_by_agent.append(_count_blocks_from_commands(accepted))

    def _format_accepted(lines: List[str]) -> str:
        if not lines:
            return "- (none)"
        return "\n".join(f"- {line}" for line in lines)

    def _format_remaining(rem: Dict[str, int], limits: Dict[str, int]) -> str:
        lines: List[str] = []
        for _key, block in task.palette.items():
            block_norm = normalize_block_id(block)
            if block_norm in ("air", "cave_air", "void_air"):
                continue
            if block_norm not in limits:
                continue
            lines.append(f"- {block_norm}: {max(0, rem.get(block_norm, 0))}")
        if not lines:
            return "(none)"
        return "\n".join(lines)

    def _format_rects(rects: List[Tuple[int, int, int, int, int, str]]) -> str:
        if not rects:
            return "(none)"
        return "\n".join(f"y={y}: ({x1}, {z1}, {x2}, {z2} {block})" for y, x1, z1, x2, z2, block in rects)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        allowed_blocks = _allowed_blocks(ctx, agent_idx, task.palette)
        allowed_norm = {normalize_block_id(b) for b in allowed_blocks}

        used = used_by_agent[agent_idx]
        remaining: Dict[str, int] = {}
        for block, limit_val in resource_limits.items():
            remaining[block] = int(limit_val) - int(used.get(block, 0))

        candidates = [
            r for r in rects_all if r[5] in allowed_norm and remaining.get(r[5], 0) > 0
        ]
        candidates.sort(key=lambda r: (-r[6], r[0], r[1], r[2], r[3], r[4], r[5]))

        selected: List[Tuple[int, int, int, int, int, str]] = []
        selected_keys = set()
        for y, x1, z1, x2, z2, block, area in candidates:
            if remaining.get(block, 0) >= area:
                selected.append((y, x1, z1, x2, z2, block))
                remaining[block] = remaining.get(block, 0) - area
                selected_keys.add((y, x1, z1, x2, z2, block))

        best_partial = None
        best_area = 0
        for y, x1, z1, x2, z2, block, area in candidates:
            if (y, x1, z1, x2, z2, block) in selected_keys:
                continue
            rem = remaining.get(block, 0)
            if rem <= 0 or area <= 0:
                continue
            px1, pz1, px2, pz2, parea = _best_subrect(
                x1=x1,
                z1=z1,
                x2=x2,
                z2=z2,
                max_area=rem,
            )
            if parea > best_area:
                best_area = parea
                best_partial = (y, px1, pz1, px2, pz2, block)

        if best_partial is not None and best_area > 0:
            selected.append(best_partial)

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
        parts.append("Accepted commands from your previous turn:")
        parts.append(_format_accepted(accepted_by_agent[agent_idx]))
        parts.append("")
        parts.append("Remaining resources after previous turn (air unlimited):")
        parts.append(_format_remaining(remaining, resource_limits))
        parts.append("")
        parts.append("Suggested rectangles to build next (greedy by size):")
        parts.append(_format_rects(selected))
        if previous_response_flag:
            prev = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
            if prev.strip():
                parts.append("")
                parts.append("Your previous raw output:")
                parts.append(prev.strip())
        prompts[agent_idx] = "\n".join(parts).strip()

    return prompts
