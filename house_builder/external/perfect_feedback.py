from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    build_expected_map,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
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
    resource_limits_text = str(ctx.get("resource_limits_text") or "").rstrip()

    max_commands_total = int(ctx.get("max_commands_total") or 600)
    limited_resource = bool(ctx.get("limited_resource", False))
    max_limits = _split_limits(max_commands_total, n)

    task = _task_from_ctx(ctx)
    resource_limits = compute_resource_limits(task, num_agents=n) if limited_resource else None

    accepted_all: List[str] = []
    allowed_blocks_by_agent: List[List[str]] = []
    for agent_idx in range(n):
        allowed = _allowed_blocks(ctx, agent_idx, task.palette)
        allowed_blocks_by_agent.append(allowed)
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

    wrong_positions = sorted(wrong_positions, key=lambda p: (-p[0][1], p[0][0], p[0][2]))

    extra_positions: List[Tuple[int, int, int]] = []
    for pos, observed_norm in observed.items():
        if pos in expected_map:
            continue
        if observed_norm and observed_norm not in ("air", "cave_air", "void_air"):
            extra_positions.append(pos)

    extra_positions = sorted(extra_positions, key=lambda p: (-p[1], p[0], p[2]))

    wrong_by_agent: Dict[int, List[Tuple[Tuple[int, int, int], str]]] = {i: [] for i in range(n)}
    if n == 1:
        wrong_by_agent[0] = wrong_positions
    else:
        allowed_sets = [
            {normalize_block_id(b) for b in allowed_blocks_by_agent[0]},
            {normalize_block_id(b) for b in (allowed_blocks_by_agent[1] or allowed_blocks_by_agent[0])},
        ]
        for idx, (pos, block) in enumerate(wrong_positions):
            in_a = block in allowed_sets[0]
            in_b = block in allowed_sets[1]
            if in_a and not in_b:
                owner = 0
            elif in_b and not in_a:
                owner = 1
            else:
                owner = idx % n
            wrong_by_agent.setdefault(owner, []).append((pos, block))

    def _format_positions(points: List[Tuple[Tuple[int, int, int], str]]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- ({x}, {y}, {z}) = {block}" for (x, y, z), block in points)

    def _format_extra(points: List[Tuple[int, int, int]]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- ({x}, {y}, {z})" for x, y, z in points)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        parts = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")
        if resource_limits_text and not original_prompt_flag:
            parts.append(resource_limits_text)
            parts.append("")
        parts.extend(
            [
                "Feedback (coordinate edits):",
                f"- Turn: {turn_number}",
                "- Coordinates are absolute (x, y, z).",
                "- Output /fill commands only (no markdown).",
                "- Fix ONLY the listed positions.",
                "- Use: /fill x y z x y z block",
                "",
                "Set these blocks:",
                _format_positions(wrong_by_agent.get(agent_idx, [])),
            ]
        )
        if agent_idx == n - 1:
            parts.extend(
                [
                    "",
                    "Remove extra blocks (set air) at:",
                    _format_extra(extra_positions),
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
