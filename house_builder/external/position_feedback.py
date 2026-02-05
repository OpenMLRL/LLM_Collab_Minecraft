from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    build_expected_map,
    normalize_block_id,
    unique_block_list,
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

    max_commands_total = int(ctx.get("max_commands_total") or 600)
    max_limits = _split_limits(max_commands_total, n)

    task = _task_from_ctx(ctx)
    expected_map = build_expected_map(task)

    positions: List[Tuple[Tuple[int, int, int], str]] = []
    for pos, block in expected_map.items():
        block_norm = normalize_block_id(block)
        if block_norm in ("air", "cave_air", "void_air"):
            continue
        positions.append((pos, block_norm))

    positions = sorted(positions, key=lambda p: (-p[0][1], p[0][0], p[0][2]))

    total = len(positions)
    target_count = max(1, total // 2) if total else 0

    def _format_positions(points: List[Tuple[Tuple[int, int, int], str]]) -> str:
        if not points:
            return "- (none)"
        return "\n".join(f"- ({x}, {y}, {z}) = {block}" for (x, y, z), block in points)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        allowed_blocks = _allowed_blocks(ctx, agent_idx, task.palette)
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
                "- Output /fill commands only (no markdown).",
                f"- Choose a subset of about half the positions (target size ~ {target_count}).",
                "- Use: /fill x y z x y z block",
                f"- Max commands allowed: {max_limits[agent_idx]}",
                "",
                "All target positions:",
                _format_positions(positions),
                "",
                "Available blocks (use ONLY these):",
                allowed_block_lines,
                "",
                "Format: /fill <x> <y> <z> <x> <y> <z> <block>",
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
