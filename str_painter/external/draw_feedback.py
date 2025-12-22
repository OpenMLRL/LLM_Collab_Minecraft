from __future__ import annotations

import random
import zlib
from typing import Any, Dict, List, Optional

from LLM_Collab_MC.str_painter.utils.str_painter import (
    TaskSpec,
    get_background_coords,
    get_letter_coords,
    normalize_block_id,
    parse_ascii_decisions,
)


def _render_clean_grid(
    task: TaskSpec,
    *,
    state_map: Dict[tuple[int, int, int], str],
    expected_letter_block: str,
    expected_bg_block: str | None,
) -> str:
    lines, _missing = _build_clean_grid_and_missing(
        task,
        state_map=state_map,
        expected_letter_block=expected_letter_block,
        expected_bg_block=expected_bg_block,
    )
    return "\n".join(lines)


def _build_clean_grid_and_missing(
    task: TaskSpec,
    *,
    state_map: Dict[tuple[int, int, int], str],
    expected_letter_block: str,
    expected_bg_block: str | None,
) -> tuple[List[str], List[tuple[int, int, str]]]:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0

    lines: List[str] = []
    missing: List[tuple[int, int, str]] = []
    for r, row in enumerate(task.target_rows_topdown):
        out: List[str] = []
        for x in range(width):
            ch = row[x] if x < len(row) else "."
            expected_symbol = None
            expected_block = None
            if ch == "#":
                expected_symbol = "B"
                expected_block = expected_letter_block
            elif expected_bg_block is not None:
                expected_symbol = "W"
                expected_block = expected_bg_block

            wx = task.local_bbox_from[0] + x
            wy = task.local_bbox_from[1] + (height - 1 - r)
            wz = task.local_bbox_from[2]
            pos = (int(wx), int(wy), int(wz))
            observed = normalize_block_id(state_map.get(pos, "air"))

            if expected_block is not None and observed == expected_block:
                out.append(expected_symbol or ".")
            else:
                out.append(".")
                if expected_symbol is not None:
                    missing.append((r, x, expected_symbol))
        lines.append("".join(out))
    return lines, missing


def _render_state_grid(
    task: TaskSpec,
    *,
    state_map: Dict[tuple[int, int, int], str],
    expected_letter_block: str,
    expected_bg_block: str | None,
) -> str:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0
    lines: List[str] = []
    for r in range(height):
        out = []
        for x in range(width):
            wx = task.local_bbox_from[0] + x
            wy = task.local_bbox_from[1] + (height - 1 - r)
            wz = task.local_bbox_from[2]
            pos = (int(wx), int(wy), int(wz))
            observed = normalize_block_id(state_map.get(pos, "air"))
            if observed == expected_letter_block:
                out.append("B")
            elif expected_bg_block is not None and observed == expected_bg_block:
                out.append("W")
            elif observed in ("air", "cave_air", "void_air"):
                out.append(".")
            else:
                out.append("?")
        lines.append("".join(out))
    return "\n".join(lines)


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

    expected_letter_block = normalize_block_id(allowed_blocks_agent1[0] if allowed_blocks_agent1 else "black_concrete")
    expected_bg_block = None
    if n >= 2:
        expected_bg_block = normalize_block_id(allowed_blocks_agent2[0] if allowed_blocks_agent2 else "white_concrete")
    hint2_enabled = bool(ctx.get("hint2_enabled", False))

    task = TaskSpec(
        task_id=task_id,
        csv_row_index=0,
        text=text,
        difficulty=difficulty,
        local_bbox_from=local_bbox_from,
        local_bbox_to=local_bbox_to,
        target_rows_topdown=target_rows_topdown,
    )

    symbol_map_agent1 = {"B": expected_letter_block, "b": expected_letter_block}
    symbol_map_agent2 = {}
    if expected_bg_block is not None:
        symbol_map_agent2 = {"W": expected_bg_block, "w": expected_bg_block}
    allowed_symbols_agent1 = {".", *symbol_map_agent1.keys()}
    allowed_symbols_agent2 = {".", *symbol_map_agent2.keys()}

    decisions_1 = parse_ascii_decisions(
        agent_completions[0] if agent_completions else "",
        task=task,
        symbol_map=symbol_map_agent1,
        allowed_symbols=allowed_symbols_agent1,
    )
    decisions_2 = {}
    if n >= 2:
        decisions_2 = parse_ascii_decisions(
            agent_completions[1] if len(agent_completions) > 1 else "",
            task=task,
            symbol_map=symbol_map_agent2,
            allowed_symbols=allowed_symbols_agent2,
        )

    state_map = {**decisions_1, **decisions_2}

    prev_grid = _render_state_grid(
        task,
        state_map=state_map,
        expected_letter_block=expected_letter_block,
        expected_bg_block=expected_bg_block,
    )
    clean_lines, missing = _build_clean_grid_and_missing(
        task,
        state_map=state_map,
        expected_letter_block=expected_letter_block,
        expected_bg_block=expected_bg_block,
    )
    if hint2_enabled and len(missing) > 1:
        seed_src = f"{task_id}:{turn_number}"
        seed_val = zlib.crc32(seed_src.encode("utf-8")) & 0xFFFFFFFF
        rng = random.Random(int(seed_val))
        k = rng.randint(1, len(missing) - 1)
        picks = rng.sample(missing, k)
        mutable = [list(row) for row in clean_lines]
        for r, x, sym in picks:
            if 0 <= r < len(mutable) and 0 <= x < len(mutable[r]):
                mutable[r][x] = sym
        clean_lines = ["".join(row) for row in mutable]
    clean_grid = "\n".join(clean_lines)

    prompts: List[str] = [""] * n
    for agent_idx in range(n):
        base_user = user_prompt_single if n == 1 else (user_prompt_agent1 if agent_idx == 0 else user_prompt_agent2)
        parts: List[str] = []
        if system_prompt:
            parts.append(system_prompt)
            parts.append("")

        feedback = "\n".join(
            [
                "Feedback (ASCII grids):",
                f"- Turn: {turn_number}",
                "- Output ASCII only.",
                "- Cleaned grid removes only incorrect extra-filled cells; you must still complete remaining correct cells.",
                "Previous grid:",
                prev_grid,
                "Cleaned grid (extra-filled cells removed, plus extra correct hints):",
                clean_grid,
            ]
        ).rstrip()

        # print(feedback)

        parts.append(feedback)

        if original_prompt_flag and base_user:
            parts.append("")
            parts.append(base_user)
        if previous_response_flag:
            prev = agent_completions[agent_idx] if agent_idx < len(agent_completions) else ""
            if prev.strip():
                parts.append("")
                parts.append("Your previous output:")
                parts.append(prev.strip())

        prompts[agent_idx] = "\n".join(parts).strip()

    return prompts
