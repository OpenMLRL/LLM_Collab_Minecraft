from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_MC.str_painter.utils.str_painter import (
    TaskSpec,
    get_background_coords,
    normalize_block_id,
    parse_ascii_decisions,
    score_painter_accuracy,
)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _task_from_batch_item(item: Mapping[str, Any]) -> TaskSpec:
    return TaskSpec(
        task_id=str(item.get("task_id") or ""),
        csv_row_index=_as_int(item.get("csv_row_index"), 0),
        text=str(item.get("string") or ""),
        difficulty=_as_int(item.get("difficulty"), 0),
        local_bbox_from=[_as_int(v, 0) for v in (item.get("local_bbox_from") or [0, 0, 0])],
        local_bbox_to=[_as_int(v, 0) for v in (item.get("local_bbox_to") or [0, 0, 0])],
        target_rows_topdown=[str(r) for r in (item.get("target_rows_topdown") or [])],
    )


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    """Return a reward function for str_painter based on grid match accuracy."""
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    def _as_block_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    allowed_blocks_agent1 = _as_block_list(task_cfg.get("block_agent1"))
    if not allowed_blocks_agent1:
        allowed_blocks_agent1 = ["black_concrete"]

    allowed_blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if not allowed_blocks_agent2:
        allowed_blocks_agent2 = ["white_concrete"]

    expected_letter_block = normalize_block_id(allowed_blocks_agent1[0])
    expected_bg_block = normalize_block_id(allowed_blocks_agent2[0])
    symbol_map_agent1 = {"B": expected_letter_block, "b": expected_letter_block}
    symbol_map_agent2 = {"W": expected_bg_block, "w": expected_bg_block}
    allowed_symbols_agent1 = {".", *symbol_map_agent1.keys()}
    allowed_symbols_agent2 = {".", *symbol_map_agent2.keys()}

    debug_cfg = cfg.get("debug") or {}
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    debug_enabled = bool(debug_cfg.get("enabled", False)) or (os.environ.get("STR_PAINTER_DEBUG_ASCII") == "1")
    debug_max_prints = _as_int(debug_cfg.get("max_prints"), 0)
    if debug_enabled and debug_max_prints <= 0:
        debug_max_prints = 10
    debug_every_n_calls = _as_int(debug_cfg.get("every_n_calls"), 0)
    debug_state = {"calls": 0, "printed": 0}

    def _render_ascii(task: TaskSpec, obs: Mapping[tuple[int, int, int], str]) -> str:
        height = len(task.target_rows_topdown)
        width = len(task.target_rows_topdown[0]) if height else 0
        lines: List[str] = []
        for r, row in enumerate(task.target_rows_topdown):
            out = []
            for x in range(width):
                wx = task.local_bbox_from[0] + x
                wy = task.local_bbox_from[1] + (height - 1 - r)
                wz = task.local_bbox_from[2]
                block = normalize_block_id(obs.get((wx, wy, wz), "air"))
                if block == expected_letter_block:
                    out.append("B")
                elif block == expected_bg_block:
                    out.append("W")
                elif block in ("air", "cave_air", "void_air"):
                    out.append(".")
                else:
                    out.append("?")
            lines.append("".join(out))
        return "\n".join(lines)

    def _maybe_debug_print(
        task: TaskSpec,
        reward: float,
        metrics: Mapping[str, Any],
        obs: Mapping[tuple[int, int, int], str],
        turn_idx: int | None,
    ) -> None:
        if not debug_enabled:
            return
        debug_state["calls"] += 1
        if debug_state["printed"] >= debug_max_prints:
            return
        if debug_every_n_calls > 0 and (debug_state["calls"] % debug_every_n_calls) != 0:
            return
        debug_state["printed"] += 1
        penalty = metrics.get("background_penalty")
        penalty_str = f" penalty={float(penalty):.3f}" if penalty is not None else ""
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        prefix = (
            f"[str_painter debug] {task.task_id} text={task.text!r}{turn_str} "
            f"reward={reward:.4f} "
            f"letter_acc={float(metrics.get('letter_acc', 0.0)):.3f} "
            f"bg_acc={float(metrics.get('background_acc', 0.0)):.3f}"
            f"{penalty_str}"
        )
        print(prefix, flush=True)
        print(_render_ascii(task, obs), flush=True)

    if num_agents == 1:
        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
            turn_idx = None
            if isinstance(batch_item, Mapping):
                turn_idx = batch_item.get("_str_painter_turn")

            completion = agent1_completions[0] if agent1_completions else ""
            decisions = parse_ascii_decisions(
                completion,
                task=task,
                symbol_map=symbol_map_agent1,
                allowed_symbols=allowed_symbols_agent1,
            )
            metrics = score_painter_accuracy(
                task=task,
                state=decisions,
                letter_block=expected_letter_block,
                background_block=None,
            )
            background_coords = get_background_coords(task)
            background_total = len(background_coords)
            background_filled = 0
            for pos in background_coords:
                if normalize_block_id(decisions.get(pos, "air")) == expected_letter_block:
                    background_filled += 1
            if background_total > 0:
                background_penalty = 2.0 * (float(background_filled) ** 2) / float(background_total ** 2)
            else:
                background_penalty = 0.0

            reward = float(metrics.get("overall_acc", 0.0)) - background_penalty
            metrics = {
                **metrics,
                "background_total": background_total,
                "background_filled": background_filled,
                "background_penalty": background_penalty,
            }
            if debug_enabled:
                _maybe_debug_print(task, reward, metrics, decisions, turn_idx)
            return [reward]

        return reward_fn

    if num_agents != 2:
        raise ValueError("num_agents must be 1 or 2")

    def reward_fn(
        agent1_completions: List[str],
        agent2_completions: List[str],
        *,
        batch_items: List[Mapping[str, Any]] | None = None,
    ) -> List[float]:
        batch_item = (batch_items or [{}])[0]
        task = _task_from_batch_item(batch_item)
        turn_idx = None
        if isinstance(batch_item, Mapping):
            turn_idx = batch_item.get("_str_painter_turn")

        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""
        decisions_1 = parse_ascii_decisions(
            c1,
            task=task,
            symbol_map=symbol_map_agent1,
            allowed_symbols=allowed_symbols_agent1,
        )
        decisions_2 = parse_ascii_decisions(
            c2,
            task=task,
            symbol_map=symbol_map_agent2,
            allowed_symbols=allowed_symbols_agent2,
        )
        merged = {**decisions_1, **decisions_2}
        metrics = score_painter_accuracy(
            task=task,
            state=merged,
            letter_block=expected_letter_block,
            background_block=expected_bg_block,
        )
        reward = float(metrics.get("overall_acc", 0.0))
        if debug_enabled:
            _maybe_debug_print(task, reward, metrics, merged, turn_idx)
        return [reward]

    return reward_fn
