from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_MC.utils.str_builder import (
    TaskSpec,
    extract_command_lines,
    score_str_builder,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
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
    """Return a CoMLRL reward function for str_builder.

    Reward = score_shape_overlap + score_components (range ~[-1, 2]).
    """
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands"), 600)

    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    chamfer_sigma = float(task_cfg.get("chamfer_sigma") or data_cfg.get("chamfer_sigma") or 2.0)

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
        # Backwards compatible fallback.
        b0 = str(task_cfg.get("block_even") or "white_concrete").strip()
        b1 = str(task_cfg.get("block_odd") or "black_concrete").strip()
        allowed_blocks_agent1 = [b0, b1]

    allowed_blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if not allowed_blocks_agent2:
        # Backwards compatible fallback.
        allowed_blocks_agent2 = [str(task_cfg.get("block_agent2") or "red_concrete").strip() or "red_concrete"]

    def _reward_from_metrics(metrics: Mapping[str, Any]) -> float:
        try:
            return float(metrics.get("score_shape_overlap", 0.0)) + float(metrics.get("score_components", 0.0))
        except Exception:
            return 0.0

    def _coverage_ratio(metrics: Mapping[str, Any]) -> float:
        try:
            target = float(metrics.get("target_blocks", 0.0))
            built = float(metrics.get("built_blocks", 0.0))
            iou = float(metrics.get("overlap_iou", 0.0))
        except Exception:
            return 0.0
        if target <= 0.0 or iou <= 0.0:
            return 0.0
        inter = (iou * (target + built)) / (1.0 + iou)
        return max(0.0, min(1.0, inter / target))

    def _non_target_fill_ratio(task: TaskSpec, metrics: Mapping[str, Any]) -> float:
        height = len(task.target_rows_topdown)
        width = len(task.target_rows_topdown[0]) if height else 0
        total = float(max(0, height * width))
        if total <= 0.0:
            return 0.0
        try:
            target = float(metrics.get("target_blocks", 0.0))
            built = float(metrics.get("built_blocks", 0.0))
            iou = float(metrics.get("overlap_iou", 0.0))
        except Exception:
            return 0.0
        non_target = max(0.0, total - target)
        if non_target <= 0.0:
            return 0.0
        if iou <= 0.0:
            inter = 0.0
        else:
            inter = (iou * (target + built)) / (1.0 + iou)
        false_pos = max(0.0, built - inter)
        return max(0.0, min(1.0, false_pos / non_target))

    debug_cfg = cfg.get("debug") or {}
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    debug_enabled = bool(debug_cfg.get("enabled", False)) or (os.environ.get("STR_BUILDER_DEBUG_ASCII") == "1")
    debug_max_prints = _as_int(debug_cfg.get("max_prints"), 0)
    if debug_enabled and debug_max_prints <= 0:
        debug_max_prints = 10
    debug_every_n_calls = _as_int(debug_cfg.get("every_n_calls"), 0)
    debug_empty_char = str(debug_cfg.get("empty_char") or ".")[:1] or "."

    debug_state = {"calls": 0, "printed": 0}

    def _block_to_color_initial(block_id: str) -> str:
        s = str(block_id or "").strip().lower()
        if s.startswith("minecraft:"):
            s = s[len("minecraft:") :]
        if not s:
            return "#"
        color = s.split("_", 1)[0]
        return (color[:1] or s[:1] or "#").upper()

    def _render_overlay_ascii(task: TaskSpec, blocks: List[Mapping[str, Any]]) -> str:
        height = len(task.target_rows_topdown)
        width = len(task.target_rows_topdown[0]) if height else 0

        obs: Dict[tuple[int, int], str] = {}
        for b in blocks:
            pos = b.get("pos")
            name = b.get("name")
            if not (isinstance(pos, list) and len(pos) == 3):
                continue
            if name is None:
                continue
            block = str(name).strip()
            if not block or block in ("air", "cave_air", "void_air"):
                continue
            obs[(int(pos[0]), int(pos[1]))] = block

        lines: List[str] = []
        for r, row in enumerate(task.target_rows_topdown):
            out = []
            for x in range(width):
                wx = task.local_bbox_from[0] + x
                wy = task.local_bbox_from[1] + (height - 1 - r)
                placed = obs.get((wx, wy))
                if placed is not None:
                    out.append(_block_to_color_initial(placed))
                elif x < len(row) and row[x] == "#":
                    out.append("#")
                else:
                    out.append(debug_empty_char)
            lines.append("".join(out))
        return "\n".join(lines)

    def _maybe_debug_print(
        *,
        task: TaskSpec,
        reward: float,
        metrics: Mapping[str, Any],
        blocks: List[Mapping[str, Any]],
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
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        prefix = (
            f"[str_builder debug] {task.task_id} text={task.text!r} diff={task.difficulty}{turn_str} "
            f"reward={reward:.4f} "
            f"s1={float(metrics.get('score_shape_overlap', 0.0)):.3f} "
            f"s2={float(metrics.get('score_components', 0.0)):.3f}"
        )
        print(prefix, flush=True)
        print(_render_overlay_ascii(task, blocks), flush=True)

    if num_agents == 1:
        max_commands_agent1 = max_commands_total

        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
            turn_idx = None
            if isinstance(batch_item, Mapping):
                turn_idx = batch_item.get("_str_builder_turn")
            world_bbox_from = task.local_bbox_from
            world_bbox_to = task.local_bbox_to

            completion = agent1_completions[0] if agent1_completions else ""
            lines = extract_command_lines(completion)
            accepted, _rejected = validate_and_normalize_mc_commands(
                lines=lines,
                allowed_blocks=allowed_blocks_agent1,
                world_bbox_from=world_bbox_from,
                world_bbox_to=world_bbox_to,
                max_commands=max_commands_agent1,
            )

            blocks = simulate_commands_to_scan_blocks(commands=accepted, world_bbox_from=world_bbox_from, world_bbox_to=world_bbox_to)
            metrics = score_str_builder(task=task, world_origin=[0, 0, 0], world_scan_blocks=blocks, chamfer_sigma=chamfer_sigma)
            reward = _reward_from_metrics(metrics)
            if _coverage_ratio(metrics) < 0.25 or _non_target_fill_ratio(task, metrics) > 0.50:
                reward = -1.0
            _maybe_debug_print(task=task, reward=reward, metrics=metrics, blocks=blocks, turn_idx=turn_idx)
            return [reward]

        return reward_fn

    if num_agents != 2:
        raise ValueError("num_agents must be 1 or 2")

    max_commands_per_agent = max(1, max_commands_total // num_agents)
    max_commands_agent1 = max_commands_per_agent + (max_commands_total % num_agents)
    max_commands_agent2 = max_commands_per_agent

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
            turn_idx = batch_item.get("_str_builder_turn")
        world_bbox_from = task.local_bbox_from
        world_bbox_to = task.local_bbox_to

        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""

        lines_1 = extract_command_lines(c1)
        lines_2 = extract_command_lines(c2)
        accepted_1, _rejected_1 = validate_and_normalize_mc_commands(
            lines=lines_1,
            allowed_blocks=allowed_blocks_agent1,
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent1,
        )
        accepted_2, _rejected_2 = validate_and_normalize_mc_commands(
            lines=lines_2,
            allowed_blocks=allowed_blocks_agent2,
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent2,
        )

        merged = [*accepted_1, *accepted_2]
        blocks = simulate_commands_to_scan_blocks(commands=merged, world_bbox_from=world_bbox_from, world_bbox_to=world_bbox_to)
        metrics = score_str_builder(task=task, world_origin=[0, 0, 0], world_scan_blocks=blocks, chamfer_sigma=chamfer_sigma)
        reward = _reward_from_metrics(metrics)
        if _coverage_ratio(metrics) < 0.25 or _non_target_fill_ratio(task, metrics) > 0.50:
            reward = -1.0
        _maybe_debug_print(task=task, reward=reward, metrics=metrics, blocks=blocks, turn_idx=turn_idx)
        return [reward]

    return reward_fn
