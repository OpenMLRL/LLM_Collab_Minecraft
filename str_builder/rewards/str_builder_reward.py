from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_Minecraft.str_builder.utils.str_builder import (
    TaskSpec,
    build_target_color_map,
    extract_command_lines,
    normalize_block_id,
    score_str_builder,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


def _log_train_metrics(metrics: Mapping[str, float], *, turn_idx: int | None) -> None:
    try:
        import wandb  # type: ignore

        run = getattr(wandb, "run", None)
        if run is None:
            return
        prefix = f"turn_{int(turn_idx)}" if turn_idx else "turn_1"
        payload = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
        wandb.log(payload, commit=False)
    except Exception:
        return


def _compute_iou(metrics: Mapping[str, Any]) -> float:
    covered = float(metrics.get("covered", 0.0))
    extra = float(metrics.get("extra_blocks", 0.0))
    target_total = float(metrics.get("target_total", 0.0))
    union = target_total + extra
    if union <= 0:
        return 0.0
    return covered / union


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
    """Return a reward function for str_builder using coverage and penalty ratios."""
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands"), 600)

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
        allowed_blocks_agent1 = ["black_concrete", "white_concrete"]

    allowed_blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if not allowed_blocks_agent2:
        allowed_blocks_agent2 = ["red_concrete"]

    allowed_blocks_per_agent = [allowed_blocks_agent1]
    if num_agents >= 2:
        allowed_blocks_per_agent.append(allowed_blocks_agent2)

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_verbose = bool(output_cfg.get("verbose", False))

    debug_cfg = cfg.get("debug") or {}
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    debug_enabled = (bool(debug_cfg.get("enabled", False)) or (os.environ.get("STR_BUILDER_DEBUG_ASCII") == "1")) and output_verbose
    debug_max_prints = _as_int(debug_cfg.get("max_prints"), 0)
    if debug_enabled and debug_max_prints <= 0:
        debug_max_prints = 10
    debug_every_n_calls = _as_int(debug_cfg.get("every_n_calls"), 0)
    debug_empty_char = str(debug_cfg.get("empty_char") or ".")[:1] or "."
    debug_raw_output = bool(debug_cfg.get("raw_output", False))

    debug_state = {"calls": 0, "printed": 0}

    def _block_to_color_initial(block_id: str) -> str:
        s = normalize_block_id(str(block_id or "")).lower()
        if not s:
            return "#"
        color = s.split("_", 1)[0]
        return (color[:1] or s[:1] or "#").upper()

    def _render_overlay(task: TaskSpec, obs_map: Mapping[tuple[int, int, int], str]) -> str:
        height = len(task.target_rows_topdown)
        width = len(task.target_rows_topdown[0]) if height else 0
        lines: List[str] = []
        for r, row in enumerate(task.target_rows_topdown):
            out: List[str] = []
            for x in range(width):
                wx = task.local_bbox_from[0] + x
                wy = task.local_bbox_from[1] + (height - 1 - r)
                wz = task.local_bbox_from[2]
                pos = (int(wx), int(wy), int(wz))
                observed = normalize_block_id(obs_map.get(pos, "air"))
                if observed not in ("air", "cave_air", "void_air"):
                    out.append(_block_to_color_initial(observed))
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
        obs_map: Mapping[tuple[int, int, int], str],
        turn_idx: int | None,
        raw_outputs: List[str] | None,
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
        coverage = float(metrics.get("coverage_ratio", metrics.get("accuracy", 0.0)))
        extra_ratio = float(metrics.get("extra_ratio", 0.0))
        adj_ratio = float(metrics.get("adjacent_same_color_ratio", 0.0))
        prefix = (
            f"[str_builder debug] {task.task_id} text={task.text!r}{turn_str} "
            f"reward={reward:.4f} "
            f"cov={coverage:.3f} "
            f"extra_r={extra_ratio:.3f} "
            f"adj_r={adj_ratio:.3f}"
        )
        print(prefix, flush=True)
        print(_render_overlay(task, obs_map), flush=True)
        if debug_raw_output and raw_outputs is not None:
            for idx, raw in enumerate(raw_outputs):
                print(f"[str_builder raw] agent{idx}:", flush=True)
                print((raw or "").rstrip(), flush=True)

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
            expected_map, _owners = build_target_color_map(
                task=task,
                allowed_blocks_per_agent=allowed_blocks_per_agent,
                num_agents=num_agents,
            )
            metrics = score_str_builder(
                task=task,
                world_scan_blocks=blocks,
                expected_map=expected_map,
                allowed_blocks_per_agent=allowed_blocks_per_agent,
            )
            reward = float(metrics.get("score_mean", 0.0))
            _log_train_metrics(
                {
                    "iou": _compute_iou(metrics),
                    "level_1": float(metrics.get("score_acc", 0.0)),
                    "level_2": -float(metrics.get("penalty_extra", 0.0)),
                    "level_3": -float(metrics.get("penalty_adj", 0.0)),
                    "level_4": -float(metrics.get("penalty_missing_palette", 0.0)),
                    "level_total": float(metrics.get("score_total", reward)),
                },
                turn_idx=turn_idx,
            )
            if debug_enabled:
                obs_map = {tuple(b["pos"]): normalize_block_id(b.get("name") or "air") for b in blocks}
                _maybe_debug_print(
                    task=task,
                    reward=reward,
                    metrics=metrics,
                    obs_map=obs_map,
                    turn_idx=turn_idx,
                    raw_outputs=[completion],
                )
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
        expected_map, _owners = build_target_color_map(
            task=task,
            allowed_blocks_per_agent=allowed_blocks_per_agent,
            num_agents=num_agents,
        )
        metrics = score_str_builder(
            task=task,
            world_scan_blocks=blocks,
            expected_map=expected_map,
            allowed_blocks_per_agent=allowed_blocks_per_agent,
        )
        reward = float(metrics.get("score_mean", 0.0))
        _log_train_metrics(
            {
                "iou": _compute_iou(metrics),
                "level_1": float(metrics.get("score_acc", 0.0)),
                "level_2": -float(metrics.get("penalty_extra", 0.0)),
                "level_3": -float(metrics.get("penalty_adj", 0.0)),
                "level_4": -float(metrics.get("penalty_missing_palette", 0.0)),
                "level_total": float(metrics.get("score_total", reward)),
            },
            turn_idx=turn_idx,
        )
        if debug_enabled:
            obs_map = {tuple(b["pos"]): normalize_block_id(b.get("name") or "air") for b in blocks}
            _maybe_debug_print(
                task=task,
                reward=reward,
                metrics=metrics,
                obs_map=obs_map,
                turn_idx=turn_idx,
                raw_outputs=[c1, c2],
            )
        return [reward]

    return reward_fn
