from __future__ import annotations

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

    Reward = score_shape_overlap + score_components + score_material_adjacent (range ~[0, 3]).
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
            return float(metrics.get("score_shape_overlap", 0.0)) + float(metrics.get("score_components", 0.0)) + float(
                metrics.get("score_material_adjacent", 0.0)
            )
        except Exception:
            return 0.0

    if num_agents == 1:
        max_commands_agent1 = max_commands_total

        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
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
            return [_reward_from_metrics(metrics)]

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
        return [_reward_from_metrics(metrics)]

    return reward_fn
