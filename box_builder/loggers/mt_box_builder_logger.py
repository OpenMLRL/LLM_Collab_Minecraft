from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from LLM_Collab_MC.box_builder.utils.box_builder import (
    TaskSpec,
    build_expected_map,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    simulate_commands_to_scan_blocks,
    unique_block_list,
    validate_and_normalize_mc_commands,
)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_block_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _is_air(block_id: str) -> bool:
    return normalize_block_id(block_id) in ("air", "cave_air", "void_air")


def _allowed_blocks_for_task(task: TaskSpec, overrides: List[str]) -> List[str]:
    if overrides:
        return unique_block_list(overrides)
    return unique_block_list(task.palette.values())


def _task_settings(cfg: Mapping[str, Any]) -> Tuple[List[str], List[str], int, bool]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    block_agent1_override = _as_block_list(task_cfg.get("block_agent1"))
    block_agent2_override = _as_block_list(task_cfg.get("block_agent2"))
    max_commands_total = _as_int(task_cfg.get("max_commands"), 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))
    return block_agent1_override, block_agent2_override, max_commands_total, limited_resource


def _compute_iou(
    *,
    task: TaskSpec,
    completions: List[str],
    block_agent1_override: List[str],
    block_agent2_override: List[str],
    max_commands_total: int,
    limited_resource: bool,
    num_agents: int,
) -> float:
    if num_agents <= 1:
        max_commands_agent1 = max_commands_total
        completion = completions[0] if completions else ""
        allowed_blocks = _allowed_blocks_for_task(task, block_agent1_override)
        resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else None
        lines = extract_command_lines(completion)
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed_blocks,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_commands_agent1,
            resource_limits=resource_limits,
        )
        merged = list(accepted)
    else:
        max_commands_per_agent = max(1, max_commands_total // num_agents)
        max_commands_agent1 = max_commands_per_agent + (max_commands_total % num_agents)
        max_commands_agent2 = max_commands_per_agent
        resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else None

        allowed_blocks_agent1 = _allowed_blocks_for_task(task, block_agent1_override)
        allowed_blocks_agent2 = _allowed_blocks_for_task(task, block_agent2_override)

        c1 = completions[0] if len(completions) > 0 else ""
        c2 = completions[1] if len(completions) > 1 else ""
        lines_1 = extract_command_lines(c1)
        lines_2 = extract_command_lines(c2)
        accepted_1, _rejected_1 = validate_and_normalize_mc_commands(
            lines=lines_1,
            allowed_blocks=allowed_blocks_agent1,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_commands_agent1,
            resource_limits=resource_limits,
        )
        accepted_2, _rejected_2 = validate_and_normalize_mc_commands(
            lines=lines_2,
            allowed_blocks=allowed_blocks_agent2,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_commands_agent2,
            resource_limits=resource_limits,
        )
        merged = [*accepted_1, *accepted_2]

    blocks = simulate_commands_to_scan_blocks(
        commands=merged,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
    )

    expected_map = build_expected_map(task)
    expected_non_air = {pos for pos, block in expected_map.items() if not _is_air(block)}
    observed_non_air = {
        (int(b.get("pos")[0]), int(b.get("pos")[1]), int(b.get("pos")[2]))
        for b in blocks
        if isinstance(b.get("pos"), list)
        and len(b.get("pos")) == 3
        and not _is_air(b.get("name") or "air")
    }

    union = expected_non_air | observed_non_air
    if not union:
        return 0.0
    intersection = expected_non_air & observed_non_air
    return float(len(intersection) / len(union))


def mt_box_builder_logger(
    agent_completions_turns: List[List[List[str]]],
    test_cases: List[str],
    entry_points: List[str],
    prompts: Optional[List[str]] = None,
    *,
    cfg: Mapping[str, Any],
    num_agents: int,
    tasks_by_prompt: Mapping[str, TaskSpec],
) -> List[Dict[str, Any]]:
    if not agent_completions_turns:
        return []
    if not prompts:
        return []

    block_agent1_override, block_agent2_override, max_commands_total, limited_resource = _task_settings(cfg)

    num_samples = len(prompts)
    num_turns = 0
    if agent_completions_turns and agent_completions_turns[0]:
        num_turns = len(agent_completions_turns[0][0]) if agent_completions_turns[0][0] else 0

    metrics_list: List[Dict[str, Any]] = []
    for sample_idx in range(num_samples):
        prompt = prompts[sample_idx]
        task = tasks_by_prompt.get(prompt)
        if task is None:
            continue
        sample_metrics: Dict[str, Any] = {"sample_id": sample_idx}
        for turn_idx in range(num_turns):
            completions: List[str] = []
            for agent_idx in range(num_agents):
                try:
                    completions.append(agent_completions_turns[agent_idx][sample_idx][turn_idx])
                except Exception:
                    completions.append("")
            iou = _compute_iou(
                task=task,
                completions=completions,
                block_agent1_override=block_agent1_override,
                block_agent2_override=block_agent2_override,
                max_commands_total=max_commands_total,
                limited_resource=limited_resource,
                num_agents=num_agents,
            )
            sample_metrics[f"turn_{turn_idx + 1}/iou"] = iou
        metrics_list.append(sample_metrics)
    return metrics_list


def aggregate_mt_box_builder_metrics(
    metrics_list: List[Dict[str, Any]], num_turns: int = 2
) -> Dict[str, float]:
    if not metrics_list:
        return {}

    aggregated: Dict[str, float] = {}
    for turn in range(1, num_turns + 1):
        key = f"turn_{turn}/iou"
        values = [m[key] for m in metrics_list if key in m]
        if not values:
            continue
        aggregated[f"turn_{turn}/avg_iou"] = float(sum(values) / len(values))
    return aggregated
