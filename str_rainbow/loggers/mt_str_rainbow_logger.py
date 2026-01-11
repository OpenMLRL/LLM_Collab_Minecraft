from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from LLM_Collab_MC.str_rainbow.utils.str_rainbow import (
    TaskSpec,
    build_target_color_map,
    extract_command_lines,
    score_str_rainbow,
    simulate_commands_to_scan_blocks,
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


def _get_allowed_blocks(cfg: Mapping[str, Any], num_agents: int) -> List[List[str]]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    blocks_agent1 = _as_block_list(task_cfg.get("block_agent1"))
    if not blocks_agent1:
        blocks_agent1 = ["black_concrete", "white_concrete"]

    blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if not blocks_agent2:
        blocks_agent2 = ["red_concrete"]

    allowed = [blocks_agent1]
    if num_agents >= 2:
        allowed.append(blocks_agent2)
    return allowed


def _max_commands_total(cfg: Mapping[str, Any]) -> int:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    return _as_int(task_cfg.get("max_commands"), 600)


def _compute_iou(
    *,
    task: TaskSpec,
    completions: List[str],
    allowed_blocks_per_agent: List[List[str]],
    max_commands_total: int,
    num_agents: int,
) -> float:
    world_bbox_from = task.local_bbox_from
    world_bbox_to = task.local_bbox_to

    if num_agents <= 1:
        max_commands_agent1 = max_commands_total
        completion = completions[0] if completions else ""
        lines = extract_command_lines(completion)
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed_blocks_per_agent[0],
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent1,
        )
        merged = list(accepted)
    else:
        max_commands_per_agent = max(1, max_commands_total // num_agents)
        max_commands_agent1 = max_commands_per_agent + (max_commands_total % num_agents)
        max_commands_agent2 = max_commands_per_agent

        c1 = completions[0] if len(completions) > 0 else ""
        c2 = completions[1] if len(completions) > 1 else ""
        lines_1 = extract_command_lines(c1)
        lines_2 = extract_command_lines(c2)
        accepted_1, _rejected_1 = validate_and_normalize_mc_commands(
            lines=lines_1,
            allowed_blocks=allowed_blocks_per_agent[0],
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent1,
        )
        accepted_2, _rejected_2 = validate_and_normalize_mc_commands(
            lines=lines_2,
            allowed_blocks=allowed_blocks_per_agent[1],
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent2,
        )
        merged = [*accepted_1, *accepted_2]

    blocks = simulate_commands_to_scan_blocks(
        commands=merged,
        world_bbox_from=world_bbox_from,
        world_bbox_to=world_bbox_to,
    )
    expected_map, _owners = build_target_color_map(
        task=task,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
        num_agents=num_agents,
    )
    metrics = score_str_rainbow(
        task=task,
        world_scan_blocks=blocks,
        expected_map=expected_map,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
    )
    covered = float(metrics.get("covered", 0.0))
    extra = float(metrics.get("extra_blocks", 0.0))
    target_total = float(metrics.get("target_total", 0.0))
    union = target_total + extra
    if union <= 0:
        return 0.0
    return covered / union


def mt_str_rainbow_logger(
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

    allowed_blocks_per_agent = _get_allowed_blocks(cfg, num_agents)
    max_commands_total = _max_commands_total(cfg)

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
                allowed_blocks_per_agent=allowed_blocks_per_agent,
                max_commands_total=max_commands_total,
                num_agents=num_agents,
            )
            sample_metrics[f"turn_{turn_idx + 1}/iou"] = iou
        metrics_list.append(sample_metrics)
    return metrics_list


def aggregate_mt_str_rainbow_metrics(
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
