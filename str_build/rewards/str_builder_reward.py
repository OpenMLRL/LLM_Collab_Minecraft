from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_Minecraft.str_build.utils.str_builder import (
    TaskSpec,
    block_to_color_key,
    build_target_color_map,
    extract_command_lines,
    normalize_block_id,
    score_str_builder,
    simulate_commands_to_scan_blocks,
    validate_and_normalize_mc_commands,
)


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


def _as_block_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def evaluate_str_builder_outputs(
    *,
    cfg: Dict[str, Any],
    num_agents: int,
    agent_completions: List[List[str]],
    batch_item: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate STR Build outputs and return reward plus canonical log metrics."""
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands"), 600)
    allowed_blocks_agent1 = _as_block_list(task_cfg.get("block_agent1"))
    if not allowed_blocks_agent1:
        raise ValueError("task.block_agent1 must be provided and non-empty")

    allowed_blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if num_agents >= 2 and not allowed_blocks_agent2:
        raise ValueError("task.block_agent2 must be provided when num_agents >= 2")

    allowed_blocks_per_agent = [allowed_blocks_agent1]
    if num_agents >= 2:
        allowed_blocks_per_agent.append(allowed_blocks_agent2)

    task = _task_from_batch_item(batch_item)
    world_bbox_from = task.local_bbox_from
    world_bbox_to = task.local_bbox_to

    if int(num_agents) == 1:
        completion = (
            agent_completions[0][0] if agent_completions and agent_completions[0] else ""
        )
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=extract_command_lines(completion),
            allowed_blocks=allowed_blocks_agent1,
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_total,
        )
        merged = accepted
    elif int(num_agents) == 2:
        max_commands_per_agent = max(1, max_commands_total // int(num_agents))
        max_commands_agent1 = max_commands_per_agent + (
            max_commands_total % int(num_agents)
        )
        max_commands_agent2 = max_commands_per_agent

        c1 = agent_completions[0][0] if agent_completions and agent_completions[0] else ""
        c2 = (
            agent_completions[1][0]
            if len(agent_completions) > 1 and agent_completions[1]
            else ""
        )

        accepted_1, _rejected_1 = validate_and_normalize_mc_commands(
            lines=extract_command_lines(c1),
            allowed_blocks=allowed_blocks_agent1,
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent1,
        )
        accepted_2, _rejected_2 = validate_and_normalize_mc_commands(
            lines=extract_command_lines(c2),
            allowed_blocks=allowed_blocks_agent2,
            world_bbox_from=world_bbox_from,
            world_bbox_to=world_bbox_to,
            max_commands=max_commands_agent2,
        )
        merged = [*accepted_1, *accepted_2]
    else:
        raise ValueError("num_agents must be 1 or 2")

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
    metrics = score_str_builder(
        task=task,
        world_scan_blocks=blocks,
        expected_map=expected_map,
        allowed_blocks_per_agent=allowed_blocks_per_agent,
    )
    reward = float(metrics.get("score_mean", 0.0))
    return {
        "reward": reward,
        "metrics": metrics,
        "blocks": blocks,
        "log_metrics": {
            "minecraft/iou": _compute_iou(metrics),
            "minecraft/level_1": float(metrics.get("score_acc", 0.0)),
            "minecraft/level_2": -float(metrics.get("penalty_extra", 0.0)),
            "minecraft/level_3": -float(metrics.get("penalty_adj", 0.0)),
            "minecraft/level_4": -float(
                metrics.get("penalty_missing_palette", 0.0)
            ),
            "minecraft/score": reward,
        },
    }


def build_ac_str_metrics_callback(*, cfg: Dict[str, Any], num_agents: int):
    def callback(rollouts: List[Any]) -> Dict[str, float]:
        return aggregate_ac_str_metrics(
            rollouts,
            cfg=cfg,
            num_agents=int(num_agents),
        )

    return callback


def aggregate_ac_str_metrics(
    rollouts: List[Any], *, cfg: Dict[str, Any], num_agents: int
) -> Dict[str, float]:
    if not rollouts:
        return {}

    grouped: Dict[tuple[int, int], List[Any]] = {}
    for sample in rollouts:
        metadata = getattr(sample, "metadata", {}) or {}
        generation_idx = int(metadata.get("generation_idx", 0))
        turn_idx = int(metadata.get("turn_idx", 0))
        grouped.setdefault((generation_idx, turn_idx), []).append(sample)

    metric_values: Dict[str, List[float]] = {}
    for (_generation_idx, turn_idx), samples in sorted(grouped.items()):
        batch_item = _first_batch_item(samples)
        if not batch_item:
            continue
        completions: List[List[str]] = [[] for _ in range(max(1, int(num_agents)))]
        for sample in samples:
            agent_idx = int(getattr(sample, "agent_idx", 0))
            if 0 <= agent_idx < len(completions):
                completions[agent_idx].append(str(getattr(sample, "completion", "") or ""))
        result = evaluate_str_builder_outputs(
            cfg=cfg,
            num_agents=num_agents,
            agent_completions=completions,
            batch_item=batch_item,
        )
        prefix = f"turn_{turn_idx + 1}/"
        for key, value in result.get("log_metrics", {}).items():
            metric_values.setdefault(prefix + key, []).append(float(value))

    return {
        key: float(sum(values) / len(values))
        for key, values in metric_values.items()
        if values
    }


def build_str_eval_logging(
    *,
    cfg: Dict[str, Any],
    num_agents: int,
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _normalize_key(value: str) -> str:
        return " ".join((value or "").split()).strip()

    item_by_prompt = {
        _normalize_key(str(item.get("prompt") or "")): dict(item)
        for item in items
        if _normalize_key(str(item.get("prompt") or ""))
    }

    def eval_logger(*, agent_completions_turns, prompts=None, **_kwargs: Any):
        if not agent_completions_turns:
            return []
        prompts = prompts or []
        sample_count = len(agent_completions_turns[0])
        records: List[Dict[str, float]] = []
        for sample_idx in range(sample_count):
            prompt = str(prompts[sample_idx]) if sample_idx < len(prompts) else ""
            item = item_by_prompt.get(_normalize_key(prompt))
            if not item:
                continue
            sample_metrics: Dict[str, float] = {}
            turn_count = 0
            for agent_turns in agent_completions_turns:
                if sample_idx < len(agent_turns):
                    turn_count = max(turn_count, len(agent_turns[sample_idx]))
            for turn_idx in range(turn_count):
                completions: List[List[str]] = []
                for agent_idx in range(num_agents):
                    completion = ""
                    if (
                        agent_idx < len(agent_completions_turns)
                        and sample_idx < len(agent_completions_turns[agent_idx])
                        and turn_idx
                        < len(agent_completions_turns[agent_idx][sample_idx])
                    ):
                        completion = agent_completions_turns[agent_idx][sample_idx][
                            turn_idx
                        ]
                    completions.append([completion])
                result = evaluate_str_builder_outputs(
                    cfg=cfg,
                    num_agents=num_agents,
                    agent_completions=completions,
                    batch_item=item,
                )
                prefix = f"turn_{turn_idx + 1}/"
                for key, value in result.get("log_metrics", {}).items():
                    sample_metrics[prefix + key] = float(value)
            if sample_metrics:
                records.append(sample_metrics)
        return records

    def eval_aggregator(metrics_list: List[Dict[str, float]], num_turns: int = 1):
        del num_turns
        values_by_key: Dict[str, List[float]] = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                values_by_key.setdefault(key, []).append(float(value))
        return {
            key: float(sum(values) / len(values))
            for key, values in values_by_key.items()
            if values
        }

    return {"eval_logger": eval_logger, "eval_aggregator": eval_aggregator}


def _first_batch_item(samples: List[Any]) -> Mapping[str, Any]:
    for sample in samples:
        metadata = getattr(sample, "metadata", {}) or {}
        item = metadata.get("batch_item")
        if isinstance(item, Mapping):
            return item
    return {}


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    """Return a reward function for str_build using coverage and penalty ratios."""
    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_verbose = bool(output_cfg.get("verbose", False))

    debug_enabled = output_verbose
    debug_empty_char = "."
    debug_raw_output = False

    def _block_to_color_initial(block_id: str) -> str:
        key = block_to_color_key(block_id)
        if key == "wood":
            return "W"
        if key == "stone":
            return "S"
        if key == "concrete":
            return "C"
        if key == "obsidian":
            return "O"
        return (key[:1] or "#").upper()

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
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        coverage = float(metrics.get("coverage_ratio", metrics.get("accuracy", 0.0)))
        extra_ratio = float(metrics.get("extra_ratio", 0.0))
        adj_ratio = float(metrics.get("adjacent_same_color_ratio", 0.0))
        prefix = (
            f"[str_build debug] {task.task_id} text={task.text!r}{turn_str} "
            f"reward={reward:.4f} "
            f"cov={coverage:.3f} "
            f"extra_r={extra_ratio:.3f} "
            f"adj_r={adj_ratio:.3f}"
        )
        print(prefix, flush=True)
        print(_render_overlay(task, obs_map), flush=True)
        if debug_raw_output and raw_outputs is not None:
            for idx, raw in enumerate(raw_outputs):
                print(f"[str_build raw] agent{idx}:", flush=True)
                print((raw or "").rstrip(), flush=True)

    if num_agents == 1:
        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
            turn_idx = None
            if isinstance(batch_item, Mapping):
                turn_idx = batch_item.get("_str_build_turn")
            completion = agent1_completions[0] if agent1_completions else ""
            result = evaluate_str_builder_outputs(
                cfg=cfg,
                num_agents=num_agents,
                agent_completions=[[completion]],
                batch_item=batch_item,
            )
            reward = float(result["reward"])
            if debug_enabled:
                obs_map = {tuple(b["pos"]): normalize_block_id(b.get("name") or "air") for b in result["blocks"]}
                _maybe_debug_print(
                    task=task,
                    reward=reward,
                    metrics=result["metrics"],
                    obs_map=obs_map,
                    turn_idx=turn_idx,
                    raw_outputs=[completion],
                )
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
            turn_idx = batch_item.get("_str_build_turn")

        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""
        result = evaluate_str_builder_outputs(
            cfg=cfg,
            num_agents=num_agents,
            agent_completions=[[c1], [c2]],
            batch_item=batch_item,
        )
        reward = float(result["reward"])
        if debug_enabled:
            obs_map = {tuple(b["pos"]): normalize_block_id(b.get("name") or "air") for b in result["blocks"]}
            _maybe_debug_print(
                task=task,
                reward=reward,
                metrics=result["metrics"],
                obs_map=obs_map,
                turn_idx=turn_idx,
                raw_outputs=[c1, c2],
            )
        return [reward]

    return reward_fn
