from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_Minecraft.house_build.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    score_house_builder,
    simulate_commands_to_scan_blocks,
    unique_block_list,
    validate_and_normalize_mc_commands,
)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


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


def _task_from_batch_item(item: Mapping[str, Any]) -> TaskSpec:
    inventory_raw = item.get("inventory") or {}
    layers_by_y = item.get("layers_by_y") or {}
    if isinstance(layers_by_y, dict):
        layers_by_y = {int(k): list(v) for k, v in layers_by_y.items()}
    return TaskSpec(
        task_id=str(item.get("task_id") or ""),
        local_bbox_from=[_as_int(v, 0) for v in (item.get("local_bbox_from") or [0, 0, 0])],
        local_bbox_to=[_as_int(v, 0) for v in (item.get("local_bbox_to") or [0, 0, 0])],
        inventory={str(k): str(v) for k, v in inventory_raw.items()},
        layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
    )


def _get_rpg_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = cfg.get("_rpg_state")
    if isinstance(state, dict):
        return state

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    player_cfg = task_cfg.get("player") or {}
    if not isinstance(player_cfg, dict):
        player_cfg = {}
    spider_cfg = task_cfg.get("spider") or {}
    if not isinstance(spider_cfg, dict):
        spider_cfg = {}

    player_hp = _as_int(player_cfg.get("hp", 0), 0)
    spider_num = _as_int(spider_cfg.get("num", 0), 0)

    atk_values_raw = spider_cfg.get("atk_values") or spider_cfg.get("atk_list") or spider_cfg.get("atk")
    atk_values: List[float] = []
    if isinstance(atk_values_raw, (list, tuple)):
        for v in atk_values_raw:
            try:
                atk_values.append(float(v))
            except Exception:
                continue
    elif atk_values_raw is not None:
        try:
            atk_val = float(atk_values_raw)
            if spider_num > 0:
                atk_values = [atk_val for _ in range(spider_num)]
            else:
                atk_values = [atk_val]
        except Exception:
            atk_values = []

    total_dmg = float(sum(atk_values))
    return {
        "player_hp": player_hp,
        "spider_num": spider_num,
        "spider_atk_values": atk_values,
        "spider_total_dmg": total_dmg,
    }


def evaluate_house_builder_outputs(
    *,
    cfg: Dict[str, Any],
    num_agents: int,
    agent_completions: List[List[str]],
    batch_item: Mapping[str, Any],
) -> Dict[str, Any]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands", 600), 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))
    block_agent1_override = _as_block_list(task_cfg.get("block_agent1"))
    block_agent2_override = _as_block_list(task_cfg.get("block_agent2"))
    task = _task_from_batch_item(batch_item)

    def _allowed_blocks_for_task(task_obj: TaskSpec, overrides: List[str]) -> List[str]:
        if overrides:
            return unique_block_list(overrides)
        return unique_block_list(task_obj.inventory.values())

    if int(num_agents) == 1:
        max_commands_agent1 = max_commands_total
        completion = agent_completions[0][0] if agent_completions and agent_completions[0] else ""
        allowed_blocks = _allowed_blocks_for_task(task, block_agent1_override)
        resource_limits = (
            compute_resource_limits(task, num_agents=num_agents)
            if limited_resource
            else None
        )
        lines = extract_command_lines(completion)
        accepted, _rejected = validate_and_normalize_mc_commands(
            lines=lines,
            allowed_blocks=allowed_blocks,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
            max_commands=max_commands_agent1,
            resource_limits=resource_limits,
        )
        blocks = simulate_commands_to_scan_blocks(
            commands=accepted,
            world_bbox_from=task.local_bbox_from,
            world_bbox_to=task.local_bbox_to,
        )
        metrics = score_house_builder(task=task, world_scan_blocks=blocks)
        reward = float(metrics.get("score_mean", 0.0))
        return {
            "reward": reward,
            "metrics": metrics,
            "blocks": blocks,
            "log_metrics": {
                "house/iou": float(metrics.get("iou", 0.0)),
                "house/structure_match": float(metrics.get("score_match", 0.0)),
                "house/spider_penalty": 0.0,
                "house/score": reward,
            },
        }

    if int(num_agents) != 2:
        raise ValueError("num_agents must be 1 or 2")

    max_commands_per_agent = max(1, max_commands_total // int(num_agents))
    max_commands_agent1 = max_commands_per_agent + (
        max_commands_total % int(num_agents)
    )
    max_commands_agent2 = max_commands_per_agent
    rpg_state = _get_rpg_state(cfg)
    player_hp_for_penalty = float(rpg_state.get("player_hp", 0) or 0)
    spider_dmg_for_penalty = float(rpg_state.get("spider_total_dmg", 0) or 0)

    def _has_kill(cmds: List[str]) -> bool:
        for cmd in cmds:
            stripped = (cmd or "").strip()
            if stripped.startswith("/"):
                stripped = stripped[1:].lstrip()
            if stripped.lower().startswith("kill"):
                return True
        return False

    c1 = agent_completions[0][0] if agent_completions and agent_completions[0] else ""
    c2 = (
        agent_completions[1][0]
        if len(agent_completions) > 1 and agent_completions[1]
        else ""
    )
    allowed_blocks_agent1 = _allowed_blocks_for_task(task, block_agent1_override)
    allowed_blocks_agent2 = _allowed_blocks_for_task(task, block_agent2_override)
    resource_limits = (
        compute_resource_limits(task, num_agents=num_agents) if limited_resource else None
    )

    accepted_1, _rejected_1 = validate_and_normalize_mc_commands(
        lines=extract_command_lines(c1),
        allowed_blocks=allowed_blocks_agent1,
        world_bbox_from=task.local_bbox_from,
        world_bbox_to=task.local_bbox_to,
        max_commands=max_commands_agent1,
        resource_limits=resource_limits,
    )
    accepted_2, _rejected_2 = validate_and_normalize_mc_commands(
        lines=extract_command_lines(c2),
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
    metrics = score_house_builder(task=task, world_scan_blocks=blocks)
    reward = float(metrics.get("score_mean", 0.0))
    spider_penalty = 0.0
    if spider_dmg_for_penalty > 0 and player_hp_for_penalty > 0:
        if not _has_kill(accepted_1) and not _has_kill(accepted_2):
            spider_penalty = (
                min(1.0, spider_dmg_for_penalty / player_hp_for_penalty) * 0.1
            )
    reward -= spider_penalty
    return {
        "reward": reward,
        "metrics": metrics,
        "blocks": blocks,
        "log_metrics": {
            "house/iou": float(metrics.get("iou", 0.0)),
            "house/structure_match": float(metrics.get("score_match", 0.0)),
            "house/spider_penalty": float(spider_penalty),
            "house/score": reward,
        },
    }


def build_ac_house_metrics_callback(*, cfg: Dict[str, Any], num_agents: int):
    def callback(rollouts: List[Any]) -> Dict[str, float]:
        return aggregate_ac_house_metrics(
            rollouts,
            cfg=cfg,
            num_agents=int(num_agents),
        )

    return callback


def aggregate_ac_house_metrics(
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
        result = evaluate_house_builder_outputs(
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


def _first_batch_item(samples: List[Any]) -> Mapping[str, Any]:
    for sample in samples:
        metadata = getattr(sample, "metadata", {}) or {}
        item = metadata.get("batch_item")
        if isinstance(item, Mapping):
            return item
    return {}


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_verbose = bool(output_cfg.get("verbose", False))

    debug_enabled = output_verbose
    debug_empty_char = "."
    debug_raw_output = False
    debug_render_layers = True

    def _render_layers(task: TaskSpec, obs_map: Mapping[tuple[int, int, int], str]) -> str:
        inventory_rev: Dict[str, str] = {}
        for key, value in task.inventory.items():
            block_norm = normalize_block_id(value)
            if block_norm and block_norm not in inventory_rev:
                inventory_rev[block_norm] = str(key)
        air_key = inventory_rev.get("air")

        min_x = min(task.local_bbox_from[0], task.local_bbox_to[0])
        max_x = max(task.local_bbox_from[0], task.local_bbox_to[0])
        min_y = min(task.local_bbox_from[1], task.local_bbox_to[1])
        max_y = max(task.local_bbox_from[1], task.local_bbox_to[1])
        min_z = min(task.local_bbox_from[2], task.local_bbox_to[2])
        max_z = max(task.local_bbox_from[2], task.local_bbox_to[2])

        lines: List[str] = []
        for y in range(min_y, max_y + 1):
            lines.append(f"y={y}:")
            for z in range(min_z, max_z + 1):
                row: List[str] = []
                for x in range(min_x, max_x + 1):
                    block = normalize_block_id(obs_map.get((x, y, z), "air"))
                    ch = inventory_rev.get(block)
                    if ch is None:
                        if block in ("air", "cave_air", "void_air"):
                            ch = air_key if air_key is not None else debug_empty_char
                        else:
                            ch = "?"
                    row.append(ch)
                lines.append("".join(row))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _maybe_debug_print(
        *,
        task: TaskSpec,
        reward: float,
        metrics: Mapping[str, Any],
        blocks: List[Mapping[str, Any]],
        turn_idx: int | None,
        raw_outputs: List[str] | None,
    ) -> None:
        if not debug_enabled:
            return
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        print(
            f"[house_build debug] {task.task_id}{turn_str} "
            f"reward={reward:.4f} match={float(metrics.get('score_match', 0.0)):.4f}",
            flush=True,
        )
        if debug_render_layers:
            obs_map = {
                (int(b.get("pos")[0]), int(b.get("pos")[1]), int(b.get("pos")[2])): normalize_block_id(b.get("name") or "air")
                for b in blocks
                if isinstance(b.get("pos"), list) and len(b.get("pos")) == 3
            }
            print(_render_layers(task, obs_map), flush=True)
        if debug_raw_output and raw_outputs is not None:
            for idx, raw in enumerate(raw_outputs):
                print(f"[house_build raw] agent{idx}:", flush=True)
                print((raw or "").rstrip(), flush=True)

    if num_agents == 1:
        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
            turn_idx = None
            if isinstance(batch_item, Mapping):
                turn_idx = batch_item.get("_house_build_turn")
            completion = agent1_completions[0] if agent1_completions else ""
            result = evaluate_house_builder_outputs(
                cfg=cfg,
                num_agents=num_agents,
                agent_completions=[[completion]],
                batch_item=batch_item,
            )
            reward = float(result["reward"])
            if debug_enabled:
                _maybe_debug_print(
                    task=task,
                    reward=reward,
                    metrics=result["metrics"],
                    blocks=result["blocks"],
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
            turn_idx = batch_item.get("_house_build_turn")

        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""
        result = evaluate_house_builder_outputs(
            cfg=cfg,
            num_agents=num_agents,
            agent_completions=[[c1], [c2]],
            batch_item=batch_item,
        )
        reward = float(result["reward"])

        if debug_enabled:
            _maybe_debug_print(
                task=task,
                reward=reward,
                metrics=result["metrics"],
                blocks=result["blocks"],
                turn_idx=turn_idx,
                raw_outputs=[c1, c2],
            )
        return [reward]

    return reward_fn
