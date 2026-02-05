from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping

from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    extract_command_lines,
    normalize_block_id,
    score_house_builder,
    simulate_commands_to_scan_blocks,
    unique_block_list,
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




def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _task_from_batch_item(item: Mapping[str, Any]) -> TaskSpec:
    palette_raw = item.get("palette") or {}
    layers_by_y = item.get("layers_by_y") or {}
    if isinstance(layers_by_y, dict):
        layers_by_y = {int(k): list(v) for k, v in layers_by_y.items()}
    return TaskSpec(
        task_id=str(item.get("task_id") or ""),
        local_bbox_from=[_as_int(v, 0) for v in (item.get("local_bbox_from") or [0, 0, 0])],
        local_bbox_to=[_as_int(v, 0) for v in (item.get("local_bbox_to") or [0, 0, 0])],
        palette={str(k): str(v) for k, v in palette_raw.items()},
        layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
    )


def _get_rpg_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = cfg.get("_rpg_state")
    if isinstance(state, dict):
        return state

    rpg_cfg = cfg.get("RPG") or cfg.get("rpg") or {}
    if not isinstance(rpg_cfg, dict):
        rpg_cfg = {}
    player_cfg = rpg_cfg.get("player") or {}
    spider_cfg = rpg_cfg.get("spider") or {}

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


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    max_commands_total = _as_int(task_cfg.get("max_commands", 600), 600)
    limited_resource = bool(task_cfg.get("limited_resource", False))

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

    block_agent1_override = _as_block_list(task_cfg.get("block_agent1"))
    block_agent2_override = _as_block_list(task_cfg.get("block_agent2"))

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_verbose = bool(output_cfg.get("verbose", False))

    debug_cfg = cfg.get("debug") or {}
    if not isinstance(debug_cfg, dict):
        debug_cfg = {}

    debug_enabled = (bool(debug_cfg.get("enabled", False)) or (os.environ.get("HOUSE_BUILDER_DEBUG") == "1")) and output_verbose
    debug_max_prints = _as_int(debug_cfg.get("max_prints", 0), 0)
    if debug_enabled and debug_max_prints <= 0:
        debug_max_prints = 10
    debug_every_n_calls = _as_int(debug_cfg.get("every_n_calls", 0), 0)
    debug_empty_char = str(debug_cfg.get("empty_char") or ".")[:1] or "."
    debug_raw_output = bool(debug_cfg.get("raw_output", False))
    debug_render_layers = bool(debug_cfg.get("render_merged_layers", True))
    debug_state = {"calls": 0, "printed": 0}
    rpg_state = _get_rpg_state(cfg)

    def _allowed_blocks_for_task(task: TaskSpec, overrides: List[str]) -> List[str]:
        if overrides:
            return unique_block_list(overrides)
        return unique_block_list(task.palette.values())

    def _render_layers(task: TaskSpec, obs_map: Mapping[tuple[int, int, int], str]) -> str:
        palette_rev: Dict[str, str] = {}
        for key, value in task.palette.items():
            block_norm = normalize_block_id(value)
            if block_norm and block_norm not in palette_rev:
                palette_rev[block_norm] = str(key)
        air_key = palette_rev.get("air")

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
                    ch = palette_rev.get(block)
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
        debug_state["calls"] += 1
        if debug_state["printed"] >= debug_max_prints:
            return
        if debug_every_n_calls > 0 and (debug_state["calls"] % debug_every_n_calls) != 0:
            return
        debug_state["printed"] += 1
        turn_str = f" turn={int(turn_idx)}" if turn_idx is not None else ""
        print(
            f"[house_builder debug] {task.task_id}{turn_str} "
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
                print(f"[house_builder raw] agent{idx}:", flush=True)
                print((raw or "").rstrip(), flush=True)

    if num_agents == 1:
        max_commands_agent1 = max_commands_total

        def reward_fn(agent1_completions: List[str], *, batch_items: List[Mapping[str, Any]] | None = None) -> List[float]:
            batch_item = (batch_items or [{}])[0]
            task = _task_from_batch_item(batch_item)
            turn_idx = None
            if isinstance(batch_item, Mapping):
                turn_idx = batch_item.get("_house_builder_turn")

            allowed_blocks = _allowed_blocks_for_task(task, block_agent1_override)
            resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else None
            completion = agent1_completions[0] if agent1_completions else ""
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
            _log_train_metrics(
                {
                    "iou": float(metrics.get("iou", 0.0)),
                    "level_1": float(metrics.get("score_match", 0.0)),
                    "level_2": 0.0,
                    "level_total": reward,
                },
                turn_idx=turn_idx,
            )
            if debug_enabled:
                _maybe_debug_print(
                    task=task,
                    reward=reward,
                    metrics=metrics,
                    blocks=blocks,
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
            turn_idx = batch_item.get("_house_builder_turn")

        allowed_blocks_agent1 = _allowed_blocks_for_task(task, block_agent1_override)
        allowed_blocks_agent2 = _allowed_blocks_for_task(task, block_agent2_override)
        resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else None

        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""

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
        metrics = score_house_builder(task=task, world_scan_blocks=blocks)
        reward = float(metrics.get("score_mean", 0.0))
        spider_penalty = 0.0
        if spider_dmg_for_penalty > 0 and player_hp_for_penalty > 0:
            if not _has_kill(accepted_1) and not _has_kill(accepted_2):
                spider_penalty = min(1.0, spider_dmg_for_penalty / player_hp_for_penalty) * 0.1
        reward -= spider_penalty
        _log_train_metrics(
            {
                "iou": float(metrics.get("iou", 0.0)),
                "level_1": float(metrics.get("score_match", 0.0)),
                "level_2": -float(spider_penalty),
                "level_total": reward,
            },
            turn_idx=turn_idx,
        )

        if debug_enabled:
            _maybe_debug_print(
                task=task,
                reward=reward,
                metrics=metrics,
                blocks=blocks,
                turn_idx=turn_idx,
                raw_outputs=[c1, c2],
            )
        return [reward]

    return reward_fn
