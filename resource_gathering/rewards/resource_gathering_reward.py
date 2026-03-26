from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from LLM_Collab_Minecraft.resource_gathering.utils.resource_gathering import (
    build_payload,
    deserialize_state,
    make_initial_state,
    task_from_item,
    transition_payload,
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


def _build_reward_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    task_cfg = cfg.get("task") or {}
    reward_cfg = cfg.get("reward_shaping") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    return {
        "path_slots": int(task_cfg.get("max_path_len", 4)),
        "comm_limit": int(task_cfg.get("comm_limit", 1)),
        "progress_reward_scale": float(reward_cfg.get("progress_reward_scale", 10.0)),
        "terminal_bonus": float(reward_cfg.get("terminal_bonus", 4.0)),
        "move_cost_scale": float(reward_cfg.get("move_cost_scale", 0.0)),
        "comm_cost_scale": float(reward_cfg.get("comm_cost_scale", 0.0)),
        "move_to_zone_bonus_scale": float(reward_cfg.get("move_to_zone_bonus_scale", 0.05)),
        "useful_comm_bonus_scale": float(reward_cfg.get("useful_comm_bonus_scale", 0.1)),
        "first_enter_zone_bonus_scale": float(reward_cfg.get("first_enter_zone_bonus_scale", 0.15)),
    }


def _state_from_batch_item(item: Mapping[str, Any], *, num_agents: int, task_max_turns: int):
    raw = item.get("_resource_state_before_turn")
    if isinstance(raw, Mapping):
        try:
            state = deserialize_state(raw, num_agents=num_agents)
            state.max_turns = max(1, int(task_max_turns))
            return state
        except Exception:
            return None
    return None


def get_reward_function(*, cfg: Dict[str, Any], num_agents: int) -> Callable[..., List[float]]:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    view = max(0, int(task_cfg.get("view", 2)))
    extraction_limit = max(0, int(task_cfg.get("extraction_limit", 2)))
    extraction_range = max(0, int(task_cfg.get("extraction_range", 2)))
    max_path_len = max(1, int(task_cfg.get("max_path_len", 4)))
    reward_config = _build_reward_config(cfg)
    debug_enabled = bool(cfg.get("debug", False))

    def _compute(
        *,
        completions: Sequence[str],
        batch_items: List[Mapping[str, Any]] | None,
    ) -> List[float]:
        batch_item = (batch_items or [{}])[0]
        task = task_from_item(batch_item)
        n_agents = max(1, int(num_agents))

        state = _state_from_batch_item(batch_item, num_agents=n_agents, task_max_turns=task.max_turns)
        if state is None:
            state = make_initial_state(task, num_agents=n_agents, max_turns=task.max_turns)

        payload = build_payload(
            task=task,
            state_before_turn=state,
            num_agents=n_agents,
            view=view,
            extraction_limit=extraction_limit,
            extraction_range=extraction_range,
            max_path_len=max_path_len,
            system_prompt="",
            user_template_single="",
            user_template_agent1="",
            user_template_agent2="",
            reward_config=reward_config,
        )
        next_payload, metrics, _actions = transition_payload(
            payload=payload,
            agent_completions=[str(x) for x in completions],
            num_agents=n_agents,
        )
        reward = float(metrics.get("reward", 0.0))
        turn_idx = int(metrics.get("turn_index") or state.turn_index)

        _log_train_metrics(
            {
                "progress_score": float(metrics.get("progress_score", 0.0)),
                "bonus_progress": float(metrics.get("bonus_progress", 0.0)),
                "bonus_terminal_complete": float(metrics.get("bonus_terminal_complete", 0.0)),
                "bonus_move_to_zone": float(metrics.get("bonus_move_to_zone", 0.0)),
                "bonus_useful_comm": float(metrics.get("bonus_useful_comm", 0.0)),
                "bonus_first_enter_zone": float(metrics.get("bonus_first_enter_zone", 0.0)),
                "delta_wood": float(metrics.get("delta_wood", 0.0)),
                "delta_stone": float(metrics.get("delta_stone", 0.0)),
                "delta_iron": float(metrics.get("delta_iron", 0.0)),
                "success": 1.0 if bool(metrics.get("success", metrics.get("completed", False))) else 0.0,
                "reward_total": reward,
            },
            turn_idx=turn_idx,
        )

        if debug_enabled:
            next_state = next_payload.get("state_before_turn") if isinstance(next_payload, Mapping) else {}
            print(
                "[resource_gathering reward] "
                f"task={getattr(task, 'task_id', 'unknown')} "
                f"turn={turn_idx} reward={reward:.3f} "
                f"progress={float(metrics.get('progress_score', 0.0)):.3f} "
                f"delta=({float(metrics.get('delta_wood', 0.0)):.0f},"
                f"{float(metrics.get('delta_stone', 0.0)):.0f},"
                f"{float(metrics.get('delta_iron', 0.0)):.0f}) "
                f"success={bool(metrics.get('success', metrics.get('completed', False)))} "
                f"next_turn={int((next_state or {}).get('turn_index', turn_idx + 1))}",
                flush=True,
            )

        return [reward]

    if num_agents == 1:

        def reward_fn(
            agent1_completions: List[str],
            *,
            batch_items: List[Mapping[str, Any]] | None = None,
            prompts: List[str] | None = None,
        ) -> List[float]:
            del prompts
            c1 = agent1_completions[0] if agent1_completions else ""
            return _compute(completions=[c1], batch_items=batch_items)

        return reward_fn

    if num_agents != 2:
        raise ValueError("num_agents must be 1 or 2")

    def reward_fn(
        agent1_completions: List[str],
        agent2_completions: List[str],
        *,
        batch_items: List[Mapping[str, Any]] | None = None,
        prompts: List[str] | None = None,
    ) -> List[float]:
        del prompts
        c1 = agent1_completions[0] if agent1_completions else ""
        c2 = agent2_completions[0] if agent2_completions else ""
        return _compute(completions=[c1, c2], batch_items=batch_items)

    return reward_fn
