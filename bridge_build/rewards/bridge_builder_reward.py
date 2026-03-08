from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    BridgeState,
    apply_turn,
    deserialize_state,
    make_initial_state,
    task_from_item,
)
from LLM_Collab_Minecraft.bridge_build.utils.debug import print_turn_debug


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


def _as_block_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    return [s] if s else []


def _state_from_batch_item(item: Mapping[str, Any], *, num_agents: int, task_max_turns: int) -> BridgeState | None:
    raw = item.get("_bridge_state_before_turn")
    if isinstance(raw, BridgeState):
        return raw
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

    max_commands_total = _as_int(task_cfg.get("max_commands", 40), 40)
    max_probe = max(0, min(2, _as_int(task_cfg.get("max_probe", 2), 2)))
    view = max(0, _as_int(task_cfg.get("view", 3), 3))

    allowed_blocks_agent1 = _as_block_list(task_cfg.get("block_agent1"))
    if not allowed_blocks_agent1:
        raise ValueError("task.block_agent1 must be provided and non-empty")

    allowed_blocks_agent2 = _as_block_list(task_cfg.get("block_agent2"))
    if num_agents >= 2 and not allowed_blocks_agent2:
        raise ValueError("task.block_agent2 must be provided when num_agents >= 2")

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
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

        allowed = [allowed_blocks_agent1]
        if n_agents >= 2:
            allowed.append(allowed_blocks_agent2)
        while len(allowed) < n_agents:
            allowed.append(list(allowed_blocks_agent1))

        result = apply_turn(
            task=task,
            state=state,
            agent_outputs=[str(x) for x in completions],
            allowed_blocks_per_agent=allowed,
            max_commands_total=max_commands_total,
            view=view,
            max_probe=max_probe,
        )

        metrics = result.metrics
        reward = float(metrics.get("reward", 0.0))
        turn_idx = int(metrics.get("turn_index") or state.turn_index)

        _log_train_metrics(
            {
                "connected": 1.0 if bool(metrics.get("connected", False)) else 0.0,
                "bonus_y_connected": float(metrics.get("bonus_y_connected", 0.0)),
                "penalty_n_adjacent": -float(metrics.get("penalty_n_adjacent", 0.0)),
                "penalty_block_cost": -float(metrics.get("penalty_block_cost", 0.0)),
                "bonus_terminal_connect": float(metrics.get("bonus_terminal_connect", 0.0)),
                "new_connected_y_count": float(metrics.get("new_connected_y_count", 0.0)),
                "new_adjacent_n_count": float(metrics.get("new_adjacent_n_count", 0.0)),
                "newly_placed_block_count": float(metrics.get("newly_placed_block_count", 0.0)),
                "level_total": reward,
            },
            turn_idx=turn_idx,
        )

        if debug_enabled:
            print_turn_debug(
                task=task,
                state=result.state,
                turn_idx=turn_idx,
                reward=reward,
                metrics=metrics,
                agent_outputs=list(completions),
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
