from __future__ import annotations

from typing import Any, Dict

from coti.api.domain import CoTIDomainSpec

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    load_tasks_from_json,
    make_initial_state,
    serialize_state,
    task_to_item,
)
from LLM_Collab_Minecraft.bridge_build.utils.prompting import apply_prompt_defaults

from .adapter import BridgeBuildAdapter
from .metric_keys import BRIDGE_BUILD_ENV_METRIC_KEYS
from .prompting import build_prompt_context
from .reward import build_reward_config


def resolve_turn_limit(cfg: Dict[str, Any], trainer_args: Any) -> int:
    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_turns_override = task_cfg.get("max_turns")
    return int(max_turns_override) if max_turns_override is not None else int(trainer_args.num_turns)


def get_domain_spec() -> CoTIDomainSpec:
    return CoTIDomainSpec(
        name="bridge_build",
        load_tasks_from_json=load_tasks_from_json,
        task_to_item=task_to_item,
        make_initial_state=make_initial_state,
        serialize_state=serialize_state,
        adapter_cls=BridgeBuildAdapter,
        state_item_key="_bridge_state_before_turn",
        apply_prompt_defaults=apply_prompt_defaults,
        build_prompt_context=build_prompt_context,
        build_reward_config=build_reward_config,
        resolve_turn_limit=resolve_turn_limit,
        env_metric_keys=BRIDGE_BUILD_ENV_METRIC_KEYS,
    )


__all__ = ["get_domain_spec", "resolve_turn_limit"]
