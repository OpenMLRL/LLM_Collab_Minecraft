from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Any, Dict, List, Mapping

import torch
from transformers import AutoTokenizer  # type: ignore

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(REPO_ROOT))

from LLM_Collab_Minecraft.resource_gathering.utils.config import apply_overrides, load_yaml, resolve_path  # noqa: E402
from LLM_Collab_Minecraft.resource_gathering.utils.prompting import apply_prompt_defaults  # noqa: E402
from LLM_Collab_Minecraft.resource_gathering.utils.resource_gathering import (  # noqa: E402
    load_tasks_from_json,
    make_initial_state,
    serialize_state,
    task_to_item,
)
from LLM_Collab_Minecraft.resource_gathering_meta.train.train_bcmaac import (  # noqa: E402
    _build_reward_config,
    _build_reward_processor,
    _map_dtype,
    _pick_devices,
    _prepare_prompt_context,
    _set_seed,
    _slice_items,
)
from LLM_Collab_Minecraft.resource_gathering_baseline.adapters import ResourceGatheringAdapter  # noqa: E402
from LLM_Collab_Minecraft.resource_gathering_baseline.trainers import (  # noqa: E402
    ResourceGatheringBaselineConfig,
    ResourceGatheringBaselineTrainer,
)


def _build_baseline_config(cfg: Dict[str, Any]) -> ResourceGatheringBaselineConfig:
    trainer_cfg = cfg.get("bcmaac") or cfg.get("maac") or {}
    baseline_cfg = cfg.get("baseline") or {}
    output_cfg = cfg.get("output") or {}
    if not isinstance(trainer_cfg, dict):
        trainer_cfg = {}
    if not isinstance(baseline_cfg, dict):
        baseline_cfg = {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    save_best_model = bool(output_cfg.get("save_best_model", False))
    best_model_dir = None
    if save_best_model:
        best_model_path = output_cfg.get("best_model_path")
        best_model_dir = os.path.abspath(str(best_model_path)) if best_model_path else os.path.join(
            os.path.abspath(str(output_cfg.get("base_dir") or ".")),
            "best_model",
        )
    return ResourceGatheringBaselineConfig(
        agent_learning_rate=float(trainer_cfg.get("agent_learning_rate", 2.5e-6)),
        critic_learning_rate=float(trainer_cfg.get("critic_learning_rate", 2.5e-6)),
        context_learning_rate=float(baseline_cfg.get("context_learning_rate", 3.0e-5)),
        rollout_buffer_size=int(trainer_cfg.get("rollout_buffer_size", 8)),
        train_batch_size=int(trainer_cfg.get("train_batch_size", 4)),
        value_loss_coef=float(trainer_cfg.get("value_loss_coef", 0.6)),
        task_loss_coef=0.0,
        entropy_coef=float(baseline_cfg.get("entropy_coef", 0.01)),
        max_grad_norm=float(baseline_cfg.get("max_grad_norm", 1.0)) if baseline_cfg.get("max_grad_norm", 1.0) is not None else None,
        max_new_tokens=int(trainer_cfg.get("max_new_tokens", 160)),
        temperature=float((cfg.get("agent_model") or {}).get("temperature", 0.6)),
        top_p=float((cfg.get("agent_model") or {}).get("top_p", 0.6)),
        top_k=None if (cfg.get("agent_model") or {}).get("top_k") in (None, "none", "null") else int((cfg.get("agent_model") or {}).get("top_k")),
        num_train_epochs=int(trainer_cfg.get("num_train_epochs", 150)),
        num_agents=int(trainer_cfg.get("num_agents", 2)),
        num_turns=int(trainer_cfg.get("num_turns", 4)),
        discount=float(trainer_cfg.get("discount", 0.9)),
        critic_type=str(trainer_cfg.get("critic_type", "v")).strip().lower(),
        critic_backbone=str(baseline_cfg.get("critic_backbone", "structured")).strip().lower(),
        logging_steps=int(trainer_cfg.get("logging_steps", 20)),
        eval_interval=int(trainer_cfg.get("eval_interval", 10)),
        eval_num_samples=int(trainer_cfg.get("eval_num_samples", 2)),
        eval_batch_size=int(trainer_cfg.get("eval_batch_size", 1)),
        context_hidden_dim=int(baseline_cfg.get("context_hidden_dim", 128)),
        context_cnn_channels=int(baseline_cfg.get("context_cnn_channels", 32)),
        context_scalar_hidden_dim=int(baseline_cfg.get("context_scalar_hidden_dim", 64)),
        value_head_hidden_dim=int(baseline_cfg.get("value_head_hidden_dim")) if baseline_cfg.get("value_head_hidden_dim") is not None else None,
        actor_condition_dim=int(baseline_cfg.get("actor_condition_dim")) if baseline_cfg.get("actor_condition_dim") is not None else None,
        critic_condition_dim=int(baseline_cfg.get("critic_condition_dim")) if baseline_cfg.get("critic_condition_dim") is not None else None,
        actor_prompt_context_scale=0.0,
        actor_response_context_scale=0.0,
        preference_loss_coef=float(baseline_cfg.get("preference_loss_coef", 0.1)),
        comm_preference_loss_scale=float(baseline_cfg.get("comm_preference_loss_scale", 0.35)),
        probe_preference_loss_scale=float(baseline_cfg.get("probe_preference_loss_scale", 0.0)),
        cmds_preference_loss_scale=float(baseline_cfg.get("cmds_preference_loss_scale", 0.0)),
        path_preference_loss_scale=float(baseline_cfg.get("path_preference_loss_scale", 1.0)),
        score_chunk_size=int(baseline_cfg.get("score_chunk_size", 2)),
        actor_gradient_checkpointing=bool(baseline_cfg.get("actor_gradient_checkpointing", True)),
        best_model_metric=str(output_cfg.get("best_metric", "eval/avg_return")).strip() if save_best_model else None,
        best_model_mode=str(output_cfg.get("best_metric_mode", "max")).strip().lower(),
        best_model_dir=best_model_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train resource_gathering baseline without meta latent.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "resource_gathering_baseline", "configs", "resource_gathering_baseline_data_dev.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="key.path=value overrides",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.override:
        cfg = apply_overrides(cfg, [item for item in args.override if item])
    apply_prompt_defaults(cfg)

    trainer_args = _build_baseline_config(cfg)
    _set_seed(int(cfg.get("seed", 42)))

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(args.config, dataset_cfg.get("json_path"))
    tasks = load_tasks_from_json(json_path)
    turn_limit = int((cfg.get("task") or {}).get("max_turns", trainer_args.num_turns))

    items: List[Dict[str, Any]] = []
    for task in tasks:
        item = task_to_item(task)
        item["max_turns"] = int(turn_limit)
        item["_resource_state_before_turn"] = serialize_state(
            make_initial_state(task, num_agents=trainer_args.num_agents, max_turns=turn_limit)
        )
        items.append(item)

    train_items = _slice_items(items, dataset_cfg.get("train_split", "[:]"))
    eval_items = _slice_items(items, dataset_cfg.get("eval_split")) if dataset_cfg.get("eval_split") else []

    model_cfg = cfg.get("agent_model") or {}
    critic_model_cfg = cfg.get("critic_model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not isinstance(critic_model_cfg, dict):
        critic_model_cfg = {}
    model_name = str(model_cfg.get("name") or "")
    agent_names = cfg.get("agents")
    critic_name = str(critic_model_cfg.get("name") or "") or model_name
    critic_names = cfg.get("critics")
    if not model_name and not agent_names:
        raise ValueError("agent_model.name or agents must be provided")
    tokenizer_source = model_name or (agent_names[0] if isinstance(agent_names, (list, tuple)) and agent_names else "")
    if not tokenizer_source:
        raise ValueError("Failed to resolve tokenizer source for resource_gathering_baseline")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter = ResourceGatheringAdapter(
        prompt_ctx=_prepare_prompt_context(cfg, num_agents=trainer_args.num_agents),
        num_agents=trainer_args.num_agents,
        task_ids=[str(item["task_id"]) for item in items],
        task_specs=tasks,
        tokenizer=tokenizer,
        external_mode=str((cfg.get("external") or {}).get("mode", "empty_feedback")),
        original_prompt=bool((cfg.get("external") or {}).get("original_prompt", True)),
        previous_response=bool((cfg.get("external") or {}).get("previous_response", False)),
        debug=bool(cfg.get("debug", False)),
        reward_config=_build_reward_config(cfg),
    )

    model_kwargs: Dict[str, Any] = {}
    critic_model_kwargs: Dict[str, Any] = {}
    dtype = _map_dtype(model_cfg.get("dtype") or model_cfg.get("torch_dtype"))
    critic_dtype = _map_dtype(critic_model_cfg.get("dtype") or critic_model_cfg.get("torch_dtype"))
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if critic_dtype is not None:
        critic_model_kwargs["torch_dtype"] = critic_dtype

    trainer = ResourceGatheringBaselineTrainer(
        agent_model=model_name or None,
        critic_model=critic_name or None,
        agents=[str(x) for x in agent_names] if isinstance(agent_names, (list, tuple)) else None,
        critics=[str(x) for x in critic_names] if isinstance(critic_names, (list, tuple)) else None,
        adapter=adapter,
        train_items=train_items,
        eval_items=eval_items,
        args=trainer_args,
        model_config={"model_kwargs": model_kwargs, "critic_model_kwargs": critic_model_kwargs},
        reward_processor=_build_reward_processor(cfg),
        wandb_config=(cfg.get("wandb") if isinstance(cfg.get("wandb"), dict) and bool((cfg.get("wandb") or {}).get("enabled", True)) else None),
        agent_devices=_pick_devices(cfg.get("bcmaac") or cfg.get("maac") or {}, "agent_devices"),
        critic_devices=_pick_devices(cfg.get("bcmaac") or cfg.get("maac") or {}, "critic_devices"),
        verbose=bool((cfg.get("output") or {}).get("verbose", True)),
    )
    trainer.train()

    output_cfg = cfg.get("output") or {}
    if isinstance(output_cfg, dict) and bool(output_cfg.get("save_final_model", False)):
        save_path_cfg = output_cfg.get("save_path")
        if save_path_cfg:
            save_path = str(save_path_cfg)
        else:
            save_path = os.path.join(os.path.abspath(str(output_cfg.get("base_dir") or ".")), "final_model")
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
