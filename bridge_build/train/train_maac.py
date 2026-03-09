from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Mapping

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"PyYAML is required. Install pyyaml. Error: {e}")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(REPO_ROOT))
COMLRL_ROOT = os.path.join(os.path.dirname(REPO_ROOT), "CoMLRL")
if COMLRL_ROOT not in sys.path:
    sys.path.insert(0, COMLRL_ROOT)

from datasets import Dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch  # type: ignore

from comlrl.trainers.actor_critic import MAACTrainer  # type: ignore
from comlrl.utils.reward_processor import RewardProcessors  # type: ignore

from LLM_Collab_Minecraft.bridge_build.external import (
    get_external_transition as external_get_transition,
    set_context_resolver as external_set_context_resolver,
)
from LLM_Collab_Minecraft.bridge_build.rewards.bridge_builder_reward import get_reward_function
from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    build_payload,
    deserialize_state,
    make_initial_state,
    render_agent_user_prompt,
    serialize_state,
    split_command_limits,
    task_from_item,
    task_to_item,
    transition_payload,
    load_tasks_from_json,
)
from LLM_Collab_Minecraft.bridge_build.utils.config import apply_overrides, load_yaml, resolve_path
from LLM_Collab_Minecraft.bridge_build.utils.prompting import apply_prompt_defaults
from LLM_Collab_Minecraft.bridge_build.utils.trainer_args import (
    get_maac_args,
    get_agent_sampling_config,
)


def _slice_items(items: List[Dict[str, Any]], split_expr: Any) -> List[Dict[str, Any]]:
    if not split_expr:
        return items
    s = str(split_expr).strip()
    if not s:
        return items
    m = re.search(r"\[\s*(?P<start>-?[^:\]]*)\s*:\s*(?P<end>-?[^\]]*)\s*\]", s)
    if not m and ":" in s:
        m = re.match(r"\s*(?P<start>-?[^:]*)\s*:\s*(?P<end>-?.*)\s*$", s)
    if not m:
        return items
    start_raw = (m.group("start") or "").strip()
    end_raw = (m.group("end") or "").strip()
    total = len(items)

    def _parse_index(raw: str):
        if raw in ("", "+"):
            return None
        if raw.endswith("%"):
            try:
                pct = float(raw[:-1].strip())
            except ValueError:
                return None
            return int(total * pct / 100.0)
        try:
            return int(raw)
        except ValueError:
            try:
                frac = float(raw)
            except ValueError:
                return None
            if 0 <= frac <= 1:
                return int(total * frac)
            return None

    start = _parse_index(start_raw)
    end = _parse_index(end_raw)
    return items[slice(start, end)]


def _map_dtype(dtype_cfg: Any) -> Any:
    if isinstance(dtype_cfg, torch.dtype):
        return dtype_cfg
    if not isinstance(dtype_cfg, str):
        return None
    s = dtype_cfg.strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    if s == "auto":
        return "auto"
    return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _render_prompt(
    *,
    tokenizer: Any | None,
    system_prompt: str,
    user_prompt: str,
    use_chat_template: bool,
) -> str:
    system_prompt = (system_prompt or "").rstrip()
    user_prompt = (user_prompt or "").rstrip()
    if (
        use_chat_template
        and tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    if system_prompt:
        return system_prompt + "\n\n" + user_prompt
    return user_prompt


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


def _normalize_key(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _prepare_prompt_context(cfg: Dict[str, Any], *, num_agents: int) -> Dict[str, Any]:
    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    agent1_blocks = _as_block_list(task_cfg.get("block_agent1"))
    if not agent1_blocks:
        raise ValueError("task.block_agent1 must be provided and non-empty")

    agent2_blocks = _as_block_list(task_cfg.get("block_agent2"))
    if num_agents > 1 and not agent2_blocks:
        raise ValueError("task.block_agent2 must be provided when num_agents > 1")
    if not agent2_blocks:
        agent2_blocks = list(agent1_blocks)

    return {
        "use_chat_template": bool(prompt_cfg.get("use_chat_template", False)),
        "system_prompt": str(prompt_cfg.get("system") or "").rstrip(),
        "user_template_single": str(prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent1": str(prompt_cfg.get("user_template_agent1") or prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent2": str(prompt_cfg.get("user_template_agent2") or prompt_cfg.get("user_template") or "").rstrip(),
        "view": max(0, int(task_cfg.get("view", 3))),
        "max_probe": max(0, min(2, int(task_cfg.get("max_probe", 2)))),
        "max_commands_total": max(1, int(task_cfg.get("max_commands", 40))),
        "agent1_blocks": agent1_blocks,
        "agent2_blocks": agent2_blocks,
    }


def _build_formatters(
    *,
    prompt_ctx: Dict[str, Any],
    num_agents: int,
    tokenizer: Any | None = None,
) -> List[Any]:
    n = max(1, int(num_agents))
    limits = split_command_limits(
        max_commands_total=int(prompt_ctx["max_commands_total"]),
        num_agents=n,
    )

    def _allowed_for(idx: int) -> List[str]:
        if idx == 0:
            return list(prompt_ctx["agent1_blocks"])
        return list(prompt_ctx["agent2_blocks"])

    def _template_for(idx: int) -> str:
        if n == 1:
            return str(prompt_ctx["user_template_single"])
        if idx == 0:
            return str(prompt_ctx["user_template_agent1"])
        return str(prompt_ctx["user_template_agent2"])

    def _render_item(item: Dict[str, Any], idx: int) -> str:
        task = task_from_item(item)
        raw_state = item.get("_bridge_state_before_turn")
        if isinstance(raw_state, Mapping):
            state = deserialize_state(raw_state, num_agents=n)
        else:
            state = make_initial_state(task, num_agents=n, max_turns=task.max_turns)

        user_prompt = render_agent_user_prompt(
            task=task,
            state=state,
            agent_idx=idx,
            view=int(prompt_ctx["view"]),
            allowed_blocks=_allowed_for(idx),
            max_probe=int(prompt_ctx["max_probe"]),
            max_commands=int(limits[idx]),
            user_template=_template_for(idx),
            feedback="",
        )
        return _render_prompt(
            tokenizer=tokenizer,
            system_prompt=str(prompt_ctx["system_prompt"]),
            user_prompt=user_prompt,
            use_chat_template=bool(prompt_ctx["use_chat_template"]),
        )

    def _formatter(item: Dict[str, Any], *, idx: int, external_prompts: Any = None) -> str:
        if external_prompts is not None:
            return str(external_prompts)
        return _render_item(item, idx)

    if n == 1:
        return [lambda item, external_prompts=None, idx=0: _formatter(item, idx=idx, external_prompts=external_prompts)]
    return [
        lambda item, external_prompts=None, idx=0: _formatter(item, idx=idx, external_prompts=external_prompts),
        lambda item, external_prompts=None, idx=1: _formatter(item, idx=idx, external_prompts=external_prompts),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BridgeBuild with MAAC (CoMLRL MAACTrainer)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "bridge_build", "configs", "bridge_build_maac_config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="key.path=value overrides (space or comma-separated)",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    override_items: List[str] = []
    if args.override:
        for item in args.override:
            if item is None:
                continue
            part = str(item).strip()
            if part:
                override_items.append(part)
    if override_items:
        cfg = apply_overrides(cfg, override_items)
    apply_prompt_defaults(cfg)

    maac_cfg = cfg.get("maac") or {}
    if not isinstance(maac_cfg, dict):
        maac_cfg = {}
    seed_value = int(cfg.get("seed", maac_cfg.get("seed", 42)))
    _set_seed(seed_value)

    num_agents = int(maac_cfg.get("num_agents") or 1)
    if num_agents not in (1, 2):
        raise ValueError("maac.num_agents must be 1 or 2")
    configured_num_turns = max(1, int(maac_cfg.get("num_turns") or 1))

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(args.config, dataset_cfg.get("json_path"))

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    max_turns_override_raw = task_cfg.get("max_turns")
    max_turns_override = int(max_turns_override_raw) if max_turns_override_raw is not None else None
    turn_limit = int(max_turns_override) if max_turns_override is not None else int(configured_num_turns)

    tasks = load_tasks_from_json(json_path)
    items: List[Dict[str, Any]] = []
    for t in tasks:
        task_item = task_to_item(t)
        task_item["max_turns"] = int(turn_limit)
        initial_state = make_initial_state(
            t,
            num_agents=num_agents,
            max_turns=turn_limit,
        )
        task_item["_bridge_state_before_turn"] = serialize_state(initial_state)
        task_item["prompt"] = f"bridge_build:{t.task_id}"
        items.append(task_item)

    train_split = dataset_cfg.get("train_split", "[:]")
    eval_split = dataset_cfg.get("eval_split")
    train_items = _slice_items(items, train_split)
    eval_items = _slice_items(items, eval_split) if eval_split else []
    train_ds = Dataset.from_list(train_items)
    eval_ds = Dataset.from_list(eval_items) if eval_items else None

    model_cfg = cfg.get("agent_model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    critic_model_cfg = cfg.get("critic_model") or {}
    if not isinstance(critic_model_cfg, dict):
        critic_model_cfg = {}

    model_name = str(model_cfg.get("name") or "")
    agent_names = cfg.get("agents")
    if not model_name and not agent_names:
        raise ValueError("agent_model.name or agents is required")
    if agent_names is not None:
        if not isinstance(agent_names, (list, tuple)) or not all(isinstance(x, str) for x in agent_names):
            raise ValueError("agents must be a list of model names.")
        agent_names = [str(x) for x in agent_names]

    critic_names = None
    critics_field = cfg.get("critics")
    if critics_field is not None:
        if not isinstance(critics_field, (list, tuple)) or not all(isinstance(x, str) for x in critics_field):
            raise ValueError("critics must be a list of model names.")
        critic_names = [str(x) for x in critics_field]

    model_kwargs: Dict[str, Any] = {}
    dtype = _map_dtype(model_cfg.get("dtype") or model_cfg.get("torch_dtype"))
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    tokenizer_source = agent_names[0] if agent_names else model_name
    if not tokenizer_source:
        raise ValueError("agent_model.name or agents must be provided.")
    if agent_names:
        tokenizers = [AutoTokenizer.from_pretrained(name) for name in agent_names]
    else:
        tokenizers = [AutoTokenizer.from_pretrained(tokenizer_source)]
    for tok in tokenizers:
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    tokenizer = tokenizers[0]

    sampling_cfg = get_agent_sampling_config(cfg)
    maac_args = get_maac_args(cfg, sampling_cfg=sampling_cfg)

    prompt_ctx = _prepare_prompt_context(cfg, num_agents=num_agents)
    formatters = _build_formatters(prompt_ctx=prompt_ctx, num_agents=num_agents, tokenizer=tokenizer)

    dataset_prompt_map: Dict[str, Dict[str, Any]] = {}
    dataset_payload_map: Dict[str, Dict[str, Any]] = {}

    critic_model_kwargs: Dict[str, Any] = {}
    if isinstance(critic_model_cfg, dict):
        critic_dtype = _map_dtype(critic_model_cfg.get("dtype") or critic_model_cfg.get("torch_dtype"))
        if critic_dtype is not None:
            critic_model_kwargs["torch_dtype"] = critic_dtype

    def _payload_from_item(item: Mapping[str, Any]) -> Dict[str, Any]:
        task = task_from_item(item)
        raw_state = item.get("_bridge_state_before_turn")
        if isinstance(raw_state, Mapping):
            state = deserialize_state(raw_state, num_agents=num_agents)
        else:
            state = make_initial_state(task, num_agents=num_agents, max_turns=task.max_turns)
        return build_payload(
            task=task,
            state_before_turn=state,
            num_agents=num_agents,
            view=int(prompt_ctx["view"]),
            max_probe=int(prompt_ctx["max_probe"]),
            max_commands_total=int(prompt_ctx["max_commands_total"]),
            allowed_blocks_agent1=list(prompt_ctx["agent1_blocks"]),
            allowed_blocks_agent2=list(prompt_ctx["agent2_blocks"]),
            system_prompt=str(prompt_ctx["system_prompt"]),
            user_template_single=str(prompt_ctx["user_template_single"]),
            user_template_agent1=str(prompt_ctx["user_template_agent1"]),
            user_template_agent2=str(prompt_ctx["user_template_agent2"]),
        )

    def _dataset_key_from_item(item: Mapping[str, Any]) -> str:
        return _normalize_key(str(item.get("prompt") or ""))

    def _register_dataset_prompts(items_list: List[Dict[str, Any]], turn_idx: int) -> None:
        del turn_idx
        for item in items_list:
            ds_key = _dataset_key_from_item(item)
            if ds_key and ds_key not in dataset_prompt_map:
                dataset_prompt_map[ds_key] = dict(item)

            payload = _payload_from_item(item)
            if ds_key:
                dataset_payload_map[ds_key] = copy.deepcopy(payload)

    _register_dataset_prompts(train_items, 1)
    if eval_items:
        _register_dataset_prompts(eval_items, 1)

    def _reward_batch_item(
        *,
        batch_items: List[Mapping[str, Any]] | None,
    ) -> Dict[str, Any]:
        if not batch_items:
            raise KeyError("bridge_build reward_func requires stable batch_items; prompt-based fallback has been removed")
        return dict(batch_items[0] or {})

    def _require_dataset_item(dataset_key: str) -> Dict[str, Any]:
        ds_key = _normalize_key(dataset_key)
        if not ds_key:
            raise KeyError("Missing stable bridge_build dataset key")
        base_item = dataset_prompt_map.get(ds_key)
        if base_item is None:
            raise KeyError(f"Failed to resolve stable bridge_build dataset key: {ds_key}")
        return dict(base_item)

    reward_base = get_reward_function(cfg=cfg, num_agents=num_agents)
    if num_agents == 1:

        def reward_func(
            prompts: List[str],
            agent1_completions: List[str],
            *,
            batch_items: List[Mapping[str, Any]] | None = None,
        ) -> List[float]:
            batch_item = _reward_batch_item(batch_items=batch_items)
            return reward_base(agent1_completions, batch_items=[batch_item], prompts=prompts)

    else:

        def reward_func(
            prompts: List[str],
            agent1_completions: List[str],
            agent2_completions: List[str],
            *,
            batch_items: List[Mapping[str, Any]] | None = None,
        ) -> List[float]:
            batch_item = _reward_batch_item(batch_items=batch_items)
            return reward_base(
                agent1_completions,
                agent2_completions,
                batch_items=[batch_item],
                prompts=prompts,
            )

    reward_processor = None
    rp_cfg = cfg.get("reward_processor") or {}
    if isinstance(rp_cfg, dict) and rp_cfg.get("enabled", False):
        scale = rp_cfg.get("scale_factor")
        shift = rp_cfg.get("shift")
        if scale is not None:
            reward_processor = RewardProcessors.scale(factor=float(scale))
        if shift is not None:
            shift_proc = RewardProcessors.shift(value=float(shift))
            if reward_processor is None:
                reward_processor = shift_proc
            else:
                prev = reward_processor
                reward_processor = (lambda p=prev, s=shift_proc: (lambda x: s(p(x))))()

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        output_cfg = {}
    output_dir = output_cfg.get("base_dir", os.path.join(os.getcwd(), "output"))
    output_verbose = bool(output_cfg.get("verbose", False))
    external_cfg = cfg.get("external") or {}
    if not isinstance(external_cfg, dict):
        external_cfg = {}

    wandb_cfg = cfg.get("wandb")
    wandb_config = None
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled", True):
        dir_val = wandb_cfg.get("dir") or output_cfg.get("base_dir")
        if dir_val:
            dir_val = str(dir_val)
        dataset_type = str(dataset_cfg.get("type") or "bridge_build")
        try:
            num_turns_val = int(getattr(maac_args, "num_turns", 1))
        except Exception:
            num_turns_val = 1
        tags = wandb_cfg.get("tags", ["maac", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"])
        if not isinstance(tags, list):
            tags = ["maac", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"]
        run_name = wandb_cfg.get("name") or wandb_cfg.get("run_name") or f"{dataset_type}-maac"
        wandb_config = {
            "project": wandb_cfg.get("project", "bridge_build"),
            "entity": wandb_cfg.get("entity", None),
            "name": run_name,
            "dir": dir_val,
            "tags": tags,
            "config_sections": {
                "dataset": dataset_cfg,
                "agent_model": model_cfg,
                "output": output_cfg,
                "external": external_cfg,
                "trainer": maac_cfg,
            },
        }
        if wandb_config.get("dir"):
            os.environ.setdefault("WANDB_DIR", str(wandb_config["dir"]))

    import LLM_Collab_Minecraft.bridge_build.external as external_mod  # type: ignore

    external_mod.VERBOSE = bool(output_verbose)

    is_multi_turn = False
    try:
        is_multi_turn = int(getattr(maac_args, "num_turns", 1)) > 1
    except Exception:
        is_multi_turn = False

    trainer_kwargs: Dict[str, Any] = {
        "tokenizer": tokenizers if agent_names else tokenizer,
        "reward_func": reward_func,
        "formatters": formatters,
        "args": maac_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "model_config": {
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": critic_model_kwargs,
            "critic_value_head_hidden_dim": maac_cfg.get("critic_value_head_hidden_dim"),
        },
        "wandb_config": wandb_config,
    }
    trainer_kwargs["agent_model"] = model_name or None
    if agent_names:
        trainer_kwargs["agents"] = agent_names
    critic_name = str(critic_model_cfg.get("name") or "").strip() or None
    if critic_name:
        trainer_kwargs["critic_model"] = critic_name
    if critic_names:
        trainer_kwargs["critics"] = critic_names
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if is_multi_turn:
        def _resolver(prompt: str) -> Any:
            ds_key = _normalize_key(prompt)
            if not ds_key:
                return None
            payload = dataset_payload_map.get(ds_key)
            if payload is not None:
                return payload
            base_item = dataset_prompt_map.get(ds_key)
            if base_item is None:
                return None
            payload = _payload_from_item(base_item)
            dataset_payload_map[ds_key] = copy.deepcopy(payload)
            return payload

        external_set_context_resolver(_resolver)

        external_mode = str(external_cfg.get("mode") or "empty_feedback")
        original_prompt_flag = bool(external_cfg.get("original_prompt", True))
        previous_response_flag = bool(external_cfg.get("previous_response", False))
        num_agents_default = int(num_agents)

        def external_transition_wrapper(prompt: str, agent_completions: Any, num_agents: int | None = None, **_kwargs: Any) -> Any:
            n_agents = int(num_agents) if num_agents is not None else num_agents_default
            prompt_history = _kwargs.get("prompt_history_per_agent")
            response_history = _kwargs.get("response_history_per_agent")
            ds_key = _normalize_key(str(prompt or ""))
            base_item_for_reset = _require_dataset_item(ds_key)

            # Reset per-rollout state at the first external transition call of each rollout.
            is_first_external_turn = False
            try:
                is_first_external_turn = bool(prompt_history) and len(prompt_history[0]) <= 1
            except Exception:
                is_first_external_turn = False
            if is_first_external_turn:
                dataset_payload_map[ds_key] = _payload_from_item(base_item_for_reset)

            prompts = external_get_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=n_agents,
                mode=external_mode,
                original_prompt=original_prompt_flag,
                previous_response=previous_response_flag,
                prompt_history_per_agent=prompt_history,
                response_history_per_agent=response_history,
            )

            base_item = dict(base_item_for_reset)

            # The per-dataset payload map is the authoritative rollout state.
            payload = dataset_payload_map.get(ds_key)
            if payload is None:
                payload = _payload_from_item(base_item)

            try:
                next_payload, _metrics, _actions = transition_payload(
                    payload=payload,
                    agent_completions=list(agent_completions),
                    num_agents=n_agents,
                )
            except Exception:
                next_payload = payload

            dataset_payload_map[ds_key] = copy.deepcopy(next_payload)

            try:
                turn_idx = int(len(prompt_history[0]) + 1) if prompt_history else 2
            except Exception:
                raw_state = next_payload.get("state_before_turn") if isinstance(next_payload, Mapping) else None
                if isinstance(raw_state, Mapping):
                    turn_idx = int(raw_state.get("turn_index") or 2)
                else:
                    turn_idx = 2
            return prompts

        trainer_kwargs["external_transition"] = external_transition_wrapper

    trainer = MAACTrainer(**trainer_kwargs)
    trainer.verbose = bool(output_verbose)
    trainer.train()

    if bool(output_cfg.get("save_final_model", False)):
        save_path_cfg = output_cfg.get("save_path")
        if save_path_cfg:
            save_path = str(save_path_cfg)
        else:
            save_path = os.path.join(os.path.abspath(str(output_dir)), "final_model")
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
