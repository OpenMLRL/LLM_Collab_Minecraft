from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"PyYAML is required. Install pyyaml. Error: {e}")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(REPO_ROOT))

from datasets import Dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch  # type: ignore

from comlrl.trainers.reinforce import MAGRPOTrainer  # type: ignore
from comlrl.utils.reward_processor import RewardProcessors  # type: ignore

from LLM_Collab_Minecraft.str_builder.external import (
    get_external_transition as external_get_transition,
    set_context_resolver as external_set_context_resolver,
)
from LLM_Collab_Minecraft.str_builder.rewards.str_builder_reward import get_reward_function
from LLM_Collab_Minecraft.str_builder.utils.config import apply_overrides, load_yaml, resolve_path
from LLM_Collab_Minecraft.str_builder.utils.patches import apply_default_patches
from LLM_Collab_Minecraft.str_builder.utils.prompting import apply_graph_setting, apply_prompt_defaults
from LLM_Collab_Minecraft.str_builder.utils.str_builder import load_tasks_from_csv
from LLM_Collab_Minecraft.str_builder.utils.trainer_args import get_trainer_args


def _slice_items(items: List[Dict[str, Any]], split_expr: Any) -> List[Dict[str, Any]]:
    if not split_expr:
        return items
    s = str(split_expr).strip()
    if not s:
        return items
    m = re.search(r"\[\s*(?P<start>-?\d*)\s*:\s*(?P<end>-?\d*)\s*\]", s)
    if not m and ":" in s:
        m = re.match(r"\s*(?P<start>-?\d*)\s*:\s*(?P<end>-?\d*)\s*$", s)
    if not m:
        return items
    start_raw = m.group("start")
    end_raw = m.group("end")
    start = int(start_raw) if start_raw not in (None, "", "+") else None
    end = int(end_raw) if end_raw not in (None, "", "+") else None
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


def _render_prompt(
    *,
    tokenizer: Any | None,
    system_prompt: str,
    user_prompt: str,
    use_chat_template: bool,
) -> str:
    system_prompt = (system_prompt or "").rstrip()
    user_prompt = (user_prompt or "").rstrip()
    if use_chat_template and tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
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


def _build_formatters(cfg: Dict[str, Any], *, num_agents: int, tokenizer: Any | None = None) -> List[Any]:
    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    provide_graph = bool(prompt_cfg.get("provide_graph", True))
    use_chat_template = bool(prompt_cfg.get("use_chat_template", False))
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "").rstrip()
    user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
    user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()
    user_template = apply_graph_setting(user_template, provide_graph=provide_graph)
    user_template_agent1 = apply_graph_setting(user_template_agent1, provide_graph=provide_graph)
    user_template_agent2 = apply_graph_setting(user_template_agent2, provide_graph=provide_graph)

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

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

    agent1_blocks = _as_block_list(task_cfg.get("block_agent1"))
    if not agent1_blocks:
        b0 = str(task_cfg.get("block_even") or "white_concrete").strip()
        b1 = str(task_cfg.get("block_odd") or "black_concrete").strip()
        agent1_blocks = [b0, b1]

    agent2_blocks = _as_block_list(task_cfg.get("block_agent2"))
    if not agent2_blocks:
        agent2_blocks = [str(task_cfg.get("block_agent2") or "red_concrete").strip() or "red_concrete"]

    block_agent1_lines = "\n".join(f"- {b}" for b in agent1_blocks)
    block_agent2_lines = "\n".join(f"- {b}" for b in agent2_blocks)

    def _prompt_override(item: Dict[str, Any]) -> str | None:
        p = item.get("prompt")
        if isinstance(p, str) and "\n" in p:
            return p.strip()
        return None

    def _render(item: Dict[str, Any], tmpl: str) -> str:
        override = _prompt_override(item)
        if override is not None:
            return override
        w_from = item.get("local_bbox_from") or [0, 0, 0]
        w_to = item.get("local_bbox_to") or [0, 0, 0]
        target_rows = item.get("target_rows_topdown") or []
        target_ascii = "" if not provide_graph else "\n".join(str(r) for r in target_rows)
        user = tmpl.format(
            task_id=str(item.get("task_id") or ""),
            text=str(item.get("string") or ""),
            difficulty=int(item.get("difficulty") or 0),
            world_bbox_from=json.dumps(w_from, separators=(",", ":")),
            world_bbox_to=json.dumps(w_to, separators=(",", ":")),
            target_ascii=target_ascii,
            block_agent1_lines=block_agent1_lines,
            block_agent2_lines=block_agent2_lines,
        ).rstrip()
        return _render_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user,
            use_chat_template=use_chat_template,
        )

    if num_agents == 1:
        return [lambda item: _render(item, user_template)]

    return [
        lambda item: _render(item, user_template_agent1),
        lambda item: _render(item, user_template_agent2),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train str_builder with GRPO (CoMLRL MAGRPOTrainer num_agents=1)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "str_builder", "configs", "str_builder_magrpo_config.yaml"),
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
            for part in str(item).split(","):
                part = part.strip()
                if part:
                    override_items.append(part)
    if override_items:
        cfg = apply_overrides(cfg, ",".join(override_items))
    apply_prompt_defaults(cfg)

    seed_val = cfg.get("seed", None)
    if seed_val is None:
        try:
            import secrets

            seed = int(secrets.randbits(32))
        except Exception:
            seed = int(time.time()) & 0x7FFFFFFF
    else:
        seed = int(seed_val)

    magrpo_cfg = cfg.get("magrpo") or {}
    if not isinstance(magrpo_cfg, dict):
        magrpo_cfg = {}
    num_agents = int(magrpo_cfg.get("num_agents") or 1)
    if num_agents not in (1, 2):
        raise ValueError("magrpo.num_agents must be 1 or 2")

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    csv_path = resolve_path(args.config, dataset_cfg.get("csv_path"))
    spacing = int(dataset_cfg.get("spacing") or 1)
    local_z = int(dataset_cfg.get("local_z") or 0)

    tasks = load_tasks_from_csv(csv_path, spacing=spacing, local_z=local_z)
    items: List[Dict[str, Any]] = []
    for t in tasks:
        items.append(
            {
                "task_id": t.task_id,
                "csv_row_index": t.csv_row_index,
                "string": t.text,
                "difficulty": t.difficulty,
                "local_bbox_from": t.local_bbox_from,
                "local_bbox_to": t.local_bbox_to,
                "target_rows_topdown": t.target_rows_topdown,
                "prompt": f"str_builder:{t.task_id}",
            }
        )

    train_split = dataset_cfg.get("train_split", "[:]")
    eval_split = dataset_cfg.get("eval_split")
    train_items = _slice_items(items, train_split)
    eval_items = _slice_items(items, eval_split) if eval_split else []
    train_ds = Dataset.from_list(train_items)
    eval_ds = Dataset.from_list(eval_items) if eval_items else None

    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_name = str(model_cfg.get("name") or "")
    if not model_name:
        raise ValueError("model.name is required")
    tokenizer_kwargs = model_cfg.get("tokenizer_kwargs") or {}
    model_kwargs = model_cfg.get("model_kwargs") or {}
    if not isinstance(tokenizer_kwargs, dict):
        tokenizer_kwargs = {}
    if not isinstance(model_kwargs, dict):
        model_kwargs = {}

    dtype = _map_dtype(model_cfg.get("dtype") or model_cfg.get("torch_dtype") or model_kwargs.get("torch_dtype"))
    if dtype is not None and "torch_dtype" not in model_kwargs:
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    agents = []
    for _ in range(num_agents):
        agent = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        enable_gc = bool(model_cfg.get("gradient_checkpointing", True))
        if enable_gc:
            if hasattr(agent, "config"):
                agent.config.use_cache = False
            agent.gradient_checkpointing_enable()
        agents.append(agent)

    magrpo_args = get_trainer_args(cfg)
    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=tokenizer)
    reward_func = get_reward_function(cfg=cfg, num_agents=num_agents)

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
        dataset_type = str(dataset_cfg.get("type") or "str_builder")
        try:
            num_turns_val = int(getattr(magrpo_args, "num_turns", 1))
        except Exception:
            num_turns_val = 1
        tags = wandb_cfg.get(
            "tags",
            ["magrpo", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"],
        )
        if not isinstance(tags, list):
            tags = ["magrpo", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"]
        run_name = (
            wandb_cfg.get("name")
            or wandb_cfg.get("run_name")
            or f"{dataset_type}-magrpo"
        )
        wandb_config = {
            "project": wandb_cfg.get("project", "str_builder"),
            "entity": wandb_cfg.get("entity", None),
            "name": run_name,
            "dir": dir_val,
            "tags": tags,
            "config_sections": {
                "dataset": dataset_cfg,
                "model": model_cfg,
                "output": output_cfg,
                "external": external_cfg,
                "trainer": magrpo_cfg,
            },
        }
        if wandb_config.get("dir"):
            os.environ.setdefault("WANDB_DIR", str(wandb_config["dir"]))

    import LLM_Collab_Minecraft.str_builder.external as external_mod  # type: ignore

    external_mod.VERBOSE = bool(output_verbose)
    apply_default_patches(cfg)

    is_multi_turn = False
    try:
        is_multi_turn = int(getattr(magrpo_args, "num_turns", 1)) > 1
    except Exception:
        is_multi_turn = False

    trainer_kwargs: Dict[str, Any] = {
        "agents": agents,
        "num_agents": num_agents,
        "reward_func": reward_func,
        "formatters": formatters,
        "args": magrpo_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "tokenizer": tokenizer,
        "wandb_config": wandb_config,
        "dataset_type": str(dataset_cfg.get("type") or "str_builder"),
    }
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if is_multi_turn:
        def _normalize_key(s: str) -> str:
            return " ".join((s or "").split()).strip()

        prompt_cfg = cfg.get("prompt") or {}
        if not isinstance(prompt_cfg, dict):
            prompt_cfg = {}
        provide_graph = bool(prompt_cfg.get("provide_graph", True))
        system_prompt = str(prompt_cfg.get("system") or "").rstrip()
        user_template = str(prompt_cfg.get("user_template") or "").rstrip()
        user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
        user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()
        user_template = apply_graph_setting(user_template, provide_graph=provide_graph)
        user_template_agent1 = apply_graph_setting(user_template_agent1, provide_graph=provide_graph)
        user_template_agent2 = apply_graph_setting(user_template_agent2, provide_graph=provide_graph)

        task_cfg = cfg.get("task") or {}
        if not isinstance(task_cfg, dict):
            task_cfg = {}

        max_commands_total = int(task_cfg.get("max_commands") or 600)

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

        agent1_blocks = _as_block_list(task_cfg.get("block_agent1"))
        if not agent1_blocks:
            b0 = str(task_cfg.get("block_even") or "white_concrete").strip()
            b1 = str(task_cfg.get("block_odd") or "black_concrete").strip()
            agent1_blocks = [b0, b1]
        agent2_blocks = _as_block_list(task_cfg.get("block_agent2"))
        if not agent2_blocks:
            agent2_blocks = [str(task_cfg.get("block_agent2") or "red_concrete").strip() or "red_concrete"]

        block_agent1_lines = "\n".join(f"- {b}" for b in agent1_blocks)
        block_agent2_lines = "\n".join(f"- {b}" for b in agent2_blocks)

        context_map: Dict[str, Any] = {}

        def _register_split(ds: Any) -> None:
            if ds is None:
                return
            try:
                n = len(ds)
            except Exception:
                return
            for idx in range(n):
                try:
                    item = ds[idx]
                except Exception:
                    continue
                w_from = item.get("local_bbox_from") or [0, 0, 0]
                w_to = item.get("local_bbox_to") or [0, 0, 0]
                target_rows = item.get("target_rows_topdown") or []
                target_ascii = "" if not provide_graph else "\n".join(str(r) for r in target_rows)
                fmt_kwargs = {
                    "task_id": str(item.get("task_id") or ""),
                    "text": str(item.get("string") or ""),
                    "difficulty": int(item.get("difficulty") or 0),
                    "world_bbox_from": json.dumps(w_from, separators=(",", ":")),
                    "world_bbox_to": json.dumps(w_to, separators=(",", ":")),
                    "target_ascii": target_ascii,
                    "block_agent1_lines": block_agent1_lines,
                    "block_agent2_lines": block_agent2_lines,
                }
                payload = {
                    "system_prompt": system_prompt,
                    "user_prompt_single": user_template.format(**fmt_kwargs).rstrip(),
                    "user_prompt_agent1": user_template_agent1.format(**fmt_kwargs).rstrip(),
                    "user_prompt_agent2": user_template_agent2.format(**fmt_kwargs).rstrip(),
                    "task_id": str(item.get("task_id") or ""),
                    "text": str(item.get("string") or ""),
                    "difficulty": int(item.get("difficulty") or 0),
                    "local_bbox_from": [int(v) for v in (w_from or [0, 0, 0])],
                    "local_bbox_to": [int(v) for v in (w_to or [0, 0, 0])],
                    "target_rows_topdown": [str(r) for r in target_rows],
                    "allowed_blocks_agent1": list(agent1_blocks),
                    "allowed_blocks_agent2": list(agent2_blocks),
                    "max_commands_total": max_commands_total,
                }

                ds_key = _normalize_key(str(item.get("prompt") or ""))
                if ds_key:
                    context_map[ds_key] = payload

                for fmt in formatters:
                    try:
                        p = fmt(item)
                    except Exception:
                        p = ""
                    key = _normalize_key(str(p))
                    if key:
                        context_map[key] = payload

        _register_split(train_ds)
        _register_split(eval_ds)

        def _resolver(p: str) -> Any:
            return context_map.get(_normalize_key(p))

        external_set_context_resolver(_resolver)

        external_mode = str(external_cfg.get("mode") or "perfect_feedback")
        original_prompt_flag = bool(external_cfg.get("original_prompt", True))
        previous_response_flag = bool(external_cfg.get("previous_response", False))

        num_agents_default = int(num_agents)

        def external_transition_wrapper(prompt: str, agent_completions: Any, num_agents: int | None = None, **_kwargs: Any) -> Any:
            n_agents = int(num_agents) if num_agents is not None else num_agents_default
            return external_get_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=n_agents,
                mode=external_mode,
                original_prompt=original_prompt_flag,
                previous_response=previous_response_flag,
                prompt_history_per_agent=_kwargs.get("prompt_history_per_agent"),
                response_history_per_agent=_kwargs.get("response_history_per_agent"),
            )

        trainer_kwargs["external_transition"] = external_transition_wrapper

    trainer = MAGRPOTrainer(**trainer_kwargs)
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
