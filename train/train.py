from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"PyYAML is required. Install pyyaml. Error: {e}")


# Make repo importable as a package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(REPO_ROOT))

from datasets import Dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch  # type: ignore

from comlrl.trainers.magrpo import MAGRPOTrainer  # type: ignore
from comlrl.utils.reward_processor import RewardProcessors  # type: ignore

from LLM_Collab_MC.rewards.str_builder_reward import get_reward_function
from LLM_Collab_MC.utils.config import apply_overrides, expand_jobid_placeholder, load_yaml, resolve_path
from LLM_Collab_MC.utils.patches import apply_default_patches
from LLM_Collab_MC.utils.str_builder import load_tasks_from_csv
from LLM_Collab_MC.utils.trainer_args import get_trainer_args


def _split_list(items: List[Dict[str, Any]], split_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not items:
        return [], []
    r = max(0.0, min(1.0, float(split_ratio)))
    n_train = int(round(len(items) * r))
    n_train = max(1, min(len(items) - 1, n_train)) if len(items) >= 2 else len(items)
    import random

    rng = random.Random(int(seed))
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    train_idxs = set(idxs[:n_train])
    train = [items[i] for i in range(len(items)) if i in train_idxs]
    eval_ = [items[i] for i in range(len(items)) if i not in train_idxs]
    return train, eval_


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


def _build_formatters(cfg: Dict[str, Any], *, num_agents: int) -> List[Any]:
    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "").rstrip()
    user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
    user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()

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
        # Backwards compatible fallback.
        b0 = str(task_cfg.get("block_even") or "white_concrete").strip()
        b1 = str(task_cfg.get("block_odd") or "black_concrete").strip()
        agent1_blocks = [b0, b1]

    agent2_blocks = _as_block_list(task_cfg.get("block_agent2"))
    if not agent2_blocks:
        # Backwards compatible fallback.
        agent2_blocks = [str(task_cfg.get("block_agent2") or "red_concrete").strip() or "red_concrete"]

    block_agent1_lines = "\n".join(f"- {b}" for b in agent1_blocks)
    block_agent2_lines = "\n".join(f"- {b}" for b in agent2_blocks)

    def _render(item: Dict[str, Any], tmpl: str) -> str:
        w_from = item.get("local_bbox_from") or [0, 0, 0]
        w_to = item.get("local_bbox_to") or [0, 0, 0]
        user = tmpl.format(
            task_id=str(item.get("task_id") or ""),
            text=str(item.get("string") or ""),
            difficulty=int(item.get("difficulty") or 0),
            world_bbox_from=json.dumps(w_from, separators=(",", ":")),
            world_bbox_to=json.dumps(w_to, separators=(",", ":")),
            block_agent1_lines=block_agent1_lines,
            block_agent2_lines=block_agent2_lines,
        ).rstrip()
        if system_prompt:
            return system_prompt + "\n\n" + user
        return user

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
        default=os.path.join(REPO_ROOT, "configs", "str_builder_config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--override", type=str, default=None, help="key.path=value overrides, comma-separated")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.override:
        cfg = apply_overrides(cfg, str(args.override))

    run_name = str(cfg.get("run_name") or "str_builder_grpo")
    seed = int(cfg.get("seed", 42))
    fixed_seed = bool(cfg.get("fixed_seed", True))
    if not fixed_seed:
        try:
            import secrets

            seed = int(secrets.randbits(32))
        except Exception:
            seed = int(time.time()) & 0x7FFFFFFF

    collab_cfg = cfg.get("collab") or {}
    if not isinstance(collab_cfg, dict):
        collab_cfg = {}
    num_agents = int(collab_cfg.get("num_agents") or 1)
    if num_agents not in (1, 2):
        raise ValueError("collab.num_agents must be 1 or 2")

    data_cfg = cfg.get("data") or {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}
    csv_path = resolve_path(args.config, data_cfg.get("csv_path"))
    split_ratio = float(data_cfg.get("split_ratio") or 0.8)
    spacing = int(data_cfg.get("spacing") or 1)
    local_z = int(data_cfg.get("local_z") or 0)

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

    train_items, eval_items = _split_list(items, split_ratio=split_ratio, seed=seed)
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
        try:
            if hasattr(agent, "config"):
                agent.config.use_cache = False
        except Exception:
            pass
        if bool(model_cfg.get("gradient_checkpointing", True)):
            try:
                agent.gradient_checkpointing_enable()
            except Exception:
                pass
        agents.append(agent)

    magrpo_args = get_trainer_args(cfg)
    formatters = _build_formatters(cfg, num_agents=num_agents)
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

    wandb_cfg = cfg.get("wandb")
    wandb_config = None
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled", False):
        dir_val = wandb_cfg.get("dir") or wandb_cfg.get("output_dir")
        if dir_val:
            dir_val = expand_jobid_placeholder(str(dir_val))
        wandb_config = {
            "project": wandb_cfg.get("project", "str_builder"),
            "entity": wandb_cfg.get("entity", None),
            "name": f"{run_name}_{num_agents}agents",
            "dir": dir_val,
            "tags": ["str_builder", f"agents_{num_agents}"],
        }
        if wandb_config.get("dir"):
            os.environ.setdefault("WANDB_DIR", str(wandb_config["dir"]))

    apply_default_patches(cfg)

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
        "dataset_type": "str_builder",
    }
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    trainer = MAGRPOTrainer(**trainer_kwargs)
    trainer.train()

    out_cfg = cfg.get("output") or {}
    if bool(out_cfg.get("save_final_model", False)):
        save_path_cfg = out_cfg.get("save_path")
        if save_path_cfg:
            save_path = expand_jobid_placeholder(str(save_path_cfg))
        else:
            save_path = os.path.join(os.path.abspath(magrpo_args.output_dir), "final_model")
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
