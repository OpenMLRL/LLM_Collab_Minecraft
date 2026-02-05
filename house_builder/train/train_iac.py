from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Mapping

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"PyYAML is required. Install pyyaml. Error: {e}")


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(REPO_ROOT))

from datasets import Dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore
import torch  # type: ignore

from comlrl.trainers.actor_critic import IACTrainer  # type: ignore
from comlrl.utils.reward_processor import RewardProcessors  # type: ignore

from LLM_Collab_Minecraft.house_builder.external import (
    get_external_transition as external_get_transition,
    set_context_resolver as external_set_context_resolver,
)
from LLM_Collab_Minecraft.house_builder.rewards.house_builder_reward import get_reward_function
from LLM_Collab_Minecraft.house_builder.utils.house_builder import (
    TaskSpec,
    compute_resource_limits,
    format_layers_text,
    legend_lines,
    load_tasks_from_json,
    normalize_block_id,
    unique_block_list,
)
from LLM_Collab_Minecraft.house_builder.utils.config import apply_overrides, load_yaml, resolve_path
from LLM_Collab_Minecraft.house_builder.utils.patches import apply_default_patches
from LLM_Collab_Minecraft.house_builder.utils.prompting import apply_prompt_defaults
from LLM_Collab_Minecraft.house_builder.utils.trainer_args import get_iac_args


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


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _prepare_rpg_state(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rpg_cfg = cfg.get("RPG") or cfg.get("rpg") or {}
    if not isinstance(rpg_cfg, dict):
        rpg_cfg = {}

    player_cfg = rpg_cfg.get("player") or {}
    spider_cfg = rpg_cfg.get("spider") or {}

    player_hp = _as_int(player_cfg.get("hp", 20), 20)

    spider_num = max(0, _as_int(spider_cfg.get("num", 0), 0))
    atk_low_raw = spider_cfg.get("atk_low", spider_cfg.get("atk"))
    atk_high_raw = spider_cfg.get("atk_high", spider_cfg.get("atk"))
    atk_low = _as_int(atk_low_raw, 0)
    atk_high = _as_int(atk_high_raw, atk_low)
    if atk_high < atk_low:
        atk_low, atk_high = atk_high, atk_low

    rng = random.Random(int(seed))
    atk_values: List[int] = []
    for _ in range(spider_num):
        try:
            atk_values.append(int(rng.randint(atk_low, atk_high)))
        except Exception:
            atk_values.append(int(atk_low))
    total_dmg = float(sum(atk_values))

    spider_cfg = dict(spider_cfg)
    spider_cfg["atk_low"] = atk_low
    spider_cfg["atk_high"] = atk_high
    spider_cfg["atk_values"] = atk_values
    spider_cfg["dmg"] = total_dmg
    spider_cfg["num"] = spider_num

    player_cfg = dict(player_cfg)
    player_cfg["hp"] = player_hp

    rpg_cfg["player"] = player_cfg
    rpg_cfg["spider"] = spider_cfg
    cfg["RPG"] = rpg_cfg

    rpg_state = {
        "player_hp": player_hp,
        "spider_num": spider_num,
        "spider_atk_low": atk_low,
        "spider_atk_high": atk_high,
        "spider_atk_values": atk_values,
        "spider_total_dmg": total_dmg,
    }
    cfg["_rpg_state"] = rpg_state
    return rpg_state


def _rpg_placeholders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = cfg.get("_rpg_state")
    if not isinstance(state, dict):
        state = {}
    spider_atk_values = state.get("spider_atk_values") or []
    if isinstance(spider_atk_values, (int, float)):
        spider_atk_values = [spider_atk_values]
    try:
        atk_iter = list(spider_atk_values)
    except Exception:
        atk_iter = []
    spider_atk_list = "[" + ", ".join(str(v) for v in atk_iter) + "]" if atk_iter else "[]"
    return {
        "player_hp": state.get("player_hp", 0),
        "spider_num": state.get("spider_num", 0),
        "spider_atk": spider_atk_list,
        "spider_dmg": state.get("spider_total_dmg", 0),
    }


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
    use_chat_template = bool(prompt_cfg.get("use_chat_template", False))
    include_air_rects = bool(prompt_cfg.get("include_air_rects", False))
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "").rstrip()
    user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
    user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    limited_resource = bool(task_cfg.get("limited_resource", False))
    rpg_kwargs = _rpg_placeholders(cfg)

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

    def _prompt_override(item: Dict[str, Any]) -> str | None:
        p = item.get("prompt")
        if isinstance(p, str) and "\n" in p:
            return p.strip()
        return None

    def _allowed_blocks(item: Dict[str, Any], overrides: List[str]) -> List[str]:
        if overrides:
            return unique_block_list(overrides)
        palette = item.get("palette") or {}
        if isinstance(palette, dict):
            return unique_block_list(palette.values())
        return []

    def _format_resource_limits(task: TaskSpec) -> str:
        if not limited_resource:
            return ""
        limits = compute_resource_limits(task, num_agents=num_agents)
        if not limits:
            return ""
        lines: List[str] = []
        for _key, block in task.palette.items():
            block_norm = normalize_block_id(block)
            if block_norm in ("air", "cave_air", "void_air"):
                continue
            limit_val = limits.get(block_norm)
            if limit_val is None:
                continue
            lines.append(f"- {block_norm}: {limit_val}")
        if not lines:
            return ""
        return "Resource limits per agent (air unlimited):\n" + "\n".join(lines)

    def _render(item: Dict[str, Any], tmpl: str) -> str:
        override = _prompt_override(item)
        if override is not None:
            return override

        w_from = item.get("local_bbox_from") or [0, 0, 0]
        w_to = item.get("local_bbox_to") or [0, 0, 0]
        palette = item.get("palette") or {}
        layers_by_y = item.get("layers_by_y") or {}

        task = TaskSpec(
            task_id=str(item.get("task_id") or ""),
            local_bbox_from=[int(v) for v in w_from],
            local_bbox_to=[int(v) for v in w_to],
            palette={str(k): str(v) for k, v in palette.items()},
            layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
        )

        layers_text = format_layers_text(task, world_from=w_from, include_air=include_air_rects)
        legend = legend_lines(task.palette)

        allowed_blocks_agent1 = _allowed_blocks(item, block_agent1_override)
        allowed_blocks_agent2 = _allowed_blocks(item, block_agent2_override)
        block_agent1_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent1)
        block_agent2_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent2)

        user = tmpl.format(
            task_id=str(item.get("task_id") or ""),
            world_bbox_from=json.dumps(w_from, separators=(",", ":")),
            world_bbox_to=json.dumps(w_to, separators=(",", ":")),
            legend_lines=legend,
            layers_text=layers_text,
            block_agent1_lines=block_agent1_lines,
            block_agent2_lines=block_agent2_lines,
            spider_num=rpg_kwargs.get("spider_num"),
            player_hp=rpg_kwargs.get("player_hp"),
            spider_atk=rpg_kwargs.get("spider_atk"),
            spider_dmg=rpg_kwargs.get("spider_dmg"),
        ).rstrip()
        resource_limits_text = _format_resource_limits(task)
        if resource_limits_text:
            user = user + "\n\n" + resource_limits_text
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
    parser = argparse.ArgumentParser(description="Train house_builder with IAC (CoMLRL IACTrainer)")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "house_builder", "configs", "house_builder_iac_config.yaml"),
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

    _prepare_rpg_state(cfg, seed)

    iac_cfg = cfg.get("iac") or {}
    if not isinstance(iac_cfg, dict):
        iac_cfg = {}
    num_agents = int(iac_cfg.get("num_agents") or 1)
    if num_agents not in (1, 2):
        raise ValueError("iac.num_agents must be 1 or 2")

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(args.config, dataset_cfg.get("json_path"))
    tasks = load_tasks_from_json(json_path)
    items: List[Dict[str, Any]] = []
    for idx, t in enumerate(tasks, start=1):
        items.append(
            {
                "task_id": t.task_id,
                "dataset_index": idx,
                "local_bbox_from": t.local_bbox_from,
                "local_bbox_to": t.local_bbox_to,
                "palette": t.palette,
                "layers_by_y": {str(k): [str(r) for r in v] for k, v in t.layers_by_y.items()},
                "prompt": f"house_builder:{t.task_id}",
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

    iac_args = get_iac_args(cfg, model_name=model_name)
    formatters = _build_formatters(cfg, num_agents=num_agents, tokenizer=tokenizer)
    prompt_to_item: Dict[str, Dict[str, Any]] = {}
    dataset_prompt_map: Dict[str, Dict[str, Any]] = {}
    layers_key_map: Dict[str, Dict[str, Any]] = {}

    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    include_air_rects = bool(prompt_cfg.get("include_air_rects", False))

    def _normalize_key(s: str) -> str:
        return " ".join((s or "").split()).strip()

    def _register_prompt(prompt: str, item: Mapping[str, Any], turn_idx: int) -> None:
        key = _normalize_key(prompt)
        if not key:
            return
        mapped = dict(item)
        mapped["_house_builder_turn"] = int(turn_idx)
        prompt_to_item[key] = mapped

    _layers_re = re.compile(
        r"Layers \(ascending WORLD y\).*?\n- Format:.*?\n(.*?)\n\s*WORLD bbox \(inclusive\):",
        re.DOTALL,
    )

    def _layers_key_from_item(item: Mapping[str, Any]) -> str:
        w_from = item.get("local_bbox_from") or [0, 0, 0]
        w_to = item.get("local_bbox_to") or [0, 0, 0]
        palette = item.get("palette") or {}
        layers_by_y = item.get("layers_by_y") or {}
        try:
            task = TaskSpec(
                task_id=str(item.get("task_id") or ""),
                local_bbox_from=[int(v) for v in w_from],
                local_bbox_to=[int(v) for v in w_to],
                palette={str(k): str(v) for k, v in palette.items()},
                layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
            )
            layers_text = format_layers_text(task, world_from=w_from, include_air=include_air_rects)
        except Exception:
            return ""
        return _normalize_key(layers_text)

    def _extract_layers_key(prompt: str) -> str:
        if not prompt:
            return ""
        match = _layers_re.search(prompt)
        if not match:
            return ""
        return _normalize_key(match.group(1))

    def _extract_turn_idx(prompt: str) -> int | None:
        if not prompt:
            return None
        match = re.search(r"\bTurn\s*:\s*(\d+)\b", prompt)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _register_dataset_prompts(items_list: List[Dict[str, Any]], turn_idx: int) -> None:
        for item in items_list:
            ds_key = _normalize_key(str(item.get("prompt") or ""))
            if ds_key and ds_key not in dataset_prompt_map:
                dataset_prompt_map[ds_key] = dict(item)
            layers_key = _layers_key_from_item(item)
            if layers_key and layers_key not in layers_key_map:
                layers_key_map[layers_key] = dict(item)
            for fmt in formatters:
                try:
                    prompt = fmt(item)
                except Exception:
                    prompt = ""
                if prompt:
                    _register_prompt(prompt, item, turn_idx)

    _register_dataset_prompts(train_items, 1)
    if eval_items:
        _register_dataset_prompts(eval_items, 1)

    def _lookup_item(prompts: List[str]) -> Dict[str, Any]:
        for p in prompts or []:
            key = _normalize_key(p)
            if key in prompt_to_item:
                return prompt_to_item[key]
            layers_key = _extract_layers_key(p)
            if layers_key and layers_key in layers_key_map:
                item = dict(layers_key_map[layers_key])
                turn_idx = _extract_turn_idx(p)
                if turn_idx is not None:
                    item["_house_builder_turn"] = int(turn_idx)
                return item
        return {}

    reward_base = get_reward_function(cfg=cfg, num_agents=num_agents)
    if num_agents == 1:

        def reward_func(prompts: List[str], agent1_completions: List[str]) -> List[float]:
            batch_item = _lookup_item(prompts)
            return reward_base(agent1_completions, batch_items=[batch_item])

    else:

        def reward_func(
            prompts: List[str],
            agent1_completions: List[str],
            agent2_completions: List[str],
        ) -> List[float]:
            batch_item = _lookup_item(prompts)
            return reward_base(agent1_completions, agent2_completions, batch_items=[batch_item])

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
        try:
            num_turns_val = int(getattr(iac_args, "num_turns", 1))
        except Exception:
            num_turns_val = 1
        dataset_type = str(dataset_cfg.get("type") or "house_builder")
        tags = wandb_cfg.get(
            "tags",
            ["iac", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"],
        )
        if not isinstance(tags, list):
            tags = ["iac", dataset_type, f"agents_{num_agents}", f"turns_{num_turns_val}"]
        run_name = (
            wandb_cfg.get("name")
            or wandb_cfg.get("run_name")
            or f"{dataset_type}-iac"
        )
        wandb_config = {
            "project": wandb_cfg.get("project", "house_builder"),
            "entity": wandb_cfg.get("entity", None),
            "name": run_name,
            "dir": dir_val,
            "tags": tags,
            "config_sections": {
                "dataset": dataset_cfg,
                "model": model_cfg,
                "output": output_cfg,
                "external": external_cfg,
                "trainer": iac_cfg,
            },
        }
        if wandb_config.get("dir"):
            os.environ.setdefault("WANDB_DIR", str(wandb_config["dir"]))

    import LLM_Collab_Minecraft.house_builder.external as external_mod  # type: ignore

    external_mod.VERBOSE = bool(output_verbose)
    apply_default_patches(cfg)

    is_multi_turn = False
    try:
        is_multi_turn = int(getattr(iac_args, "num_turns", 1)) > 1
    except Exception:
        is_multi_turn = False

    trainer_kwargs: Dict[str, Any] = {
        "model": model_name,
        "tokenizer": tokenizer,
        "reward_func": reward_func,
        "formatters": formatters,
        "args": iac_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "model_config": {
            "tokenizer_kwargs": tokenizer_kwargs,
            "model_kwargs": model_kwargs,
            "critic_model_kwargs": iac_cfg.get("critic_model_kwargs", model_kwargs),
            "critic_value_head_hidden_dim": iac_cfg.get("critic_value_head_hidden_dim"),
            "value_head_hidden_dim": iac_cfg.get("value_head_hidden_dim"),
        },
        "wandb_config": wandb_config,
    }
    if reward_processor is not None:
        trainer_kwargs["reward_processor"] = reward_processor

    if is_multi_turn:
        prompt_cfg = cfg.get("prompt") or {}
        if not isinstance(prompt_cfg, dict):
            prompt_cfg = {}
        system_prompt = str(prompt_cfg.get("system") or "").rstrip()
        user_template = str(prompt_cfg.get("user_template") or "").rstrip()
        user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
        user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()
        include_air_rects = bool(prompt_cfg.get("include_air_rects", False))

        task_cfg = cfg.get("task") or {}
        if not isinstance(task_cfg, dict):
            task_cfg = {}

        max_commands_total = int(task_cfg.get("max_commands") or 600)
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

        def _allowed_blocks(item: Dict[str, Any], overrides: List[str]) -> List[str]:
            if overrides:
                return unique_block_list(overrides)
            palette = item.get("palette") or {}
            if isinstance(palette, dict):
                return unique_block_list(palette.values())
            return []

        rpg_kwargs = _rpg_placeholders(cfg)
        context_map: Dict[str, Any] = {}

        def _register_split(ds: Dataset) -> None:
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
                palette = item.get("palette") or {}
                layers_by_y = item.get("layers_by_y") or {}

                task = TaskSpec(
                    task_id=str(item.get("task_id") or ""),
                    local_bbox_from=[int(v) for v in w_from],
                    local_bbox_to=[int(v) for v in w_to],
                    palette={str(k): str(v) for k, v in palette.items()},
                    layers_by_y={int(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
                )
                layers_text = format_layers_text(task, world_from=w_from, include_air=include_air_rects)
                legend = legend_lines(task.palette)
                resource_limits = compute_resource_limits(task, num_agents=num_agents) if limited_resource else {}
                resource_limits_lines: List[str] = []
                for _key, block in task.palette.items():
                    block_norm = normalize_block_id(block)
                    if block_norm in ("air", "cave_air", "void_air"):
                        continue
                    limit_val = resource_limits.get(block_norm)
                    if limit_val is None:
                        continue
                    resource_limits_lines.append(f"- {block_norm}: {limit_val}")
                resource_limits_text = ""
                if resource_limits_lines:
                    resource_limits_text = "Resource limits per agent (air unlimited):\n" + "\n".join(resource_limits_lines)

                allowed_blocks_agent1 = _allowed_blocks(item, block_agent1_override)
                allowed_blocks_agent2 = _allowed_blocks(item, block_agent2_override)
                block_agent1_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent1)
                block_agent2_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent2)

                fmt_kwargs = {
                    "task_id": str(item.get("task_id") or ""),
                    "world_bbox_from": json.dumps(w_from, separators=(",", ":")),
                    "world_bbox_to": json.dumps(w_to, separators=(",", ":")),
                    "legend_lines": legend,
                    "layers_text": layers_text,
                    "block_agent1_lines": block_agent1_lines,
                    "block_agent2_lines": block_agent2_lines,
                    "spider_num": rpg_kwargs.get("spider_num"),
                    "player_hp": rpg_kwargs.get("player_hp"),
                    "spider_atk": rpg_kwargs.get("spider_atk"),
                    "spider_dmg": rpg_kwargs.get("spider_dmg"),
                }
                base_user_single = user_template.format(**fmt_kwargs).rstrip()
                base_user_agent1 = user_template_agent1.format(**fmt_kwargs).rstrip()
                base_user_agent2 = user_template_agent2.format(**fmt_kwargs).rstrip()
                if resource_limits_text:
                    base_user_single = base_user_single + "\n\n" + resource_limits_text
                    base_user_agent1 = base_user_agent1 + "\n\n" + resource_limits_text
                    base_user_agent2 = base_user_agent2 + "\n\n" + resource_limits_text

                payload = {
                    "system_prompt": system_prompt,
                    "user_prompt_single": base_user_single,
                    "user_prompt_agent1": base_user_agent1,
                    "user_prompt_agent2": base_user_agent2,
                    "task_id": str(item.get("task_id") or ""),
                    "local_bbox_from": [int(v) for v in (w_from or [0, 0, 0])],
                    "local_bbox_to": [int(v) for v in (w_to or [0, 0, 0])],
                    "palette": {str(k): str(v) for k, v in (palette or {}).items()},
                    "layers_by_y": {str(k): [str(r) for r in v] for k, v in (layers_by_y or {}).items()},
                    "allowed_blocks_agent1": list(allowed_blocks_agent1),
                    "allowed_blocks_agent2": list(allowed_blocks_agent2),
                    "max_commands_total": max_commands_total,
                    "limited_resource": limited_resource,
                    "resource_limits_text": resource_limits_text,
                    "rpg_state": cfg.get("_rpg_state"),
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
        modification_limit = external_cfg.get("lim")
        if modification_limit is None:
            modification_limit = external_cfg.get("modification_limit")
        common_prefix = external_cfg.get("common_prefix")
        common_suffix = external_cfg.get("common_suffix")

        num_agents_default = int(num_agents)

        def external_transition_wrapper(prompt: str, agent_completions: Any, num_agents: int | None = None, **_kwargs: Any) -> Any:
            n_agents = int(num_agents) if num_agents is not None else num_agents_default
            prompt_history = _kwargs.get("prompt_history_per_agent")
            response_history = _kwargs.get("response_history_per_agent")
            prompts = external_get_transition(
                prompt=prompt,
                agent_completions=agent_completions,
                num_agents=n_agents,
                mode=external_mode,
                limit=modification_limit,
                original_prompt=original_prompt_flag,
                previous_response=previous_response_flag,
                common_prefix=common_prefix,
                common_suffix=common_suffix,
                prompt_history_per_agent=prompt_history,
                response_history_per_agent=response_history,
            )
            try:
                turn_idx = int(len(prompt_history[0]) + 1) if prompt_history else 2
            except Exception:
                turn_idx = 2
            item_key = _normalize_key(str(prompt or ""))
            item = dataset_prompt_map.get(item_key)
            if item and isinstance(prompts, (list, tuple)):
                for p in prompts:
                    _register_prompt(str(p), item, turn_idx)
            return prompts

        trainer_kwargs["external_transition"] = external_transition_wrapper

    trainer = IACTrainer(**trainer_kwargs)
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
