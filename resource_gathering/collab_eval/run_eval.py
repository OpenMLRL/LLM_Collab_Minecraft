"""Parallel / pipeline collaboration eval for resource_gathering.

Runs *training-free* multi-agent evaluation using the same eval data as the
matching baseline0 BCMAAC config. The two modes are:

* parallel  -- both agents generate their JSON action from the same payload
               within each turn (matches the baseline0 trainer's rollout).
* pipeline  -- agents act serially within each turn: agent 0 acts, the env
               transitions, agent 1 then sees the updated payload before it
               acts and the env transitions again.

Per-task metric is `episode_success` (1.0 iff any turn reports success=True),
matching the trainer's `eval/episode_success` rollup.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Mapping

import torch


HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
MC_ROOT = os.path.dirname(TASK_DIR)
REPO_ROOT = os.path.dirname(MC_ROOT)
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "CoTI")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from coti.train.config_utils import (  # noqa: E402
    apply_overrides,
    load_yaml,
    map_dtype,
    resolve_path,
    set_seed,
    slice_items,
)

from LLM_Collab_Minecraft.resource_gathering.coti.domain import get_domain_spec  # noqa: E402


NOOP_COMPLETION = '{"comm":{},"probe":[],"cmds":[],"path":[]}'

TASK_TYPE = "resource_gathering"

DEFAULT_CONFIG = os.path.join(
    TASK_DIR, "configs", "coti", "bcmaac_baseline0_data_full.yaml"
)


def _build_eval_items(cfg: Dict[str, Any], domain_spec: Any, num_agents: int, num_turns: int, config_path: str) -> tuple[List[Dict[str, Any]], List[Any], List[Dict[str, Any]], List[Any]]:
    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    json_path = resolve_path(config_path, dataset_cfg.get("json_path"))
    tasks = list(domain_spec.load_tasks_from_json(json_path))
    items: List[Dict[str, Any]] = []
    for task in tasks:
        item = domain_spec.task_to_item(task)
        item["max_turns"] = num_turns
        item[domain_spec.state_item_key] = domain_spec.serialize_state(
            domain_spec.make_initial_state(task, num_agents=num_agents, max_turns=num_turns)
        )
        items.append(item)
    eval_split = dataset_cfg.get("eval_split")
    eval_items = slice_items(items, eval_split) if eval_split else list(items)
    eval_task_ids = {str(item.get("task_id")) for item in eval_items}
    eval_tasks = [task for task in tasks if str(getattr(task, "task_id", "")) in eval_task_ids]
    return items, tasks, eval_items, eval_tasks


def _load_agent(model_cfg: Mapping[str, Any], device: str) -> tuple[Any, Any]:
    name = str(model_cfg.get("name") or "").strip()
    if not name:
        raise ValueError("agent_model.name is required")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = map_dtype(model_cfg.get("dtype") or model_cfg.get("torch_dtype")) or torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype if dtype != "auto" else None)
    model.to(device)
    model.eval()
    return tokenizer, model


def _generate(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float, top_k: int | None, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(getattr(tokenizer, "model_max_length", 4096) or 4096))
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = int(input_ids.size(1))
    do_sample = float(temperature) > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(max(temperature, 1e-5))
        gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _success_value(metrics: Mapping[str, Any]) -> float:
    raw = metrics.get("success")
    if raw is None:
        return 0.0
    try:
        return 1.0 if bool(raw) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _run_parallel(item: Mapping[str, Any], *, adapter: Any, num_agents: int, num_turns: int, generate_fn: Any) -> float:
    payload = adapter.reset_item_state(item)
    previous_completions = ["" for _ in range(num_agents)]
    previous_metrics: Dict[str, Any] = {}
    episode_success = 0.0
    for turn_idx in range(num_turns):
        if turn_idx == 0:
            prompts = adapter.initial_prompts(payload)
        else:
            prompts = adapter.followup_prompts(
                payload=payload,
                metrics=previous_metrics,
                previous_completions=previous_completions,
            )
        completions = [generate_fn(prompts[i]) for i in range(num_agents)]
        next_payload, metrics, _ = adapter.transition(payload, completions)
        episode_success = max(episode_success, _success_value(metrics))
        payload = next_payload
        previous_completions = list(completions)
        previous_metrics = dict(metrics)
        if bool(metrics.get("terminated", False)):
            break
    return episode_success


def _run_pipeline(item: Mapping[str, Any], *, adapter: Any, num_agents: int, num_turns: int, generate_fn: Any) -> float:
    payload = adapter.reset_item_state(item)
    last_completions = ["" for _ in range(num_agents)]
    previous_metrics: Dict[str, Any] = {}
    episode_success = 0.0
    terminated = False
    for turn_idx in range(num_turns):
        if turn_idx == 0:
            prompts = adapter.initial_prompts(payload)
        else:
            prompts = adapter.followup_prompts(
                payload=payload,
                metrics=previous_metrics,
                previous_completions=last_completions,
            )
        for active in range(num_agents):
            completion = generate_fn(prompts[active])
            sub = [completion if i == active else NOOP_COMPLETION for i in range(num_agents)]
            next_payload, metrics, _ = adapter.transition(payload, sub)
            episode_success = max(episode_success, _success_value(metrics))
            payload = next_payload
            last_completions[active] = completion
            previous_metrics = dict(metrics)
            if bool(metrics.get("terminated", False)):
                terminated = True
                break
            prompts = adapter.followup_prompts(
                payload=payload,
                metrics=previous_metrics,
                previous_completions=last_completions,
            )
        if terminated:
            break
    return episode_success


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Parallel / pipeline eval for {TASK_TYPE}")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="YAML config path (defaults to baseline0 config).")
    parser.add_argument("--mode", type=str, choices=["parallel", "pipeline"], required=True)
    parser.add_argument("--num-samples", type=int, default=50, help="Cap on number of eval episodes to run/print.")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Stochastic rollouts per task; per-task score is averaged across rollouts.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", type=str, nargs="*", default=None, help="key.path=value overrides for the config.")
    args = parser.parse_args()

    out_path = os.path.join(os.getcwd(), f"{TASK_TYPE}.out")
    out_fh = open(out_path, "w", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        out_fh.write(msg + "\n")
        out_fh.flush()

    cfg = load_yaml(args.config)
    if args.override:
        cfg = apply_overrides(cfg, [item for item in args.override if item])
    domain_spec = get_domain_spec()
    domain_spec.apply_prompt_defaults(cfg)

    set_seed(int(cfg.get("seed", 42)))

    bcmaac_cfg = cfg.get("bcmaac") or cfg.get("maac") or {}
    if not isinstance(bcmaac_cfg, dict):
        bcmaac_cfg = {}
    num_agents = int(bcmaac_cfg.get("num_agents", 2))

    class _TurnLimitProxy:
        def __init__(self, n: int) -> None:
            self.num_turns = int(n)

    num_turns = int(domain_spec.resolve_turn_limit(cfg, _TurnLimitProxy(int(bcmaac_cfg.get("num_turns", 4)))))

    all_items, all_tasks, eval_items, _eval_tasks = _build_eval_items(
        cfg, domain_spec, num_agents=num_agents, num_turns=num_turns, config_path=args.config,
    )
    cap = max(1, int(args.num_samples))
    eval_items = eval_items[:cap]
    if not eval_items:
        log("No eval items resolved from config; aborting.")
        out_fh.close()
        return 1

    model_cfg = cfg.get("agent_model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    tokenizer, model = _load_agent(model_cfg, device=args.device)

    prompt_ctx = dict(domain_spec.build_prompt_context(cfg, num_agents))
    prompt_ctx["coti_variant"] = "baseline0"
    adapter = domain_spec.adapter_cls(
        prompt_ctx=prompt_ctx,
        num_agents=num_agents,
        task_ids=[str(item["task_id"]) for item in all_items],
        task_specs=all_tasks,
        tokenizer=tokenizer,
        external_mode=str((cfg.get("external") or {}).get("mode", "empty_feedback")),
        original_prompt=bool((cfg.get("external") or {}).get("original_prompt", True)),
        previous_response=bool((cfg.get("external") or {}).get("previous_response", False)),
        debug=False,
        reward_config=domain_spec.build_reward_config(cfg),
    )

    max_new_tokens = int(bcmaac_cfg.get("max_new_tokens", 160))
    temperature = float(model_cfg.get("temperature", 0.5))
    top_p = float(model_cfg.get("top_p", 0.6))
    top_k_raw = model_cfg.get("top_k")
    top_k = None if top_k_raw in (None, "none", "null") else int(top_k_raw)

    def _generate_fn(prompt: str) -> str:
        return _generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            device=args.device,
        )

    runner = _run_parallel if args.mode == "parallel" else _run_pipeline

    num_rollouts = max(1, int(args.num_rollouts))

    log(
        f"[collab_eval] task={TASK_TYPE} mode={args.mode} num_agents={num_agents} "
        f"num_turns={num_turns} num_samples={len(eval_items)} num_rollouts={num_rollouts} "
        f"model={model_cfg.get('name')}"
    )

    total = 0.0
    for idx, item in enumerate(eval_items):
        rollout_scores = [
            float(runner(item, adapter=adapter, num_agents=num_agents, num_turns=num_turns, generate_fn=_generate_fn))
            for _ in range(num_rollouts)
        ]
        score = sum(rollout_scores) / float(num_rollouts)
        total += score
        log(
            f"[collab_eval][{idx + 1}/{len(eval_items)}] task_id={item.get('task_id')} "
            f"episode_success={score:.6f} (rollouts={num_rollouts})"
        )

    mean = total / float(len(eval_items))
    log(
        f"[collab_eval][summary] mode={args.mode} samples={len(eval_items)} rollouts={num_rollouts} "
        f"episode_success: total={total:.6f} mean={mean:.6f}"
    )
    out_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
