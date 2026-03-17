from __future__ import annotations

import json
import math
import os
import random
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from LLM_Collab_Minecraft.bridge_build_meta.adapters import AgentActionCandidates, BridgeBuildAdapter
from LLM_Collab_Minecraft.bridge_build_meta.models import (
    BeliefEncoder,
    CausalLMWithContextValueHead,
    StructuredValueCritic,
)


_ENV_METRIC_KEYS: Tuple[str, ...] = (
    "bonus_gap_st",
    "bonus_cc_merge",
    "bonus_y_connected",
    "penalty_n_adjacent",
    "penalty_block_cost",
    "bonus_terminal_connect",
    "new_cc_merge_count",
    "new_connected_y_count",
    "new_adjacent_n_count",
    "newly_placed_block_count",
    "num_valid_probes",
    "comm_tokens",
    "connected",
    "gap_st",
    "cc_component_count",
    "connected_y_count",
    "n_adjacent_count",
    "y_uncovered_count",
    "bonus_move_progress",
    "agent_a_target_distance",
    "agent_b_target_distance",
    "target_distance_total",
)


@dataclass
class BCMAACConfig:
    agent_learning_rate: float = 2.5e-6
    critic_learning_rate: float = 2.5e-6
    context_learning_rate: float = 1.0e-4
    rollout_buffer_size: int = 1
    train_batch_size: int = 1
    value_loss_coef: float = 0.6
    task_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: Optional[float] = 1.0
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = None
    num_train_epochs: int = 150
    num_agents: int = 2
    num_turns: int = 4
    discount: float = 0.9
    critic_type: str = "v"
    critic_backbone: str = "structured"
    logging_steps: int = 20
    eval_interval: int = 10
    eval_num_samples: int = 2
    eval_batch_size: int = 1
    context_hidden_dim: int = 128
    context_cnn_channels: int = 32
    context_scalar_hidden_dim: int = 64
    value_head_hidden_dim: Optional[int] = None
    actor_condition_dim: Optional[int] = None
    critic_condition_dim: Optional[int] = None
    actor_prompt_context_scale: float = 1.0
    actor_response_context_scale: float = 0.15
    score_chunk_size: int = 0
    actor_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1.")
        if self.num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if self.rollout_buffer_size < 1:
            raise ValueError("rollout_buffer_size must be >= 1.")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1.")
        if self.critic_type not in ("v", "q"):
            raise ValueError("critic_type must be 'v' or 'q'.")
        if str(self.critic_backbone).strip().lower() not in ("structured", "text"):
            raise ValueError("critic_backbone must be 'structured' or 'text'.")
        if self.logging_steps < 1:
            raise ValueError("logging_steps must be >= 1.")
        if self.actor_prompt_context_scale < 0.0:
            raise ValueError("actor_prompt_context_scale must be >= 0.")
        if self.actor_response_context_scale < 0.0:
            raise ValueError("actor_response_context_scale must be >= 0.")
        if self.score_chunk_size < 0:
            raise ValueError("score_chunk_size must be >= 0.")


@dataclass
class TurnRecord:
    prompts: List[str]
    completions: List[str]
    sequences: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    prompt_lens: List[int]
    response_lens: List[int]
    action_traces: List[List["ActionChoiceTrace"]]
    meta_obs: List[Any]
    reward: float
    metrics: Dict[str, Any]
    task_index: int
    terminated: bool


@dataclass
class EpisodeTrajectory:
    task_id: str
    turns: List[TurnRecord] = field(default_factory=list)


@dataclass
class ActionChoiceTrace:
    field_name: str
    response_prefix: str
    candidate_texts: List[str]
    chosen_index: int


class BCMAACTrainer:
    """Belief-conditioned actor-critic specialized for bridge_build."""

    def __init__(
        self,
        *,
        agent_model: Optional[Union[str, PreTrainedModel]] = None,
        critic_model: Optional[Union[str, PreTrainedModel]] = None,
        agents: Optional[Sequence[Union[str, PreTrainedModel]]] = None,
        critics: Optional[Sequence[Union[str, PreTrainedModel]]] = None,
        adapter: BridgeBuildAdapter,
        train_items: Sequence[Mapping[str, Any]],
        eval_items: Optional[Sequence[Mapping[str, Any]]] = None,
        args: Optional[BCMAACConfig] = None,
        model_config: Optional[Dict[str, Any]] = None,
        reward_processor: Optional[Any] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        agent_devices: Optional[Union[str, Sequence[str]]] = None,
        critic_devices: Optional[Union[str, Sequence[str]]] = None,
        verbose: bool = True,
    ) -> None:
        self.args = args or BCMAACConfig()
        self.adapter = adapter
        self.train_items = [dict(item) for item in train_items]
        self.eval_items = [dict(item) for item in (eval_items or [])]
        self.model_config = model_config or {}
        self.reward_processor = reward_processor or (lambda x: x)
        self.wandb_config = wandb_config or {}
        self.verbose = bool(verbose)
        self.critic_type = (self.args.critic_type or "v").lower()

        self.agent_devices = self._resolve_agent_devices(agent_devices)
        self.critic_device = self._resolve_critic_device(
            critic_devices,
            fallback_device=self.agent_devices[0],
        )
        self.critic_backbone = str(self.args.critic_backbone).strip().lower()
        self.agent_tokenizers = self._resolve_tokenizers(agent_model, agents)
        self.critic_tokenizer = (
            self._resolve_critic_tokenizer(critic_model, critics)
            if self.critic_backbone == "text"
            else None
        )
        if self.adapter.tokenizer is None:
            self.adapter.tokenizer = self.agent_tokenizers[0]

        self.agents = self._load_actor_models(agent_model=agent_model, agents=agents)

        joint_context_dim = (
            self.args.context_hidden_dim
            if self.args.num_agents == 1
            else self.args.context_hidden_dim * 3
        )
        self.context_encoder = BeliefEncoder(
            grid_channels=self.adapter.grid_channels,
            scalar_dim=self.adapter.scalar_dim,
            hidden_dim=self.args.context_hidden_dim,
            belief_dim=self.adapter.belief_dim,
            cnn_channels=self.args.context_cnn_channels,
            scalar_hidden_dim=self.args.context_scalar_hidden_dim,
        ).to(self.critic_device)
        self.critic = self._load_critic_model(
            critic_model=critic_model,
            critics=critics,
            joint_context_dim=joint_context_dim,
        )

        for actor, device in zip(self.agents, self.agent_devices):
            actor.to(device)
        self.critic.to(self.critic_device)

        self.actor_optimizers = [
            torch.optim.AdamW(actor.parameters(), lr=self.args.agent_learning_rate)
            for actor in self.agents
        ]
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.args.critic_learning_rate,
        )
        self.context_optimizer = torch.optim.AdamW(
            self.context_encoder.parameters(),
            lr=self.args.context_learning_rate,
        )

        self.env_step = 0
        self._last_logged_step = -1
        self.wandb_initialized = False
        if self.wandb_config and wandb is not None:
            self._init_wandb()

    def train(self) -> None:
        for epoch in range(self.args.num_train_epochs):
            items = list(self.train_items)
            random.shuffle(items)
            epoch_metrics: Dict[str, List[float]] = defaultdict(list)
            buffer: List[EpisodeTrajectory] = []

            iterator = tqdm(items, desc=f"Epoch {epoch + 1}/{self.args.num_train_epochs}") if self.verbose else items
            for item in iterator:
                trajectory = self._collect_episode(item, training=True)
                buffer.append(trajectory)

                if len(buffer) >= self.args.rollout_buffer_size:
                    self._flush_buffer(buffer=buffer, epoch_metrics=epoch_metrics)
                    buffer = []

            if buffer:
                self._flush_buffer(buffer=buffer, epoch_metrics=epoch_metrics)

            epoch_log = self._build_epoch_log(epoch_metrics)
            if self.verbose and epoch_log:
                print(f"Epoch {epoch + 1}/{self.args.num_train_epochs} metrics: {epoch_log}")
            if epoch_log:
                self._log_metrics(epoch_log)

            if self.eval_items and self.args.eval_interval > 0 and (epoch + 1) % self.args.eval_interval == 0:
                eval_summary = self.evaluate()
                if self.verbose:
                    print(f"Eval @ epoch {epoch + 1}: {eval_summary}")
                self._log_metrics(eval_summary)

    def evaluate(self) -> Dict[str, float]:
        max_items = min(len(self.eval_items), max(1, int(self.args.eval_num_samples)))
        if max_items <= 0:
            return {}
        trajectories: List[EpisodeTrajectory] = []
        for item in self.eval_items[:max_items]:
            trajectory = self._collect_episode(item, training=False)
            trajectories.append(trajectory)
        _advantages, rollout_metrics = self._compute_rollout_statistics(trajectories)
        flat = self._flatten_turn_metrics(rollout_metrics)
        return {f"eval/{key}": value for key, value in flat.items()}

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.context_encoder.save_path = output_dir  # type: ignore[attr-defined]
        torch.save(self.context_encoder.state_dict(), os.path.join(output_dir, "context_encoder.pt"))
        if self.critic_backbone == "text":
            self.critic.model.save_pretrained(os.path.join(output_dir, "critic"))
            if self.critic_tokenizer is not None:
                self.critic_tokenizer.save_pretrained(os.path.join(output_dir, "critic"))
            torch.save(self.critic.state_dict(), os.path.join(output_dir, "critic_context_head.pt"))
        else:
            torch.save(self.critic.state_dict(), os.path.join(output_dir, "structured_critic.pt"))
        for idx, actor in enumerate(self.agents):
            actor_dir = os.path.join(output_dir, f"agent_{idx}")
            actor.model.save_pretrained(actor_dir)
            self.agent_tokenizers[idx].save_pretrained(actor_dir)
            torch.save(actor.state_dict(), os.path.join(actor_dir, "context_head.pt"))

    def _collect_episode(self, item: Mapping[str, Any], *, training: bool) -> EpisodeTrajectory:
        self.context_encoder.eval()
        self.critic.eval()
        for actor in self.agents:
            actor.eval()
        payload = self.adapter.reset_item_state(item)
        task_id = str(item.get("task_id") or payload_to_task_id(payload))
        trajectory = EpisodeTrajectory(task_id=task_id)

        hidden_states: List[Optional[torch.Tensor]] = [None for _ in range(self.args.num_agents)]
        prompt_history: List[List[str]] = [[] for _ in range(self.args.num_agents)]
        response_history: List[List[str]] = [[] for _ in range(self.args.num_agents)]
        previous_completions: List[str] = ["" for _ in range(self.args.num_agents)]
        previous_metrics: Dict[str, Any] = {}

        for turn_idx in range(self.args.num_turns):
            if turn_idx == 0:
                prompts = self.adapter.initial_prompts(payload)
            else:
                prompts = self.adapter.followup_prompts(
                    payload=payload,
                    metrics=previous_metrics,
                    previous_completions=previous_completions,
                    prompt_history_per_agent=prompt_history,
                    response_history_per_agent=response_history,
                )

            for agent_idx, prompt in enumerate(prompts):
                prompt_history[agent_idx].append(prompt)

            meta_obs = self.adapter.build_meta_observations(payload)
            action_candidates = self.adapter.build_action_candidates(payload)
            contexts: List[torch.Tensor] = []
            with torch.no_grad():
                for agent_idx in range(self.args.num_agents):
                    encoder_out = self.context_encoder.forward_step(
                        grid=meta_obs[agent_idx].grid.unsqueeze(0).to(self.critic_device),
                        scalars=meta_obs[agent_idx].scalars.unsqueeze(0).to(self.critic_device),
                        hidden=hidden_states[agent_idx],
                    )
                    hidden_states[agent_idx] = encoder_out.hidden.detach()
                    contexts.append(encoder_out.context.detach())

            sequences: List[torch.Tensor] = []
            attention_masks: List[torch.Tensor] = []
            prompt_lens: List[int] = []
            response_lens: List[int] = []
            completions: List[str] = []
            action_traces: List[List[ActionChoiceTrace]] = []
            with torch.no_grad():
                for agent_idx in range(self.args.num_agents):
                    gen = self._sample_constrained_action(
                        model=self.agents[agent_idx],
                        tokenizer=self.agent_tokenizers[agent_idx],
                        prompt=prompts[agent_idx],
                        context_vec=contexts[agent_idx],
                        action_candidates=action_candidates[agent_idx],
                        training=training,
                    )
                    sequences.append(gen["sequence"].cpu())
                    attention_masks.append(gen["attention_mask"].cpu())
                    prompt_lens.append(int(gen["prompt_len"]))
                    response_lens.append(int(gen["response_len"]))
                    completions.append(str(gen["completion"]))
                    action_traces.append(list(gen["action_traces"]))

            next_payload, metrics, _actions = self.adapter.transition(payload, completions)
            reward = float(self.reward_processor(float(metrics.get("reward", 0.0))))
            self.adapter.debug_turn(
                payload=next_payload,
                turn_idx=turn_idx + 1,
                reward=reward,
                metrics=metrics,
                agent_outputs=completions,
            )

            trajectory.turns.append(
                TurnRecord(
                    prompts=list(prompts),
                    completions=list(completions),
                    sequences=sequences,
                    attention_masks=attention_masks,
                    prompt_lens=prompt_lens,
                    response_lens=response_lens,
                    action_traces=action_traces,
                    meta_obs=meta_obs,
                    reward=reward,
                    metrics=dict(metrics),
                    task_index=int(meta_obs[0].task_index),
                    terminated=bool(metrics.get("terminated", False)),
                )
            )

            previous_completions = list(completions)
            previous_metrics = dict(metrics)
            for agent_idx, comp in enumerate(completions):
                response_history[agent_idx].append(comp)
            payload = next_payload
            self.env_step += 1
            if bool(metrics.get("terminated", False)):
                break

        return trajectory

    def _update(self, trajectories: Sequence[EpisodeTrajectory]) -> Dict[str, float]:
        if not trajectories:
            return {}

        normalized_advantages, rollout_metrics = self._compute_rollout_statistics(trajectories)
        total_turns = max(1, sum(len(traj.turns) for traj in trajectories))

        self.context_encoder.train()
        self.critic.train()
        for actor in self.agents:
            actor.train()

        per_turn_policy_loss: Dict[int, List[float]] = defaultdict(list)
        per_turn_value_loss: Dict[int, List[float]] = defaultdict(list)
        per_turn_belief_loss: Dict[int, List[float]] = defaultdict(list)
        per_turn_belief_accuracy: Dict[int, List[float]] = defaultdict(list)
        extra_loss_accum = {
            "belief_loss": 0.0,
            "belief_accuracy": 0.0,
            "entropy": 0.0,
        }

        self.context_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad()

        for traj_idx, traj in enumerate(trajectories):
            returns = self._discounted_returns(traj)
            hidden_states: List[Optional[torch.Tensor]] = [None for _ in range(self.args.num_agents)]

            for turn_idx, turn in enumerate(traj.turns):
                contexts: List[torch.Tensor] = []
                turn_belief_losses: List[torch.Tensor] = []
                turn_belief_accuracies: List[torch.Tensor] = []
                for agent_idx in range(self.args.num_agents):
                    obs = turn.meta_obs[agent_idx]
                    hidden_in = hidden_states[agent_idx]
                    if hidden_in is not None:
                        hidden_in = hidden_in.detach()
                    encoder_out = self.context_encoder.forward_step(
                        grid=obs.grid.unsqueeze(0).to(self.critic_device),
                        scalars=obs.scalars.unsqueeze(0).to(self.critic_device),
                        hidden=hidden_in,
                    )
                    hidden_states[agent_idx] = encoder_out.hidden.detach()
                    contexts.append(encoder_out.context)
                    if bool(obs.belief_mask.any()):
                        masked_logits = encoder_out.belief_logits.view(-1)[obs.belief_mask.to(self.critic_device)]
                        masked_target = obs.belief_target.to(self.critic_device, dtype=masked_logits.dtype)[
                            obs.belief_mask.to(self.critic_device)
                        ]
                        belief_loss = F.binary_cross_entropy_with_logits(masked_logits, masked_target)
                        belief_preds = (masked_logits >= 0).to(dtype=masked_target.dtype)
                        belief_accuracy = (belief_preds == masked_target).to(dtype=torch.float32).mean()
                        turn_belief_losses.append(belief_loss)
                        turn_belief_accuracies.append(belief_accuracy)

                joint_context = self.adapter.build_joint_context(contexts)
                value = self._critic_value(joint_context, turn=turn)
                ret = torch.tensor([returns[turn_idx]], device=self.critic_device, dtype=value.dtype)
                value_loss_turn = (value - ret).pow(2).mean()

                actor_turn_loss_values: List[float] = []
                entropy_turn_values: List[float] = []
                adv_value = float(normalized_advantages[traj_idx][turn_idx])
                num_actor_terms = max(1, self.args.num_agents)
                for agent_idx in range(self.args.num_agents):
                    actor_device = self.agent_devices[agent_idx]
                    prompt_tokens = turn.sequences[agent_idx][: int(turn.prompt_lens[agent_idx])].unsqueeze(0).to(actor_device)
                    actor_loss_value, entropy_value = self._backprop_action_traces(
                        model=self.agents[agent_idx],
                        tokenizer=self.agent_tokenizers[agent_idx],
                        prompt_tokens=prompt_tokens,
                        action_traces=turn.action_traces[agent_idx],
                        context_vec=contexts[agent_idx].detach(),
                        advantage=float(adv_value),
                        loss_scale=float(total_turns) * float(num_actor_terms),
                    )
                    actor_turn_loss_values.append(float(actor_loss_value))
                    entropy_turn_values.append(float(entropy_value))

                belief_loss_turn = (
                    torch.stack(turn_belief_losses).mean()
                    if turn_belief_losses
                    else torch.zeros((), device=self.critic_device, dtype=ret.dtype)
                )
                belief_accuracy_turn = (
                    torch.stack(turn_belief_accuracies).mean()
                    if turn_belief_accuracies
                    else torch.zeros((), device=self.critic_device, dtype=ret.dtype)
                )
                shared_turn_loss = (
                    self.args.value_loss_coef * value_loss_turn
                    + self.args.task_loss_coef * belief_loss_turn
                ) / float(total_turns)
                shared_turn_loss.backward()

                actor_loss_turn = (
                    sum(actor_turn_loss_values) / float(len(actor_turn_loss_values))
                    if actor_turn_loss_values
                    else 0.0
                )
                entropy_turn = (
                    sum(entropy_turn_values) / float(len(entropy_turn_values))
                    if entropy_turn_values
                    else 0.0
                )

                per_turn_policy_loss[turn_idx].append(float(actor_loss_turn))
                per_turn_value_loss[turn_idx].append(float(value_loss_turn.detach().item()))
                if turn_belief_losses:
                    per_turn_belief_loss[turn_idx].append(float(belief_loss_turn.detach().item()))
                    per_turn_belief_accuracy[turn_idx].append(float(belief_accuracy_turn.detach().item()))
                extra_loss_accum["belief_loss"] += float(belief_loss_turn.detach().item())
                extra_loss_accum["belief_accuracy"] += float(belief_accuracy_turn.detach().item())
                extra_loss_accum["entropy"] += float(entropy_turn)

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            self._clip_grad_norm(self.context_encoder, float(self.args.max_grad_norm))
            self._clip_grad_norm(self.critic, float(self.args.max_grad_norm))
            for actor in self.agents:
                self._clip_grad_norm(actor, float(self.args.max_grad_norm))

        self.context_optimizer.step()
        self.critic_optimizer.step()
        for optimizer in self.actor_optimizers:
            optimizer.step()

        for turn_idx, values in per_turn_policy_loss.items():
            if values:
                rollout_metrics.setdefault(turn_idx, {})["policy_loss"] = float(sum(values) / len(values))
        for turn_idx, values in per_turn_value_loss.items():
            if values:
                rollout_metrics.setdefault(turn_idx, {})["value_loss"] = float(sum(values) / len(values))
        for turn_idx, values in per_turn_belief_loss.items():
            if values:
                rollout_metrics.setdefault(turn_idx, {})["belief_loss"] = float(sum(values) / len(values))
                rollout_metrics.setdefault(turn_idx, {})["task_loss"] = float(sum(values) / len(values))
        for turn_idx, values in per_turn_belief_accuracy.items():
            if values:
                rollout_metrics.setdefault(turn_idx, {})["belief_accuracy"] = float(sum(values) / len(values))

        rollout_metrics.setdefault(0, {})["belief_loss"] = float(extra_loss_accum["belief_loss"] / float(total_turns))
        rollout_metrics.setdefault(0, {})["belief_accuracy"] = float(extra_loss_accum["belief_accuracy"] / float(total_turns))
        rollout_metrics.setdefault(0, {})["task_loss"] = float(extra_loss_accum["belief_loss"] / float(total_turns))
        rollout_metrics.setdefault(0, {})["entropy"] = float(extra_loss_accum["entropy"] / float(total_turns))
        return self._flatten_turn_metrics(rollout_metrics)

    def _sample_completion(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt: str,
        context_vec: torch.Tensor,
        training: bool,
    ) -> Dict[str, Any]:
        model_device = self._module_device(model)
        max_ctx = self._max_context_length(model.model, tokenizer)
        prompt_budget = max(1, max_ctx - 1)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_budget,
        )
        prompt_input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded["attention_mask"].to(model_device)
        prompt_len = int(prompt_input_ids.size(1))
        context_vec = context_vec.to(model_device)

        with torch.no_grad():
            context_mask = torch.full_like(
                prompt_input_ids,
                float(self.args.actor_prompt_context_scale),
                device=model_device,
                dtype=torch.float32,
            )
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
                context_vec=context_vec,
                context_mask=context_mask,
                use_cache=True,
                output_values=False,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            generated: List[torch.Tensor] = []
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            max_steps = max(1, min(int(self.args.max_new_tokens), max_ctx - prompt_len))

            for _ in range(max_steps):
                next_token = self._sample_token(next_token_logits, training=training)
                generated.append(next_token)
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                token_fragment = tokenizer.decode(
                    next_token.view(-1),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if "}" in token_fragment:
                    partial_ids = torch.stack(generated, dim=0).view(1, -1)
                    partial_completion = tokenizer.decode(
                        partial_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if self._completion_has_action_json(partial_completion):
                        break
                input_ids = next_token.view(1, 1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=model_device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    context_vec=context_vec,
                    context_mask=torch.full_like(
                        input_ids,
                        float(self.args.actor_response_context_scale),
                        dtype=torch.float32,
                        device=model_device,
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_values=False,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        if generated:
            generated_ids = torch.stack(generated, dim=0).view(1, -1)
        else:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
            generated_ids = torch.tensor([[int(pad_id)]], device=model_device)
        full_sequence = torch.cat([prompt_input_ids, generated_ids], dim=1)
        full_attention_mask = torch.ones_like(full_sequence, device=model_device)
        completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {
            "sequence": full_sequence.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "prompt_len": prompt_len,
            "response_len": int(generated_ids.size(1)),
            "completion": completion,
        }

    def _sample_constrained_action(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt: str,
        context_vec: torch.Tensor,
        action_candidates: AgentActionCandidates,
        training: bool,
    ) -> Dict[str, Any]:
        candidate_values: Dict[str, Sequence[Any]] = {
            "comm": list(action_candidates.comm_options or [{}]),
            "probe": list(action_candidates.probe_options or [[]]),
            "cmds": list(action_candidates.cmd_options or [[]]),
            "path": list(action_candidates.path_options or [[]]),
        }
        response_budget = self._serialize_action_json(
            {
                field_name: self._select_longest_json_value(values)
                for field_name, values in candidate_values.items()
            }
        )
        prompt_input_ids = self._prepare_prompt_input_ids(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            response_budget=response_budget,
        )
        context_vec = context_vec.to(self._module_device(model))
        selected_values: Dict[str, Any] = {}
        choice_traces: List[ActionChoiceTrace] = []
        field_order = ("comm", "probe", "cmds", "path")
        for field_name in field_order:
            response_prefix = self._build_action_prefix(selected_values, field_name=field_name)
            candidate_texts = [
                self._serialize_json_value(value)
                for value in (candidate_values.get(field_name) or [self._default_action_value(field_name)])
            ]
            scores = self._score_constrained_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_input_ids,
                response_prefix=response_prefix,
                candidate_texts=candidate_texts,
                context_vec=context_vec,
            )
            chosen_index = self._select_candidate_index(scores=scores, training=training)
            selected_values[field_name] = (candidate_values[field_name] or [self._default_action_value(field_name)])[chosen_index]
            choice_traces.append(
                ActionChoiceTrace(
                    field_name=field_name,
                    response_prefix=response_prefix,
                    candidate_texts=list(candidate_texts),
                    chosen_index=int(chosen_index),
                )
            )

        completion = self._serialize_action_json(selected_values)
        return {
            "completion": completion,
            "action_traces": choice_traces,
            **self._encode_prompt_completion(
                tokenizer=tokenizer,
                prompt_input_ids=prompt_input_ids,
                completion=completion,
                device=self._module_device(model),
            ),
        }

    def _completion_has_action_json(self, text: str) -> bool:
        raw = str(text or "")
        if not raw:
            return False
        action_keys = ("comm", "probe", "cmds", "path")
        start = raw.find("{")
        if start < 0:
            return False

        depth = 0
        in_string = False
        quote_char = ""
        escape = False
        for idx in range(start, len(raw)):
            cur = raw[idx]
            if in_string:
                if escape:
                    escape = False
                    continue
                if cur == "\\":
                    escape = True
                    continue
                if cur == quote_char:
                    in_string = False
                continue
            if cur in ('"', "'"):
                in_string = True
                quote_char = cur
                continue
            if cur == "{":
                depth += 1
                continue
            if cur != "}":
                continue
            depth -= 1
            if depth != 0:
                continue
            candidate = raw[start : idx + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                continue
            return isinstance(obj, dict) and any(key in obj for key in action_keys)
        return False

    def _sample_token(self, logits: torch.Tensor, *, training: bool) -> torch.Tensor:
        logits = logits.squeeze(0)
        temperature = float(self.args.temperature if training else max(self.args.temperature, 1e-6))
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        if self.args.top_k is not None and self.args.top_k > 0:
            top_k = min(int(self.args.top_k), logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            threshold = values[..., -1, None]
            logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
        if self.args.top_p is not None and self.args.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            sorted_mask = cumulative > float(self.args.top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            filtered = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_indices, filtered)
        probs = torch.softmax(logits, dim=-1)
        if not training:
            return torch.argmax(probs, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _backprop_action_traces(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt_tokens: torch.Tensor,
        action_traces: Sequence[ActionChoiceTrace],
        context_vec: torch.Tensor,
        advantage: float,
        loss_scale: float,
    ) -> Tuple[float, float]:
        if not action_traces:
            return 0.0, 0.0
        model_device = self._module_device(model)
        prompt_tokens = prompt_tokens.to(model_device)
        context_vec = context_vec.to(model_device).detach()
        advantage_tensor = torch.tensor(float(advantage), device=model_device, dtype=torch.float32)
        total_actor_loss = 0.0
        entropy_values: List[float] = []
        num_traces = max(1, len(action_traces))

        for trace in action_traces:
            scores = self._score_constrained_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_tokens,
                response_prefix=trace.response_prefix,
                candidate_texts=trace.candidate_texts,
                context_vec=context_vec,
            )
            scaled_scores = scores / max(float(self.args.temperature), 1e-6)
            log_probs = F.log_softmax(scaled_scores, dim=-1)
            probs = log_probs.exp()
            chosen_logprob = log_probs[int(trace.chosen_index)]
            trace_entropy = -(probs * log_probs).sum()
            trace_loss = -(chosen_logprob * advantage_tensor.to(dtype=chosen_logprob.dtype))
            scaled_trace_loss = (
                trace_loss
                - float(self.args.entropy_coef) * (trace_entropy / float(num_traces))
            ) / max(float(loss_scale), 1.0)
            scaled_trace_loss.backward()
            total_actor_loss += float(trace_loss.detach().item())
            entropy_values.append(float(trace_entropy.detach().item()))

        entropy_mean = float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0
        return total_actor_loss, entropy_mean

    def _sequence_logprob(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt_tokens: torch.Tensor,
        action_traces: Sequence[ActionChoiceTrace],
        context_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not action_traces:
            zero = torch.zeros(1, device=prompt_tokens.device, dtype=torch.float32)
            return zero, zero
        model_device = self._module_device(model)
        prompt_tokens = prompt_tokens.to(model_device)
        context_vec = context_vec.to(model_device)
        choice_log_probs: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        for trace in action_traces:
            scores = self._score_constrained_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt_input_ids=prompt_tokens,
                response_prefix=trace.response_prefix,
                candidate_texts=trace.candidate_texts,
                context_vec=context_vec,
            )
            scaled_scores = scores / max(float(self.args.temperature), 1e-6)
            log_probs = F.log_softmax(scaled_scores, dim=-1)
            probs = log_probs.exp()
            choice_log_probs.append(log_probs[int(trace.chosen_index)])
            entropy_terms.append(-(probs * log_probs).sum())
        return torch.stack(choice_log_probs).sum().view(1), torch.stack(entropy_terms).mean().view(1)

    def _prepare_prompt_input_ids(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt: str,
        response_budget: str,
    ) -> torch.Tensor:
        model_device = self._module_device(model)
        max_ctx = self._max_context_length(model.model, tokenizer)
        suffix_ids = tokenizer(
            response_budget,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        prompt_budget = max(1, max_ctx - int(suffix_ids.size(1)))
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_budget,
            add_special_tokens=False,
        )
        return encoded["input_ids"].to(model_device)

    def _encode_prompt_completion(
        self,
        *,
        tokenizer: Any,
        prompt_input_ids: torch.Tensor,
        completion: str,
        device: torch.device,
    ) -> Dict[str, Any]:
        completion_ids = tokenizer(
            completion,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        if completion_ids.numel() == 0:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
            completion_ids = torch.tensor([[int(pad_id)]], dtype=prompt_input_ids.dtype, device=device)
        full_sequence = torch.cat([prompt_input_ids, completion_ids], dim=1)
        full_attention_mask = torch.ones_like(full_sequence, device=device)
        return {
            "sequence": full_sequence.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "prompt_len": int(prompt_input_ids.size(1)),
            "response_len": int(completion_ids.size(1)),
        }

    def _serialize_action_json(self, values: Mapping[str, Any]) -> str:
        ordered = {
            "comm": copy.deepcopy(values.get("comm", {})),
            "probe": copy.deepcopy(values.get("probe", [])),
            "cmds": copy.deepcopy(values.get("cmds", [])),
            "path": copy.deepcopy(values.get("path", [])),
        }
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))

    def _build_action_prefix(self, selected_values: Mapping[str, Any], *, field_name: str) -> str:
        prefix_parts: List[str] = ["{"]
        ordered_fields = ("comm", "probe", "cmds", "path")
        first = True
        for name in ordered_fields:
            if name == field_name:
                if not first:
                    prefix_parts.append(",")
                prefix_parts.append(json.dumps(name, ensure_ascii=False))
                prefix_parts.append(":")
                break
            if name not in selected_values:
                break
            if not first:
                prefix_parts.append(",")
            prefix_parts.append(json.dumps(name, ensure_ascii=False))
            prefix_parts.append(":")
            prefix_parts.append(self._serialize_json_value(selected_values[name]))
            first = False
        return "".join(prefix_parts)

    def _serialize_json_value(self, value: Any) -> str:
        return json.dumps(copy.deepcopy(value), ensure_ascii=False, separators=(",", ":"))

    def _select_longest_json_value(self, values: Sequence[Any]) -> Any:
        if not values:
            return {}
        return max(values, key=lambda item: len(self._serialize_json_value(item)))

    def _default_action_value(self, field_name: str) -> Any:
        if field_name == "comm":
            return {}
        return []

    def _score_constrained_candidates(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt_input_ids: torch.Tensor,
        response_prefix: str,
        candidate_texts: Sequence[str],
        context_vec: torch.Tensor,
    ) -> torch.Tensor:
        if not candidate_texts:
            return torch.zeros(1, device=prompt_input_ids.device, dtype=torch.float32)
        model_device = self._module_device(model)
        prompt_input_ids = prompt_input_ids.to(model_device)
        context_vec = context_vec.to(model_device)
        if tokenizer is None:
            raise ValueError("tokenizer is required for constrained scoring.")
        prefix_ids = tokenizer(
            response_prefix,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(model_device)
        candidate_id_list: List[torch.Tensor] = []
        for text in candidate_texts:
            candidate_ids = tokenizer(
                str(text),
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"].to(model_device)
            if candidate_ids.numel() == 0:
                pad_id = getattr(tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
                candidate_ids = torch.tensor([[int(pad_id)]], dtype=prompt_input_ids.dtype, device=model_device)
            candidate_id_list.append(candidate_ids)

        chunk_size = int(self.args.score_chunk_size or 0)
        if chunk_size <= 0:
            chunk_size = len(candidate_id_list)
        chunk_scores: List[torch.Tensor] = []
        for start in range(0, len(candidate_id_list), chunk_size):
            chunk_scores.append(
                self._score_constrained_candidate_chunk(
                    model=model,
                    prompt_input_ids=prompt_input_ids,
                    prefix_ids=prefix_ids,
                    candidate_id_list=candidate_id_list[start : start + chunk_size],
                    context_vec=context_vec,
                    pad_id=int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0),
                )
            )
        return torch.cat(chunk_scores, dim=0)

    def _score_constrained_candidate_chunk(
        self,
        *,
        model: CausalLMWithContextValueHead,
        prompt_input_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        candidate_id_list: Sequence[torch.Tensor],
        context_vec: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        model_device = self._module_device(model)
        prompt_len = int(prompt_input_ids.size(1))
        prefix_len = int(prefix_ids.size(1))
        batch_size = max(1, len(candidate_id_list))
        max_len = max(prompt_len + prefix_len + int(ids.size(1)) for ids in candidate_id_list)
        sequences = torch.full(
            (batch_size, max_len),
            int(pad_id),
            dtype=prompt_input_ids.dtype,
            device=model_device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=model_device)
        candidate_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.float32, device=model_device)

        for row, candidate_ids in enumerate(candidate_id_list):
            full_ids = torch.cat([prompt_input_ids, prefix_ids, candidate_ids], dim=1)
            seq_len = int(full_ids.size(1))
            sequences[row, :seq_len] = full_ids.squeeze(0)
            attention_mask[row, :seq_len] = 1
            candidate_len = int(candidate_ids.size(1))
            start_idx = max(prompt_len + prefix_len - 1, 0)
            end_idx = start_idx + candidate_len
            candidate_mask[row, start_idx:end_idx] = 1.0

        context_mask = torch.full(
            (batch_size, max_len),
            float(self.args.actor_response_context_scale),
            dtype=torch.float32,
            device=model_device,
        )
        if prompt_len > 0:
            context_mask[:, :prompt_len] = float(self.args.actor_prompt_context_scale)

        outputs = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            context_vec=context_vec.expand(batch_size, -1),
            context_mask=context_mask,
            output_values=False,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_targets = sequences[:, 1:]
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs * candidate_mask
        denom = candidate_mask.sum(dim=-1).clamp(min=1.0)
        return masked_log_probs.sum(dim=-1) / denom

    def _select_candidate_index(self, *, scores: torch.Tensor, training: bool) -> int:
        if scores.numel() <= 1:
            return 0
        if not training:
            return int(torch.argmax(scores).item())
        scaled_scores = scores / max(float(self.args.temperature), 1e-6)
        probs = torch.softmax(scaled_scores, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _critic_value_from_text(self, prompt: str, joint_context: torch.Tensor) -> torch.Tensor:
        max_ctx = self._max_context_length(self.critic.model, self.critic_tokenizer)
        encoded = self.critic_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_ctx,
        )
        input_ids = encoded["input_ids"].to(self.critic_device)
        attention_mask = encoded["attention_mask"].to(self.critic_device)
        joint_context = joint_context.to(self.critic_device)
        context_mask = torch.ones_like(input_ids, dtype=torch.float32, device=self.critic_device)
        outputs = self.critic(
            input_ids=input_ids,
            attention_mask=attention_mask,
            context_vec=joint_context,
            context_mask=context_mask,
            output_values=True,
        )
        if outputs.values is None:
            raise RuntimeError("Critic value head returned None.")
        last_index = int(input_ids.size(1) - 1)
        return outputs.values[:, last_index]

    def _critic_value(self, joint_context: torch.Tensor, *, turn: TurnRecord) -> torch.Tensor:
        if self.critic_backbone == "structured":
            return self.critic(joint_context.to(self.critic_device))
        critic_input = self._build_critic_input(
            turn.prompts,
            action_completions=turn.completions if self.critic_type == "q" else None,
        )
        return self._critic_value_from_text(critic_input, joint_context)

    def _build_critic_input(
        self,
        prompts: Sequence[str],
        action_completions: Optional[Sequence[str]] = None,
    ) -> str:
        base = "\n\n".join([f"[Agent {idx}] {prompt}" for idx, prompt in enumerate(prompts)])
        if self.critic_type == "v":
            return base
        action_completions = list(action_completions or [])
        action_lines = ["[Joint Action]"]
        for idx, comp in enumerate(action_completions):
            action_lines.append(f"[Agent {idx} action]\n{comp}")
        return base + "\n\n" + "\n\n".join(action_lines)

    def _discounted_returns(self, trajectory: EpisodeTrajectory) -> List[float]:
        out = [0.0 for _ in trajectory.turns]
        future = 0.0
        for idx in reversed(range(len(trajectory.turns))):
            future = float(trajectory.turns[idx].reward) + float(self.args.discount) * future
            out[idx] = future
        return out

    def _normalize_advantages(self, advantages: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not advantages:
            return []
        flat = torch.stack([adv.view(-1)[0].to(torch.float32) for adv in advantages])
        if flat.numel() <= 1:
            return [adv.to(torch.float32) for adv in advantages]
        mean = flat.mean()
        std = flat.std(unbiased=False).clamp(min=1e-6)
        return [((adv.to(torch.float32) - mean) / std) for adv in advantages]

    def _compute_rollout_statistics(
        self,
        trajectories: Sequence[EpisodeTrajectory],
    ) -> Tuple[List[List[float]], Dict[int, Dict[str, float]]]:
        self.context_encoder.eval()
        self.critic.eval()
        for actor in self.agents:
            actor.eval()

        advantages: List[List[float]] = []
        flat_advantages: List[torch.Tensor] = []
        turn_rewards: Dict[int, List[float]] = defaultdict(list)
        turn_returns: Dict[int, List[float]] = defaultdict(list)
        turn_values: Dict[int, List[float]] = defaultdict(list)
        turn_targets: Dict[int, List[float]] = defaultdict(list)
        turn_env_metrics: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        with torch.no_grad():
            for traj in trajectories:
                returns = self._discounted_returns(traj)
                hidden_states: List[Optional[torch.Tensor]] = [None for _ in range(self.args.num_agents)]
                traj_advantages: List[float] = []

                for turn_idx, turn in enumerate(traj.turns):
                    contexts: List[torch.Tensor] = []
                    for agent_idx in range(self.args.num_agents):
                        obs = turn.meta_obs[agent_idx]
                        hidden_in = hidden_states[agent_idx]
                        if hidden_in is not None:
                            hidden_in = hidden_in.detach()
                        encoder_out = self.context_encoder.forward_step(
                            grid=obs.grid.unsqueeze(0).to(self.critic_device),
                            scalars=obs.scalars.unsqueeze(0).to(self.critic_device),
                            hidden=hidden_in,
                        )
                        hidden_states[agent_idx] = encoder_out.hidden.detach()
                        contexts.append(encoder_out.context.detach())

                    joint_context = self.adapter.build_joint_context(contexts)
                    value = self._critic_value(joint_context, turn=turn).detach().view(-1)[0]
                    turn_rewards[turn_idx].append(float(turn.reward))
                    turn_returns[turn_idx].append(float(returns[turn_idx]))
                    turn_values[turn_idx].append(float(value.item()))
                    turn_targets[turn_idx].append(float(returns[turn_idx]))
                    for key in _ENV_METRIC_KEYS:
                        if key not in turn.metrics:
                            continue
                        raw_value = turn.metrics.get(key)
                        if isinstance(raw_value, bool):
                            turn_env_metrics[turn_idx][key].append(1.0 if raw_value else 0.0)
                            continue
                        try:
                            turn_env_metrics[turn_idx][key].append(float(raw_value))
                        except (TypeError, ValueError):
                            continue
                    adv = torch.tensor(
                        float(returns[turn_idx]) - float(value.item()),
                        device=self.critic_device,
                        dtype=torch.float32,
                    )
                    traj_advantages.append(float(adv.item()))
                    flat_advantages.append(adv)

                advantages.append(traj_advantages)

        normalized_flat = self._normalize_advantages(flat_advantages)
        cursor = 0
        normalized_advantages: List[List[float]] = []
        for traj_advantages in advantages:
            count = len(traj_advantages)
            normalized_advantages.append(
                [float(normalized_flat[cursor + idx].item()) for idx in range(count)]
            )
            cursor += count
        rollout_metrics: Dict[int, Dict[str, float]] = {}
        for turn_idx in sorted(set(turn_rewards.keys()) | set(turn_returns.keys()) | set(turn_values.keys()) | set(turn_targets.keys())):
            metrics: Dict[str, float] = {}
            if turn_rewards.get(turn_idx):
                metrics["reward_mean"] = float(sum(turn_rewards[turn_idx]) / len(turn_rewards[turn_idx]))
            if turn_returns.get(turn_idx):
                metrics["expected_return"] = float(sum(turn_returns[turn_idx]) / len(turn_returns[turn_idx]))
            if turn_values.get(turn_idx):
                metrics["value_pred_mean"] = float(sum(turn_values[turn_idx]) / len(turn_values[turn_idx]))
            if turn_targets.get(turn_idx):
                metrics["value_target_mean"] = float(sum(turn_targets[turn_idx]) / len(turn_targets[turn_idx]))
            if turn_env_metrics.get(turn_idx):
                for key, values in turn_env_metrics[turn_idx].items():
                    if values:
                        metrics[key] = float(sum(values) / len(values))
            if metrics:
                rollout_metrics[turn_idx] = metrics
        return normalized_advantages, rollout_metrics

    def _summarize_epoch(self, metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, values in metrics.items():
            vals = [float(v) for v in values if v is not None]
            if vals:
                out[key] = float(sum(vals) / len(vals))
        return out

    def _flatten_turn_metrics(
        self,
        metrics_by_turn: Mapping[int, Mapping[str, float]],
    ) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        for turn_idx in sorted(metrics_by_turn.keys()):
            prefix = f"turn_{turn_idx + 1}/"
            for key, value in metrics_by_turn[turn_idx].items():
                flat[prefix + key] = float(value)
        return flat

    def _build_epoch_log(self, epoch_metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
        epoch_log: Dict[str, float] = {}
        for turn_idx in range(max(1, int(self.args.num_turns))):
            prefix = f"turn_{turn_idx + 1}/"

            def _maybe_log(metric_key: str, epoch_key: str) -> None:
                values = epoch_metrics.get(prefix + metric_key)
                if values:
                    epoch_log[prefix + epoch_key] = float(sum(values) / len(values))

            _maybe_log("reward_mean", "epoch_reward_mean")
            _maybe_log("expected_return", "epoch_avg_return")
            _maybe_log("value_pred_mean", "epoch_value_pred_mean")
            _maybe_log("value_target_mean", "epoch_value_target_mean")
            _maybe_log("policy_loss", "epoch_policy_loss")
            _maybe_log("value_loss", "epoch_value_loss")
            _maybe_log("belief_loss", "epoch_belief_loss")
            _maybe_log("belief_accuracy", "epoch_belief_accuracy")
            if turn_idx == 0:
                _maybe_log("task_loss", "epoch_task_loss")
                _maybe_log("entropy", "epoch_entropy")
        return epoch_log

    def _flush_buffer(
        self,
        *,
        buffer: Sequence[EpisodeTrajectory],
        epoch_metrics: Dict[str, List[float]],
    ) -> None:
        batch_size = max(1, int(self.args.train_batch_size))
        for start in range(0, len(buffer), batch_size):
            batch = list(buffer[start : start + batch_size])
            if not batch:
                continue
            update_metrics = self._update(batch)
            for key, value in update_metrics.items():
                epoch_metrics[key].append(value)
            if self._should_log_train():
                self._log_metrics(update_metrics)

    def _log_metrics(self, metrics: Mapping[str, float]) -> None:
        if not metrics:
            return
        if not self.wandb_initialized or wandb is None:
            return
        wandb.log(dict(metrics), step=self.env_step)

    def _should_log_train(self) -> bool:
        interval = int(getattr(self.args, "logging_steps", 1))
        if interval <= 1:
            self._last_logged_step = self.env_step
            return True
        if self._last_logged_step < 0 or (self.env_step - self._last_logged_step) >= interval:
            self._last_logged_step = self.env_step
            return True
        return False

    def _init_wandb(self) -> None:
        if wandb is None or not self.wandb_config:
            return
        init_kwargs = {
            "project": self.wandb_config.get("project", "bridge_build_meta"),
            "name": self.wandb_config.get("name", self.wandb_config.get("run_name", "bcmaac")),
            "entity": self.wandb_config.get("entity"),
            "config": {
                "algorithm": "BCMAAC",
                "num_agents": self.args.num_agents,
                "num_turns": self.args.num_turns,
                "critic_type": self.args.critic_type,
                "critic_backbone": self.critic_backbone,
                "context_hidden_dim": self.args.context_hidden_dim,
            },
        }
        wandb_dir = self.wandb_config.get("dir")
        if wandb_dir:
            os.makedirs(str(wandb_dir), exist_ok=True)
            init_kwargs["dir"] = str(wandb_dir)
        tags = self.wandb_config.get("tags")
        if isinstance(tags, list):
            init_kwargs["tags"] = tags
        wandb.init(**init_kwargs)
        self.wandb_initialized = True

    def _resolve_tokenizers(
        self,
        agent_model: Optional[Union[str, PreTrainedModel]],
        agents: Optional[Sequence[Union[str, PreTrainedModel]]],
    ) -> List[Any]:
        if agents is not None:
            names = [agent for agent in agents if isinstance(agent, str)]
            if len(names) != len(agents):
                raise ValueError("When passing explicit agents, each agent must be a pretrained identifier string.")
        elif isinstance(agent_model, str):
            names = [agent_model for _ in range(self.args.num_agents)]
        else:
            raise ValueError("agent_model or agents must be provided as pretrained identifier strings.")

        tokenizers = [AutoTokenizer.from_pretrained(name) for name in names]
        for tok in tokenizers:
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        return tokenizers

    def _resolve_critic_tokenizer(
        self,
        critic_model: Optional[Union[str, PreTrainedModel]],
        critics: Optional[Sequence[Union[str, PreTrainedModel]]],
    ) -> Any:
        if critics is not None:
            if len(critics) != 1 or not isinstance(critics[0], str):
                raise ValueError("critics must contain exactly one pretrained identifier string.")
            name = critics[0]
        elif isinstance(critic_model, str):
            name = critic_model
        else:
            name = self.agent_tokenizers[0].name_or_path
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_actor_models(
        self,
        *,
        agent_model: Optional[Union[str, PreTrainedModel]],
        agents: Optional[Sequence[Union[str, PreTrainedModel]]],
    ) -> List[CausalLMWithContextValueHead]:
        if agents is not None:
            sources = list(agents)
        elif agent_model is not None:
            sources = [agent_model for _ in range(self.args.num_agents)]
        else:
            raise ValueError("agent_model or agents must be provided.")
        out: List[CausalLMWithContextValueHead] = []
        for source in sources:
            if isinstance(source, PreTrainedModel):
                base = source
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    str(source),
                    **self._filter_model_kwargs(self.model_config.get("model_kwargs", {})),
                )
            self._maybe_enable_gradient_checkpointing(
                base,
                enabled=bool(self.args.actor_gradient_checkpointing),
            )
            wrapped = CausalLMWithContextValueHead(
                base_model=base,
                context_dim=self.args.context_hidden_dim,
                attach_value_head=False,
                value_head_hidden_dim=None,
                value_context_dim=self.args.actor_condition_dim,
            )
            out.append(wrapped)
        return out

    def _load_critic_model(
        self,
        *,
        critic_model: Optional[Union[str, PreTrainedModel]],
        critics: Optional[Sequence[Union[str, PreTrainedModel]]],
        joint_context_dim: int,
    ) -> nn.Module:
        if self.critic_backbone == "structured":
            hidden_dim = self.args.value_head_hidden_dim or max(256, joint_context_dim)
            return StructuredValueCritic(
                input_dim=joint_context_dim,
                hidden_dim=hidden_dim,
            )
        if critics is not None:
            if len(critics) != 1:
                raise ValueError("bridge_build_meta expects exactly one critic.")
            source = critics[0]
        elif critic_model is not None:
            source = critic_model
        else:
            raise ValueError("critic_model or critics must be provided.")

        if isinstance(source, PreTrainedModel):
            base = source
        else:
            base = AutoModelForCausalLM.from_pretrained(
                str(source),
                **self._filter_model_kwargs(self.model_config.get("critic_model_kwargs", {})),
            )
        return CausalLMWithContextValueHead(
            base_model=base,
            context_dim=(self.args.context_hidden_dim if self.args.num_agents == 1 else self.args.context_hidden_dim * 3),
            attach_value_head=True,
            value_head_hidden_dim=self.args.value_head_hidden_dim,
            value_context_dim=self.args.critic_condition_dim,
        )

    def _maybe_enable_gradient_checkpointing(
        self,
        model: nn.Module,
        *,
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    def _filter_model_kwargs(self, cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(cfg, Mapping):
            return {}
        out: Dict[str, Any] = {}
        if cfg.get("torch_dtype") is not None:
            out["torch_dtype"] = cfg.get("torch_dtype")
        if cfg.get("dtype") is not None and "torch_dtype" not in out:
            out["torch_dtype"] = cfg.get("dtype")
        return out

    def _resolve_agent_devices(
        self,
        agent_devices: Optional[Union[str, Sequence[str]]],
    ) -> List[torch.device]:
        raw_devices: List[str] = []
        if isinstance(agent_devices, (list, tuple)):
            raw_devices = [str(device).strip() for device in agent_devices if str(device).strip()]
        elif isinstance(agent_devices, str) and agent_devices.strip():
            raw_devices = [agent_devices.strip()]
        if not raw_devices:
            raw_devices = ["cuda:0" if torch.cuda.is_available() else "cpu"]
        resolved = [torch.device(device) for device in raw_devices]
        return [resolved[idx % len(resolved)] for idx in range(self.args.num_agents)]

    def _resolve_critic_device(
        self,
        critic_devices: Optional[Union[str, Sequence[str]]],
        *,
        fallback_device: torch.device,
    ) -> torch.device:
        if isinstance(critic_devices, (list, tuple)):
            for device in critic_devices:
                text = str(device).strip()
                if text:
                    return torch.device(text)
        elif isinstance(critic_devices, str) and critic_devices.strip():
            return torch.device(critic_devices.strip())
        return fallback_device

    def _module_device(self, module: nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        return self.critic_device

    def _clip_grad_norm(self, module: nn.Module, max_norm: float) -> None:
        params = [param for param in module.parameters() if param.requires_grad and param.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    def _max_context_length(self, model: nn.Module, tokenizer: Any) -> int:
        config = getattr(model, "config", None)
        for attr in ("max_position_embeddings", "n_positions"):
            value = getattr(config, attr, None)
            if value is not None:
                return int(value)
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max is None or int(tok_max) <= 0 or int(tok_max) > 100_000:
            return 2048
        return int(tok_max)


def payload_to_task_id(payload: Mapping[str, Any]) -> str:
    task = payload.get("task") or {}
    if isinstance(task, Mapping):
        return str(task.get("task_id") or "")
    return ""
