from __future__ import annotations

import math
import os
import random
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

from LLM_Collab_Minecraft.bridge_build_meta.adapters import BridgeBuildAdapter
from LLM_Collab_Minecraft.bridge_build_meta.models import (
    BeliefEncoder,
    CausalLMWithContextValueHead,
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
        if self.logging_steps < 1:
            raise ValueError("logging_steps must be >= 1.")


@dataclass
class TurnRecord:
    prompts: List[str]
    completions: List[str]
    sequences: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    prompt_lens: List[int]
    response_lens: List[int]
    meta_obs: List[Any]
    reward: float
    metrics: Dict[str, Any]
    task_index: int
    terminated: bool


@dataclass
class EpisodeTrajectory:
    task_id: str
    turns: List[TurnRecord] = field(default_factory=list)


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

        self.device = self._resolve_device(agent_devices, critic_devices)
        self.agent_tokenizers = self._resolve_tokenizers(agent_model, agents)
        self.critic_tokenizer = self._resolve_critic_tokenizer(critic_model, critics)
        if self.adapter.tokenizer is None:
            self.adapter.tokenizer = self.agent_tokenizers[0]

        self.agents = self._load_actor_models(agent_model=agent_model, agents=agents)
        self.critic = self._load_critic_model(critic_model=critic_model, critics=critics)

        joint_context_dim = (
            self.args.context_hidden_dim
            if self.args.num_agents == 1
            else self.args.context_hidden_dim * 3
        )
        self.context_encoder = BeliefEncoder(
            grid_channels=self.adapter.grid_channels,
            scalar_dim=self.adapter.scalar_dim,
            hidden_dim=self.args.context_hidden_dim,
            task_vocab_size=self.adapter.task_vocab_size,
            cnn_channels=self.args.context_cnn_channels,
            scalar_hidden_dim=self.args.context_scalar_hidden_dim,
        ).to(self.device)

        for actor in self.agents:
            actor.to(self.device)
        self.critic.to(self.device)

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

                rollout_metrics = self._summarize_trajectory(trajectory)
                for key, value in rollout_metrics.items():
                    epoch_metrics[key].append(value)

                if len(buffer) >= self.args.rollout_buffer_size:
                    update_metrics = self._update(buffer)
                    for key, value in update_metrics.items():
                        epoch_metrics[key].append(value)
                    self._log_metrics(update_metrics)
                    buffer = []

            if buffer:
                update_metrics = self._update(buffer)
                for key, value in update_metrics.items():
                    epoch_metrics[key].append(value)
                self._log_metrics(update_metrics)

            summary = self._summarize_epoch(epoch_metrics)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.args.num_train_epochs} metrics: {summary}")
            self._log_metrics({f"train/{k}": v for k, v in summary.items()})

            if self.eval_items and self.args.eval_interval > 0 and (epoch + 1) % self.args.eval_interval == 0:
                eval_summary = self.evaluate()
                if self.verbose:
                    print(f"Eval @ epoch {epoch + 1}: {eval_summary}")
                self._log_metrics({f"eval/{k}": v for k, v in eval_summary.items()})

    def evaluate(self) -> Dict[str, float]:
        max_items = min(len(self.eval_items), max(1, int(self.args.eval_num_samples)))
        if max_items <= 0:
            return {}
        metrics: Dict[str, List[float]] = defaultdict(list)
        for item in self.eval_items[:max_items]:
            trajectory = self._collect_episode(item, training=False)
            traj_metrics = self._summarize_trajectory(trajectory)
            for key, value in traj_metrics.items():
                metrics[key].append(value)
        return self._summarize_epoch(metrics)

    def save_model(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.context_encoder.save_path = output_dir  # type: ignore[attr-defined]
        torch.save(self.context_encoder.state_dict(), os.path.join(output_dir, "context_encoder.pt"))
        self.critic.model.save_pretrained(os.path.join(output_dir, "critic"))
        self.critic_tokenizer.save_pretrained(os.path.join(output_dir, "critic"))
        torch.save(self.critic.state_dict(), os.path.join(output_dir, "critic_context_head.pt"))
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
            contexts: List[torch.Tensor] = []
            with torch.no_grad():
                for agent_idx in range(self.args.num_agents):
                    encoder_out = self.context_encoder.forward_step(
                        grid=meta_obs[agent_idx].grid.unsqueeze(0).to(self.device),
                        scalars=meta_obs[agent_idx].scalars.unsqueeze(0).to(self.device),
                        hidden=hidden_states[agent_idx],
                    )
                    hidden_states[agent_idx] = encoder_out.hidden.detach()
                    contexts.append(encoder_out.context.detach())

            sequences: List[torch.Tensor] = []
            attention_masks: List[torch.Tensor] = []
            prompt_lens: List[int] = []
            response_lens: List[int] = []
            completions: List[str] = []
            for agent_idx in range(self.args.num_agents):
                gen = self._sample_completion(
                    model=self.agents[agent_idx],
                    tokenizer=self.agent_tokenizers[agent_idx],
                    prompt=prompts[agent_idx],
                    context_vec=contexts[agent_idx],
                    training=training,
                )
                sequences.append(gen["sequence"].cpu())
                attention_masks.append(gen["attention_mask"].cpu())
                prompt_lens.append(int(gen["prompt_len"]))
                response_lens.append(int(gen["response_len"]))
                completions.append(str(gen["completion"]))

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

        normalized_advantages = self._compute_normalized_turn_advantages(trajectories)
        total_turns = max(1, sum(len(traj.turns) for traj in trajectories))

        self.context_encoder.train()
        self.critic.train()
        for actor in self.agents:
            actor.train()

        metrics_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "task_loss": 0.0,
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
                turn_task_losses: List[torch.Tensor] = []
                for agent_idx in range(self.args.num_agents):
                    obs = turn.meta_obs[agent_idx]
                    hidden_in = hidden_states[agent_idx]
                    if hidden_in is not None:
                        hidden_in = hidden_in.detach()
                    encoder_out = self.context_encoder.forward_step(
                        grid=obs.grid.unsqueeze(0).to(self.device),
                        scalars=obs.scalars.unsqueeze(0).to(self.device),
                        hidden=hidden_in,
                    )
                    hidden_states[agent_idx] = encoder_out.hidden.detach()
                    contexts.append(encoder_out.context)
                    target = torch.tensor([obs.task_index], device=self.device, dtype=torch.long)
                    turn_task_losses.append(F.cross_entropy(encoder_out.task_logits, target))

                joint_context = self.adapter.build_joint_context(contexts)
                critic_input = self._build_critic_input(
                    turn.prompts,
                    action_completions=turn.completions if self.critic_type == "q" else None,
                )
                value = self._critic_value_from_text(critic_input, joint_context)
                ret = torch.tensor([returns[turn_idx]], device=self.device, dtype=value.dtype)
                value_loss_turn = (value - ret).pow(2).mean()

                actor_turn_losses: List[torch.Tensor] = []
                entropy_turn_terms: List[torch.Tensor] = []
                adv_value = float(normalized_advantages[traj_idx][turn_idx])
                adv_tensor = torch.tensor([adv_value], device=self.device, dtype=ret.dtype)
                for agent_idx in range(self.args.num_agents):
                    sequence = turn.sequences[agent_idx].unsqueeze(0).to(self.device)
                    attention_mask = turn.attention_masks[agent_idx].unsqueeze(0).to(self.device)
                    prompt_len = int(turn.prompt_lens[agent_idx])
                    response_len = int(turn.response_lens[agent_idx])
                    logprob, entropy = self._sequence_logprob(
                        model=self.agents[agent_idx],
                        sequences=sequence,
                        attention_mask=attention_mask,
                        prompt_len=prompt_len,
                        response_len=response_len,
                        context_vec=contexts[agent_idx],
                    )
                    actor_turn_losses.append(-(logprob.squeeze(0) * adv_tensor.to(logprob.dtype)))
                    entropy_turn_terms.append(entropy.squeeze(0))

                actor_loss_turn = (
                    torch.stack(actor_turn_losses).mean()
                    if actor_turn_losses
                    else torch.zeros((), device=self.device, dtype=ret.dtype)
                )
                task_loss_turn = (
                    torch.stack(turn_task_losses).mean()
                    if turn_task_losses
                    else torch.zeros((), device=self.device, dtype=ret.dtype)
                )
                entropy_turn = (
                    torch.stack(entropy_turn_terms).mean()
                    if entropy_turn_terms
                    else torch.zeros((), device=self.device, dtype=ret.dtype)
                )

                turn_loss = (
                    actor_loss_turn
                    + self.args.value_loss_coef * value_loss_turn
                    + self.args.task_loss_coef * task_loss_turn
                    - self.args.entropy_coef * entropy_turn
                ) / float(total_turns)
                turn_loss.backward()

                metrics_accum["policy_loss"] += float(actor_loss_turn.detach().item())
                metrics_accum["value_loss"] += float(value_loss_turn.detach().item())
                metrics_accum["task_loss"] += float(task_loss_turn.detach().item())
                metrics_accum["entropy"] += float(entropy_turn.detach().item())

        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            modules: List[nn.Module] = [self.context_encoder, self.critic, *self.agents]
            torch.nn.utils.clip_grad_norm_(
                [p for module in modules for p in module.parameters() if p.requires_grad],
                max_norm=float(self.args.max_grad_norm),
            )

        self.context_optimizer.step()
        self.critic_optimizer.step()
        for optimizer in self.actor_optimizers:
            optimizer.step()

        return {
            "policy_loss": float(metrics_accum["policy_loss"] / float(total_turns)),
            "value_loss": float(metrics_accum["value_loss"] / float(total_turns)),
            "task_loss": float(metrics_accum["task_loss"] / float(total_turns)),
            "entropy": float(metrics_accum["entropy"] / float(total_turns)),
        }

    def _sample_completion(
        self,
        *,
        model: CausalLMWithContextValueHead,
        tokenizer: Any,
        prompt: str,
        context_vec: torch.Tensor,
        training: bool,
    ) -> Dict[str, Any]:
        max_ctx = self._max_context_length(model.model, tokenizer)
        prompt_budget = max(1, max_ctx - 1)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_budget,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = int(input_ids.size(1))

        with torch.no_grad():
            context_mask = torch.ones_like(input_ids, device=self.device, dtype=torch.float32)
            outputs = model(
                input_ids=input_ids,
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
                input_ids = next_token.view(1, 1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        if generated:
            generated_ids = torch.stack(generated, dim=0).view(1, -1)
        else:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
            generated_ids = torch.tensor([[int(pad_id)]], device=self.device)
        full_sequence = torch.cat([encoded["input_ids"].to(self.device), generated_ids], dim=1)
        full_attention_mask = torch.ones_like(full_sequence, device=self.device)
        completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return {
            "sequence": full_sequence.squeeze(0),
            "attention_mask": full_attention_mask.squeeze(0),
            "prompt_len": prompt_len,
            "response_len": int(generated_ids.size(1)),
            "completion": completion,
        }

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

    def _sequence_logprob(
        self,
        *,
        model: CausalLMWithContextValueHead,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        response_len: int,
        context_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_mask = torch.zeros_like(sequences, dtype=torch.float32, device=self.device)
        context_mask[:, :prompt_len] = 1.0
        outputs = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            context_vec=context_vec,
            context_mask=context_mask,
            output_values=False,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_targets = sequences[:, 1:]
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
        start_index = max(prompt_len - 1, 0)
        end_index = start_index + response_len
        response_log_probs = token_log_probs[:, start_index:end_index]
        if float(self.args.entropy_coef) != 0.0:
            token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            response_entropy = token_entropy[:, start_index:end_index]
            entropy = response_entropy.mean(dim=-1)
        else:
            entropy = torch.zeros(
                sequences.size(0),
                device=sequences.device,
                dtype=response_log_probs.dtype,
            )
        return response_log_probs.sum(dim=-1), entropy

    def _critic_value_from_text(self, prompt: str, joint_context: torch.Tensor) -> torch.Tensor:
        max_ctx = self._max_context_length(self.critic.model, self.critic_tokenizer)
        encoded = self.critic_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_ctx,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        context_mask = torch.ones_like(input_ids, dtype=torch.float32, device=self.device)
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

    def _compute_normalized_turn_advantages(
        self,
        trajectories: Sequence[EpisodeTrajectory],
    ) -> List[List[float]]:
        self.context_encoder.eval()
        self.critic.eval()
        for actor in self.agents:
            actor.eval()

        advantages: List[List[float]] = []
        flat_advantages: List[torch.Tensor] = []

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
                            grid=obs.grid.unsqueeze(0).to(self.device),
                            scalars=obs.scalars.unsqueeze(0).to(self.device),
                            hidden=hidden_in,
                        )
                        hidden_states[agent_idx] = encoder_out.hidden.detach()
                        contexts.append(encoder_out.context.detach())

                    joint_context = self.adapter.build_joint_context(contexts)
                    critic_input = self._build_critic_input(
                        turn.prompts,
                        action_completions=turn.completions if self.critic_type == "q" else None,
                    )
                    value = self._critic_value_from_text(critic_input, joint_context).detach().view(-1)[0]
                    adv = torch.tensor(
                        float(returns[turn_idx]) - float(value.item()),
                        device=self.device,
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
        return normalized_advantages

    def _summarize_trajectory(self, trajectory: EpisodeTrajectory) -> Dict[str, float]:
        if not trajectory.turns:
            return {}
        total_reward = float(sum(turn.reward for turn in trajectory.turns))
        final_metrics = trajectory.turns[-1].metrics
        return {
            "reward_sum": total_reward,
            "final_reward": float(trajectory.turns[-1].reward),
            "final_gap": float(final_metrics.get("gap_st") if final_metrics.get("gap_st") is not None else 0.0),
            "final_connected": 1.0 if bool(final_metrics.get("connected", False)) else 0.0,
        }

    def _summarize_epoch(self, metrics: Mapping[str, Sequence[float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, values in metrics.items():
            vals = [float(v) for v in values if v is not None]
            if vals:
                out[key] = float(sum(vals) / len(vals))
        return out

    def _log_metrics(self, metrics: Mapping[str, float]) -> None:
        if not metrics:
            return
        if not self.wandb_initialized or wandb is None:
            return
        wandb.log(dict(metrics), step=self.env_step)

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
    ) -> CausalLMWithContextValueHead:
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

    def _filter_model_kwargs(self, cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(cfg, Mapping):
            return {}
        out: Dict[str, Any] = {}
        if cfg.get("torch_dtype") is not None:
            out["torch_dtype"] = cfg.get("torch_dtype")
        if cfg.get("dtype") is not None and "torch_dtype" not in out:
            out["torch_dtype"] = cfg.get("dtype")
        return out

    def _resolve_device(
        self,
        agent_devices: Optional[Union[str, Sequence[str]]],
        critic_devices: Optional[Union[str, Sequence[str]]],
    ) -> torch.device:
        del critic_devices
        if isinstance(agent_devices, (list, tuple)) and agent_devices:
            return torch.device(str(agent_devices[0]))
        if isinstance(agent_devices, str) and agent_devices.strip():
            return torch.device(agent_devices.strip())
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
