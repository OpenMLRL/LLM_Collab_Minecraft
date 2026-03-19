from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from LLM_Collab_Minecraft.bridge_build_meta.trainers.bcmaac import (
    _ENV_METRIC_KEYS,
    ActionChoiceTrace,
    BCMAACConfig,
    BCMAACTrainer,
    EpisodeTrajectory,
    TurnRecord,
)

from ..models import ObservationEncoder


BridgeBuildBaselineConfig = BCMAACConfig


class BridgeBuildBaselineTrainer(BCMAACTrainer):
    """Aligned bridge_build baseline with the same decoder/reward stack but no meta latent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args.task_loss_coef = 0.0
        self.args.actor_prompt_context_scale = 0.0
        self.args.actor_response_context_scale = 0.0
        self.context_encoder = ObservationEncoder(
            grid_channels=self.adapter.grid_channels,
            scalar_dim=self.adapter.scalar_dim,
            hidden_dim=self.args.context_hidden_dim,
            cnn_channels=self.args.context_cnn_channels,
            scalar_hidden_dim=self.args.context_scalar_hidden_dim,
        ).to(self.critic_device)
        self.context_optimizer = torch.optim.AdamW(
            self.context_encoder.parameters(),
            lr=self.args.context_learning_rate,
        )

    def _zero_actor_context(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.args.context_hidden_dim, device=device, dtype=dtype)

    def _encode_turn_contexts(self, meta_obs: Sequence[Any]) -> List[torch.Tensor]:
        contexts: List[torch.Tensor] = []
        for obs in meta_obs:
            encoder_out = self.context_encoder.forward_step(
                grid=obs.grid.unsqueeze(0).to(self.critic_device),
                scalars=obs.scalars.unsqueeze(0).to(self.critic_device),
            )
            contexts.append(encoder_out.context)
        return contexts

    def _collect_episode(self, item: Mapping[str, Any], *, training: bool) -> EpisodeTrajectory:
        self.context_encoder.eval()
        self.critic.eval()
        for actor in self.agents:
            actor.eval()
        payload = self.adapter.reset_item_state(item)
        task_id = str(item.get("task_id") or payload.get("task_id") or "unknown_task")
        trajectory = EpisodeTrajectory(task_id=task_id)

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
            with torch.no_grad():
                critic_contexts = [ctx.detach() for ctx in self._encode_turn_contexts(meta_obs)]

            sequences: List[torch.Tensor] = []
            attention_masks: List[torch.Tensor] = []
            prompt_lens: List[int] = []
            response_lens: List[int] = []
            completions: List[str] = []
            action_traces: List[List[ActionChoiceTrace]] = []
            with torch.no_grad():
                for agent_idx in range(self.args.num_agents):
                    actor_context = self._zero_actor_context(
                        1,
                        device=self.agent_devices[agent_idx],
                        dtype=critic_contexts[agent_idx].dtype,
                    )
                    gen = self._sample_constrained_action(
                        model=self.agents[agent_idx],
                        tokenizer=self.agent_tokenizers[agent_idx],
                        prompt=prompts[agent_idx],
                        context_vec=actor_context,
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
        per_turn_preference_loss: Dict[int, List[float]] = defaultdict(list)
        extra_loss_accum = {
            "entropy": 0.0,
            "preference_loss": 0.0,
        }

        self.context_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for optimizer in self.actor_optimizers:
            optimizer.zero_grad()

        for traj_idx, traj in enumerate(trajectories):
            returns = self._discounted_returns(traj)

            for turn_idx, turn in enumerate(traj.turns):
                critic_contexts = self._encode_turn_contexts(turn.meta_obs)
                joint_context = self.adapter.build_joint_context(critic_contexts)
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
                    actor_context = self._zero_actor_context(
                        1,
                        device=actor_device,
                        dtype=critic_contexts[agent_idx].dtype,
                    )
                    actor_loss_value, entropy_value, preference_loss_value = self._backprop_action_traces(
                        model=self.agents[agent_idx],
                        tokenizer=self.agent_tokenizers[agent_idx],
                        prompt_tokens=prompt_tokens,
                        action_traces=turn.action_traces[agent_idx],
                        context_vec=actor_context,
                        advantage=float(adv_value),
                        loss_scale=float(total_turns) * float(num_actor_terms),
                    )
                    actor_turn_loss_values.append(float(actor_loss_value))
                    entropy_turn_values.append(float(entropy_value))
                    per_turn_preference_loss[turn_idx].append(float(preference_loss_value))
                    extra_loss_accum["preference_loss"] += float(preference_loss_value)

                shared_turn_loss = (self.args.value_loss_coef * value_loss_turn) / float(total_turns)
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
        for turn_idx, values in per_turn_preference_loss.items():
            if values:
                rollout_metrics.setdefault(turn_idx, {})["preference_loss"] = float(sum(values) / len(values))

        rollout_metrics.setdefault(0, {})["task_loss"] = 0.0
        rollout_metrics.setdefault(0, {})["entropy"] = float(extra_loss_accum["entropy"] / float(total_turns))
        rollout_metrics.setdefault(0, {})["preference_loss"] = float(extra_loss_accum["preference_loss"] / float(total_turns))
        return self._flatten_turn_metrics(rollout_metrics)

    def _compute_rollout_statistics(
        self,
        trajectories: Sequence[EpisodeTrajectory],
    ) -> tuple[List[List[float]], Dict[int, Dict[str, float]]]:
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
                traj_advantages: List[float] = []

                for turn_idx, turn in enumerate(traj.turns):
                    critic_contexts = [ctx.detach() for ctx in self._encode_turn_contexts(turn.meta_obs)]
                    joint_context = self.adapter.build_joint_context(critic_contexts)
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
        all_turns = set(turn_rewards.keys()) | set(turn_returns.keys()) | set(turn_values.keys()) | set(turn_targets.keys())
        for turn_idx in sorted(all_turns):
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
