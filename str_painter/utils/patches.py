from __future__ import annotations

from typing import Any, Dict
import re

from comlrl.trainers.magrpo import MAGRPOTrainer  # type: ignore


def patch_trainer_generation_for_memory() -> None:
    try:
        orig = MAGRPOTrainer._generate_completions  # type: ignore[attr-defined]
    except Exception:
        return

    def wrapped(self, agent, batch_items, agent_idx=0, num_return_sequences=1, max_new_tokens=None, **kwargs):
        try:
            kwargs.setdefault("output_scores", False)
            kwargs.setdefault("use_cache", False)
            import torch as _torch  # local import

            eff_max_new = max_new_tokens
            if eff_max_new is None:
                try:
                    eff_max_new = int(getattr(self.args, "max_new_tokens", 512))
                except Exception:
                    eff_max_new = 512
            with _torch.no_grad():
                return orig(
                    self,
                    agent,
                    batch_items,
                    agent_idx=agent_idx,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=eff_max_new,
                    **kwargs,
                )
        except Exception:
            return orig(
                self,
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=num_return_sequences,
                max_new_tokens=(max_new_tokens if max_new_tokens is not None else getattr(self.args, "max_new_tokens", 512)),
                **kwargs,
            )

    try:
        MAGRPOTrainer._generate_completions = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


def patch_single_agent_returns() -> None:
    try:
        orig = MAGRPOTrainer._train_step_returns  # type: ignore[attr-defined]
    except Exception:
        return

    def wrapped(self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs):
        try:
            n_turns = int(getattr(self.args, "num_turns", 1))
            if self.num_agents == 1 and n_turns == 1:
                import numpy as _np  # type: ignore

                num_gens = int(getattr(self.args, "num_generations", 2))
                comps = self._generate_completions_with_external_prompts(
                    self.agents[0],
                    [batch_item],
                    agent_idx=0,
                    num_return_sequences=num_gens,
                    max_new_tokens=getattr(self.args, "max_new_tokens", 128),
                    external_prompts=None,
                    **kwargs,
                )
                completions0 = comps.get("completions", [[]])[0]
                prompts0 = comps.get("prompts", [""])[0]
                rewards_vec = self._compute_rewards([prompts0], [completions0], batch_items=[batch_item])
                returns_vec = list(map(float, rewards_vec))

                self.optimizers[0].zero_grad()
                agent_loss = self._compute_loss_with_gradients(self.agents[0], comps, returns_vec)
                agent_loss.backward()
                self.optimizers[0].step()

                if epoch_turn_rewards and len(epoch_turn_rewards) > 0:
                    epoch_turn_rewards[0].append(float(_np.mean(rewards_vec)) if rewards_vec else 0.0)
                if epoch_turn_returns and len(epoch_turn_returns) > 0:
                    epoch_turn_returns[0].append(float(_np.mean(returns_vec)) if returns_vec else 0.0)

                batch_loss = float(_np.mean(_np.abs(returns_vec or [0.0])))
                stats = {
                    "batch_mean_reward": float(_np.mean(rewards_vec)) if rewards_vec else 0.0,
                    "batch_expected_return": float(_np.mean(returns_vec)) if returns_vec else 0.0,
                }
                return batch_loss, {0: stats}
        except Exception:
            pass
        return orig(self, batch_item, epoch_turn_rewards, epoch_turn_returns, **kwargs)

    try:
        MAGRPOTrainer._train_step_returns = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


def patch_debug_turn_tracking() -> None:
    try:
        orig = MAGRPOTrainer._generate_completions_with_external_prompts  # type: ignore[attr-defined]
    except Exception:
        return

    turn_re = re.compile(r"\bturn\s*[:#-]?\s*(\d+)\b", re.IGNORECASE)

    def wrapped(self, agent, batch_items, agent_idx=0, num_return_sequences=1, max_new_tokens=128, external_prompts=None, **kwargs):
        turn_idx = 1
        if external_prompts is not None:
            try:
                m = turn_re.search(str(external_prompts))
                if m:
                    turn_idx = int(m.group(1))
                else:
                    turn_idx = 2
            except Exception:
                turn_idx = 2
        for item in batch_items or []:
            try:
                item["_str_painter_turn"] = int(turn_idx)
            except Exception:
                pass
        return orig(
            self,
            agent,
            batch_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            external_prompts=external_prompts,
            **kwargs,
        )

    try:
        MAGRPOTrainer._generate_completions_with_external_prompts = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


def apply_default_patches(cfg: Dict[str, Any] | None = None) -> None:
    gates = (cfg or {}).get("patches", {}) if isinstance(cfg, dict) else {}
    if gates.get("generation_memory", True):
        patch_trainer_generation_for_memory()
    if gates.get("single_agent_returns", True):
        patch_single_agent_returns()
    if gates.get("debug_turn_tracking", True):
        patch_debug_turn_tracking()
