from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .conditioning import ContextTokenBias, ContextValueProjection


@dataclass
class ActorCriticOutput:
    logits: torch.Tensor
    values: Optional[torch.Tensor]
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        if hidden_dim is not None and hidden_dim > 0:
            self.pre = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
            )
            last_dim = hidden_dim
        else:
            self.pre = None
            last_dim = input_dim
        self.out = nn.Linear(last_dim, 1)
        self._init_parameters()

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pre is not None:
            hidden_states = self.pre(hidden_states)
        return self.out(hidden_states)


class CausalLMWithContextValueHead(nn.Module):
    """
    Wrap a causal LM with belief-conditioned token biasing and an optional value head.

    The context vector is injected as an additive bias on selected token embeddings.
    This keeps the implementation compatible with decoder-only LMs while exposing a
    hidden side-channel that does not rely on textual prompt hacks.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        *,
        context_dim: int,
        value_head_hidden_dim: Optional[int] = None,
        attach_value_head: bool = True,
        value_context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = base_model
        config = getattr(base_model, "config", None)
        if config is None:
            raise ValueError("Base model must provide a config with hidden size.")
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Unsupported backbone: missing hidden size on config.")

        self.hidden_size = int(hidden_size)
        self.context_dim = int(context_dim)
        self.context_bias = ContextTokenBias(self.context_dim, self.hidden_size)
        self.value_context_proj = ContextValueProjection(
            self.context_dim,
            proj_dim=value_context_dim,
        )
        value_input_dim = self.hidden_size + self.value_context_proj.output_dim
        self.value_head = ValueHead(value_input_dim, value_head_hidden_dim) if attach_value_head else None

        base_params = list(base_model.parameters())
        if base_params:
            base_dtype = base_params[0].dtype
            self.context_bias.to(dtype=base_dtype)
            self.value_context_proj.to(dtype=base_dtype)
            if self.value_head is not None:
                self.value_head.to(dtype=base_dtype)

    def _context_bias(
        self,
        *,
        context_vec: Optional[torch.Tensor],
        seq_len: int,
        context_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if context_vec is None:
            return None
        bias = self.context_bias(context_vec.to(dtype=dtype))
        bias = bias.unsqueeze(1).expand(-1, seq_len, -1)
        if context_mask is None:
            return bias
        mask = context_mask.unsqueeze(-1).to(dtype=dtype)
        return bias * mask

    def forward(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_vec: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        output_values: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ActorCriticOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        if inputs_embeds is None:
            embeds = self.model.get_input_embeddings()(input_ids)
        else:
            embeds = inputs_embeds

        bias = self._context_bias(
            context_vec=context_vec,
            seq_len=embeds.size(1),
            context_mask=context_mask,
            dtype=embeds.dtype,
        )
        if bias is not None:
            embeds = embeds + bias

        outputs: CausalLMOutputWithCrossAttentions = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            inputs_embeds=embeds,
            **kwargs,
        )

        values: Optional[torch.Tensor] = None
        hidden_states = outputs.hidden_states[-1]
        if output_values and self.value_head is not None:
            if context_vec is None:
                ctx = torch.zeros(
                    hidden_states.size(0),
                    self.context_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                ctx = context_vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
            ctx_proj = self.value_context_proj(ctx).unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            values = self.value_head(torch.cat([hidden_states, ctx_proj], dim=-1)).squeeze(-1)

        return ActorCriticOutput(
            logits=outputs.logits,
            values=values,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
        )
