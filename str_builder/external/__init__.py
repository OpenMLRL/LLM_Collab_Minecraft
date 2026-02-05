from __future__ import annotations

"""External transition utilities for multi-turn training on str_builder.

CoMLRL's MAGRPOTrainer supports multi-turn rollouts via an `external_transition`
callback. This module provides a small adapter that:

- Resolves a dataset-level prompt key (e.g., "str_builder:...") into a context dict.
- Builds next-turn prompts for each agent using `mode=perfect_feedback`.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import perfect_feedback
from . import position_feedback
from . import score_feedback


# Verbose toggle for external previews
VERBOSE = False


# Context resolver API
_context_resolver: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None


def set_context_resolver(fn: Callable[[str], Optional[Dict[str, Any]]]) -> None:
    """Register a resolver that maps prompt -> context dict."""
    global _context_resolver
    _context_resolver = fn


def get_context(prompt: str) -> Optional[Dict[str, Any]]:
    if _context_resolver is None:
        return None
    try:
        return _context_resolver(prompt)
    except Exception:
        return None


def get_external_transition(
    prompt: str,
    agent_completions: Union[List[str], Tuple[str, ...]],
    num_agents: int = 2,
    mode: str = "perfect_feedback",
    *,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
    **kwargs,
) -> Union[List[str], Tuple[str, ...]]:
    n = int(num_agents)
    if n <= 0:
        raise ValueError("num_agents must be >= 1")
    if not isinstance(agent_completions, (list, tuple)) or len(agent_completions) != n:
        raise ValueError(f"Expected {n} agent completions, got {len(agent_completions) if isinstance(agent_completions, (list, tuple)) else 'invalid type'}")

    ctx = get_context(prompt) or {}
    mode_key = (mode or "").strip().lower()

    original_prompt_flag = bool(kwargs.get("original_prompt", True))
    previous_response_flag = bool(kwargs.get("previous_response", False))

    if mode_key in ("perfect_feedback", "perfect-feedback", "feedback"):
        prompts = perfect_feedback.format_followup_prompts(
            ctx=ctx,
            agent_completions=list(agent_completions),
            num_agents=n,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            prompt_history_per_agent=prompt_history_per_agent,
            response_history_per_agent=response_history_per_agent,
        )
        if VERBOSE:
            print("\n" + "=" * 60)
            print("EXTERNAL MODE PREVIEW: perfect_feedback")
            for i, p in enumerate(prompts):
                print("-" * 60)
                print(f"AGENT {i} PROMPT:\n{p}")
            print("=" * 60 + "\n")
        return prompts
    if mode_key in ("position_feedback", "position-feedback"):
        prompts = position_feedback.format_followup_prompts(
            ctx=ctx,
            agent_completions=list(agent_completions),
            num_agents=n,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            prompt_history_per_agent=prompt_history_per_agent,
            response_history_per_agent=response_history_per_agent,
        )
        if VERBOSE:
            print("\n" + "=" * 60)
            print("EXTERNAL MODE PREVIEW: position_feedback")
            for i, p in enumerate(prompts):
                print("-" * 60)
                print(f"AGENT {i} PROMPT:\n{p}")
            print("=" * 60 + "\n")
        return prompts
    if mode_key in ("score_feedback", "score-feedback", "score"):
        prompts = score_feedback.format_followup_prompts(
            ctx=ctx,
            agent_completions=list(agent_completions),
            num_agents=n,
            original_prompt_flag=original_prompt_flag,
            previous_response_flag=previous_response_flag,
            prompt_history_per_agent=prompt_history_per_agent,
            response_history_per_agent=response_history_per_agent,
        )
        if VERBOSE:
            print("\n" + "=" * 60)
            print("EXTERNAL MODE PREVIEW: score_feedback")
            for i, p in enumerate(prompts):
                print("-" * 60)
                print(f"AGENT {i} PROMPT:\n{p}")
            print("=" * 60 + "\n")
        return prompts

    raise NotImplementedError("External mode not implemented: " + str(mode))
