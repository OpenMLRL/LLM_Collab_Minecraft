from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.resource_gathering.utils.resource_gathering import (
    payload_to_state,
    render_prompts_from_payload,
    transition_payload,
)


def format_followup_prompts(
    *,
    ctx: Dict[str, Any],
    agent_completions: List[str],
    num_agents: int = 2,
    original_prompt_flag: bool = True,
    previous_response_flag: bool = False,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
) -> List[str]:
    del prompt_history_per_agent
    del response_history_per_agent

    n = int(num_agents)
    if n <= 0:
        raise ValueError("num_agents must be >= 1")
    if len(agent_completions) != n:
        raise ValueError(f"Expected {n} agent completions, got {len(agent_completions)}")

    next_payload, _metrics, _actions = transition_payload(
        payload=ctx,
        agent_completions=agent_completions,
        num_agents=n,
    )
    next_state = payload_to_state(next_payload)

    prompts = render_prompts_from_payload(
        payload=next_payload,
        state=next_state,
        num_agents=n,
        feedback_by_agent=["" for _ in range(n)],
        include_system=bool(original_prompt_flag),
    )

    if previous_response_flag:
        for i in range(n):
            prev = (agent_completions[i] if i < len(agent_completions) else "").strip()
            if prev:
                prompts[i] = prompts[i].rstrip() + "\n\nYour previous action JSON:\n" + prev

    return prompts
