from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    payload_to_state,
    render_prompts_from_payload,
    transition_payload,
)


def _turn_number(prompt_history_per_agent: Optional[List[List[str]]]) -> int:
    if not prompt_history_per_agent:
        return 1
    try:
        return int(len(prompt_history_per_agent[0]) + 1)
    except Exception:
        return 1


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
    del response_history_per_agent
    n = int(num_agents)
    if n <= 0:
        raise ValueError("num_agents must be >= 1")
    if len(agent_completions) != n:
        raise ValueError(f"Expected {n} agent completions, got {len(agent_completions)}")

    turn_no = _turn_number(prompt_history_per_agent)

    next_payload, metrics, _actions = transition_payload(
        payload=ctx,
        agent_completions=agent_completions,
        num_agents=n,
    )
    next_state = payload_to_state(next_payload)

    reward = float(metrics.get("reward", 0.0))
    feedback = "\n".join(
        [
            "Score feedback:",
            f"- Turn: {turn_no}",
            f"- reward: {reward:.4f}",
            f"- bonus_y_connected: {float(metrics.get('bonus_y_connected', 0.0)):.4f}",
            f"- penalty_n_adjacent: {float(metrics.get('penalty_n_adjacent', 0.0)):.4f}",
            f"- penalty_block_cost: {float(metrics.get('penalty_block_cost', 0.0)):.4f}",
            f"- bonus_terminal_connect: {float(metrics.get('bonus_terminal_connect', 0.0)):.4f}",
            f"- new_connected_y_count: {int(metrics.get('new_connected_y_count', 0))}",
            f"- new_adjacent_n_count: {int(metrics.get('new_adjacent_n_count', 0))}",
            f"- newly_placed_block_count: {int(metrics.get('newly_placed_block_count', 0))}",
            f"- connected(S,T): {bool(metrics.get('connected', False))}",
        ]
    )
    feedbacks = [feedback for _ in range(n)]

    if original_prompt_flag:
        prompts = render_prompts_from_payload(
            payload=next_payload,
            state=next_state,
            num_agents=n,
            feedback_by_agent=feedbacks,
            include_system=True,
        )
    else:
        prompts = list(feedbacks)

    if previous_response_flag:
        for i in range(n):
            prev = (agent_completions[i] if i < len(agent_completions) else "").strip()
            if prev:
                prompts[i] = prompts[i].rstrip() + "\n\nYour previous action JSON:\n" + prev

    return prompts
