from __future__ import annotations

from typing import Any, Dict, List, Optional

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    get_agent_observation,
    payload_to_state,
    payload_to_task,
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
    task = payload_to_task(next_payload)
    state = payload_to_state(next_payload)
    view = int(next_payload.get("view") or 3)

    feedbacks: List[str] = []
    for i in range(n):
        obs = get_agent_observation(task, state, agent_idx=i, view=view)
        feedbacks.append(
            "\n".join(
                [
                    "Position feedback:",
                    f"- Turn: {turn_no}",
                    f"- Agent: {'A' if i == 0 else 'B'}",
                    f"- Current position: {obs.get('current_pos')}",
                    f"- Visible land count: {len(obs.get('visible_land_coords') or [])}",
                    f"- Visible pillar-candidate count: {len(obs.get('visible_p_candidates') or [])}",
                    f"- Known probe-result count: {len(obs.get('known_probe_results') or [])}",
                    f"- connected(S,T): {bool(metrics.get('connected', False))}",
                ]
            )
        )

    if original_prompt_flag:
        prompts = render_prompts_from_payload(
            payload=next_payload,
            state=state,
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
