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


def _build_feedback(metrics: Dict[str, Any], *, turn_number: int, agent_idx: int) -> str:
    connected = bool(metrics.get("connected", False))
    reward = float(metrics.get("reward", 0.0))
    n_adj = int(metrics.get("n_adjacent_count", 0))
    y_uncovered = int(metrics.get("y_uncovered_count", 0))
    y_connected = int(metrics.get("connected_y_count", 0))
    probe_cnt = int(metrics.get("num_valid_probes", 0))
    comm_tokens = int(metrics.get("comm_tokens", 0))
    return "\n".join(
        [
            "Perfect feedback:",
            f"- Turn: {turn_number}",
            f"- Agent: {'A' if agent_idx == 0 else 'B'}",
            f"- reward: {reward:.4f}",
            f"- connected(S,T): {connected}",
            f"- gap_ST: {metrics.get('gap_st', None)} / {int(metrics.get('max_gap_st', 0))}",
            f"- CC components: {int(metrics.get('cc_component_count', 0))} / {int(metrics.get('initial_cc_component_count', 0))}",
            f"- Y connected count: {y_connected}",
            f"- N adjacent count: {n_adj}",
            f"- Y uncovered count: {y_uncovered}",
            f"- bonus_gap_st: {float(metrics.get('bonus_gap_st', 0.0)):.4f}",
            f"- bonus_cc_merge: {float(metrics.get('bonus_cc_merge', 0.0)):.4f}",
            f"- bonus_y_connected: {float(metrics.get('bonus_y_connected', 0.0)):.4f}",
            f"- penalty_n_adjacent: {float(metrics.get('penalty_n_adjacent', 0.0)):.4f}",
            f"- penalty_block_cost: {float(metrics.get('penalty_block_cost', 0.0)):.4f}",
            f"- bonus_terminal_connect: {float(metrics.get('bonus_terminal_connect', 0.0)):.4f}",
            f"- valid probes: {probe_cnt}",
            f"- comm tokens: {comm_tokens}",
            "- Target: gather information, avoid new N adjacency, connect more Y, then connect S/T.",
        ]
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

    feedbacks = [_build_feedback(metrics, turn_number=turn_no, agent_idx=i) for i in range(n)]

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
