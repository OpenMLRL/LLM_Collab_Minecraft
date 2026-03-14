from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from LLM_Collab_Minecraft.bridge_build.utils.bridge_builder import (
    build_payload,
    compute_visible_cells,
    deserialize_state,
    get_agent_observation,
    make_initial_state,
    payload_allowed_blocks,
    payload_to_state,
    payload_to_task,
    render_agent_user_prompt,
    serialize_state,
    split_command_limits,
    task_from_item,
    transition_payload,
)


Coord = Tuple[int, int]


@dataclass
class AgentMetaObservation:
    grid: torch.Tensor
    scalars: torch.Tensor
    task_index: int
    task_id: str
    turn_index: int


class BridgeBuildAdapter:
    """Task adapter that exposes bridge_build as a structured belief-learning problem."""

    channel_names: Sequence[str] = (
        "visibility",
        "visible_land",
        "visible_filled",
        "visible_candidate",
        "visible_anchor_s",
        "visible_anchor_t",
        "known_y",
        "known_n",
        "msg_y",
        "msg_n",
        "msg_candidate",
        "self_pos",
    )

    def __init__(
        self,
        *,
        prompt_ctx: Dict[str, Any],
        num_agents: int,
        task_ids: Sequence[str],
        tokenizer: Any | None = None,
        external_mode: str = "empty_feedback",
        original_prompt: bool = True,
        previous_response: bool = False,
        debug: bool = False,
    ) -> None:
        self.prompt_ctx = dict(prompt_ctx)
        self.num_agents = int(num_agents)
        self.tokenizer = tokenizer
        self.external_mode = str(external_mode or "empty_feedback").strip().lower()
        self.original_prompt = bool(original_prompt)
        self.previous_response = bool(previous_response)
        self.debug = bool(debug)
        self.task_id_to_index = {str(task_id): idx for idx, task_id in enumerate(task_ids)}
        self.max_probe = int(self.prompt_ctx["max_probe"])
        self.view = int(self.prompt_ctx["view"])
        self.max_commands_total = int(self.prompt_ctx["max_commands_total"])

    @property
    def grid_channels(self) -> int:
        return len(self.channel_names)

    @property
    def scalar_dim(self) -> int:
        return 8

    @property
    def task_vocab_size(self) -> int:
        return len(self.task_id_to_index)

    def item_to_payload(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        task = task_from_item(item)
        raw_state = item.get("_bridge_state_before_turn")
        if isinstance(raw_state, Mapping):
            state = deserialize_state(raw_state, num_agents=self.num_agents)
        else:
            state = make_initial_state(task, num_agents=self.num_agents, max_turns=task.max_turns)
        return build_payload(
            task=task,
            state_before_turn=state,
            num_agents=self.num_agents,
            view=self.view,
            max_probe=self.max_probe,
            max_commands_total=self.max_commands_total,
            allowed_blocks_agent1=list(self.prompt_ctx["agent1_blocks"]),
            allowed_blocks_agent2=list(self.prompt_ctx["agent2_blocks"]),
            system_prompt=str(self.prompt_ctx["system_prompt"]),
            user_template_single=str(self.prompt_ctx["user_template_single"]),
            user_template_agent1=str(self.prompt_ctx["user_template_agent1"]),
            user_template_agent2=str(self.prompt_ctx["user_template_agent2"]),
        )

    def reset_item_state(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        return self.item_to_payload(item)

    def transition(
        self,
        payload: Mapping[str, Any],
        agent_completions: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
        return transition_payload(
            payload=payload,
            agent_completions=list(agent_completions),
            num_agents=self.num_agents,
        )

    def render_prompts(
        self,
        payload: Mapping[str, Any],
        *,
        feedback_by_agent: Optional[Sequence[str]] = None,
    ) -> List[str]:
        task = payload_to_task(payload)
        state = payload_to_state(payload)
        limits = split_command_limits(
            max_commands_total=int(payload.get("max_commands_total") or self.max_commands_total),
            num_agents=self.num_agents,
        )
        allowed = payload_allowed_blocks(payload, num_agents=self.num_agents)
        feedbacks = list(feedback_by_agent or [])
        while len(feedbacks) < self.num_agents:
            feedbacks.append("")

        prompts: List[str] = []
        for idx in range(self.num_agents):
            template = self.prompt_ctx["user_template_single"]
            if self.num_agents > 1:
                template = self.prompt_ctx["user_template_agent1"] if idx == 0 else self.prompt_ctx["user_template_agent2"]
            user_prompt = render_agent_user_prompt(
                task=task,
                state=state,
                agent_idx=idx,
                view=self.view,
                allowed_blocks=allowed[idx],
                max_probe=self.max_probe,
                max_commands=int(limits[idx]),
                user_template=str(template),
                feedback=str(feedbacks[idx]),
            )
            prompts.append(
                self._render_prompt(
                    system_prompt=str(self.prompt_ctx["system_prompt"]),
                    user_prompt=user_prompt,
                )
            )
        return prompts

    def initial_prompts(self, payload: Mapping[str, Any]) -> List[str]:
        return self.render_prompts(payload, feedback_by_agent=["" for _ in range(self.num_agents)])

    def followup_prompts(
        self,
        *,
        payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
        previous_completions: Sequence[str],
        prompt_history_per_agent: Optional[List[List[str]]] = None,
        response_history_per_agent: Optional[List[List[str]]] = None,
    ) -> List[str]:
        del response_history_per_agent
        turn_no = self._turn_number(prompt_history_per_agent)
        feedbacks = self._build_feedbacks(payload=payload, metrics=metrics, turn_no=turn_no)
        prompts = (
            self.render_prompts(payload, feedback_by_agent=feedbacks)
            if self.original_prompt
            else list(feedbacks)
        )
        if self.previous_response:
            for idx, prev in enumerate(previous_completions):
                prev_txt = str(prev or "").strip()
                if prev_txt:
                    prompts[idx] = prompts[idx].rstrip() + "\n\nYour previous action JSON:\n" + prev_txt
        return prompts

    def build_meta_observations(self, payload: Mapping[str, Any]) -> List[AgentMetaObservation]:
        task = payload_to_task(payload)
        state = payload_to_state(payload)
        outputs: List[AgentMetaObservation] = []
        for idx in range(self.num_agents):
            obs = get_agent_observation(task, state, agent_idx=idx, view=self.view)
            visible = compute_visible_cells(
                state.vision_origins[idx],
                view=self.view,
                width=task.width,
                height=task.height,
            )
            channels = self._build_channels(
                task=task,
                obs=obs,
                visible=visible,
                width=task.width,
                height=task.height,
            )
            scalars = self._build_scalars(
                obs=obs,
                width=task.width,
                height=task.height,
            )
            task_id = str(task.task_id)
            outputs.append(
                AgentMetaObservation(
                    grid=channels,
                    scalars=scalars,
                    task_index=int(self.task_id_to_index[task_id]),
                    task_id=task_id,
                    turn_index=int(obs["turn_index"]),
                )
            )
        return outputs

    def build_joint_context(self, contexts: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(contexts) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} agent contexts, got {len(contexts)}")
        if self.num_agents == 1:
            return contexts[0]
        left, right = contexts[0], contexts[1]
        return torch.cat([left, right, left * right], dim=-1)

    def debug_turn(
        self,
        *,
        payload: Mapping[str, Any],
        turn_idx: int,
        reward: float,
        metrics: Mapping[str, Any],
        agent_outputs: Sequence[str],
    ) -> None:
        if not self.debug:
            return
        from LLM_Collab_Minecraft.bridge_build.utils.debug import print_turn_debug

        task = payload_to_task(payload)
        state = payload_to_state(payload)
        print_turn_debug(
            task=task,
            state=state,
            turn_idx=turn_idx,
            reward=float(reward),
            metrics=metrics,
            agent_outputs=list(agent_outputs),
        )

    def _render_prompt(self, *, system_prompt: str, user_prompt: str) -> str:
        tokenizer = self.tokenizer
        use_chat_template = bool(self.prompt_ctx.get("use_chat_template", False))
        system_prompt = (system_prompt or "").rstrip()
        user_prompt = (user_prompt or "").rstrip()
        if (
            use_chat_template
            and tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        ):
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                return tokenizer.apply_chat_template(messages, tokenize=False)
        if system_prompt:
            return system_prompt + "\n\n" + user_prompt
        return user_prompt

    def _turn_number(self, prompt_history_per_agent: Optional[List[List[str]]]) -> int:
        if not prompt_history_per_agent:
            return 1
        try:
            return int(len(prompt_history_per_agent[0]) + 1)
        except Exception:
            return 1

    def _build_feedbacks(
        self,
        *,
        payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
        turn_no: int,
    ) -> List[str]:
        mode = self.external_mode
        if mode in ("", "empty", "empty_feedback", "empty-feedback"):
            return ["" for _ in range(self.num_agents)]
        if mode in ("score", "score_feedback", "score-feedback"):
            return [self._score_feedback(metrics=metrics, turn_no=turn_no) for _ in range(self.num_agents)]
        if mode in ("perfect", "perfect_feedback", "perfect-feedback", "feedback"):
            return [
                self._perfect_feedback(metrics=metrics, turn_no=turn_no, agent_idx=idx)
                for idx in range(self.num_agents)
            ]
        if mode in ("position", "position_feedback", "position-feedback"):
            task = payload_to_task(payload)
            state = payload_to_state(payload)
            return [
                self._position_feedback(
                    task=task,
                    state=state,
                    metrics=metrics,
                    turn_no=turn_no,
                    agent_idx=idx,
                )
                for idx in range(self.num_agents)
            ]
        raise NotImplementedError(f"Unsupported external mode for bridge_build_meta: {self.external_mode}")

    def _score_feedback(self, *, metrics: Mapping[str, Any], turn_no: int) -> str:
        return "\n".join(
            [
                "Score feedback:",
                f"- Turn: {turn_no}",
                f"- reward: {float(metrics.get('reward', 0.0)):.4f}",
                f"- bonus_gap_st: {float(metrics.get('bonus_gap_st', 0.0)):.4f}",
                f"- gap_ST: {metrics.get('gap_st', None)} / {int(metrics.get('max_gap_st', 0))}",
                f"- bonus_cc_merge: {float(metrics.get('bonus_cc_merge', 0.0)):.4f}",
                f"- CC components: {int(metrics.get('cc_component_count', 0))} / {int(metrics.get('initial_cc_component_count', 0))}",
                f"- new_cc_merge_count: {int(metrics.get('new_cc_merge_count', 0))}",
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

    def _perfect_feedback(self, *, metrics: Mapping[str, Any], turn_no: int, agent_idx: int) -> str:
        return "\n".join(
            [
                "Perfect feedback:",
                f"- Turn: {turn_no}",
                f"- Agent: {'A' if agent_idx == 0 else 'B'}",
                f"- reward: {float(metrics.get('reward', 0.0)):.4f}",
                f"- connected(S,T): {bool(metrics.get('connected', False))}",
                f"- gap_ST: {metrics.get('gap_st', None)} / {int(metrics.get('max_gap_st', 0))}",
                f"- CC components: {int(metrics.get('cc_component_count', 0))} / {int(metrics.get('initial_cc_component_count', 0))}",
                f"- Y connected count: {int(metrics.get('connected_y_count', 0))}",
                f"- N adjacent count: {int(metrics.get('n_adjacent_count', 0))}",
                f"- Y uncovered count: {int(metrics.get('y_uncovered_count', 0))}",
                f"- valid probes: {int(metrics.get('num_valid_probes', 0))}",
                f"- comm tokens: {int(metrics.get('comm_tokens', 0))}",
                "- Target: gather information, avoid new N adjacency, connect more Y, then connect S/T.",
            ]
        )

    def _position_feedback(
        self,
        *,
        task: Any,
        state: Any,
        metrics: Mapping[str, Any],
        turn_no: int,
        agent_idx: int,
    ) -> str:
        obs = get_agent_observation(task, state, agent_idx=agent_idx, view=self.view)
        return "\n".join(
            [
                "Position feedback:",
                f"- Turn: {turn_no}",
                f"- Agent: {'A' if agent_idx == 0 else 'B'}",
                f"- Current position: {obs.get('current_pos')}",
                f"- Visible land count: {len(obs.get('visible_land_coords') or [])}",
                f"- Visible pillar-candidate count: {len(obs.get('visible_p_candidates') or [])}",
                f"- Known probe-result count: {len(obs.get('known_probe_results') or [])}",
                f"- connected(S,T): {bool(metrics.get('connected', False))}",
            ]
        )

    def _coord_set(self, coords: Iterable[Sequence[int]]) -> List[Coord]:
        out: List[Coord] = []
        for item in coords:
            if item is None or len(item) < 2:
                continue
            out.append((int(item[0]), int(item[1])))
        return out

    def _build_channels(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        visible: Iterable[Coord],
        width: int,
        height: int,
    ) -> torch.Tensor:
        channels = torch.zeros(self.grid_channels, height, width, dtype=torch.float32)
        visible_set = set(visible)
        self._fill_coords(channels[0], visible_set)
        self._fill_coords(channels[1], self._coord_set(obs.get("visible_land_coords") or []))
        self._fill_coords(channels[2], self._coord_set(obs.get("visible_filled_coords") or []))
        self._fill_coords(channels[3], self._coord_set(obs.get("visible_p_candidates") or []))

        s_coords = []
        t_coords = []
        for anchor in obs.get("visible_anchors") or []:
            if not isinstance(anchor, Mapping):
                continue
            coord = anchor.get("coord") or []
            if len(coord) < 2:
                continue
            item = (int(coord[0]), int(coord[1]))
            if str(anchor.get("kind")) == "S":
                s_coords.append(item)
            elif str(anchor.get("kind")) == "T":
                t_coords.append(item)
        self._fill_coords(channels[4], s_coords)
        self._fill_coords(channels[5], t_coords)

        known_y: List[Coord] = []
        known_n: List[Coord] = []
        for result in obs.get("known_probe_results") or []:
            if not isinstance(result, Mapping):
                continue
            coord = result.get("coord") or []
            if len(coord) < 2:
                continue
            item = (int(coord[0]), int(coord[1]))
            pillar_type = str(result.get("type") or "")
            if pillar_type == "Y":
                known_y.append(item)
            elif pillar_type == "N":
                known_n.append(item)
        self._fill_coords(channels[6], known_y)
        self._fill_coords(channels[7], known_n)

        msg_y: List[Coord] = []
        msg_n: List[Coord] = []
        msg_candidate: List[Coord] = []
        for message in obs.get("received_messages") or []:
            facts = self._extract_message_facts(message)
            msg_y.extend(facts["y"])
            msg_n.extend(facts["n"])
            msg_candidate.extend(facts["candidate"])
        self._fill_coords(channels[8], msg_y)
        self._fill_coords(channels[9], msg_n)
        self._fill_coords(channels[10], msg_candidate)

        current_pos = obs.get("current_pos") or [0, 0]
        self._fill_coords(channels[11], [(int(current_pos[0]), int(current_pos[1]))])
        return channels

    def _build_scalars(
        self,
        *,
        obs: Mapping[str, Any],
        width: int,
        height: int,
    ) -> torch.Tensor:
        pos = obs.get("current_pos") or [0, 0]
        known = obs.get("known_probe_results") or []
        num_known_y = 0
        num_known_n = 0
        for item in known:
            if not isinstance(item, Mapping):
                continue
            pillar_type = str(item.get("type") or "")
            if pillar_type == "Y":
                num_known_y += 1
            elif pillar_type == "N":
                num_known_n += 1
        max_turns = max(1, int(obs.get("max_turns") or 1))
        turn_idx = int(obs.get("turn_index") or 1)
        scalars = torch.tensor(
            [
                float(turn_idx) / float(max_turns),
                float(int(pos[0])) / float(max(1, width - 1)),
                float(int(pos[1])) / float(max(1, height - 1)),
                float(num_known_y) / 8.0,
                float(num_known_n) / 8.0,
                float(len(obs.get("received_messages") or [])) / 4.0,
                1.0 if bool(obs.get("connected", False)) else 0.0,
                1.0 if bool(obs.get("terminated", False)) else 0.0,
            ],
            dtype=torch.float32,
        )
        return scalars

    def _extract_message_facts(self, message: Any) -> Dict[str, List[Coord]]:
        out = {"y": [], "n": [], "candidate": []}
        if not isinstance(message, Mapping):
            return out

        discovered = message.get("discovered_pillars") or []
        typed_facts = message.get("typed_facts") or []
        candidates = message.get("candidate_coords") or []

        for fact in list(discovered) + list(typed_facts):
            if not isinstance(fact, Mapping):
                continue
            coord = fact.get("coord")
            if coord is None:
                x = fact.get("x")
                z = fact.get("z")
                if x is None or z is None:
                    continue
                coord = [x, z]
            if len(coord) < 2:
                continue
            item = (int(coord[0]), int(coord[1]))
            pillar_type = str(fact.get("type") or "")
            if pillar_type == "Y":
                out["y"].append(item)
            elif pillar_type == "N":
                out["n"].append(item)

        for coord in candidates:
            if coord is None or len(coord) < 2:
                continue
            out["candidate"].append((int(coord[0]), int(coord[1])))

        return out

    def _fill_coords(self, channel: torch.Tensor, coords: Iterable[Coord]) -> None:
        h, w = channel.shape
        for x, z in coords:
            if 0 <= int(x) < w and 0 <= int(z) < h:
                channel[int(z), int(x)] = 1.0
