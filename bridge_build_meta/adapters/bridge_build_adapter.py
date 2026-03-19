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
    belief_target: torch.Tensor
    belief_mask: torch.Tensor
    task_index: int
    task_id: str
    turn_index: int


@dataclass
class AgentActionCandidates:
    comm_options: List[Dict[str, Any]]
    probe_options: List[List[List[int]]]
    cmd_options: List[List[str]]
    path_options: List[List[List[int]]]
    comm_preference_scores: Optional[List[float]] = None
    probe_preference_scores: Optional[List[float]] = None
    cmd_preference_scores: Optional[List[float]] = None
    path_preference_scores: Optional[List[float]] = None


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
        task_specs: Optional[Sequence[Any]] = None,
        tokenizer: Any | None = None,
        external_mode: str = "empty_feedback",
        original_prompt: bool = True,
        previous_response: bool = False,
        debug: bool = False,
        reward_config: Optional[Mapping[str, Any]] = None,
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
        task_specs = list(task_specs or [])
        self.max_width = max((int(task.width) for task in task_specs), default=1)
        self.max_height = max((int(task.height) for task in task_specs), default=1)
        reward_cfg = dict(reward_config or {})
        self.n_adjacent_penalty_scale = float(reward_cfg.get("n_adjacent_penalty_scale", 1.5))
        self.cc_merge_bonus_scale = float(reward_cfg.get("cc_merge_bonus_scale", 0.5))
        self.move_progress_bonus_total = float(reward_cfg.get("move_progress_bonus_total", 2.5))

    @property
    def grid_channels(self) -> int:
        return len(self.channel_names)

    @property
    def scalar_dim(self) -> int:
        return 8

    @property
    def task_vocab_size(self) -> int:
        return len(self.task_id_to_index)

    @property
    def belief_dim(self) -> int:
        return int(self.max_width * self.max_height)

    def item_to_payload(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        task = task_from_item(item)
        raw_state = item.get("_bridge_state_before_turn")
        if isinstance(raw_state, Mapping):
            state = deserialize_state(raw_state, num_agents=self.num_agents)
        else:
            state = make_initial_state(task, num_agents=self.num_agents, max_turns=task.max_turns)
        payload = build_payload(
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
        payload["reward_state"] = {
            "move_distance_norm": float(
                max(1.0, self._movement_distance_total(task=task, state=state))
            ),
        }
        return payload

    def reset_item_state(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        return self.item_to_payload(item)

    def transition(
        self,
        payload: Mapping[str, Any],
        agent_completions: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
        task = payload_to_task(payload)
        prev_state = payload_to_state(payload)
        next_payload, metrics, actions = transition_payload(
            payload=payload,
            agent_completions=list(agent_completions),
            num_agents=self.num_agents,
        )
        reward_state = payload.get("reward_state") or {}
        move_distance_norm = float(reward_state.get("move_distance_norm", 0.0))
        if move_distance_norm <= 0.0:
            move_distance_norm = float(max(1.0, self._movement_distance_total(task=task, state=prev_state)))
        next_payload["reward_state"] = {"move_distance_norm": move_distance_norm}
        next_state = payload_to_state(next_payload)
        adjusted_metrics = self._adjust_reward_metrics(
            metrics=dict(metrics),
            task=task,
            prev_state=prev_state,
            next_state=next_state,
            move_distance_norm=move_distance_norm,
        )
        return next_payload, adjusted_metrics, actions

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
                    belief_target=self._build_belief_target(task=task),
                    belief_mask=self._build_belief_mask(task=task, obs=obs),
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

    def build_action_candidates(self, payload: Mapping[str, Any]) -> List[AgentActionCandidates]:
        task = payload_to_task(payload)
        state = payload_to_state(payload)
        allowed = payload_allowed_blocks(payload, num_agents=self.num_agents)
        limits = split_command_limits(
            max_commands_total=int(payload.get("max_commands_total") or self.max_commands_total),
            num_agents=self.num_agents,
        )
        outputs: List[AgentActionCandidates] = []
        for agent_idx in range(self.num_agents):
            obs = get_agent_observation(task, state, agent_idx=agent_idx, view=self.view)
            visible = compute_visible_cells(
                state.vision_origins[agent_idx],
                view=self.view,
                width=task.width,
                height=task.height,
            )
            unknown_visible = self._visible_unknown_candidates(
                task=task,
                state=state,
                agent_idx=agent_idx,
                visible=visible,
            )
            comm_options = self._build_comm_options(
                obs=obs,
                unknown_visible=unknown_visible,
            )
            probe_options = self._build_probe_options(
                obs=obs,
                unknown_visible=unknown_visible,
            )
            cmd_options = self._build_cmd_options(
                task=task,
                state=state,
                agent_idx=agent_idx,
                visible=visible,
                allowed_blocks=allowed[agent_idx],
                max_commands=int(limits[agent_idx]),
                obs=obs,
            )
            path_options = self._build_path_options(
                task=task,
                state=state,
                agent_idx=agent_idx,
                unknown_visible=unknown_visible,
                obs=obs,
            )
            outputs.append(
                AgentActionCandidates(
                    comm_options=comm_options,
                    probe_options=probe_options,
                    cmd_options=cmd_options,
                    path_options=path_options,
                    comm_preference_scores=self._build_comm_preference_scores(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        comm_options=comm_options,
                    ),
                    probe_preference_scores=self._build_probe_preference_scores(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        probe_options=probe_options,
                    ),
                    cmd_preference_scores=self._build_cmd_preference_scores(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        cmd_options=cmd_options,
                    ),
                    path_preference_scores=self._build_path_preference_scores(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        path_options=path_options,
                        cmd_options=cmd_options,
                    ),
                )
            )
        return outputs

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

    def _coord_list(self, coord: Coord) -> List[int]:
        return [int(coord[0]), int(coord[1])]

    def _dedupe_jsonable_options(self, options: Sequence[Any]) -> List[Any]:
        seen: set[str] = set()
        out: List[Any] = []
        for item in options:
            key = str(item)
            try:
                import json

                key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except Exception:
                key = str(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(copy.deepcopy(item))
        return out

    def _neighbors4(self, coord: Coord) -> List[Coord]:
        x, z = int(coord[0]), int(coord[1])
        return [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)]

    def _in_bounds(self, coord: Coord, *, task: Any) -> bool:
        return 0 <= int(coord[0]) < int(task.width) and 0 <= int(coord[1]) < int(task.height)

    def _agent_goal_hints(self, *, task: Any, obs: Mapping[str, Any], agent_idx: int) -> List[Coord]:
        target_kind = "T" if int(agent_idx) == 0 else "S"
        visible_anchors = obs.get("visible_anchors") or []
        out: List[Coord] = []
        for item in visible_anchors:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("kind") or "") != target_kind:
                continue
            coord = item.get("coord")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            out.append((int(coord[0]), int(coord[1])))
        if out:
            return out
        if int(agent_idx) == 0:
            return [(int(task.width) - 1, int(task.height) - 1)]
        return [(0, 0)]

    def _visible_unknown_candidates(
        self,
        *,
        task: Any,
        state: Any,
        agent_idx: int,
        visible: Iterable[Coord],
    ) -> List[Coord]:
        visible_set = {(int(x), int(z)) for x, z in visible}
        known = {(int(x), int(z)) for x, z in state.known_pillars[agent_idx].keys()}
        coords = [
            (int(x), int(z))
            for x, z in task.candidate_pillars
            if (int(x), int(z)) in visible_set and (int(x), int(z)) not in known
        ]
        return list(sorted(coords))

    def _build_comm_options(
        self,
        *,
        obs: Mapping[str, Any],
        unknown_visible: Sequence[Coord],
    ) -> List[Dict[str, Any]]:
        known_items = obs.get("known_probe_results") or []
        discovered: List[Dict[str, Any]] = []
        for item in known_items:
            if not isinstance(item, Mapping):
                continue
            coord = item.get("coord")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            discovered.append(
                {
                    "x": int(coord[0]),
                    "z": int(coord[1]),
                    "type": str(item.get("type") or ""),
                    "source": "self_probe",
                }
            )
        candidate_coords = [self._coord_list(coord) for coord in list(unknown_visible)[:8]]
        options: List[Dict[str, Any]] = [{}]
        if discovered:
            options.append({"discovered_pillars": discovered})
        if discovered or candidate_coords:
            options.append(
                {
                    "discovered_pillars": discovered,
                    "candidate_coords": candidate_coords,
                }
            )
        return self._dedupe_jsonable_options(options)

    def _rank_coords(
        self,
        *,
        coords: Sequence[Coord],
        origin: Coord,
        targets: Sequence[Coord],
    ) -> List[Coord]:
        def _score(coord: Coord) -> Tuple[int, int, int, int]:
            dist_origin = abs(int(coord[0]) - int(origin[0])) + abs(int(coord[1]) - int(origin[1]))
            dist_target = self._min_manhattan_distance(origin=coord, targets=targets)
            return (dist_target, dist_origin, int(coord[1]), int(coord[0]))

        return list(sorted(((int(x), int(z)) for x, z in coords), key=_score))

    def _build_probe_options(
        self,
        *,
        obs: Mapping[str, Any],
        unknown_visible: Sequence[Coord],
    ) -> List[List[List[int]]]:
        current_pos = obs.get("current_pos") or [0, 0]
        origin = (int(current_pos[0]), int(current_pos[1]))
        ranked = self._rank_coords(coords=unknown_visible, origin=origin, targets=[origin])
        options: List[List[List[int]]] = [[]]
        capped = ranked[: max(0, self.max_probe)]
        for count in range(1, min(self.max_probe, len(capped)) + 1):
            options.append([self._coord_list(coord) for coord in capped[:count]])
        return self._dedupe_jsonable_options(options)

    def _coord_priority_score(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        coord: Coord,
    ) -> float:
        cell = (int(coord[0]), int(coord[1]))
        current_pos_raw = obs.get("current_pos") or [0, 0]
        current_pos = (int(current_pos_raw[0]), int(current_pos_raw[1]))
        goal_hints = self._agent_goal_hints(task=task, obs=obs, agent_idx=agent_idx)
        closing_targets = self._closing_path_targets(task=task, obs=obs, agent_idx=agent_idx)
        frontier = self._known_frontier_cells(task=task, obs=obs)
        known_n = self._known_n_cells(obs=obs)

        dist_cur = abs(int(cell[0]) - int(current_pos[0])) + abs(int(cell[1]) - int(current_pos[1]))
        frontier_touch = sum(1 for nb in self._neighbors4(cell) if nb in frontier)
        n_touch = sum(1 for nb in self._neighbors4(cell) if nb in known_n)

        score = 0.0
        if closing_targets:
            dist_closing = self._min_manhattan_distance(origin=cell, targets=closing_targets)
            score += max(0.0, 2.4 - 0.8 * float(dist_closing))
        if frontier:
            dist_frontier = self._min_manhattan_distance(origin=cell, targets=list(frontier))
            score += max(0.0, 1.2 - 0.35 * float(dist_frontier))
        if goal_hints:
            dist_goal = self._min_manhattan_distance(origin=cell, targets=goal_hints)
            score += max(0.0, 0.8 - 0.12 * float(dist_goal))
        score += 0.25 * float(frontier_touch)
        score -= 0.08 * float(dist_cur)
        score -= 0.55 * float(n_touch)
        return score

    def _build_comm_preference_scores(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        comm_options: Sequence[Mapping[str, Any]],
    ) -> Optional[List[float]]:
        if not comm_options:
            return None
        scores: List[float] = []
        for option in comm_options:
            discovered = option.get("discovered_pillars") or []
            candidates = option.get("candidate_coords") or []
            if not discovered and not candidates:
                scores.append(0.0)
                continue

            score = 0.0
            for fact in discovered:
                if not isinstance(fact, Mapping):
                    continue
                coord = fact.get("coord")
                if coord is None:
                    x = fact.get("x")
                    z = fact.get("z")
                    if x is None or z is None:
                        continue
                    coord = [x, z]
                if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                    continue
                priority = max(
                    0.0,
                    self._coord_priority_score(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        coord=(int(coord[0]), int(coord[1])),
                    ),
                )
                pillar_type = str(fact.get("type") or "")
                if pillar_type == "Y":
                    score += 1.4 + 0.45 * priority
                elif pillar_type == "N":
                    score += 0.9 + 0.30 * priority
                else:
                    score += 0.25 + 0.20 * priority

            candidate_bonus = 0.0
            for coord in list(candidates)[:4]:
                if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                    continue
                priority = max(
                    0.0,
                    self._coord_priority_score(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        coord=(int(coord[0]), int(coord[1])),
                    ),
                )
                candidate_bonus += 0.2 + 0.35 * priority
            score += candidate_bonus
            scores.append(float(score))
        return scores

    def _build_probe_preference_scores(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        probe_options: Sequence[Sequence[Sequence[int]]],
    ) -> Optional[List[float]]:
        if not probe_options:
            return None
        scores: List[float] = []
        for option in probe_options:
            if not option:
                scores.append(0.0)
                continue
            priority_sum = 0.0
            for coord in option:
                if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                    continue
                priority_sum += max(
                    0.0,
                    self._coord_priority_score(
                        task=task,
                        obs=obs,
                        agent_idx=agent_idx,
                        coord=(int(coord[0]), int(coord[1])),
                    ),
                )
            scores.append(float(priority_sum + 0.3 * float(len(option))))
        return scores

    def _greedy_path(
        self,
        *,
        start: Coord,
        target: Coord,
        task: Any,
        max_steps: int,
    ) -> List[List[int]]:
        cur = (int(start[0]), int(start[1]))
        goal = (int(target[0]), int(target[1]))
        path: List[List[int]] = [self._coord_list(cur)]
        for _ in range(max(0, int(max_steps))):
            if cur == goal:
                break
            dx = int(goal[0]) - int(cur[0])
            dz = int(goal[1]) - int(cur[1])
            step = (
                int(cur[0] + (1 if dx > 0 else -1 if dx < 0 else 0)),
                int(cur[1] + (1 if dz > 0 else -1 if dz < 0 else 0)),
            )
            if not self._in_bounds(step, task=task):
                break
            cur = step
            path.append(self._coord_list(cur))
        return path

    def _build_path_options(
        self,
        *,
        task: Any,
        state: Any,
        agent_idx: int,
        unknown_visible: Sequence[Coord],
        obs: Mapping[str, Any],
    ) -> List[List[List[int]]]:
        current_pos = (int(state.agent_positions[agent_idx][0]), int(state.agent_positions[agent_idx][1]))
        options: List[List[List[int]]] = [[self._coord_list(current_pos)]]
        goal_hints = self._agent_goal_hints(task=task, obs=obs, agent_idx=agent_idx)
        if goal_hints:
            options.append(
                self._greedy_path(
                    start=current_pos,
                    target=goal_hints[0],
                    task=task,
                    max_steps=max(1, min(5, int(self.view) + 2)),
                )
            )
        if unknown_visible:
            ranked_unknown = self._rank_coords(coords=unknown_visible, origin=current_pos, targets=goal_hints or [current_pos])
            options.append(
                self._greedy_path(
                    start=current_pos,
                    target=ranked_unknown[0],
                    task=task,
                    max_steps=max(1, min(5, int(self.view) + 2)),
                )
            )
        closing_targets = self._closing_path_targets(
            task=task,
            obs=obs,
            agent_idx=agent_idx,
        )
        max_steps = max(1, min(5, int(self.view) + 2))
        for target in closing_targets[:2]:
            options.append(
                self._greedy_path(
                    start=current_pos,
                    target=target,
                    task=task,
                    max_steps=max_steps,
                )
            )
        return self._dedupe_jsonable_options(options)

    def _format_fill_command(self, *, start: Coord, end: Coord, block: str) -> str:
        x1, z1 = int(start[0]), int(start[1])
        x2, z2 = int(end[0]), int(end[1])
        return f"/fill {min(x1, x2)} {min(z1, z2)} {max(x1, x2)} {max(z1, z2)} {block}"

    def _known_message_facts(self, *, obs: Mapping[str, Any]) -> Dict[str, List[Coord]]:
        out = {"y": [], "n": [], "candidate": []}
        for message in obs.get("received_messages") or []:
            facts = self._extract_message_facts(message)
            out["y"].extend(facts["y"])
            out["n"].extend(facts["n"])
            out["candidate"].extend(facts["candidate"])
        return out

    def _known_typed_cells(self, *, obs: Mapping[str, Any]) -> set[Coord]:
        out: set[Coord] = set()
        for item in obs.get("known_probe_results") or []:
            if not isinstance(item, Mapping):
                continue
            coord = item.get("coord")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            pillar_type = str(item.get("type") or "")
            if pillar_type not in {"Y", "N"}:
                continue
            out.add((int(coord[0]), int(coord[1])))
        facts = self._known_message_facts(obs=obs)
        out.update((int(x), int(z)) for x, z in facts["y"])
        out.update((int(x), int(z)) for x, z in facts["n"])
        return out

    def _known_candidate_cells(self, *, obs: Mapping[str, Any]) -> set[Coord]:
        out: set[Coord] = set()
        for coord in obs.get("visible_p_candidates") or []:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                out.add((int(coord[0]), int(coord[1])))
        for item in obs.get("known_probe_results") or []:
            if not isinstance(item, Mapping):
                continue
            coord = item.get("coord")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            out.add((int(coord[0]), int(coord[1])))
        facts = self._known_message_facts(obs=obs)
        out.update((int(x), int(z)) for x, z in facts["y"])
        out.update((int(x), int(z)) for x, z in facts["n"])
        out.update((int(x), int(z)) for x, z in facts["candidate"])
        return out

    def _known_n_cells(self, *, obs: Mapping[str, Any]) -> set[Coord]:
        out: set[Coord] = set()
        for item in obs.get("known_probe_results") or []:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("type") or "") != "N":
                continue
            coord = item.get("coord")
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            out.add((int(coord[0]), int(coord[1])))
        facts = self._known_message_facts(obs=obs)
        out.update((int(x), int(z)) for x, z in facts["n"])
        return out

    def _visible_land_cells(self, *, obs: Mapping[str, Any]) -> set[Coord]:
        out: set[Coord] = set()
        for coord in obs.get("visible_land_coords") or []:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                out.add((int(coord[0]), int(coord[1])))
        return out

    def _visible_filled_cells(self, *, obs: Mapping[str, Any]) -> set[Coord]:
        out: set[Coord] = set()
        for coord in obs.get("visible_filled_coords") or []:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                out.add((int(coord[0]), int(coord[1])))
        return out

    def _known_traversable_cells(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        extra_filled: Optional[Iterable[Coord]] = None,
    ) -> set[Coord]:
        traversable = {
            (int(x), int(z)) for x, z in [*task.anchors_s, *task.anchors_t]
        }
        traversable.update(self._visible_filled_cells(obs=obs))
        traversable.update(self._known_candidate_cells(obs=obs))
        if extra_filled is not None:
            traversable.update((int(x), int(z)) for x, z in extra_filled)
        return traversable

    def _known_gap_estimate(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        extra_filled: Optional[Iterable[Coord]] = None,
    ) -> Optional[int]:
        starts = {(int(x), int(z)) for x, z in task.anchors_s}
        targets = {(int(x), int(z)) for x, z in task.anchors_t}
        if not starts or not targets:
            return None

        traversable = self._known_traversable_cells(task=task, obs=obs, extra_filled=extra_filled)
        immutable = self._visible_land_cells(obs=obs) | self._known_candidate_cells(obs=obs)

        dist: Dict[Coord, int] = {pos: 0 for pos in starts}
        queue: List[Coord] = list(starts)
        from collections import deque

        q = deque(queue)
        while q:
            cur = q.popleft()
            cur_dist = dist[cur]
            if cur in targets:
                return int(cur_dist)
            for nb in self._neighbors4(cur):
                if not self._in_bounds(nb, task=task):
                    continue
                if nb in traversable:
                    next_dist = cur_dist
                elif nb in immutable:
                    continue
                else:
                    next_dist = cur_dist + 1
                prev = dist.get(nb)
                if prev is not None and prev <= next_dist:
                    continue
                dist[nb] = next_dist
                if next_dist == cur_dist:
                    q.appendleft(nb)
                else:
                    q.append(nb)
        return None

    def _known_connecting_path(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        extra_filled: Optional[Iterable[Coord]] = None,
    ) -> Optional[List[Coord]]:
        starts = {(int(x), int(z)) for x, z in task.anchors_s}
        targets = {(int(x), int(z)) for x, z in task.anchors_t}
        if not starts or not targets:
            return None

        traversable = self._known_traversable_cells(task=task, obs=obs, extra_filled=extra_filled)
        immutable = self._visible_land_cells(obs=obs) | self._known_candidate_cells(obs=obs)

        from collections import deque

        dist: Dict[Coord, int] = {pos: 0 for pos in starts}
        prev: Dict[Coord, Optional[Coord]] = {pos: None for pos in starts}
        q = deque(sorted(starts))
        found: Optional[Coord] = None
        while q:
            cur = q.popleft()
            cur_dist = dist[cur]
            if cur in targets:
                found = cur
                break
            for nb in self._neighbors4(cur):
                if not self._in_bounds(nb, task=task):
                    continue
                if nb in traversable:
                    next_dist = cur_dist
                elif nb in immutable:
                    continue
                else:
                    next_dist = cur_dist + 1
                prev_dist = dist.get(nb)
                if prev_dist is not None and prev_dist <= next_dist:
                    continue
                dist[nb] = next_dist
                prev[nb] = cur
                if next_dist == cur_dist:
                    q.appendleft(nb)
                else:
                    q.append(nb)
        if found is None:
            return None
        path_rev: List[Coord] = []
        cur: Optional[Coord] = found
        while cur is not None:
            path_rev.append(cur)
            cur = prev.get(cur)
        return list(reversed(path_rev))

    def _orient_path_for_agent(self, *, path: Sequence[Coord], agent_idx: int) -> List[Coord]:
        oriented = [(int(x), int(z)) for x, z in path]
        if int(agent_idx) == 0:
            return oriented
        return list(reversed(oriented))

    def _closing_path_targets(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
    ) -> List[Coord]:
        path = self._known_connecting_path(task=task, obs=obs)
        if not path:
            return []
        known_traversable = self._known_traversable_cells(task=task, obs=obs)
        oriented = self._orient_path_for_agent(path=path, agent_idx=agent_idx)
        missing_cells = [cell for cell in oriented if cell not in known_traversable]
        if missing_cells:
            targets: List[Coord] = [missing_cells[0]]
            frontier_cells = [
                cell
                for cell in oriented
                if cell in known_traversable and any(nb == missing_cells[0] for nb in self._neighbors4(cell))
            ]
            if frontier_cells:
                targets.append(frontier_cells[0])
            deduped: List[Coord] = []
            seen: set[Coord] = set()
            for cell in targets:
                if cell in seen:
                    continue
                seen.add(cell)
                deduped.append(cell)
            return deduped
        frontier_cells = [
            cell
            for cell in oriented
            if cell in known_traversable and any(nb not in known_traversable for nb in self._neighbors4(cell))
        ]
        return frontier_cells[:1]

    def _parse_fill_command_cells(self, cmd: str) -> set[Coord]:
        raw = str(cmd or "").strip()
        if not raw.startswith("/fill "):
            return set()
        parts = raw.split()
        if len(parts) < 6:
            return set()
        try:
            x1, z1, x2, z2 = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        except Exception:
            return set()
        out: set[Coord] = set()
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for z in range(min(z1, z2), max(z1, z2) + 1):
                out.add((int(x), int(z)))
        return out

    def _score_cmd_option(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        option: Sequence[str],
    ) -> float:
        cmds = [str(cmd) for cmd in option if str(cmd).strip()]
        if not cmds:
            return 0.0
        current_pos_raw = obs.get("current_pos") or [0, 0]
        current_pos = (int(current_pos_raw[0]), int(current_pos_raw[1]))
        goal_hints = self._agent_goal_hints(task=task, obs=obs, agent_idx=agent_idx)
        base_gap = self._known_gap_estimate(task=task, obs=obs)
        cmd_cells: set[Coord] = set()
        for cmd in cmds:
            cmd_cells.update(self._parse_fill_command_cells(cmd))
        if not cmd_cells:
            return 0.0
        next_gap = self._known_gap_estimate(task=task, obs=obs, extra_filled=cmd_cells)
        gap_gain = 0.0
        if base_gap is not None and next_gap is not None:
            gap_gain = float(base_gap - next_gap)

        known_traversable = self._known_traversable_cells(task=task, obs=obs)
        known_n = self._known_n_cells(obs=obs)
        frontier_touch = sum(1 for cell in cmd_cells if any(nb in known_traversable for nb in self._neighbors4(cell)))
        n_touch = sum(1 for cell in cmd_cells if any(nb in known_n for nb in self._neighbors4(cell)))
        cell_count = len(cmd_cells)
        dist_cur = min(
            abs(int(cell[0]) - int(current_pos[0])) + abs(int(cell[1]) - int(current_pos[1]))
            for cell in cmd_cells
        )
        dist_goal = min(
            self._min_manhattan_distance(origin=cell, targets=goal_hints)
            for cell in cmd_cells
        ) if goal_hints else 0
        return (
            4.0 * gap_gain
            + 0.35 * float(frontier_touch)
            - 0.25 * float(cell_count)
            - 0.7 * float(n_touch)
            - 0.05 * float(dist_cur)
            - 0.03 * float(dist_goal)
        )

    def _build_cmd_preference_scores(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        cmd_options: Sequence[Sequence[str]],
    ) -> Optional[List[float]]:
        if not cmd_options:
            return None
        return [
            float(self._score_cmd_option(task=task, obs=obs, agent_idx=agent_idx, option=option))
            for option in cmd_options
        ]

    def _known_frontier_cells(self, *, task: Any, obs: Mapping[str, Any]) -> set[Coord]:
        known_traversable = self._known_traversable_cells(task=task, obs=obs)
        immutable = self._visible_land_cells(obs=obs) | self._known_candidate_cells(obs=obs)
        frontier: set[Coord] = set()
        for cell in known_traversable:
            for nb in self._neighbors4(cell):
                if not self._in_bounds(nb, task=task):
                    continue
                if nb in known_traversable or nb in immutable:
                    continue
                frontier.add(nb)
        return frontier

    def _build_path_preference_scores(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        path_options: Sequence[Sequence[Sequence[int]]],
        cmd_options: Sequence[Sequence[str]],
    ) -> Optional[List[float]]:
        if not path_options:
            return None
        goal_hints = self._agent_goal_hints(task=task, obs=obs, agent_idx=agent_idx)
        frontier = self._known_frontier_cells(task=task, obs=obs)
        current_pos_raw = obs.get("current_pos") or [0, 0]
        current_pos = (int(current_pos_raw[0]), int(current_pos_raw[1]))
        cmd_scores = self._build_cmd_preference_scores(
            task=task,
            obs=obs,
            agent_idx=agent_idx,
            cmd_options=cmd_options,
        ) or []
        top_cmd_cells: set[Coord] = set()
        for score, cmds in sorted(zip(cmd_scores, cmd_options), key=lambda item: item[0], reverse=True)[:2]:
            if score <= 0.0:
                continue
            for cmd in cmds:
                top_cmd_cells.update(self._parse_fill_command_cells(str(cmd)))
        target_cells = top_cmd_cells or frontier
        scores: List[float] = []
        for option in path_options:
            if not option:
                scores.append(0.0)
                continue
            endpoint_raw = option[-1]
            endpoint = (int(endpoint_raw[0]), int(endpoint_raw[1]))
            path_len = max(0, len(option) - 1)
            dist_goal = self._min_manhattan_distance(origin=endpoint, targets=goal_hints) if goal_hints else 0
            dist_frontier = self._min_manhattan_distance(origin=endpoint, targets=list(frontier)) if frontier else 0
            dist_target = self._min_manhattan_distance(origin=endpoint, targets=list(target_cells)) if target_cells else dist_goal
            moved = 0.0 if endpoint == current_pos else 1.0
            scores.append(
                0.4 * moved
                - 0.18 * float(path_len)
                - 0.22 * float(dist_frontier)
                - 0.12 * float(dist_goal)
                - 0.28 * float(dist_target)
            )
        return scores

    def _extend_line_command(
        self,
        *,
        seed: Coord,
        targets: Sequence[Coord],
        placeable: set[Coord],
        block: str,
    ) -> Optional[str]:
        if not targets:
            return None
        target = targets[0]
        dx = int(target[0]) - int(seed[0])
        dz = int(target[1]) - int(seed[1])
        axes: List[Coord] = []
        if abs(dx) >= abs(dz) and dx != 0:
            axes.append((1 if dx > 0 else -1, 0))
        if dz != 0:
            axes.append((0, 1 if dz > 0 else -1))
        if not axes and dx != 0:
            axes.append((1 if dx > 0 else -1, 0))
        for step_x, step_z in axes:
            end = seed
            for _ in range(2):
                nxt = (int(end[0]) + int(step_x), int(end[1]) + int(step_z))
                if nxt not in placeable:
                    break
                end = nxt
            if end != seed:
                return self._format_fill_command(start=seed, end=end, block=block)
        return None

    def _build_closing_cmd_options(
        self,
        *,
        task: Any,
        obs: Mapping[str, Any],
        agent_idx: int,
        placeable: set[Coord],
        block: str,
        max_commands: int,
    ) -> List[List[str]]:
        if max_commands <= 0 or not placeable:
            return []
        path = self._known_connecting_path(task=task, obs=obs)
        if not path:
            return []
        known_traversable = self._known_traversable_cells(task=task, obs=obs)
        oriented = self._orient_path_for_agent(path=path, agent_idx=agent_idx)
        visible_missing = [cell for cell in oriented if cell not in known_traversable and cell in placeable]
        if not visible_missing:
            return []

        closing_cmds: List[str] = []
        for cell in visible_missing[:2]:
            closing_cmds.append(self._format_fill_command(start=cell, end=cell, block=block))

        if len(visible_missing) >= 2:
            segment: List[Coord] = [visible_missing[0]]
            for cell in visible_missing[1:]:
                prev = segment[-1]
                if abs(int(cell[0]) - int(prev[0])) + abs(int(cell[1]) - int(prev[1])) != 1:
                    break
                same_axis = int(cell[0]) == int(prev[0]) or int(cell[1]) == int(prev[1])
                if not same_axis:
                    break
                segment.append(cell)
                if len(segment) >= 3:
                    break
            if len(segment) >= 2:
                closing_cmds.append(
                    self._format_fill_command(start=segment[0], end=segment[-1], block=block)
                )

        closing_cmds = [str(cmd) for cmd in self._dedupe_jsonable_options(closing_cmds)]
        options: List[List[str]] = []
        for cmd in closing_cmds:
            options.append([cmd])
        if max_commands >= 2 and len(closing_cmds) >= 2:
            options.append([closing_cmds[0], closing_cmds[1]])
        return self._dedupe_jsonable_options(options)

    def _build_cmd_options(
        self,
        *,
        task: Any,
        state: Any,
        agent_idx: int,
        visible: Iterable[Coord],
        allowed_blocks: Sequence[str],
        max_commands: int,
        obs: Mapping[str, Any],
    ) -> List[List[str]]:
        if max_commands <= 0:
            return [[]]
        block = str((list(allowed_blocks) or ["oak_planks"])[0])
        visible_set = {(int(x), int(z)) for x, z in visible}
        immutable = {(int(x), int(z)) for x, z in task.land_cells} | {
            (int(x), int(z)) for x, z in task.candidate_pillars
        }
        placeable = {coord for coord in visible_set if coord not in immutable}
        if not placeable:
            return [[]]

        current_pos = (int(state.agent_positions[agent_idx][0]), int(state.agent_positions[agent_idx][1]))
        goal_hints = self._agent_goal_hints(task=task, obs=obs, agent_idx=agent_idx)
        visible_anchors = {
            (int(item["coord"][0]), int(item["coord"][1]))
            for item in (obs.get("visible_anchors") or [])
            if isinstance(item, Mapping) and isinstance(item.get("coord"), (list, tuple)) and len(item.get("coord")) >= 2
        }
        visible_filled = {
            (int(x), int(z)) for x, z in [tuple(coord) for coord in (obs.get("visible_filled_coords") or []) if len(coord) >= 2]
        }
        known_true = {
            (int(item["coord"][0]), int(item["coord"][1]))
            for item in (obs.get("known_probe_results") or [])
            if isinstance(item, Mapping)
            and str(item.get("type") or "") == "Y"
            and isinstance(item.get("coord"), (list, tuple))
            and len(item.get("coord")) >= 2
        }
        focus = set([current_pos]) | visible_anchors | visible_filled | known_true

        def _seed_score(cell: Coord) -> Tuple[float, int, int]:
            adj_focus = sum(1 for nb in self._neighbors4(cell) if nb in focus)
            dist_goal = self._min_manhattan_distance(origin=cell, targets=goal_hints or [current_pos])
            dist_cur = abs(int(cell[0]) - int(current_pos[0])) + abs(int(cell[1]) - int(current_pos[1]))
            return (-(3.0 * float(adj_focus) - 0.35 * float(dist_goal) - 0.15 * float(dist_cur)), int(cell[1]), int(cell[0]))

        ranked = list(sorted(placeable, key=_seed_score))
        command_pool: List[str] = []
        for seed in ranked[:6]:
            command_pool.append(self._format_fill_command(start=seed, end=seed, block=block))
            line_cmd = self._extend_line_command(
                seed=seed,
                targets=goal_hints or [current_pos],
                placeable=placeable,
                block=block,
            )
            if line_cmd is not None:
                command_pool.append(line_cmd)
        command_pool = self._dedupe_jsonable_options(command_pool)
        options: List[List[str]] = [[]]
        for cmd in command_pool[: min(3, len(command_pool))]:
            options.append([str(cmd)])
        if max_commands >= 2 and len(command_pool) >= 2:
            options.append([str(command_pool[0]), str(command_pool[1])])
        options.extend(
            self._build_closing_cmd_options(
                task=task,
                obs=obs,
                agent_idx=agent_idx,
                placeable=placeable,
                block=block,
                max_commands=max_commands,
            )
        )
        return self._dedupe_jsonable_options(options)

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
            if coord is None:
                continue
            if isinstance(coord, Mapping):
                x = coord.get("x")
                z = coord.get("z")
                if x is None or z is None:
                    continue
                out["candidate"].append((int(x), int(z)))
                continue
            if len(coord) < 2:
                continue
            out["candidate"].append((int(coord[0]), int(coord[1])))

        return out

    def _fill_coords(self, channel: torch.Tensor, coords: Iterable[Coord]) -> None:
        h, w = channel.shape
        for x, z in coords:
            if 0 <= int(x) < w and 0 <= int(z) < h:
                channel[int(z), int(x)] = 1.0

    def _belief_index(self, coord: Coord) -> int:
        x, z = int(coord[0]), int(coord[1])
        return int(z * self.max_width + x)

    def _build_belief_target(self, *, task: Any) -> torch.Tensor:
        target = torch.zeros(self.belief_dim, dtype=torch.float32)
        for coord in task.true_pillars:
            target[self._belief_index(coord)] = 1.0
        return target

    def _build_belief_mask(self, *, task: Any, obs: Mapping[str, Any]) -> torch.Tensor:
        mask = torch.zeros(self.belief_dim, dtype=torch.bool)
        resolved = self._known_typed_cells(obs=obs)
        for coord in task.candidate_pillars:
            cell = (int(coord[0]), int(coord[1]))
            if cell in resolved:
                continue
            mask[self._belief_index(cell)] = True
        return mask

    def _movement_target_anchors(self, *, task: Any, agent_idx: int) -> Sequence[Coord]:
        if int(agent_idx) == 0:
            return [(int(x), int(z)) for x, z in task.anchors_t]
        return [(int(x), int(z)) for x, z in task.anchors_s]

    def _min_manhattan_distance(self, *, origin: Coord, targets: Sequence[Coord]) -> int:
        if not targets:
            return 0
        ox, oz = int(origin[0]), int(origin[1])
        return min(abs(ox - int(tx)) + abs(oz - int(tz)) for tx, tz in targets)

    def _movement_distances(self, *, task: Any, state: Any) -> List[int]:
        distances: List[int] = []
        for agent_idx, pos in enumerate(state.agent_positions):
            targets = self._movement_target_anchors(task=task, agent_idx=agent_idx)
            distances.append(
                self._min_manhattan_distance(
                    origin=(int(pos[0]), int(pos[1])),
                    targets=targets,
                )
            )
        return distances

    def _movement_distance_total(self, *, task: Any, state: Any) -> float:
        return float(sum(self._movement_distances(task=task, state=state)))

    def _adjust_reward_metrics(
        self,
        *,
        metrics: Dict[str, Any],
        task: Any,
        prev_state: Any,
        next_state: Any,
        move_distance_norm: float,
    ) -> Dict[str, Any]:
        reward = float(metrics.get("reward", 0.0))

        base_cc_merge = float(metrics.get("bonus_cc_merge", 0.0))
        adjusted_cc_merge = base_cc_merge * self.cc_merge_bonus_scale
        metrics["bonus_cc_merge"] = adjusted_cc_merge
        reward += adjusted_cc_merge - base_cc_merge

        base_penalty = float(metrics.get("penalty_n_adjacent", 0.0))
        adjusted_penalty = base_penalty * self.n_adjacent_penalty_scale
        metrics["penalty_n_adjacent"] = adjusted_penalty
        reward -= adjusted_penalty - base_penalty

        prev_distances = self._movement_distances(task=task, state=prev_state)
        next_distances = self._movement_distances(task=task, state=next_state)
        prev_total_distance = float(sum(prev_distances))
        next_total_distance = float(sum(next_distances))
        if self.move_progress_bonus_total > 0.0 and move_distance_norm > 0.0:
            move_bonus = self.move_progress_bonus_total * (
                (prev_total_distance - next_total_distance) / move_distance_norm
            )
        else:
            move_bonus = 0.0
        reward += move_bonus

        metrics["bonus_move_progress"] = float(move_bonus)
        metrics["agent_a_target_distance"] = float(next_distances[0]) if next_distances else 0.0
        metrics["agent_b_target_distance"] = float(next_distances[1]) if len(next_distances) > 1 else 0.0
        metrics["target_distance_total"] = float(next_total_distance)
        metrics["reward"] = reward
        return metrics
