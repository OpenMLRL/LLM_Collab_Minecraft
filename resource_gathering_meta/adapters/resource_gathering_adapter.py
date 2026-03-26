from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


_USER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_NNGAMES_ROOT = os.path.join(_USER_ROOT, "NNGames")
if os.path.isdir(_NNGAMES_ROOT) and _NNGAMES_ROOT not in sys.path:
    sys.path.insert(0, _NNGAMES_ROOT)

from NNGames.resource_gathering.envs.nn_env import ResourceGatheringEnv as NNResourceGatheringEnv

from LLM_Collab_Minecraft.resource_gathering.utils.resource_gathering import (
    build_payload,
    compute_visible_cells,
    deserialize_state,
    get_agent_observation,
    make_initial_state,
    payload_to_state,
    payload_to_task,
    render_agent_user_prompt,
    serialize_state,
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


class ResourceGatheringAdapter:
    channel_names: Sequence[str] = tuple(NNResourceGatheringEnv.channel_names)

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
        task_specs = list(task_specs or [])
        self.max_width = max((int(task.width) for task in task_specs), default=1)
        self.max_height = max((int(task.height) for task in task_specs), default=1)
        self.reward_config = dict(reward_config or {})
        self.comm_limit = max(1, int(self.reward_config.get("comm_limit", self.prompt_ctx.get("comm_limit", 1))))

    @property
    def grid_channels(self) -> int:
        return len(self.channel_names)

    @property
    def scalar_dim(self) -> int:
        return 11

    @property
    def task_vocab_size(self) -> int:
        return len(self.task_id_to_index)

    @property
    def belief_dim(self) -> int:
        return int(self.max_width * self.max_height)

    def item_to_payload(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        task = task_from_item(item)
        raw_state = item.get("_resource_state_before_turn")
        if isinstance(raw_state, Mapping):
            state = deserialize_state(raw_state, num_agents=self.num_agents)
        else:
            state = make_initial_state(task, num_agents=self.num_agents, max_turns=task.max_turns)
        return build_payload(
            task=task,
            state_before_turn=state,
            num_agents=self.num_agents,
            view=int(self.prompt_ctx["view"]),
            extraction_range=int(self.prompt_ctx["extraction_range"]),
            max_path_len=int(self.prompt_ctx["max_path_len"]),
            system_prompt=str(self.prompt_ctx["system_prompt"]),
            user_template_single=str(self.prompt_ctx["user_template_single"]),
            user_template_agent1=str(self.prompt_ctx["user_template_agent1"]),
            user_template_agent2=str(self.prompt_ctx["user_template_agent2"]),
            reward_config=self.reward_config,
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
                view=int(payload.get("view") or self.prompt_ctx["view"]),
                extraction_range=int(payload.get("extraction_range") or self.prompt_ctx["extraction_range"]),
                max_path_len=int(payload.get("max_path_len") or self.prompt_ctx["max_path_len"]),
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
        feedbacks = self._build_feedbacks(metrics=metrics)
        prompts = self.render_prompts(payload, feedback_by_agent=feedbacks) if self.original_prompt else list(feedbacks)
        if self.previous_response:
            for idx, prev in enumerate(previous_completions):
                prev_txt = str(prev or "").strip()
                if prev_txt:
                    prompts[idx] = prompts[idx].rstrip() + "\n\nYour previous action JSON:\n" + prev_txt
        return prompts

    def build_meta_observations(self, payload: Mapping[str, Any]) -> List[AgentMetaObservation]:
        task = payload_to_task(payload)
        state = payload_to_state(payload)
        env = self._build_env(task=task, state=state, payload=payload)
        obs_bundle = env._build_observation()
        outputs: List[AgentMetaObservation] = []
        for obs in obs_bundle["agents"]:
            task_id = str(obs["task_id"])
            outputs.append(
                AgentMetaObservation(
                    grid=obs["grid"],
                    scalars=obs["scalars"],
                    belief_target=obs["belief_target"],
                    belief_mask=obs["belief_mask"],
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
        env = self._build_env(task=task, state=state, payload=payload)
        outputs: List[AgentActionCandidates] = []
        for agent_idx in range(self.num_agents):
            obs = get_agent_observation(
                task,
                state,
                agent_idx=agent_idx,
                view=int(payload.get("view") or self.prompt_ctx["view"]),
                extraction_range=int(payload.get("extraction_range") or self.prompt_ctx["extraction_range"]),
            )
            current_pos = tuple(int(v) for v in obs["current_pos"])
            path_options = self._build_path_options(
                env=env,
                task=task,
                state=state,
                agent_idx=agent_idx,
                current_pos=current_pos,
                visible_resource_counts=obs["visible_resource_counts"],
            )
            comm_options, comm_scores = self._build_comm_options(
                env=env,
                task=task,
                state=state,
                agent_idx=agent_idx,
                visible_resource_counts=obs["visible_resource_counts"],
            )
            path_scores = self._build_path_preference_scores(
                env=env,
                task=task,
                state=state,
                agent_idx=agent_idx,
                path_options=path_options,
            )
            outputs.append(
                AgentActionCandidates(
                    comm_options=comm_options,
                    probe_options=[[]],
                    cmd_options=[[]],
                    path_options=path_options,
                    comm_preference_scores=comm_scores,
                    probe_preference_scores=[0.0],
                    cmd_preference_scores=[0.0],
                    path_preference_scores=path_scores,
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
        task = payload_to_task(payload)
        state = payload_to_state(payload)
        print(
            f"[resource_gathering debug] task={task.task_id} family={task.family or 'unknown'} "
            f"turn={turn_idx}/{state.max_turns} reward={float(reward):.3f} "
            f"progress={float(metrics.get('progress_score', 0.0)):.3f} success={bool(metrics.get('success', metrics.get('completed', False)))}",
            flush=True,
        )
        for agent_idx, output in enumerate(agent_outputs):
            print(f"  agent_{agent_idx} output={str(output).strip()}", flush=True)

    def _build_env(
        self,
        *,
        task: Any,
        state: Any,
        payload: Mapping[str, Any],
    ) -> NNResourceGatheringEnv:
        env = NNResourceGatheringEnv(
            tasks=[task],
            num_agents=self.num_agents,
            view=int(payload.get("view") or self.prompt_ctx["view"]),
            max_turns=int(state.max_turns),
            extraction_limit=0,
            extraction_range=int(payload.get("extraction_range") or self.prompt_ctx["extraction_range"]),
            path_slots=max(1, int(payload.get("max_path_len") or self.prompt_ctx["max_path_len"])),
            comm_limit=self.comm_limit,
            progress_reward_scale=float(self.reward_config.get("progress_reward_scale", 10.0)),
            terminal_bonus=float(self.reward_config.get("terminal_bonus", 4.0)),
            move_cost_scale=float(self.reward_config.get("move_cost_scale", 0.0)),
            comm_cost_scale=float(self.reward_config.get("comm_cost_scale", 0.0)),
            wasted_extraction_penalty=0.0,
            move_to_zone_bonus_scale=float(self.reward_config.get("move_to_zone_bonus_scale", 0.05)),
            useful_comm_bonus_scale=float(self.reward_config.get("useful_comm_bonus_scale", 0.1)),
            first_enter_zone_bonus_scale=float(self.reward_config.get("first_enter_zone_bonus_scale", 0.15)),
            debug=False,
        )
        env.current_task_index = 0
        env.current_state = copy.deepcopy(state)
        return env

    def _manhattan(self, left: Coord, right: Coord) -> int:
        return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))

    def _chebyshev(self, left: Coord, right: Coord) -> int:
        return max(abs(int(left[0]) - int(right[0])), abs(int(left[1]) - int(right[1])))

    def _rank_coords(self, *, coords: Sequence[Coord], origin: Coord) -> List[Coord]:
        unique = {(int(coord[0]), int(coord[1])) for coord in coords}
        return list(
            sorted(
                unique,
                key=lambda coord: (
                    self._chebyshev(origin, coord),
                    self._manhattan(origin, coord),
                    int(coord[1]),
                    int(coord[0]),
                ),
            )
        )

    def _spread_coords(self, *, coords: Sequence[Coord], min_gap: int, limit: int) -> List[Coord]:
        chosen: List[Coord] = []
        for coord in coords:
            item = (int(coord[0]), int(coord[1]))
            if any(self._manhattan(item, existing) < int(min_gap) for existing in chosen):
                continue
            chosen.append(item)
            if len(chosen) >= int(limit):
                break
        return chosen

    def _path_prefix(self, path: Sequence[Sequence[int]], *, keep_steps: int) -> List[List[int]]:
        if not path:
            return []
        keep_len = min(len(path), max(1, int(keep_steps) + 1))
        return [[int(point[0]), int(point[1])] for point in path[:keep_len]]

    def _resource_coord_groups(
        self,
        *,
        env: NNResourceGatheringEnv,
        task: Any,
        state: Any,
        agent_idx: int,
        visible_resource_counts: Mapping[Coord, Any],
    ) -> Tuple[List[Coord], List[Coord]]:
        target_names = set(env._agent_target_resources(task=task, collected=state.collected, agent_idx=agent_idx))
        target_coords: List[Coord] = []
        distractor_coords: List[Coord] = []
        for coord, counts in visible_resource_counts.items():
            positive = [
                resource_name
                for resource_name in ("wood", "stone", "iron")
                if env._resource_count(counts, resource_name) > 0
            ]
            if not positive:
                continue
            item = (int(coord[0]), int(coord[1]))
            if target_names and any(name in target_names for name in positive):
                target_coords.append(item)
            else:
                distractor_coords.append(item)
        return target_coords, distractor_coords

    def _build_comm_options(
        self,
        *,
        env: NNResourceGatheringEnv,
        task: Any,
        state: Any,
        agent_idx: int,
        visible_resource_counts: Mapping[Coord, Any],
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        teammate_idx = 1 - int(agent_idx)
        teammate_targets = set(env._agent_target_resources(task=task, collected=state.collected, agent_idx=teammate_idx))
        self_targets = set(env._agent_target_resources(task=task, collected=state.collected, agent_idx=agent_idx))
        current_pos = tuple(int(v) for v in state.agent_positions[agent_idx])
        useful_facts: List[Tuple[float, Dict[str, Any]]] = []
        distractor_facts: List[Tuple[float, Dict[str, Any]]] = []
        for coord, counts in visible_resource_counts.items():
            item = (int(coord[0]), int(coord[1]))
            for resource_name in ("wood", "stone", "iron"):
                count = env._resource_count(counts, resource_name)
                if count <= 0:
                    continue
                fact = {
                    "x": int(coord[0]),
                    "z": int(coord[1]),
                    "type": str(resource_name),
                    "count": int(count),
                }
                base_score = float(min(4, count)) - 0.1 * float(self._manhattan(current_pos, item))
                if resource_name in teammate_targets:
                    useful_facts.append((base_score + 1.5, fact))
                elif resource_name in self_targets:
                    distractor_facts.append((base_score + 0.4, fact))
                else:
                    distractor_facts.append((base_score, fact))

        useful_facts.sort(key=lambda item: (-item[0], item[1]["z"], item[1]["x"], item[1]["type"]))
        distractor_facts.sort(key=lambda item: (-item[0], item[1]["z"], item[1]["x"], item[1]["type"]))
        options: List[Dict[str, Any]] = [{}]
        scores: List[float] = [0.0]

        for score, fact in useful_facts[:2]:
            options.append({"resource_facts": [fact]})
            scores.append(float(max(0.1, score + 0.35)))
        if len(useful_facts) >= 2:
            options.append({"resource_facts": [useful_facts[0][1], useful_facts[1][1]]})
            scores.append(float(max(0.2, useful_facts[0][0] + useful_facts[1][0] + 0.5)))
        if len(useful_facts) >= 3:
            options.append({"resource_facts": [useful_facts[-1][1]]})
            scores.append(float(max(0.08, 0.2 + 0.15 * max(0.0, useful_facts[-1][0]))))
            options.append({"resource_facts": [useful_facts[0][1], useful_facts[-1][1]]})
            scores.append(float(max(0.12, useful_facts[0][0] + 0.1 * max(0.0, useful_facts[-1][0]))))

        for score, fact in distractor_facts[:2]:
            options.append({"resource_facts": [fact]})
            scores.append(float(max(0.05, 0.25 + 0.2 * max(0.0, score))))
        if len(distractor_facts) >= 2:
            options.append({"resource_facts": [distractor_facts[0][1], distractor_facts[1][1]]})
            scores.append(
                float(
                    max(
                        0.1,
                        0.35 + 0.15 * max(0.0, distractor_facts[0][0]) + 0.15 * max(0.0, distractor_facts[1][0]),
                    )
                )
            )

        if useful_facts and distractor_facts:
            options.append({"resource_facts": [useful_facts[0][1], distractor_facts[0][1]]})
            scores.append(float(max(0.15, useful_facts[0][0] + 0.15 * max(0.0, distractor_facts[0][0]))))
        return self._dedupe_options(options, scores)

    def _build_path_options(
        self,
        *,
        env: NNResourceGatheringEnv,
        task: Any,
        state: Any,
        agent_idx: int,
        current_pos: Coord,
        visible_resource_counts: Mapping[Coord, Any],
    ) -> List[List[List[int]]]:
        options: List[List[List[int]]] = [[[int(current_pos[0]), int(current_pos[1])]]]
        max_steps = max(1, int(self.prompt_ctx["max_path_len"]))
        zone = env._work_zone(task=task, state=state, agent_idx=agent_idx)
        ranked_zone = self._rank_coords(coords=list(zone), origin=current_pos)
        if ranked_zone:
            nearest_path = self._greedy_path(start=current_pos, target=ranked_zone[0], task=task, max_steps=max_steps)
            options.append(nearest_path)
            if len(nearest_path) > 2:
                midpoint_steps = max(1, (len(nearest_path) - 1) // 2)
                options.append(self._path_prefix(nearest_path, keep_steps=midpoint_steps))
            for coord in self._spread_coords(coords=ranked_zone[1:], min_gap=2, limit=2):
                options.append(self._greedy_path(start=current_pos, target=coord, task=task, max_steps=max_steps))

        target_coords, distractor_coords = self._resource_coord_groups(
            env=env,
            task=task,
            state=state,
            agent_idx=agent_idx,
            visible_resource_counts=visible_resource_counts,
        )
        for coord in self._rank_coords(coords=target_coords, origin=current_pos)[:2]:
            options.append(self._greedy_path(start=current_pos, target=coord, task=task, max_steps=max_steps))

        message_maps = env._message_maps(state.inbox[agent_idx])
        target_names = env._agent_target_resources(task=task, collected=state.collected, agent_idx=agent_idx)
        hinted_coords: List[Coord] = []
        for name in target_names:
            for coord in message_maps.get(name, {}).keys():
                hinted_coords.append((int(coord[0]), int(coord[1])))
        for coord in self._spread_coords(coords=self._rank_coords(coords=hinted_coords, origin=current_pos), min_gap=2, limit=2):
            options.append(self._greedy_path(start=current_pos, target=coord, task=task, max_steps=max_steps))

        for coord in self._rank_coords(coords=distractor_coords, origin=current_pos)[:2]:
            options.append(self._greedy_path(start=current_pos, target=coord, task=task, max_steps=max_steps))

        center = (int(task.width // 2), int(task.height // 2))
        options.append(self._greedy_path(start=current_pos, target=center, task=task, max_steps=max_steps))
        return self._dedupe_path_options(options)

    def _build_path_preference_scores(
        self,
        *,
        env: NNResourceGatheringEnv,
        task: Any,
        state: Any,
        agent_idx: int,
        path_options: Sequence[Sequence[Sequence[int]]],
    ) -> List[float]:
        zone = env._work_zone(task=task, state=state, agent_idx=agent_idx)
        scores: List[float] = []
        for option in path_options:
            if not option:
                scores.append(0.0)
                continue
            end = tuple(int(v) for v in option[-1][:2])
            dist = env._distance_to_zone(end, zone)
            scores.append(0.0 if dist is None else float(max(0, 6 - dist)))
        return scores

    def _greedy_path(self, *, start: Coord, target: Coord, task: Any, max_steps: int) -> List[List[int]]:
        cur = (int(start[0]), int(start[1]))
        goal = (int(target[0]), int(target[1]))
        path: List[List[int]] = [[int(cur[0]), int(cur[1])]]
        for _ in range(max(0, int(max_steps))):
            if cur == goal:
                break
            step_x = 0 if goal[0] == cur[0] else (1 if goal[0] > cur[0] else -1)
            step_z = 0 if goal[1] == cur[1] else (1 if goal[1] > cur[1] else -1)
            nxt = (int(cur[0] + step_x), int(cur[1] + step_z))
            if not (0 <= nxt[0] < int(task.width) and 0 <= nxt[1] < int(task.height)):
                break
            cur = nxt
            path.append([int(cur[0]), int(cur[1])])
        return path

    def _dedupe_options(self, options: Sequence[Dict[str, Any]], scores: Sequence[float]) -> Tuple[List[Dict[str, Any]], List[float]]:
        out_options: List[Dict[str, Any]] = []
        out_scores: List[float] = []
        seen: set[str] = set()
        for option, score in zip(options, scores):
            key = json.dumps(option, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            out_options.append(copy.deepcopy(option))
            out_scores.append(float(score))
        return out_options, out_scores

    def _dedupe_path_options(self, options: Sequence[Sequence[Sequence[int]]]) -> List[List[List[int]]]:
        out: List[List[List[int]]] = []
        seen: set[str] = set()
        for option in options:
            key = json.dumps(option, ensure_ascii=False, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            out.append(copy.deepcopy(list(option)))
        return out

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

    def _build_feedbacks(self, *, metrics: Mapping[str, Any]) -> List[str]:
        text = (
            f"Last reward={float(metrics.get('reward', 0.0)):.3f}; "
            f"progress={float(metrics.get('progress_score', 0.0)):.3f}; "
            f"totals wood={int(metrics.get('collected_wood', 0))}, "
            f"stone={int(metrics.get('collected_stone', 0))}, "
            f"iron={int(metrics.get('collected_iron', 0))}."
        )
        return [text for _ in range(self.num_agents)]
