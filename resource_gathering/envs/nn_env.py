from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch


Coord = Tuple[int, int]
ResourceTriple = Tuple[int, int, int]
_PATH_DELTAS: Sequence[Coord] = (
    (0, 0),  # stop
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
)
_DELTA_TO_PATH_INDEX: Mapping[Coord, int] = {delta: idx for idx, delta in enumerate(_PATH_DELTAS)}
_RESOURCE_ORDER: Sequence[str] = ("wood", "stone", "iron")
_RESOURCE_BITS: Mapping[str, int] = {
    "wood": 1,
    "stone": 2,
    "iron": 4,
}
_COMM_TYPE_DIM = 4  # noop + wood + stone + iron
_COMM_TYPE_TO_RESOURCE: Mapping[int, str] = {
    1: "wood",
    2: "stone",
    3: "iron",
}
_TUPLE_TOKEN_RE = re.compile(
    r"^\(?\s*(?P<wood>\d+)\s*[,:\|]\s*(?P<stone>\d+)\s*[,:\|]\s*(?P<iron>\d+)\s*\)?$"
)


@dataclass(frozen=True)
class ResourceTaskSpec:
    task_id: str
    width: int
    height: int
    rows_topdown: Tuple[str, ...]
    resources: Tuple[Tuple[ResourceTriple, ...], ...]
    agent_starts: Tuple[Coord, ...]
    goal_wood: int
    goal_stone: int
    goal_iron: int
    max_turns: int
    family: str = ""


@dataclass
class ResourceGatheringState:
    turn_index: int
    max_turns: int
    agent_positions: List[Coord]
    vision_origins: List[Set[Coord]]
    inbox: List[List[Any]]
    resources: Dict[Coord, ResourceTriple]
    collected: Dict[str, int]
    entered_work_zones: List[bool]
    completed: bool
    terminated: bool


@dataclass
class DecodedAgentAction:
    message_obj: Dict[str, Any]
    comm_items: int
    auto_harvest: List[Coord]
    path: List[Coord]
    path_valid: bool


@dataclass(frozen=True)
class ResourceGatheringActionSpec:
    head_dims: Dict[str, int]
    extraction_slots: int
    path_slots: int
    comm_cells: int
    noop_index: int
    stop_index: int


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_coords(coords: Iterable[Coord]) -> List[Coord]:
    return sorted({(int(x), int(z)) for x, z in coords}, key=lambda item: (item[1], item[0]))


def _in_bounds(coord: Coord, *, width: int, height: int) -> bool:
    x, z = int(coord[0]), int(coord[1])
    return 0 <= x < int(width) and 0 <= z < int(height)


def _parse_coord(value: Any) -> Coord | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, Mapping):
        if "x" in value and "z" in value:
            return (int(value["x"]), int(value["z"]))
    return None


def _parse_row_tokens(row: Any) -> List[str]:
    if isinstance(row, str):
        return [token.strip() for token in row.strip().split() if token.strip()]
    if isinstance(row, (list, tuple)):
        return [str(token).strip() for token in row if str(token).strip()]
    raise ValueError(f"Unsupported row type: {type(row)}")


def _parse_rows(rows_obj: Any) -> List[List[str]]:
    if isinstance(rows_obj, str):
        raw_rows = [line.strip() for line in rows_obj.splitlines() if line.strip()]
        return [_parse_row_tokens(row) for row in raw_rows]
    if isinstance(rows_obj, list):
        return [_parse_row_tokens(row) for row in rows_obj]
    raise ValueError("task rows must be a string or list")


def _parse_resource_token(token: str) -> ResourceTriple:
    text = str(token).strip()
    if text in (".", "_", "0"):
        return (0, 0, 0)
    match = _TUPLE_TOKEN_RE.match(text)
    if not match:
        raise ValueError(f"Invalid resource token: {token}")
    return (
        int(match.group("wood")),
        int(match.group("stone")),
        int(match.group("iron")),
    )


def _rows_to_task(
    *,
    task_id: str,
    rows: Sequence[Sequence[str]],
    goal: Mapping[str, Any] | None,
    family: str,
    agent_starts_override: Sequence[Any] | None,
) -> ResourceTaskSpec:
    if not rows:
        raise ValueError(f"{task_id}: map rows are empty")
    width = len(rows[0])
    if width <= 0:
        raise ValueError(f"{task_id}: map row width must be > 0")
    for row_idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"{task_id}: inconsistent row width at row={row_idx}")

    height = len(rows)
    parsed_rows: List[str] = []
    resources: List[List[ResourceTriple]] = []
    starts: Dict[str, Coord] = {}

    for z, row in enumerate(rows):
        resource_row: List[ResourceTriple] = []
        normalized_tokens: List[str] = []
        for x, token in enumerate(row):
            text = str(token).strip()
            if text in ("A", "B", "AB", "BA"):
                if "A" in text:
                    starts["A"] = (int(x), int(z))
                if "B" in text:
                    starts["B"] = (int(x), int(z))
                resource_row.append((0, 0, 0))
                normalized_tokens.append(text)
            else:
                triple = _parse_resource_token(text)
                resource_row.append(triple)
                normalized_tokens.append(f"({triple[0]},{triple[1]},{triple[2]})")
        resources.append(resource_row)
        parsed_rows.append(" ".join(normalized_tokens))

    if agent_starts_override:
        parsed_override = [_parse_coord(item) for item in agent_starts_override]
        if any(coord is None for coord in parsed_override):
            raise ValueError(f"{task_id}: invalid agent_starts override")
        starts = {
            "A": tuple(parsed_override[0]),  # type: ignore[arg-type]
            "B": tuple(parsed_override[1]),  # type: ignore[arg-type]
        }

    if "A" not in starts or "B" not in starts:
        raise ValueError(f"{task_id}: rows must define start markers A and B")

    goal_map = dict(goal or {})
    goal_wood = max(1, int(goal_map.get("wood", 1)))
    goal_stone = max(1, int(goal_map.get("stone", 1)))
    goal_iron = max(1, int(goal_map.get("iron", 1)))
    resolved_max_turns = max(1, int(2 * max(width, height)))

    return ResourceTaskSpec(
        task_id=task_id,
        width=int(width),
        height=int(height),
        rows_topdown=tuple(parsed_rows),
        resources=tuple(tuple(tuple(cell) for cell in row) for row in resources),
        agent_starts=(starts["A"], starts["B"]),
        goal_wood=goal_wood,
        goal_stone=goal_stone,
        goal_iron=goal_iron,
        max_turns=resolved_max_turns,
        family=str(family or ""),
    )


def load_tasks_from_json(json_path: str) -> List[ResourceTaskSpec]:
    path = Path(json_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset.json_path not found: {path}")

    raw = _load_json(path)
    if isinstance(raw, list):
        task_objs = raw
    elif isinstance(raw, dict):
        task_objs = [raw]
    else:
        raise ValueError(f"dataset.json_path must contain object or list, got {type(raw)}")

    tasks: List[ResourceTaskSpec] = []
    for idx, task_obj in enumerate(task_objs, start=1):
        if not isinstance(task_obj, Mapping):
            raise ValueError(f"task entry #{idx} must be an object")
        task_id = str(task_obj.get("task_id") or f"resource_gathering_{idx:04d}")
        rows_obj = task_obj.get("map") if "map" in task_obj else task_obj.get("rows")
        rows = _parse_rows(rows_obj)
        tasks.append(
            _rows_to_task(
                task_id=task_id,
                rows=rows,
                goal=(task_obj.get("goal") if isinstance(task_obj.get("goal"), Mapping) else None),
                family=str(task_obj.get("family") or ""),
                agent_starts_override=(
                    task_obj.get("agent_starts") if isinstance(task_obj.get("agent_starts"), Sequence) else None
                ),
            )
        )
    return tasks


def compute_visible_cells(
    origins: Iterable[Coord], *, view: int, width: int, height: int
) -> Set[Coord]:
    radius = max(0, int(view))
    out: Set[Coord] = set()
    for ox, oz in origins:
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                coord = (int(ox + dx), int(oz + dz))
                if _in_bounds(coord, width=width, height=height):
                    out.add(coord)
    return out


def _clone_state(state: ResourceGatheringState) -> ResourceGatheringState:
    return ResourceGatheringState(
        turn_index=int(state.turn_index),
        max_turns=int(state.max_turns),
        agent_positions=[(int(x), int(z)) for x, z in state.agent_positions],
        vision_origins=[set((int(x), int(z)) for x, z in origins) for origins in state.vision_origins],
        inbox=[list(messages) for messages in state.inbox],
        resources={
            (int(x), int(z)): (int(counts[0]), int(counts[1]), int(counts[2]))
            for (x, z), counts in state.resources.items()
        },
        collected={str(name): int(value) for name, value in state.collected.items()},
        entered_work_zones=[bool(value) for value in state.entered_work_zones],
        completed=bool(state.completed),
        terminated=bool(state.terminated),
    )


class ResourceGatheringEnv:
    """NN-friendly resource gathering task with partial observability and communication."""

    channel_names: Sequence[str] = (
        "visibility",
        "visible_wood",
        "visible_stone",
        "visible_iron",
        "msg_wood",
        "msg_stone",
        "msg_iron",
        "self_pos",
        "teammate_pos",
        "extraction_reach",
    )

    def __init__(
        self,
        *,
        tasks: Sequence[ResourceTaskSpec],
        num_agents: int = 2,
        view: int = 2,
        max_turns: Optional[int] = None,
        extraction_limit: int = 3,
        extraction_range: int = 2,
        path_slots: Optional[int] = None,
        comm_limit: int = 3,
        progress_reward_scale: float = 6.0,
        terminal_bonus: float = 2.0,
        move_cost_scale: float = 0.01,
        comm_cost_scale: float = 0.01,
        wasted_extraction_penalty: float = 0.05,
        move_to_zone_bonus_scale: float = 0.2,
        useful_comm_bonus_scale: float = 0.5,
        first_enter_zone_bonus_scale: float = 0.75,
        debug: bool = False,
    ) -> None:
        self.tasks = list(tasks)
        if not self.tasks:
            raise ValueError("tasks must be non-empty")
        self.num_agents = int(num_agents)
        if self.num_agents != 2:
            raise ValueError("resource_gathering currently expects exactly 2 agents")
        self.view = int(view)
        self.max_turns = int(max_turns) if max_turns is not None else None
        self.extraction_limit = max(0, int(extraction_limit))
        self.extraction_range = max(0, int(extraction_range))
        self.comm_slots = max(1, int(comm_limit))
        self.debug = bool(debug)
        self.progress_reward_scale = float(progress_reward_scale)
        self.terminal_bonus = float(terminal_bonus)
        self.move_cost_scale = float(move_cost_scale)
        self.comm_cost_scale = float(comm_cost_scale)
        self.wasted_extraction_penalty = float(wasted_extraction_penalty)
        self.move_to_zone_bonus_scale = float(move_to_zone_bonus_scale)
        self.useful_comm_bonus_scale = float(useful_comm_bonus_scale)
        self.first_enter_zone_bonus_scale = float(first_enter_zone_bonus_scale)

        width = int(self.tasks[0].width)
        height = int(self.tasks[0].height)
        for task in self.tasks:
            if int(task.width) != width or int(task.height) != height:
                raise ValueError("All tasks must share the same width and height.")
        self.width = width
        self.height = height
        self.num_cells = int(self.width * self.height)
        self.noop_index = int(self.num_cells)
        self.path_slots = int(path_slots) if path_slots is not None else max(self.width, self.height)
        self.max_probe = 0
        self.max_commands_total = int(self.comm_slots)

        self.task_id_to_index = {str(task.task_id): idx for idx, task in enumerate(self.tasks)}
        self.max_cell_resource = max(
            1,
            max(
                max(int(cell[0]), int(cell[1]), int(cell[2]))
                for task in self.tasks
                for row in task.resources
                for cell in row
            ),
        )
        self.max_goal_value = max(
            1,
            max(
                int(max(task.goal_wood, task.goal_stone, task.goal_iron))
                for task in self.tasks
            ),
        )

        head_dims: Dict[str, int] = {}
        for slot in range(self.path_slots):
            head_dims[f"path_{slot}"] = len(_PATH_DELTAS)
        for slot in range(self.comm_slots):
            head_dims[f"comm_coord_{slot}"] = self.num_cells + 1
            head_dims[f"comm_type_{slot}"] = _COMM_TYPE_DIM
        self.action_spec = ResourceGatheringActionSpec(
            head_dims=head_dims,
            extraction_slots=0,
            path_slots=self.path_slots,
            comm_cells=self.num_cells,
            noop_index=self.noop_index,
            stop_index=0,
        )

        self.current_task_index: Optional[int] = None
        self.current_state: Optional[ResourceGatheringState] = None

    @property
    def grid_channels(self) -> int:
        return len(self.channel_names)

    @property
    def scalar_dim(self) -> int:
        return 11

    def reset(self, task_index: Optional[int] = None) -> Dict[str, Any]:
        if task_index is None:
            task_index = random.randrange(len(self.tasks))
        self.current_task_index = int(task_index)
        task = self.tasks[self.current_task_index]
        resolved_turns = int(self.max_turns) if self.max_turns is not None else int(task.max_turns)
        resources: Dict[Coord, ResourceTriple] = {}
        for z, row in enumerate(task.resources):
            for x, counts in enumerate(row):
                triple = (int(counts[0]), int(counts[1]), int(counts[2]))
                if sum(triple) > 0:
                    resources[(int(x), int(z))] = triple
        self.current_state = ResourceGatheringState(
            turn_index=1,
            max_turns=resolved_turns,
            agent_positions=[(int(x), int(z)) for x, z in task.agent_starts[: self.num_agents]],
            vision_origins=[{(int(x), int(z))} for x, z in task.agent_starts[: self.num_agents]],
            inbox=[[] for _ in range(self.num_agents)],
            resources=resources,
            collected={"wood": 0, "stone": 0, "iron": 0},
            entered_work_zones=[],
            completed=False,
            terminated=False,
        )
        self.current_state.entered_work_zones = [
            self._is_in_work_zone(
                pos=self.current_state.agent_positions[agent_idx],
                zone=self._work_zone(task=task, state=self.current_state, agent_idx=agent_idx),
            )
            for agent_idx in range(self.num_agents)
        ]
        return self._build_observation()

    def step(self, actions: Sequence[Mapping[str, int]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.current_state is None or self.current_task_index is None:
            raise RuntimeError("Call reset() before step().")

        task = self.tasks[self.current_task_index]
        state = self.current_state
        if state.terminated:
            observation = self._build_observation()
            metrics = self._build_terminal_metrics(task=task, state=state)
            return observation, 0.0, True, {"metrics": metrics, "decoded_actions": []}

        prev_progress = self._progress_score(task=task, collected=state.collected)
        decoded_actions: List[DecodedAgentAction] = []
        for agent_idx in range(self.num_agents):
            action = dict(actions[agent_idx]) if agent_idx < len(actions) else {}
            decoded_actions.append(
                self._decode_action(
                    task=task,
                    state=state,
                    agent_idx=agent_idx,
                    action=action,
                )
            )

        next_state = _clone_state(state)
        work_zones = [self._work_zone(task=task, state=state, agent_idx=agent_idx) for agent_idx in range(self.num_agents)]
        prev_zone_distances = [
            self._distance_to_zone(state.agent_positions[agent_idx], work_zones[agent_idx]) for agent_idx in range(self.num_agents)
        ]
        prev_in_zone = [
            self._is_in_work_zone(pos=state.agent_positions[agent_idx], zone=work_zones[agent_idx])
            for agent_idx in range(self.num_agents)
        ]
        move_steps = 0
        for agent_idx, decoded in enumerate(decoded_actions):
            cur = next_state.agent_positions[agent_idx]
            next_state.vision_origins[agent_idx].add(cur)
            if decoded.path_valid:
                for coord in decoded.path:
                    next_state.vision_origins[agent_idx].add(coord)
                next_state.agent_positions[agent_idx] = decoded.path[-1]
                move_steps += max(0, len(decoded.path) - 1)

        next_zone_distances = [
            self._distance_to_zone(next_state.agent_positions[agent_idx], work_zones[agent_idx])
            for agent_idx in range(self.num_agents)
        ]
        move_toward_zone_steps = 0
        first_enter_zone_count = 0
        for agent_idx in range(self.num_agents):
            prev_distance = prev_zone_distances[agent_idx]
            next_distance = next_zone_distances[agent_idx]
            if prev_distance is not None and next_distance is not None and next_distance < prev_distance:
                move_toward_zone_steps += int(prev_distance - next_distance)
            next_in_zone = self._is_in_work_zone(pos=next_state.agent_positions[agent_idx], zone=work_zones[agent_idx])
            if not state.entered_work_zones[agent_idx] and not prev_in_zone[agent_idx] and next_in_zone:
                first_enter_zone_count += 1
            next_state.entered_work_zones[agent_idx] = bool(state.entered_work_zones[agent_idx] or next_in_zone)

        delta_wood = 0
        delta_stone = 0
        delta_iron = 0
        auto_harvest_cells = 0
        productive_harvesters = 0

        for agent_idx, decoded in enumerate(decoded_actions):
            current_pos = next_state.agent_positions[agent_idx]
            yielded, harvested_coords = self._auto_harvest(
                agent_idx=agent_idx,
                current_pos=current_pos,
                resources=next_state.resources,
            )
            decoded.auto_harvest = harvested_coords
            if harvested_coords:
                productive_harvesters += 1
            auto_harvest_cells += len(harvested_coords)
            delta_wood += int(yielded["wood"])
            delta_stone += int(yielded["stone"])
            delta_iron += int(yielded["iron"])
            next_state.collected["wood"] += int(yielded["wood"])
            next_state.collected["stone"] += int(yielded["stone"])
            next_state.collected["iron"] += int(yielded["iron"])

        for src_idx, decoded in enumerate(decoded_actions):
            if not decoded.message_obj:
                continue
            for dst_idx in range(self.num_agents):
                if dst_idx == src_idx:
                    continue
                next_state.inbox[dst_idx].append(copy.deepcopy(decoded.message_obj))

        next_progress = self._progress_score(task=task, collected=next_state.collected)
        completed = bool(self._is_completed(task=task, collected=next_state.collected))
        progress_bonus = self.progress_reward_scale * max(0.0, next_progress - prev_progress)
        terminal_bonus = self.terminal_bonus if completed and not state.completed else 0.0
        move_to_zone_bonus = self.move_to_zone_bonus_scale * float(move_toward_zone_steps)
        useful_comm_items, useful_comm_bonus = self._useful_comm_bonus(
            task=task,
            state=state,
            decoded_actions=decoded_actions,
        )
        first_enter_zone_bonus = self.first_enter_zone_bonus_scale * float(first_enter_zone_count)
        comm_items = sum(int(decoded.comm_items) for decoded in decoded_actions)
        move_penalty = self.move_cost_scale * float(move_steps)
        comm_penalty = self.comm_cost_scale * float(comm_items)
        wasted_penalty = 0.0
        reward = (
            progress_bonus
            + terminal_bonus
            + move_to_zone_bonus
            + useful_comm_bonus
            + first_enter_zone_bonus
            - move_penalty
            - comm_penalty
            - wasted_penalty
        )

        next_state.turn_index = int(state.turn_index) + 1
        next_state.completed = completed
        next_state.terminated = bool(completed or (next_state.turn_index > next_state.max_turns))
        self.current_state = next_state

        metrics = {
            "reward": float(reward),
            "bonus_progress": float(progress_bonus),
            "bonus_terminal_complete": float(terminal_bonus),
            "bonus_move_to_zone": float(move_to_zone_bonus),
            "bonus_useful_comm": float(useful_comm_bonus),
            "bonus_first_enter_zone": float(first_enter_zone_bonus),
            "penalty_move": float(move_penalty),
            "penalty_comm": float(comm_penalty),
            "penalty_wasted_extraction": float(wasted_penalty),
            "progress_score": float(next_progress),
            "goal_wood": float(task.goal_wood),
            "goal_stone": float(task.goal_stone),
            "goal_iron": float(task.goal_iron),
            "collected_wood": float(next_state.collected["wood"]),
            "collected_stone": float(next_state.collected["stone"]),
            "collected_iron": float(next_state.collected["iron"]),
            "delta_wood": float(delta_wood),
            "delta_stone": float(delta_stone),
            "delta_iron": float(delta_iron),
            "num_comm_items": float(comm_items),
            "useful_comm_items": float(useful_comm_items),
            "num_valid_extractions": 0.0,
            "wasted_extractions": 0.0,
            "auto_harvest_cells": float(auto_harvest_cells),
            "productive_harvesters": float(productive_harvesters),
            "move_steps": float(move_steps),
            "move_toward_zone_steps": float(move_toward_zone_steps),
            "first_enter_zone_count": float(first_enter_zone_count),
            "completed": bool(completed),
            "connected": bool(completed),
            "terminated": bool(next_state.terminated),
            "turn_index": int(state.turn_index),
        }
        observation = self._build_observation()
        done = bool(next_state.terminated)
        if self.debug:
            self._print_debug_turn(
                task=task,
                prev_state=state,
                next_state=next_state,
                decoded_actions=decoded_actions,
                reward=float(reward),
                progress=float(next_progress),
                metrics=metrics,
            )
        return observation, float(reward), done, {"metrics": metrics, "decoded_actions": decoded_actions}

    def expert_actions(self) -> List[Dict[str, int]]:
        if self.current_state is None or self.current_task_index is None:
            raise RuntimeError("Call reset() before requesting expert actions.")
        task = self.tasks[self.current_task_index]
        state = self.current_state
        actions: List[Dict[str, int]] = []
        for agent_idx in range(self.num_agents):
            action: Dict[str, int] = {}
            current_pos = state.agent_positions[agent_idx]
            visible = compute_visible_cells(
                state.vision_origins[agent_idx],
                view=self.view,
                width=task.width,
                height=task.height,
            )
            visible_counts = self._visible_resource_counts(state=state, visible=visible)

            teammate_targets = self._agent_target_resources(task=task, collected=state.collected, agent_idx=1 - agent_idx)
            comm_facts: List[Tuple[Coord, str, int]] = []
            for coord, counts in visible_counts.items():
                for resource_name in teammate_targets:
                    count = self._resource_count(counts, resource_name)
                    if count > 0:
                        comm_facts.append((coord, resource_name, count))
            comm_facts.sort(key=lambda item: (-int(item[2]), self._chebyshev(current_pos, item[0]), item[0][1], item[0][0]))
            for slot, (coord, resource_name, _count) in enumerate(comm_facts[: self.comm_slots]):
                action[f"comm_coord_{slot}"] = self._coord_to_index(coord)
                action[f"comm_type_{slot}"] = self._resource_name_to_comm_type(resource_name)

            zone = self._work_zone(task=task, state=state, agent_idx=agent_idx)
            target_pos = self._nearest_zone_pos(origin=current_pos, zone=zone)
            if target_pos is not None:
                self._encode_path_toward(action=action, start=current_pos, target=target_pos)

            actions.append(action)
        return actions

    def resolve_action_mask(
        self,
        *,
        agent_obs: Mapping[str, Any],
        head_name: str,
        partial_action: Mapping[str, int],
    ) -> torch.Tensor:
        mask = agent_obs["action_masks"][head_name].clone()
        if head_name.startswith("path_"):
            return self._resolve_path_mask(
                mask=mask,
                head_name=head_name,
                current_pos=tuple(int(v) for v in agent_obs["current_pos"]),
                partial_action=partial_action,
            )
        if head_name.startswith("comm_coord_"):
            return self._resolve_comm_coord_mask(mask=mask, head_name=head_name, partial_action=partial_action)
        if head_name.startswith("comm_type_"):
            return self._resolve_comm_type_mask(
                mask=mask,
                head_name=head_name,
                agent_obs=agent_obs,
                partial_action=partial_action,
            )
        return mask

    def _build_observation(self) -> Dict[str, Any]:
        if self.current_state is None or self.current_task_index is None:
            raise RuntimeError("Call reset() before requesting observation.")
        task = self.tasks[self.current_task_index]
        state = self.current_state
        agents: List[Dict[str, Any]] = []
        for agent_idx in range(self.num_agents):
            visible = compute_visible_cells(
                state.vision_origins[agent_idx],
                view=self.view,
                width=task.width,
                height=task.height,
            )
            visible_counts = self._visible_resource_counts(state=state, visible=visible)
            message_maps = self._message_maps(state.inbox[agent_idx])
            grid = self._build_grid(
                state=state,
                agent_idx=agent_idx,
                visible=visible,
                visible_counts=visible_counts,
                message_maps=message_maps,
            )
            scalars = self._build_scalars(task=task, state=state, agent_idx=agent_idx)
            belief_target, belief_mask = self._build_belief_supervision(task=task, agent_idx=agent_idx)
            agents.append(
                {
                    "grid": grid,
                    "scalars": scalars,
                    "agent_index": int(agent_idx),
                    "task_index": int(self.task_id_to_index[task.task_id]),
                    "task_id": str(task.task_id),
                    "turn_index": int(state.turn_index),
                    "belief_target": belief_target,
                    "belief_mask": belief_mask,
                    "current_pos": [
                        int(state.agent_positions[agent_idx][0]),
                        int(state.agent_positions[agent_idx][1]),
                    ],
                    "action_masks": self._action_masks(
                        task=task,
                        state=state,
                        agent_idx=agent_idx,
                        visible=visible,
                        visible_counts=visible_counts,
                    ),
                }
            )
        return {
            "agents": agents,
            "task_index": int(self.task_id_to_index[task.task_id]),
            "task_id": str(task.task_id),
            "turn_index": int(state.turn_index),
        }

    def _action_masks(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        agent_idx: int,
        visible: Set[Coord],
        visible_counts: Mapping[Coord, ResourceTriple],
    ) -> Dict[str, torch.Tensor]:
        del task
        masks: Dict[str, torch.Tensor] = {}

        path_mask = torch.ones(len(_PATH_DELTAS), dtype=torch.bool)
        for slot in range(self.path_slots):
            masks[f"path_{slot}"] = path_mask.clone()

        visible_resource_indices = [
            self._coord_to_index(coord)
            for coord, counts in visible_counts.items()
            if sum(int(value) for value in counts) > 0
        ]
        comm_coord_mask = torch.zeros(self.num_cells + 1, dtype=torch.bool)
        comm_coord_mask[self.noop_index] = True
        for index in visible_resource_indices:
            comm_coord_mask[index] = True
        for slot in range(self.comm_slots):
            masks[f"comm_coord_{slot}"] = comm_coord_mask.clone()
        for slot in range(self.comm_slots):
            mask = torch.zeros(_COMM_TYPE_DIM, dtype=torch.bool)
            mask[0] = True
            masks[f"comm_type_{slot}"] = mask
        return masks

    def _decode_action(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        agent_idx: int,
        action: Mapping[str, int],
    ) -> DecodedAgentAction:
        current = state.agent_positions[agent_idx]
        visible = compute_visible_cells(
            state.vision_origins[agent_idx],
            view=self.view,
            width=task.width,
            height=task.height,
        )
        visible_counts = self._visible_resource_counts(state=state, visible=visible)

        path = self._decode_path(current_pos=current, action=action)

        message_obj, comm_items = self._build_message(action=action, visible_counts=visible_counts)
        return DecodedAgentAction(
            message_obj=message_obj,
            comm_items=int(comm_items),
            auto_harvest=[],
            path=path,
            path_valid=True,
        )

    def _build_message(
        self,
        *,
        action: Mapping[str, int],
        visible_counts: Mapping[Coord, ResourceTriple],
    ) -> Tuple[Dict[str, Any], int]:
        resource_facts: List[Dict[str, Any]] = []
        seen_fact_keys: Set[Tuple[Coord, str]] = set()
        for slot in range(self.comm_slots):
            coord = self._index_to_coord(int(action.get(f"comm_coord_{slot}", self.noop_index)))
            type_value = int(action.get(f"comm_type_{slot}", 0))
            if coord is None or type_value <= 0:
                continue
            if coord not in visible_counts:
                continue
            counts = visible_counts[coord]
            resource_name = _COMM_TYPE_TO_RESOURCE.get(type_value)
            if resource_name is None:
                continue
            count = self._resource_count(counts, resource_name)
            fact_key = (coord, resource_name)
            if count <= 0 or fact_key in seen_fact_keys:
                continue
            resource_facts.append(
                {
                    "x": int(coord[0]),
                    "z": int(coord[1]),
                    "type": resource_name,
                    "count": int(count),
                    "source": "nn_policy",
                }
            )
            seen_fact_keys.add(fact_key)
        if not resource_facts:
            return {}, 0
        return {"resource_facts": resource_facts}, len(resource_facts)

    def _resource_name_to_comm_type(self, resource_name: str) -> int:
        for comm_type, mapped_name in _COMM_TYPE_TO_RESOURCE.items():
            if mapped_name == resource_name:
                return int(comm_type)
        return 0

    def _resolve_path_mask(
        self,
        *,
        mask: torch.Tensor,
        head_name: str,
        current_pos: Coord,
        partial_action: Mapping[str, int],
    ) -> torch.Tensor:
        slot = int(head_name.split("_")[1])
        cursor = (int(current_pos[0]), int(current_pos[1]))
        for prev_slot in range(slot):
            prev_move = int(partial_action.get(f"path_{prev_slot}", self.action_spec.stop_index))
            if prev_move == self.action_spec.stop_index:
                forced = torch.zeros_like(mask)
                forced[self.action_spec.stop_index] = True
                return forced
            if prev_move < 0 or prev_move >= len(_PATH_DELTAS):
                forced = torch.zeros_like(mask)
                forced[self.action_spec.stop_index] = True
                return forced
            dx, dz = _PATH_DELTAS[prev_move]
            nxt = (int(cursor[0] + dx), int(cursor[1] + dz))
            if not _in_bounds(nxt, width=self.width, height=self.height):
                forced = torch.zeros_like(mask)
                forced[self.action_spec.stop_index] = True
                return forced
            cursor = nxt

        mask.zero_()
        mask[self.action_spec.stop_index] = True
        for move_idx, (dx, dz) in enumerate(_PATH_DELTAS[1:], start=1):
            nxt = (int(cursor[0] + dx), int(cursor[1] + dz))
            if _in_bounds(nxt, width=self.width, height=self.height):
                mask[move_idx] = True
        return mask

    def _encode_path_toward(self, *, action: Dict[str, int], start: Coord, target: Coord) -> Coord:
        cursor = (int(start[0]), int(start[1]))
        for slot in range(self.path_slots):
            if cursor == target:
                break
            step_x = 0 if target[0] == cursor[0] else (1 if target[0] > cursor[0] else -1)
            step_z = 0 if target[1] == cursor[1] else (1 if target[1] > cursor[1] else -1)
            move_idx = int(_DELTA_TO_PATH_INDEX[(step_x, step_z)])
            action[f"path_{slot}"] = move_idx
            cursor = (int(cursor[0] + step_x), int(cursor[1] + step_z))
        return cursor

    def _resolve_comm_coord_mask(
        self,
        *,
        mask: torch.Tensor,
        head_name: str,
        partial_action: Mapping[str, int],
    ) -> torch.Tensor:
        slot = int(head_name.split("_")[-1])
        for prev_slot in range(slot):
            prev_value = int(partial_action.get(f"comm_coord_{prev_slot}", self.noop_index))
            if prev_value == self.noop_index:
                forced = torch.zeros_like(mask)
                forced[self.noop_index] = True
                return forced
        for prev_slot in range(slot):
            prev_coord = int(partial_action.get(f"comm_coord_{prev_slot}", self.noop_index))
            prev_type = int(partial_action.get(f"comm_type_{prev_slot}", 0))
            if 0 <= prev_coord < self.num_cells and prev_type > 0:
                mask[prev_coord] = False
        mask[self.noop_index] = True
        if not bool(mask.any()):
            mask[self.noop_index] = True
        return mask

    def _resolve_comm_type_mask(
        self,
        *,
        mask: torch.Tensor,
        head_name: str,
        agent_obs: Mapping[str, Any],
        partial_action: Mapping[str, int],
    ) -> torch.Tensor:
        slot = int(head_name.split("_")[-1])
        coord_value = int(partial_action.get(f"comm_coord_{slot}", self.noop_index))
        forced = torch.zeros_like(mask)
        if coord_value == self.noop_index:
            forced[0] = True
            return forced
        coord = self._index_to_coord(coord_value)
        if coord is None:
            forced[0] = True
            return forced
        grid = agent_obs["grid"]
        x, z = int(coord[0]), int(coord[1])
        mask.zero_()
        mask[0] = True
        if float(grid[1, z, x].item()) > 0.0:
            mask[1] = True
        if float(grid[2, z, x].item()) > 0.0:
            mask[2] = True
        if float(grid[3, z, x].item()) > 0.0:
            mask[3] = True
        for prev_slot in range(slot):
            prev_coord = int(partial_action.get(f"comm_coord_{prev_slot}", self.noop_index))
            prev_type = int(partial_action.get(f"comm_type_{prev_slot}", 0))
            if prev_coord == coord_value and 0 <= prev_type < _COMM_TYPE_DIM:
                mask[prev_type] = False
        mask[0] = True
        if not bool(mask.any()):
            mask[0] = True
        return mask

    def _build_grid(
        self,
        *,
        state: ResourceGatheringState,
        agent_idx: int,
        visible: Set[Coord],
        visible_counts: Mapping[Coord, ResourceTriple],
        message_maps: Mapping[str, Mapping[Coord, int]],
    ) -> torch.Tensor:
        grid = torch.zeros(self.grid_channels, self.height, self.width, dtype=torch.float32)
        norm = float(self.max_cell_resource)
        for coord in visible:
            x, z = int(coord[0]), int(coord[1])
            grid[0, z, x] = 1.0
            counts = visible_counts.get(coord, (0, 0, 0))
            grid[1, z, x] = float(counts[0]) / norm
            grid[2, z, x] = float(counts[1]) / norm
            grid[3, z, x] = float(counts[2]) / norm
        for channel_idx, resource_name in enumerate(_RESOURCE_ORDER, start=4):
            for coord, count in message_maps.get(resource_name, {}).items():
                x, z = int(coord[0]), int(coord[1])
                grid[channel_idx, z, x] = max(grid[channel_idx, z, x], float(count) / norm)
        cur = state.agent_positions[agent_idx]
        grid[7, int(cur[1]), int(cur[0])] = 1.0
        teammate_idx = 1 - agent_idx
        teammate_pos = state.agent_positions[teammate_idx]
        if teammate_pos in visible:
            grid[8, int(teammate_pos[1]), int(teammate_pos[0])] = 1.0
        for coord in self._extract_reach(cur):
            grid[9, int(coord[1]), int(coord[0])] = 1.0
        return grid

    def _build_scalars(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        agent_idx: int,
    ) -> torch.Tensor:
        pos = state.agent_positions[agent_idx]
        inbox_count = sum(
            len(message.get("resource_facts") or [])
            for message in state.inbox[agent_idx]
            if isinstance(message, Mapping)
        )
        collected = state.collected
        return torch.tensor(
            [
                float(state.turn_index) / float(max(1, state.max_turns)),
                float(pos[0]) / float(max(1, self.width - 1)),
                float(pos[1]) / float(max(1, self.height - 1)),
                float(task.goal_wood) / float(self.max_goal_value),
                float(task.goal_stone) / float(self.max_goal_value),
                float(task.goal_iron) / float(self.max_goal_value),
                min(1.0, float(collected["wood"]) / float(max(1, task.goal_wood))),
                min(1.0, float(collected["stone"]) / float(max(1, task.goal_stone))),
                min(1.0, float(collected["iron"]) / float(max(1, task.goal_iron))),
                min(1.0, float(inbox_count) / 16.0),
                1.0 if state.completed else 0.0,
            ],
            dtype=torch.float32,
        )

    def _build_belief_supervision(
        self,
        *,
        task: ResourceTaskSpec,
        agent_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.zeros(self.num_cells, dtype=torch.float32)
        mask = torch.zeros(self.num_cells, dtype=torch.bool)
        for z, row in enumerate(task.resources):
            for x, counts in enumerate(row):
                idx = self._coord_to_index((int(x), int(z)))
                wood, stone, iron = int(counts[0]), int(counts[1]), int(counts[2])
                if wood + stone + iron <= 0:
                    continue
                mask[idx] = True
                if agent_idx == 0:
                    target[idx] = 1.0 if wood > 0 else 0.0
                else:
                    target[idx] = 1.0 if (stone + iron) > 0 else 0.0
        return target, mask

    def _visible_resource_counts(
        self,
        *,
        state: ResourceGatheringState,
        visible: Iterable[Coord],
    ) -> Dict[Coord, ResourceTriple]:
        out: Dict[Coord, ResourceTriple] = {}
        for coord in visible:
            counts = state.resources.get(coord, (0, 0, 0))
            out[(int(coord[0]), int(coord[1]))] = (int(counts[0]), int(counts[1]), int(counts[2]))
        return out

    def _message_maps(self, inbox: Sequence[Any]) -> Dict[str, Dict[Coord, int]]:
        out: Dict[str, Dict[Coord, int]] = {name: {} for name in _RESOURCE_ORDER}
        for message in inbox:
            if not isinstance(message, Mapping):
                continue
            for fact in message.get("resource_facts") or []:
                if not isinstance(fact, Mapping):
                    continue
                coord = _parse_coord(fact)
                if coord is None:
                    coord = _parse_coord({"x": fact.get("x"), "z": fact.get("z")})
                if coord is None:
                    continue
                resource_name = str(fact.get("type") or "").strip().lower()
                if resource_name not in out:
                    continue
                count = max(0, int(fact.get("count") or 0))
                current = out[resource_name].get(coord, 0)
                out[resource_name][coord] = max(current, count)
        return out

    def _extract_reach(self, current_pos: Coord) -> List[Coord]:
        out: List[Coord] = []
        cx, cz = int(current_pos[0]), int(current_pos[1])
        for x in range(self.width):
            for z in range(self.height):
                coord = (int(x), int(z))
                if self._manhattan((cx, cz), coord) <= self.extraction_range:
                    out.append(coord)
        return out

    def _apply_extraction(self, *, agent_idx: int, counts: ResourceTriple) -> Tuple[Dict[str, int], ResourceTriple]:
        wood, stone, iron = int(counts[0]), int(counts[1]), int(counts[2])
        if agent_idx == 0:
            yielded = {"wood": wood, "stone": 0, "iron": 0}
            return yielded, (0, stone, iron)
        yielded = {"wood": 0, "stone": stone, "iron": iron}
        return yielded, (wood, 0, 0)

    def _auto_harvest(
        self,
        *,
        agent_idx: int,
        current_pos: Coord,
        resources: Dict[Coord, ResourceTriple],
    ) -> Tuple[Dict[str, int], List[Coord]]:
        total = {"wood": 0, "stone": 0, "iron": 0}
        harvested_coords: List[Coord] = []
        for coord in self._extract_reach(current_pos):
            counts = resources.get(coord)
            if counts is None:
                continue
            yielded, remaining = self._apply_extraction(agent_idx=agent_idx, counts=counts)
            if yielded["wood"] <= 0 and yielded["stone"] <= 0 and yielded["iron"] <= 0:
                continue
            total["wood"] += int(yielded["wood"])
            total["stone"] += int(yielded["stone"])
            total["iron"] += int(yielded["iron"])
            harvested_coords.append((int(coord[0]), int(coord[1])))
            if sum(remaining) > 0:
                resources[coord] = remaining
            else:
                resources.pop(coord, None)
        return total, harvested_coords

    def _agent_target_resources(
        self,
        *,
        task: ResourceTaskSpec,
        collected: Mapping[str, int],
        agent_idx: int,
    ) -> Tuple[str, ...]:
        if agent_idx == 0:
            return ("wood",) if int(collected.get("wood", 0)) < int(task.goal_wood) else ()
        targets: List[str] = []
        if int(collected.get("stone", 0)) < int(task.goal_stone):
            targets.append("stone")
        if int(collected.get("iron", 0)) < int(task.goal_iron):
            targets.append("iron")
        return tuple(targets)

    def _resource_count(self, counts: ResourceTriple, resource_name: str) -> int:
        if resource_name == "wood":
            return int(counts[0])
        if resource_name == "stone":
            return int(counts[1])
        if resource_name == "iron":
            return int(counts[2])
        raise ValueError(f"Unsupported resource_name={resource_name}")

    def _remaining_goal(self, *, task: ResourceTaskSpec, collected: Mapping[str, int], resource_name: str) -> int:
        goal_name = f"goal_{resource_name}"
        goal_value = int(getattr(task, goal_name))
        return max(0, goal_value - int(collected.get(resource_name, 0)))

    def _work_zone(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        agent_idx: int,
    ) -> Set[Coord]:
        target_resources = self._agent_target_resources(task=task, collected=state.collected, agent_idx=agent_idx)
        if not target_resources:
            return set()
        zone: Set[Coord] = set()
        for coord, counts in state.resources.items():
            if all(self._resource_count(counts, resource_name) <= 0 for resource_name in target_resources):
                continue
            cx, cz = int(coord[0]), int(coord[1])
            for x in range(self.width):
                for z in range(self.height):
                    pos = (int(x), int(z))
                    if self._manhattan(pos, (cx, cz)) <= self.extraction_range:
                        zone.add(pos)
        return zone

    def _distance_to_zone(self, pos: Coord, zone: Set[Coord]) -> Optional[int]:
        if not zone:
            return None
        return min(self._manhattan(pos, target) for target in zone)

    def _is_in_work_zone(self, *, pos: Coord, zone: Set[Coord]) -> bool:
        return pos in zone if zone else False

    def _nearest_zone_pos(self, *, origin: Coord, zone: Set[Coord]) -> Optional[Coord]:
        if not zone:
            return None
        return min(zone, key=lambda coord: (self._chebyshev(origin, coord), coord[1], coord[0]))

    def _chebyshev(self, left: Coord, right: Coord) -> int:
        return max(abs(int(left[0]) - int(right[0])), abs(int(left[1]) - int(right[1])))

    def _useful_comm_bonus(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        decoded_actions: Sequence[DecodedAgentAction],
    ) -> Tuple[int, float]:
        useful_items = 0
        useful_bonus = 0.0
        for src_idx, decoded in enumerate(decoded_actions):
            facts = decoded.message_obj.get("resource_facts") if isinstance(decoded.message_obj, Mapping) else None
            if not facts:
                continue
            dst_idx = 1 - src_idx
            target_resources = self._agent_target_resources(task=task, collected=state.collected, agent_idx=dst_idx)
            if not target_resources:
                continue
            known_maps = self._known_resource_maps(task=task, state=state, agent_idx=dst_idx)
            sender_score = 0.0
            seen_fact_keys: Set[Tuple[Coord, str]] = set()
            for fact in facts:
                if not isinstance(fact, Mapping):
                    continue
                coord = _parse_coord({"x": fact.get("x"), "z": fact.get("z")})
                if coord is None:
                    continue
                resource_name = str(fact.get("type") or "").strip().lower()
                if resource_name not in target_resources:
                    continue
                fact_key = (coord, resource_name)
                if fact_key in seen_fact_keys:
                    continue
                count = max(0, int(fact.get("count") or 0))
                if count <= 0:
                    continue
                if int(known_maps.get(resource_name, {}).get(coord, 0)) > 0:
                    continue
                remaining_goal = self._remaining_goal(task=task, collected=state.collected, resource_name=resource_name)
                if remaining_goal <= 0:
                    continue
                sender_score += min(1.0, float(count) / float(max(1, remaining_goal)))
                useful_items += 1
                seen_fact_keys.add(fact_key)
            useful_bonus += self.useful_comm_bonus_scale * min(1.0, sender_score)
        return useful_items, float(useful_bonus)

    def _known_resource_maps(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
        agent_idx: int,
    ) -> Dict[str, Dict[Coord, int]]:
        known_maps = self._message_maps(state.inbox[agent_idx])
        visible = compute_visible_cells(
            state.vision_origins[agent_idx],
            view=self.view,
            width=task.width,
            height=task.height,
        )
        visible_counts = self._visible_resource_counts(state=state, visible=visible)
        for coord, counts in visible_counts.items():
            for resource_name in _RESOURCE_ORDER:
                count = self._resource_count(counts, resource_name)
                if count <= 0:
                    continue
                current = int(known_maps[resource_name].get(coord, 0))
                known_maps[resource_name][coord] = max(current, count)
        return known_maps

    def _available_bits(self, counts: ResourceTriple) -> int:
        bits = 0
        if int(counts[0]) > 0:
            bits |= int(_RESOURCE_BITS["wood"])
        if int(counts[1]) > 0:
            bits |= int(_RESOURCE_BITS["stone"])
        if int(counts[2]) > 0:
            bits |= int(_RESOURCE_BITS["iron"])
        return bits

    def _progress_score(self, *, task: ResourceTaskSpec, collected: Mapping[str, int]) -> float:
        fractions = [
            min(1.0, float(collected.get("wood", 0)) / float(max(1, task.goal_wood))),
            min(1.0, float(collected.get("stone", 0)) / float(max(1, task.goal_stone))),
            min(1.0, float(collected.get("iron", 0)) / float(max(1, task.goal_iron))),
        ]
        return float(sum(fractions) / len(fractions))

    def _is_completed(self, *, task: ResourceTaskSpec, collected: Mapping[str, int]) -> bool:
        return (
            int(collected.get("wood", 0)) >= int(task.goal_wood)
            and int(collected.get("stone", 0)) >= int(task.goal_stone)
            and int(collected.get("iron", 0)) >= int(task.goal_iron)
        )

    def _build_terminal_metrics(
        self,
        *,
        task: ResourceTaskSpec,
        state: ResourceGatheringState,
    ) -> Dict[str, Any]:
        progress = self._progress_score(task=task, collected=state.collected)
        completed = bool(self._is_completed(task=task, collected=state.collected))
        return {
            "reward": 0.0,
            "bonus_progress": 0.0,
            "bonus_terminal_complete": 0.0,
            "bonus_move_to_zone": 0.0,
            "bonus_useful_comm": 0.0,
            "bonus_first_enter_zone": 0.0,
            "penalty_move": 0.0,
            "penalty_comm": 0.0,
            "penalty_wasted_extraction": 0.0,
            "progress_score": float(progress),
            "goal_wood": float(task.goal_wood),
            "goal_stone": float(task.goal_stone),
            "goal_iron": float(task.goal_iron),
            "collected_wood": float(state.collected["wood"]),
            "collected_stone": float(state.collected["stone"]),
            "collected_iron": float(state.collected["iron"]),
            "delta_wood": 0.0,
            "delta_stone": 0.0,
            "delta_iron": 0.0,
            "num_comm_items": 0.0,
            "useful_comm_items": 0.0,
            "num_valid_extractions": 0.0,
            "wasted_extractions": 0.0,
            "auto_harvest_cells": 0.0,
            "productive_harvesters": 0.0,
            "move_steps": 0.0,
            "move_toward_zone_steps": 0.0,
            "first_enter_zone_count": 0.0,
            "completed": bool(completed),
            "connected": bool(completed),
            "terminated": True,
            "turn_index": int(max(1, state.turn_index) - 1),
        }

    def _print_debug_turn(
        self,
        *,
        task: ResourceTaskSpec,
        prev_state: ResourceGatheringState,
        next_state: ResourceGatheringState,
        decoded_actions: Sequence[DecodedAgentAction],
        reward: float,
        progress: float,
        metrics: Mapping[str, Any],
    ) -> None:
        print(
            f"[resource_gathering] task={task.task_id} family={task.family or 'unknown'} "
            f"turn={prev_state.turn_index}/{prev_state.max_turns} reward={reward:.3f} "
            f"progress={progress:.3f} completed={bool(metrics.get('completed', False))}",
            flush=True,
        )
        print(
            "  totals: "
            f"wood={int(next_state.collected['wood'])}/{task.goal_wood} "
            f"stone={int(next_state.collected['stone'])}/{task.goal_stone} "
            f"iron={int(next_state.collected['iron'])}/{task.goal_iron}",
            flush=True,
        )
        print(
            "  delta: "
            f"wood=+{int(metrics.get('delta_wood', 0))} "
            f"stone=+{int(metrics.get('delta_stone', 0))} "
            f"iron=+{int(metrics.get('delta_iron', 0))} "
            f"move_steps={int(metrics.get('move_steps', 0))} "
            f"comm_items={int(metrics.get('num_comm_items', 0))} "
            f"auto_harvest_cells={int(metrics.get('auto_harvest_cells', 0))}",
            flush=True,
        )
        print(
            "  shaping: "
            f"move_to_zone={float(metrics.get('bonus_move_to_zone', 0.0)):.3f} "
            f"useful_comm={float(metrics.get('bonus_useful_comm', 0.0)):.3f} "
            f"first_enter_zone={float(metrics.get('bonus_first_enter_zone', 0.0)):.3f} "
            f"zone_steps={int(metrics.get('move_toward_zone_steps', 0))} "
            f"useful_comm_items={int(metrics.get('useful_comm_items', 0))} "
            f"first_entries={int(metrics.get('first_enter_zone_count', 0))}",
            flush=True,
        )
        for agent_idx, action in enumerate(decoded_actions):
            start = prev_state.agent_positions[agent_idx]
            end = next_state.agent_positions[agent_idx]
            facts = action.message_obj.get("resource_facts") if isinstance(action.message_obj, Mapping) else None
            print(
                f"  agent_{agent_idx}: pos {list(start)} -> {list(end)} "
                f"path={self._format_path(action.path)} "
                f"harvest={self._format_coord_list(action.auto_harvest)}",
                flush=True,
            )
            if facts:
                print(
                    f"    comm={self._format_comm_facts(facts)}",
                    flush=True,
                )

    def _format_path(self, path: Sequence[Coord]) -> str:
        if not path:
            return "[]"
        return "[" + " -> ".join(f"({int(x)},{int(z)})" for x, z in path) + "]"

    def _format_coord_list(self, coords: Sequence[Coord]) -> str:
        if not coords:
            return "[]"
        return "[" + ", ".join(f"({int(x)},{int(z)})" for x, z in coords) + "]"

    def _format_comm_facts(self, facts: Sequence[Any]) -> str:
        parts: List[str] = []
        for fact in facts:
            if not isinstance(fact, Mapping):
                continue
            x = int(fact.get("x", 0))
            z = int(fact.get("z", 0))
            resource_type = str(fact.get("type") or "?")
            count = int(fact.get("count") or 0)
            parts.append(f"({x},{z}) {resource_type}:{count}")
        if not parts:
            return "[]"
        return "[" + ", ".join(parts) + "]"

    def _coord_to_index(self, coord: Coord) -> int:
        x, z = int(coord[0]), int(coord[1])
        return int(z * self.width + x)

    def _index_to_coord(self, index: int) -> Coord | None:
        if index < 0 or index >= self.num_cells:
            return None
        x = int(index % self.width)
        z = int(index // self.width)
        return (x, z)

    def _decode_path(self, *, current_pos: Coord, action: Mapping[str, int]) -> List[Coord]:
        path: List[Coord] = [(int(current_pos[0]), int(current_pos[1]))]
        cursor = path[0]
        for slot in range(self.path_slots):
            move_idx = int(action.get(f"path_{slot}", self.action_spec.stop_index))
            if move_idx == self.action_spec.stop_index:
                break
            if move_idx < 0 or move_idx >= len(_PATH_DELTAS):
                break
            dx, dz = _PATH_DELTAS[move_idx]
            nxt = (int(cursor[0] + dx), int(cursor[1] + dz))
            if not _in_bounds(nxt, width=self.width, height=self.height):
                break
            path.append(nxt)
            cursor = nxt
        return path

    def _manhattan(self, left: Coord, right: Coord) -> int:
        return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


__all__ = [
    "DecodedAgentAction",
    "ResourceGatheringActionSpec",
    "ResourceGatheringEnv",
    "ResourceGatheringState",
    "ResourceTaskSpec",
    "_parse_rows",
    "_rows_to_task",
    "compute_visible_cells",
    "load_tasks_from_json",
]
