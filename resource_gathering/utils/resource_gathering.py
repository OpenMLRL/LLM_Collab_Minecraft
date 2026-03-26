from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_USER_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_NNGAMES_ROOT = os.path.join(_USER_ROOT, "NNGames")
if os.path.isdir(_NNGAMES_ROOT) and _NNGAMES_ROOT not in sys.path:
    sys.path.insert(0, _NNGAMES_ROOT)

from NNGames.resource_gathering.envs.nn_env import (
    DecodedAgentAction,
    ResourceGatheringEnv as NNResourceGatheringEnv,
    ResourceGatheringState,
    ResourceTaskSpec,
    _parse_rows as _nn_parse_rows,
    _rows_to_task as _nn_rows_to_task,
    compute_visible_cells,
    load_tasks_from_json as _load_tasks_from_json,
)


Coord = Tuple[int, int]
ResourceTriple = Tuple[int, int, int]
_ACTION_KEYS = frozenset({"comm", "probe", "cmds", "path"})


load_tasks_from_json = _load_tasks_from_json


def _parse_coord(value: Any) -> Coord | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (int(value[0]), int(value[1]))
        except Exception:
            return None
    if isinstance(value, Mapping):
        try:
            return (int(value.get("x")), int(value.get("z")))
        except Exception:
            return None
    return None


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1].strip()
            raw = re.sub(r"^\s*[A-Za-z0-9_-]+\s*\n", "", raw).strip()
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    quote = ""
    escape = False
    for idx in range(start, len(raw)):
        cur = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif cur == "\\":
                escape = True
            elif cur == quote:
                in_string = False
            continue
        if cur in ('"', "'"):
            in_string = True
            quote = cur
            continue
        if cur == "{":
            depth += 1
        elif cur == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                except Exception:
                    return {}
                return obj if isinstance(obj, dict) else {}
    return {}


def task_to_item(task: ResourceTaskSpec) -> Dict[str, Any]:
    return {
        "task_id": str(task.task_id),
        "family": str(task.family or ""),
        "goal": {
            "wood": int(task.goal_wood),
            "stone": int(task.goal_stone),
            "iron": int(task.goal_iron),
        },
        "rows": list(task.rows_topdown),
        "agent_starts": [[int(x), int(z)] for x, z in task.agent_starts],
        "max_turns": int(task.max_turns),
    }


def task_from_item(item: Mapping[str, Any]) -> ResourceTaskSpec:
    rows_obj = item.get("map") if "map" in item else item.get("rows")
    rows = _nn_parse_rows(rows_obj)
    task = _nn_rows_to_task(
        task_id=str(item.get("task_id") or "resource_gathering_unknown"),
        rows=rows,
        goal=(item.get("goal") if isinstance(item.get("goal"), Mapping) else None),
        family=str(item.get("family") or ""),
        agent_starts_override=(
            item.get("agent_starts") if isinstance(item.get("agent_starts"), Sequence) else None
        ),
    )
    max_turns = int(item.get("max_turns") or task.max_turns)
    return ResourceTaskSpec(
        task_id=task.task_id,
        width=task.width,
        height=task.height,
        rows_topdown=task.rows_topdown,
        resources=task.resources,
        agent_starts=task.agent_starts,
        goal_wood=task.goal_wood,
        goal_stone=task.goal_stone,
        goal_iron=task.goal_iron,
        max_turns=max(1, max_turns),
        family=task.family,
    )


def make_initial_state(task: ResourceTaskSpec, *, num_agents: int, max_turns: Optional[int] = None) -> ResourceGatheringState:
    resolved_turns = int(max_turns) if max_turns is not None else int(task.max_turns)
    resources: Dict[Coord, ResourceTriple] = {}
    for z, row in enumerate(task.resources):
        for x, counts in enumerate(row):
            triple = (int(counts[0]), int(counts[1]), int(counts[2]))
            if sum(triple) > 0:
                resources[(int(x), int(z))] = triple
    state = ResourceGatheringState(
        turn_index=1,
        max_turns=resolved_turns,
        agent_positions=[(int(x), int(z)) for x, z in task.agent_starts[: int(num_agents)]],
        vision_origins=[{(int(x), int(z))} for x, z in task.agent_starts[: int(num_agents)]],
        inbox=[[] for _ in range(int(num_agents))],
        resources=resources,
        collected={"wood": 0, "stone": 0, "iron": 0},
        entered_work_zones=[False for _ in range(int(num_agents))],
        completed=False,
        terminated=False,
    )
    env = _build_env(task=task, state=state, view=2, extraction_range=2, reward_config=None, debug=False)
    state.entered_work_zones = [
        env._is_in_work_zone(
            pos=state.agent_positions[agent_idx],
            zone=env._work_zone(task=task, state=state, agent_idx=agent_idx),
        )
        for agent_idx in range(int(num_agents))
    ]
    return state


def serialize_state(state: ResourceGatheringState) -> Dict[str, Any]:
    return {
        "turn_index": int(state.turn_index),
        "max_turns": int(state.max_turns),
        "agent_positions": [[int(x), int(z)] for x, z in state.agent_positions],
        "vision_origins": [
            [[int(x), int(z)] for x, z in sorted(origins, key=lambda item: (item[1], item[0]))]
            for origins in state.vision_origins
        ],
        "inbox": copy.deepcopy(state.inbox),
        "resources": [
            {
                "x": int(coord[0]),
                "z": int(coord[1]),
                "wood": int(counts[0]),
                "stone": int(counts[1]),
                "iron": int(counts[2]),
            }
            for coord, counts in sorted(state.resources.items(), key=lambda item: (item[0][1], item[0][0]))
        ],
        "collected": {name: int(value) for name, value in state.collected.items()},
        "entered_work_zones": [bool(v) for v in state.entered_work_zones],
        "completed": bool(state.completed),
        "terminated": bool(state.terminated),
    }


def deserialize_state(raw: Mapping[str, Any], *, num_agents: int) -> ResourceGatheringState:
    resources: Dict[Coord, ResourceTriple] = {}
    for item in raw.get("resources") or []:
        if not isinstance(item, Mapping):
            continue
        coord = _parse_coord(item)
        if coord is None:
            continue
        resources[coord] = (
            int(item.get("wood", 0)),
            int(item.get("stone", 0)),
            int(item.get("iron", 0)),
        )
    agent_positions = [
        tuple(int(v) for v in pos[:2])
        for pos in (raw.get("agent_positions") or [])[: int(num_agents)]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2
    ]
    while len(agent_positions) < int(num_agents):
        agent_positions.append((0, 0))
    vision_origins = []
    for items in (raw.get("vision_origins") or [])[: int(num_agents)]:
        origins = set()
        if isinstance(items, (list, tuple)):
            for pos in items:
                coord = _parse_coord(pos)
                if coord is not None:
                    origins.add(coord)
        vision_origins.append(origins or {agent_positions[len(vision_origins)]})
    while len(vision_origins) < int(num_agents):
        vision_origins.append({agent_positions[len(vision_origins)]})
    inbox = [list(msgs) for msgs in (raw.get("inbox") or [])[: int(num_agents)]]
    while len(inbox) < int(num_agents):
        inbox.append([])
    entered = [bool(v) for v in (raw.get("entered_work_zones") or [])[: int(num_agents)]]
    while len(entered) < int(num_agents):
        entered.append(False)
    collected_raw = raw.get("collected") or {}
    return ResourceGatheringState(
        turn_index=int(raw.get("turn_index", 1)),
        max_turns=int(raw.get("max_turns", 4)),
        agent_positions=agent_positions,
        vision_origins=vision_origins,
        inbox=inbox,
        resources=resources,
        collected={
            "wood": int(collected_raw.get("wood", 0)),
            "stone": int(collected_raw.get("stone", 0)),
            "iron": int(collected_raw.get("iron", 0)),
        },
        entered_work_zones=entered,
        completed=bool(raw.get("completed", False)),
        terminated=bool(raw.get("terminated", False)),
    )


def _build_env(
    *,
    task: ResourceTaskSpec,
    state: ResourceGatheringState,
    view: int,
    extraction_range: int,
    reward_config: Optional[Mapping[str, Any]],
    debug: bool,
) -> NNResourceGatheringEnv:
    reward_cfg = dict(reward_config or {})
    env = NNResourceGatheringEnv(
        tasks=[task],
        num_agents=len(state.agent_positions),
        view=int(view),
        max_turns=int(state.max_turns),
        extraction_limit=0,
        extraction_range=int(extraction_range),
        path_slots=max(1, int(reward_cfg.get("path_slots", 4))),
        comm_limit=max(1, int(reward_cfg.get("comm_limit", 1))),
        progress_reward_scale=float(reward_cfg.get("progress_reward_scale", 10.0)),
        terminal_bonus=float(reward_cfg.get("terminal_bonus", 4.0)),
        move_cost_scale=float(reward_cfg.get("move_cost_scale", 0.0)),
        comm_cost_scale=float(reward_cfg.get("comm_cost_scale", 0.0)),
        wasted_extraction_penalty=0.0,
        move_to_zone_bonus_scale=float(reward_cfg.get("move_to_zone_bonus_scale", 0.05)),
        useful_comm_bonus_scale=float(reward_cfg.get("useful_comm_bonus_scale", 0.1)),
        first_enter_zone_bonus_scale=float(reward_cfg.get("first_enter_zone_bonus_scale", 0.15)),
        debug=bool(debug),
    )
    env.current_task_index = 0
    env.current_state = copy.deepcopy(state)
    return env


def build_payload(
    *,
    task: ResourceTaskSpec,
    state_before_turn: ResourceGatheringState,
    num_agents: int,
    view: int,
    extraction_range: int,
    max_path_len: int,
    system_prompt: str,
    user_template_single: str,
    user_template_agent1: str,
    user_template_agent2: str,
    reward_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "task": task_to_item(task),
        "state_before_turn": serialize_state(state_before_turn),
        "num_agents": int(num_agents),
        "view": int(view),
        "extraction_range": int(extraction_range),
        "max_path_len": int(max_path_len),
        "system_prompt": str(system_prompt or "").rstrip(),
        "user_template_single": str(user_template_single or "").rstrip(),
        "user_template_agent1": str(user_template_agent1 or "").rstrip(),
        "user_template_agent2": str(user_template_agent2 or "").rstrip(),
        "reward_config": dict(reward_config or {}),
    }


def payload_to_task(payload: Mapping[str, Any]) -> ResourceTaskSpec:
    task_obj = payload.get("task") or {}
    if not isinstance(task_obj, Mapping):
        raise ValueError("payload.task must be a mapping")
    return task_from_item(task_obj)


def payload_to_state(payload: Mapping[str, Any]) -> ResourceGatheringState:
    task = payload_to_task(payload)
    num_agents = int(payload.get("num_agents") or 2)
    raw_state = payload.get("state_before_turn") or {}
    if isinstance(raw_state, Mapping):
        return deserialize_state(raw_state, num_agents=num_agents)
    return make_initial_state(task, num_agents=num_agents)


def get_agent_observation(
    task: ResourceTaskSpec,
    state: ResourceGatheringState,
    *,
    agent_idx: int,
    view: int,
    extraction_range: int,
) -> Dict[str, Any]:
    env = _build_env(
        task=task,
        state=state,
        view=view,
        extraction_range=extraction_range,
        reward_config=None,
        debug=False,
    )
    visible = compute_visible_cells(
        state.vision_origins[agent_idx],
        view=view,
        width=task.width,
        height=task.height,
    )
    visible_counts = env._visible_resource_counts(state=state, visible=visible)
    visible_resources = [
        {
            "coord": [int(coord[0]), int(coord[1])],
            "wood": int(counts[0]),
            "stone": int(counts[1]),
            "iron": int(counts[2]),
        }
        for coord, counts in sorted(visible_counts.items(), key=lambda item: (item[0][1], item[0][0]))
        if sum(int(v) for v in counts) > 0
    ]
    teammate_idx = 1 - int(agent_idx)
    teammate_pos = state.agent_positions[teammate_idx]
    return {
        "turn_index": int(state.turn_index),
        "max_turns": int(state.max_turns),
        "current_pos": [int(state.agent_positions[agent_idx][0]), int(state.agent_positions[agent_idx][1])],
        "visible_resources": visible_resources,
        "visible_resource_counts": visible_counts,
        "visible": visible,
        "visible_teammate_pos": [int(teammate_pos[0]), int(teammate_pos[1])] if teammate_pos in visible else None,
        "received_messages": copy.deepcopy(state.inbox[agent_idx]),
        "harvest_zone": [[int(x), int(z)] for x, z in env._extract_reach(state.agent_positions[agent_idx])],
    }


def build_prompt_fields(
    *,
    task: ResourceTaskSpec,
    state: ResourceGatheringState,
    agent_idx: int,
    view: int,
    extraction_range: int,
    max_path_len: int,
    feedback: str = "",
) -> Dict[str, Any]:
    obs = get_agent_observation(
        task,
        state,
        agent_idx=agent_idx,
        view=view,
        extraction_range=extraction_range,
    )
    collected = state.collected
    return {
        "agent_name": "A" if int(agent_idx) == 0 else "B",
        "origin": [0, 0],
        "map_width": int(task.width),
        "map_height": int(task.height),
        "turn_idx": int(state.turn_index),
        "max_turns": int(state.max_turns),
        "goal_wood": int(task.goal_wood),
        "goal_stone": int(task.goal_stone),
        "goal_iron": int(task.goal_iron),
        "collected_wood": int(collected["wood"]),
        "collected_stone": int(collected["stone"]),
        "collected_iron": int(collected["iron"]),
        "remaining_wood": max(0, int(task.goal_wood) - int(collected["wood"])),
        "remaining_stone": max(0, int(task.goal_stone) - int(collected["stone"])),
        "remaining_iron": max(0, int(task.goal_iron) - int(collected["iron"])),
        "view": int(view),
        "extraction_range": int(extraction_range),
        "max_path_len": int(max_path_len),
        "current_pos": obs["current_pos"],
        "visible_resources": json.dumps(obs["visible_resources"], ensure_ascii=False, separators=(",", ":")),
        "visible_teammate_pos": json.dumps(obs["visible_teammate_pos"], ensure_ascii=False),
        "received_messages": json.dumps(obs["received_messages"], ensure_ascii=False, separators=(",", ":")),
        "harvest_zone": json.dumps(obs["harvest_zone"], ensure_ascii=False, separators=(",", ":")),
        "feedback": str(feedback or "").rstrip() or "None",
    }


def render_agent_user_prompt(
    *,
    task: ResourceTaskSpec,
    state: ResourceGatheringState,
    agent_idx: int,
    view: int,
    extraction_range: int,
    max_path_len: int,
    user_template: str,
    feedback: str = "",
) -> str:
    fields = build_prompt_fields(
        task=task,
        state=state,
        agent_idx=agent_idx,
        view=view,
        extraction_range=extraction_range,
        max_path_len=max_path_len,
        feedback=feedback,
    )
    return str(user_template or "").format(**fields).rstrip()


def render_prompts_from_payload(
    *,
    payload: Mapping[str, Any],
    state: ResourceGatheringState,
    num_agents: int,
    feedback_by_agent: Sequence[str] | None = None,
    include_system: bool = True,
) -> List[str]:
    task = payload_to_task(payload)
    n = max(1, int(num_agents))
    view = int(payload.get("view") or 2)
    extraction_range = int(payload.get("extraction_range") or 2)
    max_path_len = int(payload.get("max_path_len") or 4)

    system_prompt = str(payload.get("system_prompt") or "").rstrip()
    tmpl_single = str(payload.get("user_template_single") or "").rstrip()
    tmpl_a1 = str(payload.get("user_template_agent1") or tmpl_single).rstrip()
    tmpl_a2 = str(payload.get("user_template_agent2") or tmpl_single).rstrip()

    feedback_arr = list(feedback_by_agent or [])
    while len(feedback_arr) < n:
        feedback_arr.append("")

    prompts: List[str] = []
    for idx in range(n):
        tmpl = tmpl_single if n == 1 else (tmpl_a1 if idx == 0 else tmpl_a2)
        user_prompt = render_agent_user_prompt(
            task=task,
            state=state,
            agent_idx=idx,
            view=view,
            extraction_range=extraction_range,
            max_path_len=max_path_len,
            user_template=tmpl,
            feedback=feedback_arr[idx],
        )
        if include_system and system_prompt:
            prompts.append(system_prompt + "\n\n" + user_prompt)
        else:
            prompts.append(user_prompt)
    return prompts


def _sanitize_comm_obj(
    *,
    comm_obj: Any,
    visible_counts: Mapping[Coord, ResourceTriple],
) -> Tuple[Dict[str, Any], int]:
    if not isinstance(comm_obj, Mapping):
        return {}, 0
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[Coord, str]] = set()
    for fact in comm_obj.get("resource_facts") or []:
        if not isinstance(fact, Mapping):
            continue
        coord = _parse_coord(fact)
        if coord is None or coord not in visible_counts:
            continue
        resource_name = str(fact.get("type") or "").strip().lower()
        if resource_name not in {"wood", "stone", "iron"}:
            continue
        counts = visible_counts[coord]
        if resource_name == "wood":
            count = int(counts[0])
        elif resource_name == "stone":
            count = int(counts[1])
        else:
            count = int(counts[2])
        key = (coord, resource_name)
        if count <= 0 or key in seen:
            continue
        out.append(
            {
                "x": int(coord[0]),
                "z": int(coord[1]),
                "type": resource_name,
                "count": int(count),
                "source": "nn_policy",
            }
        )
        seen.add(key)
    if not out:
        return {}, 0
    return {"resource_facts": out}, len(out)


def _decode_path(
    *,
    task: ResourceTaskSpec,
    current_pos: Coord,
    raw_path: Any,
    max_path_len: int,
) -> Tuple[List[Coord], bool]:
    start = (int(current_pos[0]), int(current_pos[1]))
    if not isinstance(raw_path, list) or not raw_path:
        return [start], True
    coords: List[Coord] = []
    for item in raw_path[: max(1, int(max_path_len) + 1)]:
        coord = _parse_coord(item)
        if coord is None:
            return [start], False
        coords.append((int(coord[0]), int(coord[1])))
    if coords[0] != start:
        return [start], False
    for prev, cur in zip(coords, coords[1:]):
        dx = abs(int(cur[0]) - int(prev[0]))
        dz = abs(int(cur[1]) - int(prev[1]))
        if max(dx, dz) > 1 or not (0 <= int(cur[0]) < int(task.width) and 0 <= int(cur[1]) < int(task.height)):
            return [start], False
    return coords, True


def transition_payload(
    *,
    payload: Mapping[str, Any],
    agent_completions: Sequence[str],
    num_agents: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[DecodedAgentAction]]:
    task = payload_to_task(payload)
    state = payload_to_state(payload)
    n = max(1, int(num_agents))
    view = int(payload.get("view") or 2)
    extraction_range = int(payload.get("extraction_range") or 2)
    max_path_len = int(payload.get("max_path_len") or 4)
    reward_cfg = payload.get("reward_config") if isinstance(payload.get("reward_config"), Mapping) else {}

    env = _build_env(
        task=task,
        state=state,
        view=view,
        extraction_range=extraction_range,
        reward_config=reward_cfg,
        debug=False,
    )

    prev_progress = env._progress_score(task=task, collected=state.collected)
    decoded_actions: List[DecodedAgentAction] = []
    for agent_idx in range(n):
        obs = get_agent_observation(task, state, agent_idx=agent_idx, view=view, extraction_range=extraction_range)
        raw = _extract_json_object(agent_completions[agent_idx] if agent_idx < len(agent_completions) else "")
        comm_obj, comm_items = _sanitize_comm_obj(
            comm_obj=raw.get("comm", {}),
            visible_counts=obs["visible_resource_counts"],
        )
        path, path_valid = _decode_path(
            task=task,
            current_pos=tuple(int(v) for v in obs["current_pos"]),
            raw_path=raw.get("path", [obs["current_pos"]]),
            max_path_len=max_path_len,
        )
        decoded_actions.append(
            DecodedAgentAction(
                message_obj=comm_obj,
                comm_items=int(comm_items),
                auto_harvest=[],
                path=path,
                path_valid=bool(path_valid),
            )
        )

    next_state = copy.deepcopy(state)
    work_zones = [env._work_zone(task=task, state=state, agent_idx=agent_idx) for agent_idx in range(n)]
    prev_zone_distances = [env._distance_to_zone(state.agent_positions[agent_idx], work_zones[agent_idx]) for agent_idx in range(n)]
    prev_in_zone = [env._is_in_work_zone(pos=state.agent_positions[agent_idx], zone=work_zones[agent_idx]) for agent_idx in range(n)]
    move_steps = 0
    for agent_idx, decoded in enumerate(decoded_actions):
        cur = next_state.agent_positions[agent_idx]
        next_state.vision_origins[agent_idx].add(cur)
        if decoded.path_valid:
            for coord in decoded.path:
                next_state.vision_origins[agent_idx].add(coord)
            next_state.agent_positions[agent_idx] = decoded.path[-1]
            move_steps += max(0, len(decoded.path) - 1)

    next_zone_distances = [env._distance_to_zone(next_state.agent_positions[agent_idx], work_zones[agent_idx]) for agent_idx in range(n)]
    move_toward_zone_steps = 0
    first_enter_zone_count = 0
    for agent_idx in range(n):
        prev_distance = prev_zone_distances[agent_idx]
        next_distance = next_zone_distances[agent_idx]
        if prev_distance is not None and next_distance is not None and next_distance < prev_distance:
            move_toward_zone_steps += int(prev_distance - next_distance)
        next_in_zone = env._is_in_work_zone(pos=next_state.agent_positions[agent_idx], zone=work_zones[agent_idx])
        if not state.entered_work_zones[agent_idx] and not prev_in_zone[agent_idx] and next_in_zone:
            first_enter_zone_count += 1
        next_state.entered_work_zones[agent_idx] = bool(state.entered_work_zones[agent_idx] or next_in_zone)

    delta_wood = 0
    delta_stone = 0
    delta_iron = 0
    auto_harvest_cells = 0
    productive_harvesters = 0
    for agent_idx, decoded in enumerate(decoded_actions):
        yielded, harvested_coords = env._auto_harvest(
            agent_idx=agent_idx,
            current_pos=next_state.agent_positions[agent_idx],
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
        for dst_idx in range(n):
            if dst_idx != src_idx:
                next_state.inbox[dst_idx].append(copy.deepcopy(decoded.message_obj))

    next_progress = env._progress_score(task=task, collected=next_state.collected)
    completed = bool(env._is_completed(task=task, collected=next_state.collected))
    progress_bonus = env.progress_reward_scale * max(0.0, next_progress - prev_progress)
    terminal_bonus = env.terminal_bonus if completed and not state.completed else 0.0
    move_to_zone_bonus = env.move_to_zone_bonus_scale * float(move_toward_zone_steps)
    useful_comm_items, useful_comm_bonus = env._useful_comm_bonus(task=task, state=state, decoded_actions=decoded_actions)
    first_enter_zone_bonus = env.first_enter_zone_bonus_scale * float(first_enter_zone_count)
    comm_items = sum(int(decoded.comm_items) for decoded in decoded_actions)
    move_penalty = env.move_cost_scale * float(move_steps)
    comm_penalty = env.comm_cost_scale * float(comm_items)
    reward = progress_bonus + terminal_bonus + move_to_zone_bonus + useful_comm_bonus + first_enter_zone_bonus - move_penalty - comm_penalty

    next_state.turn_index = int(state.turn_index) + 1
    next_state.completed = completed
    next_state.terminated = bool(completed or (next_state.turn_index > next_state.max_turns))

    next_payload = dict(payload)
    next_payload["state_before_turn"] = serialize_state(next_state)
    metrics = {
        "reward": float(reward),
        "bonus_progress": float(progress_bonus),
        "bonus_terminal_complete": float(terminal_bonus),
        "bonus_move_to_zone": float(move_to_zone_bonus),
        "bonus_useful_comm": float(useful_comm_bonus),
        "bonus_first_enter_zone": float(first_enter_zone_bonus),
        "penalty_move": float(move_penalty),
        "penalty_comm": float(comm_penalty),
        "penalty_wasted_extraction": 0.0,
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
        "success": bool(completed),
        "terminated": bool(next_state.terminated),
        "turn_index": int(state.turn_index),
    }
    return next_payload, metrics, decoded_actions
