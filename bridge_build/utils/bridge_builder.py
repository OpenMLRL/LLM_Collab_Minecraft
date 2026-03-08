from __future__ import annotations

import ast
import copy
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


Coord2D = Tuple[int, int]

_AIR_BLOCKS = {"air", "cave_air", "void_air"}
_ALLOWED_MAP_CHARS = {"#", ".", "S", "T", "Y", "N"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")
_FILL_CMD_RE = re.compile(
    r"^/?fill\s+(?P<x1>\S+)\s+(?P<z1>\S+)\s+(?P<x2>\S+)\s+(?P<z2>\S+)\s+(?P<block>\S+)\s*$",
    flags=re.IGNORECASE,
)
_ACTION_KEYS = frozenset({"comm", "probe", "cmds", "path"})


def normalize_block_id(block_id: str) -> str:
    s = (block_id or "").strip()
    if s.startswith("minecraft:"):
        s = s[len("minecraft:") :]
    return s


def _strip_markdown_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw or "```" not in raw:
        return raw
    parts = raw.split("```")
    if len(parts) < 3:
        return raw
    inner = parts[1].strip()
    inner = re.sub(r"^\s*[a-zA-Z0-9_-]+\s*\n", "", inner)
    return inner.strip()


def _parse_int_token(tok: Any) -> int | None:
    t = str(tok).strip()
    if not t:
        return None
    if t.startswith(("~", "^")):
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _parse_coord(value: Any) -> Coord2D | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _parse_int_token(value[0])
        z = _parse_int_token(value[1])
        if x is None or z is None:
            return None
        return (int(x), int(z))
    if isinstance(value, Mapping):
        x = _parse_int_token(value.get("x"))
        z = _parse_int_token(value.get("z"))
        if x is None or z is None:
            return None
        return (int(x), int(z))
    if isinstance(value, str):
        m = re.match(r"^\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*$", value)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


def _coord_key(pos: Coord2D) -> str:
    return f"{int(pos[0])},{int(pos[1])}"


def _decode_coord_key(key: str) -> Coord2D | None:
    m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$", str(key))
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def _in_bounds(pos: Coord2D, *, width: int, height: int) -> bool:
    x, z = int(pos[0]), int(pos[1])
    return 0 <= x < int(width) and 0 <= z < int(height)


def _neighbors4(pos: Coord2D) -> List[Coord2D]:
    x, z = pos
    return [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)]


def _sorted_coords(coords: Iterable[Coord2D]) -> List[Coord2D]:
    return sorted({(int(x), int(z)) for x, z in coords}, key=lambda p: (p[1], p[0]))


def _coords_to_json(coords: Iterable[Coord2D]) -> str:
    arr = [[int(x), int(z)] for x, z in _sorted_coords(coords)]
    return json.dumps(arr, ensure_ascii=False, separators=(",", ":"))


def _count_tokens(text: str) -> int:
    raw = (text or "").strip()
    if not raw:
        return 0
    return len(_TOKEN_RE.findall(raw))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    width: int
    height: int
    rows_topdown: Tuple[str, ...]
    anchors_s: Tuple[Coord2D, ...]
    anchors_t: Tuple[Coord2D, ...]
    land_cells: Tuple[Coord2D, ...]
    true_pillars: Tuple[Coord2D, ...]
    false_pillars: Tuple[Coord2D, ...]
    max_turns: int

    @property
    def candidate_pillars(self) -> Tuple[Coord2D, ...]:
        return tuple(_sorted_coords([*self.true_pillars, *self.false_pillars]))


@dataclass
class BridgeState:
    turn_index: int
    max_turns: int
    filled: Dict[Coord2D, str]
    agent_positions: List[Coord2D]
    vision_origins: List[Set[Coord2D]]
    known_pillars: List[Dict[Coord2D, str]]
    inbox: List[List[Any]]
    connected: bool
    terminated: bool


@dataclass
class ParsedAgentAction:
    comm_obj: Any
    comm_text: str
    comm_tokens: int
    probes: List[Coord2D]
    cmds: List[str]
    fills: List[Tuple[int, int, int, int, str]]
    path: List[Coord2D]
    path_valid: bool
    rejected_cmds: List[Dict[str, Any]]


@dataclass
class TurnResult:
    state: BridgeState
    metrics: Dict[str, Any]
    actions: List[ParsedAgentAction]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_rows(rows_obj: Any) -> List[str]:
    if isinstance(rows_obj, str):
        lines = [line.rstrip("\n") for line in rows_obj.splitlines()]
        rows = [ln.strip() for ln in lines if ln.strip()]
        return rows
    if isinstance(rows_obj, list):
        rows = [str(r) for r in rows_obj if str(r).strip()]
        return rows
    raise ValueError("task map must be a non-empty string or list of strings")


def _rows_to_task(*, task_id: str, rows: Sequence[str], max_turns: int | None = None) -> TaskSpec:
    if not rows:
        raise ValueError(f"{task_id}: map rows are empty")
    width = len(rows[0])
    if width <= 0:
        raise ValueError(f"{task_id}: map row width must be > 0")
    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"{task_id}: inconsistent row width at row={idx}")
        invalid = [ch for ch in row if ch not in _ALLOWED_MAP_CHARS]
        if invalid:
            raise ValueError(f"{task_id}: invalid map chars {sorted(set(invalid))}")

    height = len(rows)

    anchors_s: List[Coord2D] = []
    anchors_t: List[Coord2D] = []
    land: List[Coord2D] = []
    true_pillars: List[Coord2D] = []
    false_pillars: List[Coord2D] = []

    for z, row in enumerate(rows):
        for x, ch in enumerate(row):
            pos = (int(x), int(z))
            if ch in ("#", "S", "T"):
                land.append(pos)
            if ch == "S":
                anchors_s.append(pos)
            elif ch == "T":
                anchors_t.append(pos)
            elif ch == "Y":
                true_pillars.append(pos)
            elif ch == "N":
                false_pillars.append(pos)

    if not anchors_s:
        raise ValueError(f"{task_id}: map must contain at least one S")
    if not anchors_t:
        raise ValueError(f"{task_id}: map must contain at least one T")

    auto_turns = int(5 * max(width, height))
    if max_turns is None:
        resolved_max_turns = max(1, auto_turns)
    else:
        resolved_max_turns = max(1, int(max_turns))

    return TaskSpec(
        task_id=task_id,
        width=int(width),
        height=int(height),
        rows_topdown=tuple(rows),
        anchors_s=tuple(_sorted_coords(anchors_s)),
        anchors_t=tuple(_sorted_coords(anchors_t)),
        land_cells=tuple(_sorted_coords(land)),
        true_pillars=tuple(_sorted_coords(true_pillars)),
        false_pillars=tuple(_sorted_coords(false_pillars)),
        max_turns=resolved_max_turns,
    )


def load_tasks_from_json(json_path: str) -> List[TaskSpec]:
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

    tasks: List[TaskSpec] = []
    for idx, task_obj in enumerate(task_objs, start=1):
        if not isinstance(task_obj, Mapping):
            raise ValueError(f"task entry #{idx} must be an object")
        task_id = str(task_obj.get("task_id") or f"bridge_build_{idx:04d}")
        rows_obj = task_obj.get("map") if "map" in task_obj else task_obj.get("rows")
        rows = _parse_rows(rows_obj)
        max_turns_raw = task_obj.get("max_turns")
        max_turns = int(max_turns_raw) if max_turns_raw is not None else None
        tasks.append(_rows_to_task(task_id=task_id, rows=rows, max_turns=max_turns))

    return tasks


def task_to_item(task: TaskSpec) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "width": int(task.width),
        "height": int(task.height),
        "rows_topdown": [str(r) for r in task.rows_topdown],
        "anchors_s": [[int(x), int(z)] for x, z in task.anchors_s],
        "anchors_t": [[int(x), int(z)] for x, z in task.anchors_t],
        "land_cells": [[int(x), int(z)] for x, z in task.land_cells],
        "true_pillars": [[int(x), int(z)] for x, z in task.true_pillars],
        "false_pillars": [[int(x), int(z)] for x, z in task.false_pillars],
        "max_turns": int(task.max_turns),
    }


def task_from_item(item: Mapping[str, Any]) -> TaskSpec:
    task_id = str(item.get("task_id") or "")
    rows = [str(r) for r in (item.get("rows_topdown") or [])]
    if rows:
        return _rows_to_task(
            task_id=task_id or "bridge_build_unknown",
            rows=rows,
            max_turns=int(item.get("max_turns") or 0) or None,
        )

    width = int(item.get("width") or 0)
    height = int(item.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("task item must contain rows_topdown or positive width/height")

    # Fallback: reconstruct empty rows and inject coordinates below.
    rows_fallback = ["." * width for _ in range(height)]
    task = _rows_to_task(task_id=task_id or "bridge_build_unknown", rows=rows_fallback)

    def _as_coords(value: Any) -> List[Coord2D]:
        out: List[Coord2D] = []
        if isinstance(value, list):
            for v in value:
                p = _parse_coord(v)
                if p is not None:
                    out.append(p)
        return _sorted_coords(out)

    anchors_s = tuple(_as_coords(item.get("anchors_s"))) or task.anchors_s
    anchors_t = tuple(_as_coords(item.get("anchors_t"))) or task.anchors_t
    land = tuple(_as_coords(item.get("land_cells"))) or task.land_cells
    true_p = tuple(_as_coords(item.get("true_pillars")))
    false_p = tuple(_as_coords(item.get("false_pillars")))
    max_turns = int(item.get("max_turns") or task.max_turns)

    return TaskSpec(
        task_id=task.task_id,
        width=task.width,
        height=task.height,
        rows_topdown=task.rows_topdown,
        anchors_s=anchors_s,
        anchors_t=anchors_t,
        land_cells=land,
        true_pillars=true_p,
        false_pillars=false_p,
        max_turns=max(1, int(max_turns)),
    )


def split_command_limits(max_commands_total: int, num_agents: int) -> List[int]:
    n = max(1, int(num_agents))
    total = max(1, int(max_commands_total))
    per = max(1, total // n)
    extra = total % n
    limits = [per] * n
    limits[0] += extra
    return limits


def _is_air(block: str) -> bool:
    return normalize_block_id(block) in _AIR_BLOCKS


def clone_state(state: BridgeState) -> BridgeState:
    return BridgeState(
        turn_index=int(state.turn_index),
        max_turns=int(state.max_turns),
        filled={(int(x), int(z)): str(b) for (x, z), b in state.filled.items()},
        agent_positions=[(int(x), int(z)) for x, z in state.agent_positions],
        vision_origins=[set((int(x), int(z)) for x, z in origins) for origins in state.vision_origins],
        known_pillars=[
            {(int(x), int(z)): str(v) for (x, z), v in known.items()} for known in state.known_pillars
        ],
        inbox=[list(msgs) for msgs in state.inbox],
        connected=bool(state.connected),
        terminated=bool(state.terminated),
    )


def _traversable_cells(task: TaskSpec, filled: Mapping[Coord2D, str]) -> Set[Coord2D]:
    cells: Set[Coord2D] = set(task.land_cells)
    cells.update(task.true_pillars)
    cells.update(task.false_pillars)
    for pos, block in filled.items():
        if not _is_air(block):
            cells.add((int(pos[0]), int(pos[1])))
    return cells


def is_connected_st(task: TaskSpec, filled: Mapping[Coord2D, str]) -> bool:
    s_set = set(task.anchors_s)
    t_set = set(task.anchors_t)
    if not s_set or not t_set:
        return False

    traversable = _traversable_cells(task, filled)
    starts = [p for p in s_set if p in traversable]
    targets = {p for p in t_set if p in traversable}
    if not starts or not targets:
        return False

    q: deque[Coord2D] = deque(starts)
    seen: Set[Coord2D] = set(starts)
    while q:
        cur = q.popleft()
        if cur in targets:
            return True
        for nb in _neighbors4(cur):
            if nb in seen or nb not in traversable:
                continue
            seen.add(nb)
            q.append(nb)
    return False


def make_initial_state(task: TaskSpec, *, num_agents: int, max_turns: int | None = None) -> BridgeState:
    n = max(1, int(num_agents))
    s0 = task.anchors_s[0]
    t0 = task.anchors_t[0]
    positions: List[Coord2D] = []
    for idx in range(n):
        if idx == 0:
            positions.append(s0)
        elif idx == 1:
            positions.append(t0)
        else:
            positions.append(s0)

    state = BridgeState(
        turn_index=1,
        max_turns=max(1, int(max_turns or task.max_turns)),
        filled={},
        agent_positions=[(int(x), int(z)) for x, z in positions],
        vision_origins=[{(int(x), int(z))} for x, z in positions],
        known_pillars=[{} for _ in range(n)],
        inbox=[[] for _ in range(n)],
        connected=False,
        terminated=False,
    )
    state.connected = is_connected_st(task, state.filled)
    state.terminated = bool(state.connected)
    return state


def serialize_state(state: BridgeState) -> Dict[str, Any]:
    return {
        "turn_index": int(state.turn_index),
        "max_turns": int(state.max_turns),
        "filled": {_coord_key(p): str(b) for p, b in state.filled.items()},
        "agent_positions": [[int(x), int(z)] for x, z in state.agent_positions],
        "vision_origins": [
            [[int(x), int(z)] for x, z in _sorted_coords(origins)] for origins in state.vision_origins
        ],
        "known_pillars": [
            {_coord_key(p): str(v) for p, v in known.items()} for known in state.known_pillars
        ],
        "inbox": [copy.deepcopy(list(msgs)) for msgs in state.inbox],
        "connected": bool(state.connected),
        "terminated": bool(state.terminated),
    }


def deserialize_state(value: Mapping[str, Any], *, num_agents: int) -> BridgeState:
    n = max(1, int(num_agents))

    filled: Dict[Coord2D, str] = {}
    for k, v in dict(value.get("filled") or {}).items():
        pos = _decode_coord_key(str(k))
        if pos is not None:
            filled[pos] = str(v)

    positions: List[Coord2D] = []
    for raw in list(value.get("agent_positions") or []):
        p = _parse_coord(raw)
        if p is not None:
            positions.append(p)
    while len(positions) < n:
        positions.append((0, 0))
    positions = positions[:n]

    vision_origins: List[Set[Coord2D]] = []
    raw_vision = value.get("vision_origins") or []
    if isinstance(raw_vision, list):
        for raw_agent in raw_vision[:n]:
            cur_set: Set[Coord2D] = set()
            if isinstance(raw_agent, list):
                for rc in raw_agent:
                    p = _parse_coord(rc)
                    if p is not None:
                        cur_set.add(p)
            vision_origins.append(cur_set)
    while len(vision_origins) < n:
        idx = len(vision_origins)
        vision_origins.append({positions[idx]})

    known_pillars: List[Dict[Coord2D, str]] = []
    raw_known = value.get("known_pillars") or []
    if isinstance(raw_known, list):
        for raw_agent in raw_known[:n]:
            cur: Dict[Coord2D, str] = {}
            if isinstance(raw_agent, Mapping):
                for k, v in raw_agent.items():
                    p = _decode_coord_key(str(k))
                    if p is not None:
                        cur[p] = str(v)
            known_pillars.append(cur)
    while len(known_pillars) < n:
        known_pillars.append({})

    inbox: List[List[Any]] = []
    raw_inbox = value.get("inbox") or []
    if isinstance(raw_inbox, list):
        for raw_agent in raw_inbox[:n]:
            if isinstance(raw_agent, list):
                inbox.append(copy.deepcopy(list(raw_agent)))
            else:
                inbox.append([])
    while len(inbox) < n:
        inbox.append([])

    return BridgeState(
        turn_index=max(1, int(value.get("turn_index") or 1)),
        max_turns=max(1, int(value.get("max_turns") or 1)),
        filled=filled,
        agent_positions=positions,
        vision_origins=vision_origins,
        known_pillars=known_pillars,
        inbox=inbox,
        connected=bool(value.get("connected", False)),
        terminated=bool(value.get("terminated", False)),
    )


def _extract_action_obj(text: str) -> Dict[str, Any]:
    raw = _strip_markdown_fences(text)
    if not raw:
        return {}

    candidates: List[str] = [raw]
    first = raw.find("{")
    last = raw.rfind("}")
    if 0 <= first < last:
        mid = raw[first : last + 1]
        if mid not in candidates:
            candidates.append(mid)

    seen_candidates = set(candidates)
    n = len(raw)
    for start, ch in enumerate(raw):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        string_quote = ""
        escape = False
        for end in range(start, n):
            cur = raw[end]
            if in_string:
                if escape:
                    escape = False
                    continue
                if cur == "\\":
                    escape = True
                    continue
                if cur == string_quote:
                    in_string = False
                continue
            if cur in ('"', "'"):
                in_string = True
                string_quote = cur
                continue
            if cur == "{":
                depth += 1
                continue
            if cur != "}":
                continue
            depth -= 1
            if depth == 0:
                cand = raw[start : end + 1].strip()
                if cand and cand not in seen_candidates:
                    candidates.append(cand)
                    seen_candidates.add(cand)
                break
            if depth < 0:
                break

    parsed_action_objs: List[Dict[str, Any]] = []
    parsed_dicts: List[Dict[str, Any]] = []
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                if any(key in obj for key in _ACTION_KEYS):
                    parsed_action_objs.append(obj)
                else:
                    parsed_dicts.append(obj)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                if any(key in obj for key in _ACTION_KEYS):
                    parsed_action_objs.append(obj)
                else:
                    parsed_dicts.append(obj)
        except Exception:
            pass

    if parsed_action_objs:
        return parsed_action_objs[-1]
    if parsed_dicts:
        return parsed_dicts[-1]
    return {}


def _parse_comm(comm_raw: Any) -> Tuple[Any, str, int]:
    if comm_raw is None:
        return {}, "", 0
    if comm_raw == {} or comm_raw == [] or comm_raw == "":
        return {}, "", 0
    if not isinstance(comm_raw, dict):
        return {}, "", 0

    try:
        txt = json.dumps(comm_raw, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        txt = str(comm_raw)
    txt = txt.strip()
    if not txt or txt in ("{}", "[]"):
        return comm_raw, "", 0
    return comm_raw, txt, _count_tokens(txt)


def _parse_probe_list(raw_probe: Any, *, max_probe: int, width: int, height: int) -> List[Coord2D]:
    out: List[Coord2D] = []
    if not isinstance(raw_probe, list):
        return out
    for value in raw_probe:
        if len(out) >= int(max_probe):
            break
        p = _parse_coord(value)
        if p is None or not _in_bounds(p, width=width, height=height):
            continue
        out.append(p)
    return out


def _parse_fill_commands(
    raw_cmds: Any,
    *,
    allowed_blocks: Sequence[str],
    width: int,
    height: int,
    max_commands: int,
) -> Tuple[List[str], List[Tuple[int, int, int, int, str]], List[Dict[str, Any]]]:
    accepted_cmds: List[str] = []
    fills: List[Tuple[int, int, int, int, str]] = []
    rejected: List[Dict[str, Any]] = []

    allowed = {normalize_block_id(b) for b in allowed_blocks}
    if not isinstance(raw_cmds, list):
        return accepted_cmds, fills, rejected

    for raw in raw_cmds:
        line = str(raw).strip()
        if not line:
            continue
        if len(fills) >= int(max_commands):
            rejected.append({"line": line, "reason": f"exceeds max_commands={max_commands}"})
            continue

        m = _FILL_CMD_RE.match(line)
        if not m:
            rejected.append({"line": line, "reason": "cmd must be '/fill x1 z1 x2 z2 block'"})
            continue

        x1 = _parse_int_token(m.group("x1"))
        z1 = _parse_int_token(m.group("z1"))
        x2 = _parse_int_token(m.group("x2"))
        z2 = _parse_int_token(m.group("z2"))
        if None in (x1, z1, x2, z2):
            rejected.append({"line": line, "reason": "fill coords must be absolute integers"})
            continue

        block = normalize_block_id(m.group("block"))
        if block not in allowed:
            rejected.append({"line": line, "reason": f"block not allowed: {block}"})
            continue

        p1 = (int(x1), int(z1))
        p2 = (int(x2), int(z2))
        if not _in_bounds(p1, width=width, height=height) or not _in_bounds(p2, width=width, height=height):
            rejected.append({"line": line, "reason": "fill coord out of map"})
            continue

        nx1 = min(int(x1), int(x2))
        nx2 = max(int(x1), int(x2))
        nz1 = min(int(z1), int(z2))
        nz2 = max(int(z1), int(z2))
        fills.append((nx1, nz1, nx2, nz2, block))
        accepted_cmds.append(f"/fill {nx1} {nz1} {nx2} {nz2} {block}")

    return accepted_cmds, fills, rejected


def _parse_path(raw_path: Any, *, current_pos: Coord2D, width: int, height: int) -> Tuple[List[Coord2D], bool]:
    if not isinstance(raw_path, list) or not raw_path:
        return [current_pos], False

    parsed: List[Coord2D] = []
    for node in raw_path:
        p = _parse_coord(node)
        if p is None or not _in_bounds(p, width=width, height=height):
            return [current_pos], False
        parsed.append(p)

    if parsed[0] != current_pos:
        return [current_pos], False

    for prev, cur in zip(parsed[:-1], parsed[1:]):
        dx = abs(int(cur[0]) - int(prev[0]))
        dz = abs(int(cur[1]) - int(prev[1]))
        if dx > 1 or dz > 1:
            return [current_pos], False

    return parsed, True


def parse_agent_action(
    *,
    text: str,
    current_pos: Coord2D,
    allowed_blocks: Sequence[str],
    width: int,
    height: int,
    max_commands: int,
    max_probe: int,
) -> ParsedAgentAction:
    obj = _extract_action_obj(text)

    comm_obj, comm_text, comm_tokens = _parse_comm(obj.get("comm"))
    probes = _parse_probe_list(obj.get("probe"), max_probe=max_probe, width=width, height=height)
    cmds, fills, rejected_cmds = _parse_fill_commands(
        obj.get("cmds"),
        allowed_blocks=allowed_blocks,
        width=width,
        height=height,
        max_commands=max_commands,
    )
    path, path_valid = _parse_path(
        obj.get("path"),
        current_pos=current_pos,
        width=width,
        height=height,
    )

    return ParsedAgentAction(
        comm_obj=comm_obj,
        comm_text=comm_text,
        comm_tokens=int(comm_tokens),
        probes=probes,
        cmds=cmds,
        fills=fills,
        path=path,
        path_valid=bool(path_valid),
        rejected_cmds=rejected_cmds,
    )


def _apply_fill(
    *,
    filled: Dict[Coord2D, str],
    fill_cmd: Tuple[int, int, int, int, str],
    immutable_cells: Set[Coord2D],
) -> None:
    x1, z1, x2, z2, block = fill_cmd
    block_norm = normalize_block_id(block)
    for x in range(int(x1), int(x2) + 1):
        for z in range(int(z1), int(z2) + 1):
            pos = (int(x), int(z))
            if pos in immutable_cells:
                continue
            if _is_air(block_norm):
                filled.pop(pos, None)
            else:
                filled[pos] = block_norm


def _filled_cells(filled: Mapping[Coord2D, str]) -> Set[Coord2D]:
    return {(int(x), int(z)) for (x, z), block in filled.items() if not _is_air(block)}


def _count_n_adjacent(task: TaskSpec, filled_set: Set[Coord2D]) -> int:
    return sum(1 for pos in task.false_pillars if any(nb in filled_set for nb in _neighbors4(pos)))


def _count_connected_y(task: TaskSpec, filled_set: Set[Coord2D]) -> int:
    if not task.true_pillars:
        return 0

    traversable: Set[Coord2D] = set(task.anchors_s) | set(task.anchors_t) | set(task.true_pillars) | set(filled_set)
    q: deque[Coord2D] = deque((int(x), int(z)) for x, z in [*task.anchors_s, *task.anchors_t])
    seen: Set[Coord2D] = set(q)

    while q:
        cur = q.popleft()
        for nb in _neighbors4(cur):
            if nb in seen or nb not in traversable:
                continue
            seen.add(nb)
            q.append(nb)

    return sum(1 for pos in task.true_pillars if pos in seen)


def compute_visible_cells(
    origins: Iterable[Coord2D], *, view: int, width: int, height: int
) -> Set[Coord2D]:
    v = max(0, int(view))
    out: Set[Coord2D] = set()
    for ox, oz in origins:
        for dx in range(-v, v + 1):
            for dz in range(-v, v + 1):
                p = (int(ox + dx), int(oz + dz))
                if _in_bounds(p, width=width, height=height):
                    out.add(p)
    return out


def get_agent_observation(task: TaskSpec, state: BridgeState, *, agent_idx: int, view: int) -> Dict[str, Any]:
    idx = int(agent_idx)
    if idx < 0 or idx >= len(state.agent_positions):
        raise IndexError(f"invalid agent_idx={agent_idx}")

    visible = compute_visible_cells(
        state.vision_origins[idx],
        view=view,
        width=task.width,
        height=task.height,
    )

    visible_land = _sorted_coords(set(task.land_cells) & visible)
    visible_p = _sorted_coords(set(task.candidate_pillars) & visible)
    visible_anchors = [
        {"coord": [int(x), int(z)], "kind": "S"}
        for x, z in _sorted_coords(set(task.anchors_s) & visible)
    ]
    visible_anchors.extend(
        {"coord": [int(x), int(z)], "kind": "T"}
        for x, z in _sorted_coords(set(task.anchors_t) & visible)
    )

    known = state.known_pillars[idx]
    known_items = [
        {"coord": [int(x), int(z)], "type": str(known[(x, z)])}
        for x, z in _sorted_coords(known.keys())
    ]

    received_messages = copy.deepcopy(list(state.inbox[idx]))

    return {
        "turn_index": int(state.turn_index),
        "max_turns": int(state.max_turns),
        "current_pos": [int(state.agent_positions[idx][0]), int(state.agent_positions[idx][1])],
        "visible_anchors": visible_anchors,
        "visible_land_coords": [[int(x), int(z)] for x, z in visible_land],
        "visible_p_candidates": [[int(x), int(z)] for x, z in visible_p],
        "known_probe_results": known_items,
        "received_messages": received_messages,
        "connected": bool(state.connected),
        "terminated": bool(state.terminated),
    }


def apply_turn(
    *,
    task: TaskSpec,
    state: BridgeState,
    agent_outputs: Sequence[str],
    allowed_blocks_per_agent: Sequence[Sequence[str]],
    max_commands_total: int,
    view: int,
    max_probe: int = 2,
) -> TurnResult:
    n = len(state.agent_positions)
    if n <= 0:
        raise ValueError("state.agent_positions must be non-empty")

    if len(agent_outputs) < n:
        outs = list(agent_outputs) + [""] * (n - len(agent_outputs))
    else:
        outs = [str(x) for x in agent_outputs[:n]]

    limits = split_command_limits(max_commands_total=max_commands_total, num_agents=n)
    allowed_per_agent: List[List[str]] = []
    for i in range(n):
        if i < len(allowed_blocks_per_agent):
            allowed = [str(b) for b in allowed_blocks_per_agent[i]]
        else:
            allowed = []
        allowed_per_agent.append(allowed)

    if state.terminated:
        frozen_state = clone_state(state)
        frozen_filled_set = _filled_cells(frozen_state.filled)
        frozen_n_adjacent_count = _count_n_adjacent(task, frozen_filled_set)
        frozen_connected_y_count = _count_connected_y(task, frozen_filled_set)
        metrics = {
            "reward": 0.0,
            "bonus_y_connected": 0.0,
            "penalty_n_adjacent": 0.0,
            "penalty_block_cost": 0.0,
            "bonus_terminal_connect": 0.0,
            "connected_y_count": int(frozen_connected_y_count),
            "n_adjacent_count": int(frozen_n_adjacent_count),
            "y_uncovered_count": int(max(0, len(task.true_pillars) - frozen_connected_y_count)),
            "new_connected_y_count": 0,
            "new_adjacent_n_count": 0,
            "newly_placed_block_count": 0,
            "total_true_pillars": int(len(task.true_pillars)),
            "total_false_pillars": int(len(task.false_pillars)),
            "total_placeable_cells": int(
                max(1, (task.width * task.height) - len(task.land_cells) - len(task.candidate_pillars))
            ),
            "num_valid_probes": 0,
            "comm_tokens": 0,
            "connected": bool(state.connected),
            "terminated": True,
        }
        actions = [
            parse_agent_action(
                text=outs[i],
                current_pos=frozen_state.agent_positions[i],
                allowed_blocks=allowed_per_agent[i],
                width=task.width,
                height=task.height,
                max_commands=limits[i],
                max_probe=max_probe,
            )
            for i in range(n)
        ]
        return TurnResult(state=frozen_state, metrics=metrics, actions=actions)

    parsed_actions: List[ParsedAgentAction] = []
    for i in range(n):
        parsed_actions.append(
            parse_agent_action(
                text=outs[i],
                current_pos=state.agent_positions[i],
                allowed_blocks=allowed_per_agent[i],
                width=task.width,
                height=task.height,
                max_commands=limits[i],
                max_probe=max_probe,
            )
        )

    nxt = clone_state(state)
    immutable_cells = set(task.land_cells) | set(task.candidate_pillars)

    # Execute fills in agent order: A then B.
    for action in parsed_actions:
        for fill_cmd in action.fills:
            _apply_fill(filled=nxt.filled, fill_cmd=fill_cmd, immutable_cells=immutable_cells)

    # Resolve probes and local knowledge.
    true_set = set(task.true_pillars)
    false_set = set(task.false_pillars)
    for i, action in enumerate(parsed_actions):
        known = nxt.known_pillars[i]
        for pos in action.probes:
            if pos in true_set:
                known[pos] = "Y"
            elif pos in false_set:
                known[pos] = "N"
            else:
                known[pos] = "none"

    # Broadcast comm payloads to teammates.
    for src, action in enumerate(parsed_actions):
        if not action.comm_text:
            continue
        for dst in range(n):
            if dst == src:
                continue
            nxt.inbox[dst].append(copy.deepcopy(action.comm_obj))

    # Apply path movement and update fog-of-war origins.
    for i, action in enumerate(parsed_actions):
        cur = nxt.agent_positions[i]
        nxt.vision_origins[i].add(cur)
        if action.path_valid:
            for p in action.path:
                nxt.vision_origins[i].add(p)
            nxt.agent_positions[i] = action.path[-1]

    prev_filled_set = _filled_cells(state.filled)
    filled_set = _filled_cells(nxt.filled)
    prev_n_adjacent_count = _count_n_adjacent(task, prev_filled_set)
    n_adjacent_count = _count_n_adjacent(task, filled_set)
    prev_connected_y_count = _count_connected_y(task, prev_filled_set)
    connected_y_count = _count_connected_y(task, filled_set)
    y_uncovered_count = max(0, len(task.true_pillars) - connected_y_count)

    connected = is_connected_st(task, nxt.filled)

    probe_count = sum(len(action.probes) for action in parsed_actions)
    comm_tokens = sum(int(action.comm_tokens) for action in parsed_actions)
    total_y = max(1, len(task.true_pillars))
    total_n = max(1, len(task.false_pillars))
    total_placeable = max(1, (task.width * task.height) - len(task.land_cells) - len(task.candidate_pillars))

    new_connected_y_count = max(0, connected_y_count - prev_connected_y_count)
    new_adjacent_n_count = max(0, n_adjacent_count - prev_n_adjacent_count)
    newly_placed_block_count = len(filled_set - prev_filled_set)

    bonus_y_connected = (float(new_connected_y_count) / float(total_y)) * 5.0
    penalty_n = (float(new_adjacent_n_count) / float(total_n)) * 8.0
    penalty_block = (float(newly_placed_block_count) / float(total_placeable)) * 5.0
    bonus_terminal_connect = 10.0 if (connected and not state.connected) else 0.0
    reward = bonus_y_connected - penalty_n - penalty_block + bonus_terminal_connect

    nxt.turn_index = int(state.turn_index) + 1
    nxt.connected = bool(connected)
    nxt.terminated = bool(connected or (nxt.turn_index > nxt.max_turns))

    metrics = {
        "reward": reward,
        "bonus_y_connected": bonus_y_connected,
        "penalty_n_adjacent": penalty_n,
        "penalty_block_cost": penalty_block,
        "bonus_terminal_connect": bonus_terminal_connect,
        "connected_y_count": int(connected_y_count),
        "n_adjacent_count": int(n_adjacent_count),
        "y_uncovered_count": int(y_uncovered_count),
        "new_connected_y_count": int(new_connected_y_count),
        "new_adjacent_n_count": int(new_adjacent_n_count),
        "newly_placed_block_count": int(newly_placed_block_count),
        "total_true_pillars": int(len(task.true_pillars)),
        "total_false_pillars": int(len(task.false_pillars)),
        "total_placeable_cells": int(total_placeable),
        "num_valid_probes": int(probe_count),
        "comm_tokens": int(comm_tokens),
        "connected": bool(connected),
        "terminated": bool(nxt.terminated),
        "turn_index": int(state.turn_index),
    }

    return TurnResult(state=nxt, metrics=metrics, actions=parsed_actions)


def _format_feedback_text(feedback: Any) -> str:
    if feedback is None:
        return "(none)"
    txt = str(feedback).strip()
    return txt if txt else "(none)"


def build_prompt_fields(
    *,
    task: TaskSpec,
    state: BridgeState,
    agent_idx: int,
    view: int,
    allowed_blocks: Sequence[str],
    max_probe: int,
    max_commands: int,
    feedback: str,
) -> Dict[str, Any]:
    obs = get_agent_observation(task, state, agent_idx=agent_idx, view=view)
    role_name = "A" if int(agent_idx) == 0 else "B"

    return {
        "task_id": task.task_id,
        "agent_name": role_name,
        "turn_idx": int(obs["turn_index"]),
        "max_turns": int(obs["max_turns"]),
        "map_width": int(task.width),
        "map_height": int(task.height),
        "map_size": f"{task.width}x{task.height}",
        "origin": "(0,0)",
        "view": int(view),
        "current_pos": json.dumps(obs["current_pos"], ensure_ascii=False, separators=(",", ":")),
        "visible_anchors": json.dumps(obs["visible_anchors"], ensure_ascii=False, separators=(",", ":")),
        "visible_land_coords": json.dumps(obs["visible_land_coords"], ensure_ascii=False, separators=(",", ":")),
        "visible_p_candidates": json.dumps(obs["visible_p_candidates"], ensure_ascii=False, separators=(",", ":")),
        "known_probe_results": json.dumps(obs["known_probe_results"], ensure_ascii=False, separators=(",", ":")),
        "received_messages": json.dumps(obs["received_messages"], ensure_ascii=False, separators=(",", ":")),
        "available_blocks": json.dumps([normalize_block_id(b) for b in allowed_blocks], ensure_ascii=False, separators=(",", ":")),
        "max_probe": int(max_probe),
        "max_commands": int(max_commands),
        "feedback": _format_feedback_text(feedback),
    }


def render_agent_user_prompt(
    *,
    task: TaskSpec,
    state: BridgeState,
    agent_idx: int,
    view: int,
    allowed_blocks: Sequence[str],
    max_probe: int,
    max_commands: int,
    user_template: str,
    feedback: str = "",
) -> str:
    fields = build_prompt_fields(
        task=task,
        state=state,
        agent_idx=agent_idx,
        view=view,
        allowed_blocks=allowed_blocks,
        max_probe=max_probe,
        max_commands=max_commands,
        feedback=feedback,
    )
    return str(user_template or "").format(**fields).rstrip()


def build_payload(
    *,
    task: TaskSpec,
    state_before_turn: BridgeState,
    num_agents: int,
    view: int,
    max_probe: int,
    max_commands_total: int,
    allowed_blocks_agent1: Sequence[str],
    allowed_blocks_agent2: Sequence[str],
    system_prompt: str,
    user_template_single: str,
    user_template_agent1: str,
    user_template_agent2: str,
) -> Dict[str, Any]:
    return {
        "task": task_to_item(task),
        "state_before_turn": serialize_state(state_before_turn),
        "num_agents": int(num_agents),
        "view": int(view),
        "max_probe": int(max_probe),
        "max_commands_total": int(max_commands_total),
        "allowed_blocks_agent1": [str(b) for b in allowed_blocks_agent1],
        "allowed_blocks_agent2": [str(b) for b in allowed_blocks_agent2],
        "system_prompt": str(system_prompt or "").rstrip(),
        "user_template_single": str(user_template_single or "").rstrip(),
        "user_template_agent1": str(user_template_agent1 or "").rstrip(),
        "user_template_agent2": str(user_template_agent2 or "").rstrip(),
    }


def payload_to_task(payload: Mapping[str, Any]) -> TaskSpec:
    task_obj = payload.get("task") or {}
    if not isinstance(task_obj, Mapping):
        raise ValueError("payload.task must be a mapping")
    return task_from_item(task_obj)


def payload_to_state(payload: Mapping[str, Any]) -> BridgeState:
    task = payload_to_task(payload)
    num_agents = int(payload.get("num_agents") or 2)
    raw_state = payload.get("state_before_turn") or {}
    if isinstance(raw_state, Mapping):
        return deserialize_state(raw_state, num_agents=num_agents)
    return make_initial_state(task, num_agents=num_agents)


def payload_allowed_blocks(payload: Mapping[str, Any], *, num_agents: int) -> List[List[str]]:
    a1 = [str(b) for b in (payload.get("allowed_blocks_agent1") or []) if str(b).strip()]
    a2 = [str(b) for b in (payload.get("allowed_blocks_agent2") or []) if str(b).strip()]
    if not a1:
        raise ValueError("allowed_blocks_agent1 must be non-empty")
    blocks = [a1]
    if int(num_agents) >= 2:
        blocks.append(a2 if a2 else list(a1))
    while len(blocks) < int(num_agents):
        blocks.append(list(a1))
    return blocks


def render_prompts_from_payload(
    *,
    payload: Mapping[str, Any],
    state: BridgeState,
    num_agents: int,
    feedback_by_agent: Sequence[str] | None = None,
    include_system: bool = True,
) -> List[str]:
    task = payload_to_task(payload)
    n = max(1, int(num_agents))
    view = int(payload.get("view") or 3)
    max_probe = int(payload.get("max_probe") or 2)
    max_commands_total = int(payload.get("max_commands_total") or 40)
    limits = split_command_limits(max_commands_total=max_commands_total, num_agents=n)
    allowed = payload_allowed_blocks(payload, num_agents=n)

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
            allowed_blocks=allowed[idx],
            max_probe=max_probe,
            max_commands=limits[idx],
            user_template=tmpl,
            feedback=feedback_arr[idx],
        )
        if include_system and system_prompt:
            prompts.append(system_prompt + "\n\n" + user_prompt)
        else:
            prompts.append(user_prompt)
    return prompts


def transition_payload(
    *,
    payload: Mapping[str, Any],
    agent_completions: Sequence[str],
    num_agents: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[ParsedAgentAction]]:
    task = payload_to_task(payload)
    state = payload_to_state(payload)
    n = max(1, int(num_agents))
    view = int(payload.get("view") or 3)
    max_probe = int(payload.get("max_probe") or 2)
    max_commands_total = int(payload.get("max_commands_total") or 40)
    allowed = payload_allowed_blocks(payload, num_agents=n)

    result = apply_turn(
        task=task,
        state=state,
        agent_outputs=[str(x) for x in agent_completions],
        allowed_blocks_per_agent=allowed,
        max_commands_total=max_commands_total,
        view=view,
        max_probe=max_probe,
    )

    next_payload = dict(payload)
    next_payload["state_before_turn"] = serialize_state(result.state)
    return next_payload, dict(result.metrics), list(result.actions)
