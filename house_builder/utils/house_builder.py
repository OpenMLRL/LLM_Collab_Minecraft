from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_LEADING_LIST_RE = re.compile(r"^\s*(?:[-*]+|\d+[.)])\s*")


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


def extract_command_lines(text: str) -> List[str]:
    raw = _strip_markdown_fences(text)
    lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        s = _LEADING_LIST_RE.sub("", s).strip()
        if not s:
            continue
        lines.append(s)
    return lines


def _parse_int_token(tok: str) -> int | None:
    t = (tok or "").strip()
    if not t:
        return None
    if t.startswith(("~", "^")):
        return None
    try:
        return int(t)
    except ValueError:
        return None


def validate_and_normalize_mc_commands(
    *,
    lines: List[str],
    allowed_blocks: List[str],
    world_bbox_from: List[int],
    world_bbox_to: List[int],
    max_commands: int,
    resource_limits: Dict[str, int] | None = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    allowed = {normalize_block_id(b) for b in allowed_blocks}
    allowed.add("air")
    norm_limits: Dict[str, int] = {}
    if resource_limits:
        for k, v in resource_limits.items():
            try:
                limit_val = int(v)
            except Exception:
                continue
            if limit_val < 0:
                continue
            norm_limits[normalize_block_id(str(k))] = limit_val
    resource_used: Dict[str, int] = {}

    min_x = min(world_bbox_from[0], world_bbox_to[0])
    max_x = max(world_bbox_from[0], world_bbox_to[0])
    min_y = min(world_bbox_from[1], world_bbox_to[1])
    max_y = max(world_bbox_from[1], world_bbox_to[1])
    min_z = min(world_bbox_from[2], world_bbox_to[2])
    max_z = max(world_bbox_from[2], world_bbox_to[2])

    accepted: List[str] = []
    rejected: List[Dict[str, Any]] = []

    def _in_bbox(x: int, y: int, z: int) -> bool:
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z)

    for line in lines:
        if len(accepted) >= int(max_commands):
            rejected.append({"line": line, "reason": f"exceeds max_commands={max_commands}"})
            continue
        stripped = (line or "").strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            stripped = stripped[1:].lstrip()
        parts = stripped.split()
        if not parts:
            continue
        cmd = parts[0].lower()

        if cmd == "kill":
            kill_cmd = "/kill"
            if len(parts) > 1:
                kill_cmd = kill_cmd + " " + " ".join(parts[1:])
            accepted.append(kill_cmd)
            continue

        if cmd != "fill":
            rejected.append({"line": line, "reason": f"unsupported command: {cmd}"})
            continue

        if len(parts) < 8:
            rejected.append({"line": line, "reason": "fill needs: /fill x1 y1 z1 x2 y2 z2 block"})
            continue
        x1 = _parse_int_token(parts[1])
        y1 = _parse_int_token(parts[2])
        z1 = _parse_int_token(parts[3])
        x2 = _parse_int_token(parts[4])
        y2 = _parse_int_token(parts[5])
        z2 = _parse_int_token(parts[6])
        if None in (x1, y1, z1, x2, y2, z2):
            rejected.append({"line": line, "reason": "fill coords must be absolute integers (no ~)"})
            continue
        block = normalize_block_id(parts[7])
        if block not in allowed:
            rejected.append({"line": line, "reason": f"block not allowed: {block}"})
            continue
        if not (_in_bbox(int(x1), int(y1), int(z1)) and _in_bbox(int(x2), int(y2), int(z2))):
            rejected.append({"line": line, "reason": "fill coord out of bbox"})
            continue
        if len(parts) > 8:
            rejected.append({"line": line, "reason": "fill modes (replace/keep/...) not allowed"})
            continue
        if norm_limits and block not in ("air", "cave_air", "void_air"):
            limit_val = norm_limits.get(block)
            if limit_val is not None:
                dx = abs(int(x2) - int(x1)) + 1
                dy = abs(int(y2) - int(y1)) + 1
                dz = abs(int(z2) - int(z1)) + 1
                volume = int(dx * dy * dz)
                used = resource_used.get(block, 0)
                if used + volume > limit_val:
                    rejected.append({"line": line, "reason": f"exceeds resource limit for {block}: {used + volume}>{limit_val}"})
                    continue
                resource_used[block] = used + volume
        accepted.append(f"/fill {int(x1)} {int(y1)} {int(z1)} {int(x2)} {int(y2)} {int(z2)} {block}")

    return accepted, rejected


def simulate_commands_to_scan_blocks(
    *, commands: List[str], world_bbox_from: List[int], world_bbox_to: List[int]
) -> List[Dict[str, Any]]:
    min_x = min(world_bbox_from[0], world_bbox_to[0])
    max_x = max(world_bbox_from[0], world_bbox_to[0])
    min_y = min(world_bbox_from[1], world_bbox_to[1])
    max_y = max(world_bbox_from[1], world_bbox_to[1])
    min_z = min(world_bbox_from[2], world_bbox_to[2])
    max_z = max(world_bbox_from[2], world_bbox_to[2])

    state: Dict[Tuple[int, int, int], str] = {}

    def _set(x: int, y: int, z: int, block: str) -> None:
        state[(x, y, z)] = normalize_block_id(block)

    for cmd in commands:
        stripped = (cmd or "").strip()
        if stripped.startswith("/"):
            stripped = stripped[1:].lstrip()
        parts = stripped.split()
        if not parts:
            continue
        name = parts[0].lower()
        if name != "fill" or len(parts) < 8:
            continue
        x1 = int(parts[1])
        y1 = int(parts[2])
        z1 = int(parts[3])
        x2 = int(parts[4])
        y2 = int(parts[5])
        z2 = int(parts[6])
        block = parts[7]
        fx1 = min(x1, x2)
        fx2 = max(x1, x2)
        fy1 = min(y1, y2)
        fy2 = max(y1, y2)
        fz1 = min(z1, z2)
        fz2 = max(z1, z2)
        for yy in range(fy1, fy2 + 1):
            for xx in range(fx1, fx2 + 1):
                for zz in range(fz1, fz2 + 1):
                    _set(xx, yy, zz, block)

    blocks: List[Dict[str, Any]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                blocks.append({"pos": [x, y, z], "name": state.get((x, y, z), "air")})
    return blocks


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bbox(bbox_obj: Any) -> Tuple[List[int], List[int]]:
    if not isinstance(bbox_obj, dict):
        raise ValueError("task.bbox must be an object with from/to")
    bfrom = bbox_obj.get("from")
    bto = bbox_obj.get("to")
    if not (isinstance(bfrom, list) and isinstance(bto, list) and len(bfrom) == 3 and len(bto) == 3):
        raise ValueError("task.bbox.from/to must be 3-element lists")
    return [int(bfrom[0]), int(bfrom[1]), int(bfrom[2])], [int(bto[0]), int(bto[1]), int(bto[2])]


def _parse_layers(target_spec: Dict[str, Any]) -> Dict[int, List[str]]:
    layers_obj = target_spec.get("layers")
    if layers_obj is None:
        raise ValueError("target_spec.layers is required")

    layers: Dict[int, List[str]] = {}

    def _normalize_rows(rows_obj: Any, *, y: int) -> List[str]:
        if not isinstance(rows_obj, list) or not rows_obj:
            raise ValueError(f"target_spec.layers rows must be a non-empty list (y={y})")
        return [str(r) for r in rows_obj]

    if isinstance(layers_obj, dict):
        for y_key, rows_obj in layers_obj.items():
            y = int(str(y_key).strip())
            if y in layers:
                raise ValueError(f"duplicate layer y={y}")
            layers[y] = _normalize_rows(rows_obj, y=y)
        return layers

    if isinstance(layers_obj, list):
        for entry in layers_obj:
            if not isinstance(entry, dict):
                raise ValueError("target_spec.layers list entries must be objects")
            if "y" not in entry:
                raise ValueError("target_spec.layers entry missing 'y'")
            y = int(entry.get("y"))
            if y in layers:
                raise ValueError(f"duplicate layer y={y}")
            layers[y] = _normalize_rows(entry.get("rows"), y=y)
        return layers

    raise ValueError("target_spec.layers must be a list or mapping")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    local_bbox_from: List[int]
    local_bbox_to: List[int]
    palette: Dict[str, str]
    layers_by_y: Dict[int, List[str]]


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
        raise ValueError(f"dataset.json_path must contain an object or list, got {type(raw)}")

    tasks: List[TaskSpec] = []
    for idx, task_obj in enumerate(task_objs, start=1):
        if not isinstance(task_obj, dict):
            raise ValueError(f"Task entry {idx} must be an object")
        task_id = str(task_obj.get("task_id") or f"house_builder_{idx:04d}")

        palette_raw = task_obj.get("palette")
        if not isinstance(palette_raw, dict) or not palette_raw:
            raise ValueError(f"{task_id}: palette must be a non-empty mapping")
        palette: Dict[str, str] = {}
        for key, value in palette_raw.items():
            k = str(key)
            if len(k) != 1:
                raise ValueError(f"{task_id}: palette key must be a single character, got {k!r}")
            v = str(value).strip()
            if not v:
                raise ValueError(f"{task_id}: palette value for {k!r} is empty")
            palette[k] = v

        target_spec = task_obj.get("target_spec") or {}
        if not isinstance(target_spec, dict):
            raise ValueError(f"{task_id}: target_spec must be an object")
        layers_by_y = _parse_layers(target_spec)
        if not layers_by_y:
            raise ValueError(f"{task_id}: target_spec.layers is empty")

        width: int | None = None
        depth: int | None = None
        for y, rows in list(layers_by_y.items()):
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{task_id}: layer y={y} rows must be a non-empty list")
            normalized_rows: List[str] = []
            for row in rows:
                row_str = str(row)
                if width is None:
                    width = len(row_str)
                if len(row_str) != width:
                    raise ValueError(f"{task_id}: row width mismatch at y={y}")
                for ch in row_str:
                    if ch not in palette:
                        raise ValueError(f"{task_id}: unknown palette key {ch!r} at y={y}")
                normalized_rows.append(row_str)
            if depth is None:
                depth = len(normalized_rows)
            if len(normalized_rows) != depth:
                raise ValueError(f"{task_id}: row count mismatch at y={y}")
            layers_by_y[y] = normalized_rows

        width = width or 0
        depth = depth or 0
        layer_ys = sorted(layers_by_y)
        min_layer_y = layer_ys[0]
        max_layer_y = layer_ys[-1]

        size_hint = target_spec.get("size")
        size_x = size_y = size_z = None
        if isinstance(size_hint, list) and len(size_hint) == 3:
            try:
                size_x = int(size_hint[0])
                size_y = int(size_hint[1])
                size_z = int(size_hint[2])
            except (TypeError, ValueError) as e:
                raise ValueError(f"{task_id}: target_spec.size must be 3 integers") from e

        bbox_obj = task_obj.get("bbox")
        if bbox_obj is not None:
            local_from, local_to = _parse_bbox(bbox_obj)
            min_x = min(local_from[0], local_to[0])
            max_x = max(local_from[0], local_to[0])
            min_y = min(local_from[1], local_to[1])
            max_y = max(local_from[1], local_to[1])
            min_z = min(local_from[2], local_to[2])
            max_z = max(local_from[2], local_to[2])
            bbox_width = max_x - min_x + 1
            bbox_height = max_y - min_y + 1
            bbox_depth = max_z - min_z + 1
            if width != bbox_width or depth != bbox_depth:
                raise ValueError(
                    f"{task_id}: bbox size {bbox_width}x{bbox_height}x{bbox_depth} "
                    f"does not match layers {width}x{len(layer_ys)}x{depth}"
                )
            if size_x is not None and size_x != bbox_width:
                raise ValueError(f"{task_id}: size.x mismatch (size={size_x}, bbox={bbox_width})")
            if size_y is not None and size_y != bbox_height:
                raise ValueError(f"{task_id}: size.y mismatch (size={size_y}, bbox={bbox_height})")
            if size_z is not None and size_z != bbox_depth:
                raise ValueError(f"{task_id}: size.z mismatch (size={size_z}, bbox={bbox_depth})")
            if min_layer_y < min_y or max_layer_y > max_y:
                raise ValueError(f"{task_id}: layer y out of bbox range")
            for y in range(min_y, max_y + 1):
                if y not in layers_by_y:
                    raise ValueError(f"{task_id}: missing layer for y={y} within bbox")
            local_bbox_from = [min_x, min_y, min_z]
            local_bbox_to = [max_x, max_y, max_z]
        else:
            min_x = 0
            min_z = 0
            min_y = min_layer_y
            max_y = max_layer_y
            if size_x is not None and size_x != width:
                raise ValueError(f"{task_id}: size.x mismatch (size={size_x}, layers={width})")
            if size_y is not None and size_y != (max_y - min_y + 1):
                raise ValueError(
                    f"{task_id}: size.y mismatch (size={size_y}, layers={max_y - min_y + 1})"
                )
            if size_z is not None and size_z != depth:
                raise ValueError(f"{task_id}: size.z mismatch (size={size_z}, layers={depth})")
            for y in range(min_y, max_y + 1):
                if y not in layers_by_y:
                    raise ValueError(f"{task_id}: missing layer for y={y}")
            local_bbox_from = [min_x, min_y, min_z]
            local_bbox_to = [max(0, width - 1), max_y, max(0, depth - 1)]

        tasks.append(
            TaskSpec(
                task_id=task_id,
                local_bbox_from=local_bbox_from,
                local_bbox_to=local_bbox_to,
                palette=palette,
                layers_by_y=layers_by_y,
            )
        )

    return tasks


def build_expected_map(task: TaskSpec) -> Dict[Tuple[int, int, int], str]:
    palette_norm = {k: normalize_block_id(v) for k, v in task.palette.items()}

    min_lx = min(task.local_bbox_from[0], task.local_bbox_to[0])
    min_ly = min(task.local_bbox_from[1], task.local_bbox_to[1])
    max_ly = max(task.local_bbox_from[1], task.local_bbox_to[1])
    min_lz = min(task.local_bbox_from[2], task.local_bbox_to[2])
    max_lz = max(task.local_bbox_from[2], task.local_bbox_to[2])

    expected: Dict[Tuple[int, int, int], str] = {}
    for ly in range(min_ly, max_ly + 1):
        rows = task.layers_by_y.get(ly)
        if rows is None:
            raise ValueError(f"{task.task_id}: missing layer y={ly}")
        for rz, row in enumerate(rows):
            lz = min_lz + rz
            for rx, ch in enumerate(row):
                lx = min_lx + rx
                block = palette_norm.get(ch)
                if block is None:
                    raise ValueError(f"{task.task_id}: unknown palette key {ch!r}")
                expected[(lx, ly, lz)] = block

    return expected


def count_expected_blocks(task: TaskSpec) -> Dict[str, int]:
    expected_map = build_expected_map(task)
    counts: Dict[str, int] = {}
    for block in expected_map.values():
        block_norm = normalize_block_id(block)
        counts[block_norm] = counts.get(block_norm, 0) + 1
    return counts


def compute_resource_limits(task: TaskSpec, *, num_agents: int) -> Dict[str, int]:
    n = max(1, int(num_agents))
    counts = count_expected_blocks(task)
    limits: Dict[str, int] = {}
    palette_blocks = {normalize_block_id(v) for v in task.palette.values()}
    for block in sorted(counts.keys() | palette_blocks):
        if block in ("air", "cave_air", "void_air"):
            continue
        count = int(counts.get(block, 0))
        limit_val = (count + n - 1) // n + (count // (n * 2))
        limits[block] = int(limit_val)
    return limits


def score_house_builder(*, task: TaskSpec, world_scan_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_map = build_expected_map(task)

    observed: Dict[Tuple[int, int, int], str] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        observed[(int(pos[0]), int(pos[1]), int(pos[2]))] = normalize_block_id(str(name)) if name is not None else "air"

    def _is_air(block_id: str) -> bool:
        return normalize_block_id(block_id) in ("air", "cave_air", "void_air")

    correct = 0
    total = 0
    expected_non_air = 0
    intersection = 0
    for pos, expected in expected_map.items():
        total += 1
        obs = observed.get(pos, "air")
        if obs == expected:
            correct += 1
        if not _is_air(expected):
            expected_non_air += 1
            if not _is_air(obs):
                intersection += 1

    observed_non_air = sum(1 for v in observed.values() if not _is_air(v))
    union = expected_non_air + observed_non_air - intersection
    iou = (intersection / union) if union > 0 else 0.0

    score_match = (correct / total) if total else 0.0
    return {
        "match_correct": correct,
        "match_total": total,
        "score_match": score_match,
        "score_mean": score_match,
        "iou": iou,
    }


def rows_to_rects(
    *,
    rows: List[str],
    palette: Dict[str, str],
    min_x: int,
    min_z: int,
) -> List[Tuple[int, int, int, int, str]]:
    if not rows:
        return []
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("row width mismatch while building rectangles")

    rects: List[Tuple[int, int, int, int, str]] = []
    active: Dict[Tuple[int, int, str], List[int | str]] = {}

    for z_idx, row in enumerate(rows):
        runs: List[Tuple[int, int, str]] = []
        start = 0
        cur = row[0]
        for x in range(1, width + 1):
            if x == width or row[x] != cur:
                runs.append((start, x - 1, cur))
                start = x
                if x < width:
                    cur = row[x]

        new_active: Dict[Tuple[int, int, str], List[int | str]] = {}
        for x1, x2, ch in runs:
            block = palette.get(ch)
            if block is None:
                raise ValueError(f"unknown palette key {ch!r} in layer")
            key = (x1, x2, block)
            if key in active:
                rect = active.pop(key)
                rect[3] = min_z + z_idx
                new_active[key] = rect
            else:
                rect = [min_x + x1, min_z + z_idx, min_x + x2, min_z + z_idx, block]
                new_active[key] = rect

        for rect in active.values():
            rects.append((rect[0], rect[1], rect[2], rect[3], str(rect[4])))
        active = new_active

    for rect in active.values():
        rects.append((rect[0], rect[1], rect[2], rect[3], str(rect[4])))

    return rects


def format_layers_text(
    task: TaskSpec,
    *,
    world_from: List[int] | None = None,
    include_air: bool = False,
) -> str:
    min_y = min(task.local_bbox_from[1], task.local_bbox_to[1])
    max_y = max(task.local_bbox_from[1], task.local_bbox_to[1])
    min_x = min(task.local_bbox_from[0], task.local_bbox_to[0])
    min_z = min(task.local_bbox_from[2], task.local_bbox_to[2])
    if world_from is None:
        offset_x = 0
        offset_y = 0
        offset_z = 0
    else:
        offset_x = int(world_from[0]) - min_x
        offset_y = int(world_from[1]) - min_y
        offset_z = int(world_from[2]) - min_z

    lines: List[str] = []
    for y in range(min_y, max_y + 1):
        rows = task.layers_by_y.get(y)
        if rows is None:
            raise ValueError(f"{task.task_id}: missing layer y={y}")
        rects = rows_to_rects(rows=rows, palette=task.palette, min_x=min_x, min_z=min_z)
        if not include_air:
            rects = [r for r in rects if normalize_block_id(r[4]) not in ("air", "cave_air", "void_air")]
        y_abs = y + offset_y
        rect_parts = [
            f"({x1 + offset_x}, {y_abs}, {z1 + offset_z}, {x2 + offset_x}, {y_abs}, {z2 + offset_z} {block})"
            for x1, z1, x2, z2, block in rects
        ]
        lines.append(f"y={y_abs}: {{{', '.join(rect_parts)}}}")
    return "\n".join(lines)


def unique_block_list(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def legend_lines(palette: Dict[str, str]) -> str:
    return "\n".join(f"{k} = {v}" for k, v in palette.items())
