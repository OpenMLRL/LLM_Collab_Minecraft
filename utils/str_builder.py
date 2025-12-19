from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


_LEADING_LIST_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


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
) -> Tuple[List[str], List[Dict[str, Any]]]:
    allowed = {normalize_block_id(b) for b in allowed_blocks}
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

        if cmd == "setblock":
            if len(parts) < 5:
                rejected.append({"line": line, "reason": "setblock needs: /setblock x y z block"})
                continue
            x = _parse_int_token(parts[1])
            y = _parse_int_token(parts[2])
            z = _parse_int_token(parts[3])
            if x is None or y is None or z is None:
                rejected.append({"line": line, "reason": "setblock coords must be absolute integers (no ~)"})
                continue
            block = normalize_block_id(parts[4])
            if block not in allowed:
                rejected.append({"line": line, "reason": f"block not allowed: {block}"})
                continue
            if not _in_bbox(x, y, z):
                rejected.append({"line": line, "reason": "setblock coord out of bbox"})
                continue
            accepted.append(f"/setblock {x} {y} {z} {block}")
            continue

        if cmd == "fill":
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
            accepted.append(f"/fill {int(x1)} {int(y1)} {int(z1)} {int(x2)} {int(y2)} {int(z2)} {block}")
            continue

        rejected.append({"line": line, "reason": f"unsupported command: {cmd}"})

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
        if name == "setblock" and len(parts) >= 5:
            x = int(parts[1])
            y = int(parts[2])
            z = int(parts[3])
            block = parts[4]
            _set(x, y, z, block)
            continue
        if name == "fill" and len(parts) >= 8:
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
            continue

    blocks: List[Dict[str, Any]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                blocks.append({"pos": [x, y, z], "name": state.get((x, y, z), "air")})
    return blocks


_FONT_5X7: Dict[str, List[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def render_string_mask_rows(text: str, *, spacing: int) -> List[str]:
    text = str(text or "").strip().upper()
    if not text:
        return [""] * 7

    height = 7
    width = 5
    rows = [""] * height
    for i, ch in enumerate(text):
        glyph = _FONT_5X7.get(ch) or ["0" * width] * height
        if len(glyph) != height or any(len(r) != width for r in glyph):
            raise ValueError(f"Invalid glyph for {ch!r}")
        for r in range(height):
            rows[r] += "".join("#" if c == "1" else "." for c in glyph[r])
        if i != len(text) - 1:
            for r in range(height):
                rows[r] += "." * int(spacing)
    return rows


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    csv_row_index: int
    text: str
    difficulty: int
    local_bbox_from: List[int]
    local_bbox_to: List[int]
    target_rows_topdown: List[str]


def load_tasks_from_csv(csv_path: str, *, spacing: int, local_z: int) -> List[TaskSpec]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"dataset.csv_path not found: {path}")

    tasks: List[TaskSpec] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            text = str(row.get("string") or "").strip()
            if not text:
                continue
            try:
                difficulty = int(str(row.get("difficulty") or "").strip() or "0")
            except ValueError:
                difficulty = 0

            target_rows = render_string_mask_rows(text, spacing=spacing)
            height = len(target_rows)
            width = len(target_rows[0]) if height else 0
            if any(len(r) != width for r in target_rows):
                raise ValueError(f"Inconsistent target row widths for {text!r}")

            local_from = [0, 0, int(local_z)]
            local_to = [max(0, width - 1), max(0, height - 1), int(local_z)]
            task_id = f"str_builder_{idx:04d}"
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    csv_row_index=idx,
                    text=text,
                    difficulty=difficulty,
                    local_bbox_from=local_from,
                    local_bbox_to=local_to,
                    target_rows_topdown=target_rows,
                )
            )
    return tasks


def _is_air(block_name: str | None) -> bool:
    if block_name is None:
        return True
    b = normalize_block_id(str(block_name))
    return b in {"air", "cave_air", "void_air"}


def block_to_color_initial(block_id: str) -> str:
    s = normalize_block_id(str(block_id or "")).lower()
    if not s:
        return "#"
    color = s.split("_", 1)[0]
    return (color[:1] or s[:1] or "#").upper()


def render_target_ascii(task: TaskSpec) -> str:
    return "\n".join(str(r) for r in (task.target_rows_topdown or []))


def render_progress_overlay_ascii(
    task: TaskSpec,
    world_scan_blocks: List[Dict[str, Any]],
    *,
    empty_char: str = ".",
    missing_target_char: str = "#",
) -> str:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0

    ec = (str(empty_char or ".")[:1]) or "."
    mc = (str(missing_target_char or "#")[:1]) or "#"

    obs_block: Dict[Tuple[int, int], str] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        x, y = int(pos[0]), int(pos[1])
        if _is_air(None if name is None else str(name)):
            continue
        obs_block[(x, y)] = normalize_block_id(str(name))

    lines: List[str] = []
    for r, row in enumerate(task.target_rows_topdown):
        out: List[str] = []
        for x in range(width):
            wx = task.local_bbox_from[0] + x
            wy = task.local_bbox_from[1] + (height - 1 - r)
            placed = obs_block.get((wx, wy))
            if placed is not None:
                out.append(block_to_color_initial(placed))
            elif x < len(row) and row[x] == "#":
                out.append(mc)
            else:
                out.append(ec)
        lines.append("".join(out))
    return "\n".join(lines)


def _chamfer_distance(points_a: List[Tuple[int, int]], points_b: List[Tuple[int, int]]) -> float:
    if not points_a or not points_b:
        return float("inf")

    def _mean_nn(src: List[Tuple[int, int]], dst: List[Tuple[int, int]]) -> float:
        total = 0.0
        for ax, ay in src:
            best = float("inf")
            for bx, by in dst:
                d = math.hypot(ax - bx, ay - by)
                if d < best:
                    best = d
                    if best == 0.0:
                        break
            total += best
        return total / float(len(src))

    return 0.5 * (_mean_nn(points_a, points_b) + _mean_nn(points_b, points_a))


def _count_components_8(points: set[Tuple[int, int]]) -> int:
    if not points:
        return 0
    remaining = set(points)
    comps = 0
    neigh = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        remaining.remove(start)
        while stack:
            x, y = stack.pop()
            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if (nx, ny) in remaining:
                    remaining.remove((nx, ny))
                    stack.append((nx, ny))
        comps += 1
    return comps


def score_str_builder(
    *, task: TaskSpec, world_origin: List[int], world_scan_blocks: List[Dict[str, Any]], chamfer_sigma: float
) -> Dict[str, Any]:
    if chamfer_sigma <= 0:
        raise ValueError("chamfer_sigma must be > 0")

    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0

    t_points: List[Tuple[int, int]] = []
    for r, row in enumerate(task.target_rows_topdown):
        if len(row) != width:
            raise ValueError("target_rows_topdown has inconsistent row widths")
        for x, ch in enumerate(row):
            if ch != "#":
                continue
            lx = task.local_bbox_from[0] + x
            ly = task.local_bbox_from[1] + (height - 1 - r)
            wx = world_origin[0] + lx
            wy = world_origin[1] + ly
            t_points.append((wx, wy))
    t_set = set(t_points)

    obs_block: Dict[Tuple[int, int], str] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        x, y = int(pos[0]), int(pos[1])
        if _is_air(None if name is None else str(name)):
            continue
        obs_block[(x, y)] = normalize_block_id(str(name))

    o_set = set(obs_block.keys())
    inter = len(t_set & o_set)
    union = len(t_set | o_set)
    iou = (inter / union) if union else 1.0

    cd = _chamfer_distance(list(t_set), list(o_set))
    shape_score = 0.0 if not math.isfinite(cd) else math.exp(-cd / float(chamfer_sigma))
    score_shape_overlap = 0.5 * shape_score + 0.5 * iou

    components = _count_components_8(o_set)
    expected = max(0, len(task.text))
    components_ratio = (components / expected) if expected else 0.0
    score_components = min(components_ratio, 1.0) if expected else 0.0

    total_pairs = 0
    diff_pairs = 0
    for x, y in o_set:
        for nx, ny in ((x + 1, y), (x, y + 1)):
            if (nx, ny) not in o_set:
                continue
            total_pairs += 1
            if obs_block.get((x, y)) != obs_block.get((nx, ny)):
                diff_pairs += 1
    adj_diff_ratio = (diff_pairs / total_pairs) if total_pairs else 0.0
    adj_all_different = bool(total_pairs > 0 and diff_pairs == total_pairs)

    score_mean = (score_shape_overlap + score_components + adj_diff_ratio) / 3.0

    return {
        "target_blocks": len(t_set),
        "built_blocks": len(o_set),
        "overlap_iou": iou,
        "chamfer_distance": None if not math.isfinite(cd) else cd,
        "shape_score": shape_score,
        "score_shape_overlap": score_shape_overlap,
        "components_8": components,
        "expected_components": expected,
        "components_ratio": components_ratio,
        "score_components": score_components,
        "adjacent_pairs_4": total_pairs,
        "adjacent_pairs_4_diff_material": diff_pairs,
        "adjacent_diff_ratio": adj_diff_ratio,
        "adjacent_all_different": adj_all_different,
        "score_material_adjacent": adj_diff_ratio,
        "score_mean": score_mean,
    }
