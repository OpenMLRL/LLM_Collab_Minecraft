from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
) -> Tuple[List[str], List[Dict[str, Any]]]:
    allowed = {normalize_block_id(b) for b in allowed_blocks}
    allowed.add("air")
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

        if cmd != "setblock":
            rejected.append({"line": line, "reason": f"unsupported command: {cmd}"})
            continue

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

    blocks: List[Dict[str, Any]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                blocks.append({"pos": [x, y, z], "name": state.get((x, y, z), "air")})
    return blocks


_FONT_5X5: Dict[str, List[str]] = {
    "A": ["01110", "10001", "11111", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000"],
    "G": ["01111", "10000", "10111", "10001", "01110"],
    "H": ["10001", "10001", "11111", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "11100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001"],
    "O": ["01110", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "11110", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10011", "01111"],
    "R": ["11110", "10001", "11110", "10010", "10001"],
    "S": ["01111", "10000", "01110", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10101", "11011", "10001"],
    "X": ["10001", "01010", "00100", "01010", "10001"],
    "Y": ["10001", "01010", "00100", "00100", "00100"],
    "Z": ["11111", "00010", "00100", "01000", "11111"],
}


def render_string_mask_rows(text: str, *, spacing: int) -> List[str]:
    text = str(text or "").strip().upper()
    if not text:
        return [""] * 5

    height = 5
    width = 5
    rows = [""] * height
    for i, ch in enumerate(text):
        glyph = _FONT_5X5.get(ch) or ["0" * width] * height
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


_COLOR_PREFIXES = (
    "light_blue",
    "light_gray",
    "orange",
    "magenta",
    "yellow",
    "lime",
    "pink",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
    "white",
    "gray",
)


def block_to_color_key(block_id: str) -> str:
    s = normalize_block_id(str(block_id or "")).lower()
    if not s:
        return ""
    for color in _COLOR_PREFIXES:
        if s == color or s.startswith(color + "_"):
            return color
    return s


def render_target_ascii(task: TaskSpec) -> str:
    return "\n".join(str(r) for r in (task.target_rows_topdown or []))


def get_target_positions(task: TaskSpec) -> List[Tuple[int, int, int]]:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0
    coords: List[Tuple[int, int, int]] = []
    for r, row in enumerate(task.target_rows_topdown):
        if len(row) != width:
            raise ValueError("target_rows_topdown has inconsistent row widths")
        for x, ch in enumerate(row):
            if ch != "#":
                continue
            wx = task.local_bbox_from[0] + x
            wy = task.local_bbox_from[1] + (height - 1 - r)
            wz = task.local_bbox_from[2]
            coords.append((int(wx), int(wy), int(wz)))
    return coords


def blocks_to_map(blocks: List[Dict[str, Any]]) -> Dict[Tuple[int, int, int], str]:
    obs: Dict[Tuple[int, int, int], str] = {}
    for b in blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        obs[(int(pos[0]), int(pos[1]), int(pos[2]))] = normalize_block_id(str(name or "air"))
    return obs


def _normalize_palette(palette: List[str]) -> List[str]:
    out: List[str] = []
    for b in palette:
        s = normalize_block_id(str(b or "")).strip()
        if s:
            out.append(s)
    return out


def _agent_index_for_local(lx: int, ly: int, num_agents: int) -> int:
    n = max(1, int(num_agents))
    return int((lx + ly) % n)


def build_target_color_map(
    *,
    task: TaskSpec,
    allowed_blocks_per_agent: List[List[str]],
    num_agents: int,
) -> Tuple[Dict[Tuple[int, int, int], str], Dict[Tuple[int, int, int], int]]:
    n = max(1, int(num_agents))
    palettes: List[List[str]] = []
    for i in range(n):
        palette = allowed_blocks_per_agent[i] if i < len(allowed_blocks_per_agent) else []
        palette = _normalize_palette(palette)
        if not palette:
            palette = ["white_concrete"]
        palettes.append(palette)

    expected: Dict[Tuple[int, int, int], str] = {}
    owners: Dict[Tuple[int, int, int], int] = {}

    positions = get_target_positions(task)
    positions.sort(key=lambda p: (p[1], p[0], p[2]))

    for pos in positions:
        lx = pos[0] - task.local_bbox_from[0]
        ly = pos[1] - task.local_bbox_from[1]
        agent_idx = _agent_index_for_local(lx, ly, n)
        palette = palettes[agent_idx]
        start_idx = (lx + ly) % len(palette)

        neighbor_colors = set()
        neighbors = [
            (pos[0] - 1, pos[1], pos[2]),
            (pos[0] + 1, pos[1], pos[2]),
            (pos[0], pos[1] - 1, pos[2]),
            (pos[0], pos[1] + 1, pos[2]),
        ]
        for npos in neighbors:
            if npos in expected:
                neighbor_colors.add(expected[npos])

        chosen = None
        for offset in range(len(palette)):
            color = palette[(start_idx + offset) % len(palette)]
            if color not in neighbor_colors:
                chosen = color
                break
        if chosen is None:
            chosen = palette[start_idx]

        expected[pos] = normalize_block_id(chosen)
        owners[pos] = agent_idx

    return expected, owners


def score_str_builder(
    *,
    task: TaskSpec,
    world_scan_blocks: List[Dict[str, Any]],
    expected_map: Dict[Tuple[int, int, int], str],
    allowed_blocks_per_agent: List[List[str]] | None = None,
) -> Dict[str, Any]:
    obs_map = blocks_to_map(world_scan_blocks)
    target_positions = set(expected_map.keys())
    target_total = len(target_positions)

    covered = 0
    missing = 0
    for pos in expected_map:
        observed = normalize_block_id(obs_map.get(pos, "air"))
        if _is_air(observed):
            missing += 1
        else:
            covered += 1

    extra = 0
    for pos, observed in obs_map.items():
        if pos in target_positions:
            continue
        if not _is_air(observed):
            extra += 1

    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0
    total_cells = max(0, height * width)
    background_total = max(0, total_cells - target_total)

    coverage_ratio = (covered / target_total) if target_total else 0.0
    extra_ratio = (extra / background_total) if background_total > 0 else 0.0

    color_map: Dict[Tuple[int, int, int], str] = {}
    for pos, block_id in obs_map.items():
        if _is_air(block_id):
            continue
        color_map[pos] = block_to_color_key(block_id)

    adjacent_same_color_pairs = 0
    adjacent_color_pairs = 0
    for (x, y, z), color in color_map.items():
        neighbor = color_map.get((x + 1, y, z))
        if neighbor is not None:
            adjacent_color_pairs += 1
            if neighbor == color:
                adjacent_same_color_pairs += 1
        neighbor = color_map.get((x, y + 1, z))
        if neighbor is not None:
            adjacent_color_pairs += 1
            if neighbor == color:
                adjacent_same_color_pairs += 1

    adjacent_same_color_ratio = (
        adjacent_same_color_pairs / adjacent_color_pairs if adjacent_color_pairs > 0 else 0.0
    )

    missing_agent_palette_count = 0
    penalty_missing_palette = 0.0
    if allowed_blocks_per_agent is not None:
        observed_blocks = {block_id for block_id in obs_map.values() if not _is_air(block_id)}
        for palette in allowed_blocks_per_agent:
            normalized_palette = _normalize_palette(palette)
            if not normalized_palette:
                normalized_palette = ["white_concrete"]
            if not set(normalized_palette).issubset(observed_blocks):
                missing_agent_palette_count += 1
        penalty_missing_palette = 0.25 * float(missing_agent_palette_count)

    score_acc = 2.0 * coverage_ratio
    penalty_extra = 1.5 * extra_ratio
    penalty_adj = 1.0 * adjacent_same_color_ratio
    score_total = score_acc - penalty_extra - penalty_adj - penalty_missing_palette

    return {
        "target_total": target_total,
        "covered": covered,
        "missing": missing,
        "accuracy": coverage_ratio,
        "coverage_ratio": coverage_ratio,
        "extra_blocks": extra,
        "extra_ratio": extra_ratio,
        "adjacent_same_color_pairs": adjacent_same_color_pairs,
        "adjacent_color_pairs": adjacent_color_pairs,
        "adjacent_same_color_ratio": adjacent_same_color_ratio,
        "score_acc": score_acc,
        "penalty_extra": penalty_extra,
        "penalty_adj": penalty_adj,
        "missing_agent_palette_count": missing_agent_palette_count,
        "penalty_missing_palette": penalty_missing_palette,
        "score_total": score_total,
        "score_mean": score_total,
    }
