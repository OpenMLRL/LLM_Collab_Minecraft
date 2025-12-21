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
            task_id = f"str_painter_{idx:04d}"
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


def get_letter_coords(task: TaskSpec) -> List[Tuple[int, int, int]]:
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


def get_background_coords(task: TaskSpec) -> List[Tuple[int, int, int]]:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0
    coords: List[Tuple[int, int, int]] = []
    for r, row in enumerate(task.target_rows_topdown):
        if len(row) != width:
            raise ValueError("target_rows_topdown has inconsistent row widths")
        for x, ch in enumerate(row):
            if ch == "#":
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
        obs[(int(pos[0]), int(pos[1]), int(pos[2]))] = normalize_block_id(str(name))
    return obs


def parse_ascii_grid(text: str, *, width: int, height: int, allowed_symbols: set[str] | None = None) -> List[str]:
    allowed = allowed_symbols or {".", "B", "W", "b", "w"}
    rows: List[str] = []
    for raw in (text or "").splitlines():
        s = "".join(ch for ch in raw.strip() if ch in allowed)
        if not s:
            continue
        s = s[:width].ljust(width, ".")
        rows.append(s)
        if len(rows) >= height:
            break
    if len(rows) < height:
        rows.extend(["." * width for _ in range(height - len(rows))])
    return rows[:height]


def parse_ascii_decisions(
    text: str,
    *,
    task: TaskSpec,
    symbol_map: Dict[str, str],
    allowed_symbols: set[str] | None = None,
) -> Dict[Tuple[int, int, int], str]:
    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0
    rows = parse_ascii_grid(text, width=width, height=height, allowed_symbols=allowed_symbols)
    local_from = task.local_bbox_from
    decisions: Dict[Tuple[int, int, int], str] = {}
    for r, row in enumerate(rows):
        for x, ch in enumerate(row):
            block = symbol_map.get(ch)
            if not block:
                continue
            wx = int(local_from[0]) + x
            wy = int(local_from[1]) + (height - 1 - r)
            wz = int(local_from[2])
            decisions[(wx, wy, wz)] = normalize_block_id(block)
    return decisions


def score_painter_accuracy(
    *,
    task: TaskSpec,
    state: Dict[Tuple[int, int, int], str],
    letter_block: str,
    background_block: str | None,
) -> Dict[str, Any]:
    letter_block = normalize_block_id(letter_block)
    background_block = normalize_block_id(background_block) if background_block else None

    letter_coords = get_letter_coords(task)
    background_coords = get_background_coords(task)
    letter_total = len(letter_coords)
    bg_total = len(background_coords)

    letter_correct = 0
    for pos in letter_coords:
        if normalize_block_id(state.get(pos, "air")) == letter_block:
            letter_correct += 1

    bg_correct = 0
    if background_block is not None:
        for pos in background_coords:
            if normalize_block_id(state.get(pos, "air")) == background_block:
                bg_correct += 1
    else:
        for pos in background_coords:
            if _is_air(state.get(pos, "air")):
                bg_correct += 1

    letter_acc = (letter_correct / letter_total) if letter_total else 0.0
    if background_block is None:
        bg_acc = 1.0
    else:
        bg_acc = (bg_correct / bg_total) if bg_total else 0.0

    return {
        "letter_total": letter_total,
        "letter_correct": letter_correct,
        "letter_acc": letter_acc,
        "background_total": bg_total,
        "background_correct": bg_correct,
        "background_acc": bg_acc,
        "overall_acc": (letter_acc + bg_acc) / 2.0 if background_block is not None else letter_acc,
    }
