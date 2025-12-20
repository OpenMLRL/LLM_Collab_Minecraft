#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: pyyaml (pip install pyyaml)") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got: {type(data)}")
    return data


def _resolve_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _render_prompt(
    *,
    tokenizer: Any,
    system_prompt: str | None,
    user_prompt: str,
    use_chat_template: bool,
) -> tuple[str, list[dict[str, str]] | None]:
    system_prompt = (system_prompt or "").strip()

    if use_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return text, messages
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            return text, messages

    parts = []
    if system_prompt:
        parts.append(f"[SYSTEM]\n{system_prompt}")
    parts.append(f"[USER]\n{user_prompt}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts), None


def _strip_markdown_fences(text: str) -> str:
    raw = text.strip()
    if not raw:
        return raw
    if "```" not in raw:
        return raw

    parts = raw.split("```")
    if len(parts) < 3:
        return raw
    inner = parts[1].strip()
    inner = re.sub(r"^\s*[a-zA-Z0-9_-]+\s*\n", "", inner)
    return inner.strip()


_LEADING_LIST_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def _normalize_block_id(block_id: str) -> str:
    s = block_id.strip()
    if s.startswith("minecraft:"):
        s = s[len("minecraft:") :]
    return s


def _extract_command_lines(text: str) -> list[str]:
    raw = _strip_markdown_fences(text)
    lines: list[str] = []
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
    t = tok.strip()
    if not t:
        return None
    if t.startswith(("~", "^")):
        return None
    try:
        return int(t)
    except ValueError:
        return None


def _validate_and_normalize_mc_commands(
    *,
    lines: list[str],
    allowed_blocks: list[str],
    world_bbox_from: list[int],
    world_bbox_to: list[int],
    max_commands: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    allowed = {_normalize_block_id(b) for b in allowed_blocks}
    min_x = min(world_bbox_from[0], world_bbox_to[0])
    max_x = max(world_bbox_from[0], world_bbox_to[0])
    min_y = min(world_bbox_from[1], world_bbox_to[1])
    max_y = max(world_bbox_from[1], world_bbox_to[1])
    min_z = min(world_bbox_from[2], world_bbox_to[2])
    max_z = max(world_bbox_from[2], world_bbox_to[2])

    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []

    def _in_bbox(x: int, y: int, z: int) -> bool:
        return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z)

    for line in lines:
        if len(accepted) >= max_commands:
            rejected.append({"line": line, "reason": f"exceeds max_commands={max_commands}"})
            continue

        stripped = line.strip()
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
            block = _normalize_block_id(parts[4])
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
            block = _normalize_block_id(parts[7])
            if block not in allowed:
                rejected.append({"line": line, "reason": f"block not allowed: {block}"})
                continue
            if not (_in_bbox(x1, y1, z1) and _in_bbox(x2, y2, z2)):
                rejected.append({"line": line, "reason": "fill coord out of bbox"})
                continue
            if len(parts) > 8:
                rejected.append({"line": line, "reason": "fill modes (replace/keep/...) not allowed"})
                continue
            accepted.append(f"/fill {x1} {y1} {z1} {x2} {y2} {z2} {block}")
            continue

        rejected.append({"line": line, "reason": f"unsupported command: {cmd}"})

    return accepted, rejected


def _run_mc_executor(
    *,
    node_script: Path,
    host: str,
    port: int,
    username: str,
    version: str | None,
    step_delay_ms: int,
    scan_delay_ms: int,
    timeout_ms: int,
    pre_commands: list[str],
    commands: list[str],
    post_commands: list[str],
    scan_from: list[int] | None,
    scan_to: list[int] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pre_commands": pre_commands,
        "commands": commands,
        "post_commands": post_commands,
    }
    if scan_from is not None and scan_to is not None:
        payload["scan"] = {"from": scan_from, "to": scan_to}

    cmd = [
        "node",
        str(node_script),
        "--host",
        host,
        "--port",
        str(port),
        "--username",
        username,
        "--step-delay-ms",
        str(step_delay_ms),
        "--scan-delay-ms",
        str(scan_delay_ms),
        "--timeout-ms",
        str(timeout_ms),
    ]
    if version:
        cmd.extend(["--version", version])

    proc = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
    )
    stdout = proc.stdout or ""
    if proc.returncode != 0 and not stdout.strip():
        raise RuntimeError(f"mc_executor failed: returncode={proc.returncode} stderr={proc.stderr}")

    for line in stdout.splitlines()[::-1]:
        s = line.strip()
        if not s:
            continue
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise RuntimeError(
        "mc_executor returned no JSON object on stdout.\n"
        f"returncode={proc.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def _run_mc_executor_multi(
    *,
    node_script: Path,
    host: str,
    port: int,
    username1: str,
    username2: str,
    version: str | None,
    step_delay_ms: int,
    scan_delay_ms: int,
    timeout_ms: int,
    pre_commands_1: list[str],
    commands_1: list[str],
    post_commands_1: list[str],
    pre_commands_2: list[str],
    commands_2: list[str],
    post_commands_2: list[str],
    scan_from: list[int] | None,
    scan_to: list[int] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bots": [
            {
                "username": username1,
                "pre_commands": pre_commands_1,
                "commands": commands_1,
                "post_commands": post_commands_1,
            },
            {
                "username": username2,
                "pre_commands": pre_commands_2,
                "commands": commands_2,
                "post_commands": post_commands_2,
            },
        ]
    }
    if scan_from is not None and scan_to is not None:
        payload["scan"] = {"from": scan_from, "to": scan_to}

    cmd = [
        "node",
        str(node_script),
        "--host",
        host,
        "--port",
        str(port),
        "--username",
        username1,
        "--username2",
        username2,
        "--step-delay-ms",
        str(step_delay_ms),
        "--scan-delay-ms",
        str(scan_delay_ms),
        "--timeout-ms",
        str(timeout_ms),
    ]
    if version:
        cmd.extend(["--version", version])

    proc = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
    )
    stdout = proc.stdout or ""
    if proc.returncode != 0 and not stdout.strip():
        raise RuntimeError(f"mc_executor failed: returncode={proc.returncode} stderr={proc.stderr}")

    for line in stdout.splitlines()[::-1]:
        s = line.strip()
        if not s:
            continue
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise RuntimeError(
        "mc_executor returned no JSON object on stdout.\n"
        f"returncode={proc.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


def _compute_world_bbox(
    *,
    local_from: list[int],
    local_to: list[int],
    world_origin: list[int],
) -> tuple[list[int], list[int]]:
    return (
        [world_origin[0] + local_from[0], world_origin[1] + local_from[1], world_origin[2] + local_from[2]],
        [world_origin[0] + local_to[0], world_origin[1] + local_to[1], world_origin[2] + local_to[2]],
    )


def _simulate_commands_to_scan_blocks(
    *,
    commands: list[str],
    world_bbox_from: list[int],
    world_bbox_to: list[int],
) -> list[dict[str, Any]]:
    min_x = min(world_bbox_from[0], world_bbox_to[0])
    max_x = max(world_bbox_from[0], world_bbox_to[0])
    min_y = min(world_bbox_from[1], world_bbox_to[1])
    max_y = max(world_bbox_from[1], world_bbox_to[1])
    min_z = min(world_bbox_from[2], world_bbox_to[2])
    max_z = max(world_bbox_from[2], world_bbox_to[2])

    state: dict[tuple[int, int, int], str] = {}

    def _set(x: int, y: int, z: int, block: str) -> None:
        state[(x, y, z)] = _normalize_block_id(block)

    for cmd in commands:
        stripped = cmd.strip()
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
            for y in range(fy1, fy2 + 1):
                for x in range(fx1, fx2 + 1):
                    for z in range(fz1, fz2 + 1):
                        _set(x, y, z, block)
            continue

    blocks: list[dict[str, Any]] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            for z in range(min_z, max_z + 1):
                blocks.append({"pos": [x, y, z], "name": state.get((x, y, z), "air")})
    return blocks


_FONT_5X5: dict[str, list[str]] = {
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


def _render_string_mask_rows(text: str, *, spacing: int) -> list[str]:
    text = str(text or "").strip().upper()
    if not text:
        return [""] * 5

    height = 5
    width = 5
    rows = [""] * height
    for i, ch in enumerate(text):
        glyph = _FONT_5X5.get(ch)
        if glyph is None:
            glyph = ["0" * width] * height
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
    local_bbox_from: list[int]
    local_bbox_to: list[int]
    target_rows_topdown: list[str]


def _load_tasks_from_csv(
    csv_path: Path,
    *,
    spacing: int,
    local_z: int,
) -> list[TaskSpec]:
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv_path not found: {csv_path}")

    tasks: list[TaskSpec] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

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

            target_rows = _render_string_mask_rows(text, spacing=spacing)
            height = len(target_rows)
            width = len(target_rows[0]) if height else 0
            if any(len(r) != width for r in target_rows):
                raise ValueError(f"Inconsistent target row widths for {text!r}")

            local_from = [0, 0, local_z]
            local_to = [max(0, width - 1), max(0, height - 1), local_z]

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
    b = _normalize_block_id(str(block_name))
    return b in {"air", "cave_air", "void_air"}


def _chamfer_distance(points_a: list[tuple[int, int]], points_b: list[tuple[int, int]]) -> float:
    if not points_a or not points_b:
        return float("inf")

    def _mean_nn(src: list[tuple[int, int]], dst: list[tuple[int, int]]) -> float:
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


def _count_components_8(points: set[tuple[int, int]]) -> int:
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


def _score_str_builder(
    *,
    task: TaskSpec,
    world_origin: list[int],
    world_scan_blocks: list[dict[str, Any]],
    chamfer_sigma: float,
) -> dict[str, Any]:
    if chamfer_sigma <= 0:
        raise ValueError("chamfer_sigma must be > 0")

    height = len(task.target_rows_topdown)
    width = len(task.target_rows_topdown[0]) if height else 0

    # Target points in WORLD coords (projected to (x,y)).
    t_points: list[tuple[int, int]] = []
    for r, row in enumerate(task.target_rows_topdown):
        if len(row) != width:
            raise ValueError("target_rows_topdown has inconsistent row widths")
        for x, ch in enumerate(row):
            if ch != "#":
                continue
            lx = task.local_bbox_from[0] + x
            ly = task.local_bbox_from[1] + (height - 1 - r)  # top row -> max y
            lz = task.local_bbox_from[2]
            wx = world_origin[0] + lx
            wy = world_origin[1] + ly
            wz = world_origin[2] + lz
            _ = wz  # z is constant; ignore in metrics below
            t_points.append((wx, wy))
    t_set = set(t_points)

    # Observed blocks in WORLD coords.
    obs_block: dict[tuple[int, int], str] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        _ = z
        if _is_air(None if name is None else str(name)):
            continue
        obs_block[(x, y)] = _normalize_block_id(str(name))

    o_set = set(obs_block.keys())
    inter = len(t_set & o_set)
    union = len(t_set | o_set)
    iou = (inter / union) if union else 1.0

    cd = _chamfer_distance(list(t_set), list(o_set))
    shape_score = 0.0 if not math.isfinite(cd) else math.exp(-cd / chamfer_sigma)
    score_shape_overlap = iou

    components = _count_components_8(o_set)
    expected = max(0, len(task.text))
    components_ratio = (components / expected) if expected else 0.0
    score_components = min(components_ratio, 1.0) if expected else 0.0

    score_mean = (score_shape_overlap + score_components) / 2.0

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
        "score_mean": score_mean,
    }


@dataclass(frozen=True)
class LoadedModel:
    tokenizer: Any
    model: Any
    model_id: str


def _load_transformers_model(model_cfg: dict[str, Any]) -> LoadedModel:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing deps for transformers backend. Install: pip install torch transformers accelerate"
        ) from e

    model_name_or_path = model_cfg.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("model.model_name_or_path is required")

    tokenizer_name_or_path = model_cfg.get("tokenizer_name_or_path") or model_name_or_path
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    revision = model_cfg.get("revision")

    dtype_str = str(model_cfg.get("dtype") or "").lower().strip()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str) if dtype_str else None

    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    quant_kwargs: dict[str, Any] = {}
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "load_in_4bit/load_in_8bit requires bitsandbytes. Install: pip install bitsandbytes"
            ) from e
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

    device_map_cfg = model_cfg.get("device_map", "auto")
    explicit_device: str | None = None
    device_map: Any = None
    if isinstance(device_map_cfg, str):
        val = device_map_cfg.strip()
        if val in {"auto", "balanced", "balanced_low_0", "sequential"}:
            device_map = val
        elif val.startswith("cuda") or val == "cpu":
            explicit_device = val
            device_map = None
        else:
            device_map = val
    else:
        device_map = device_map_cfg

    token = model_cfg.get("token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def _from_pretrained_with_token(cls: Any, name_or_path: str, **kwargs: Any) -> Any:
        if token:
            try:
                return cls.from_pretrained(name_or_path, token=token, **kwargs)
            except TypeError:
                return cls.from_pretrained(name_or_path, use_auth_token=token, **kwargs)
        return cls.from_pretrained(name_or_path, **kwargs)

    try:
        tokenizer = _from_pretrained_with_token(
            AutoTokenizer,
            str(tokenizer_name_or_path),
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = _from_pretrained_with_token(
            AutoModelForCausalLM,
            str(model_name_or_path),
            trust_remote_code=trust_remote_code,
            revision=revision,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **quant_kwargs,
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(
            "Failed to load transformers model/tokenizer. "
            "If this is a gated HF repo, set HF_TOKEN / HUGGINGFACE_HUB_TOKEN, or use a public model id or local path."
        ) from e

    if explicit_device is not None:
        model.to(explicit_device)

    return LoadedModel(tokenizer=tokenizer, model=model, model_id=str(model_name_or_path))


def _generate_with_transformers(
    loaded: LoadedModel,
    prompt_text: str,
    generation_cfg: dict[str, Any],
    *,
    stream_output: bool = False,
) -> list[str]:
    import torch  # type: ignore

    tokenizer = loaded.tokenizer
    model = loaded.model

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    gen_kwargs: dict[str, Any] = {}
    for k in [
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "repetition_penalty",
        "num_return_sequences",
    ]:
        if k in generation_cfg and generation_cfg[k] is not None:
            gen_kwargs[k] = generation_cfg[k]

    if tokenizer.pad_token_id is not None:
        gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    streamer = None
    if stream_output:
        try:
            from transformers import TextStreamer  # type: ignore

            class _StderrStreamer(TextStreamer):  # type: ignore[misc]
                def on_finalized_text(self, text: str, stream_end: bool = False) -> None:  # type: ignore[override]
                    sys.stderr.write(text)
                    if stream_end:
                        sys.stderr.write("\n")
                    sys.stderr.flush()

            streamer = _StderrStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        except Exception:
            streamer = None

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **({"streamer": streamer} if streamer is not None else {}),
            **gen_kwargs,
        )

    input_len = int(input_ids.shape[-1])
    texts: list[str] = []
    for seq in out_ids:
        gen = seq[input_len:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    default_config = Path(__file__).resolve().parent / "config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks.")
    parser.add_argument("--dry-run", action="store_true", help="Load tasks and print prompts without loading the model.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-stage progress logs (useful when debugging slow runs).",
    )
    parser.add_argument(
        "--print-llm-output",
        action="store_true",
        help="Print raw model output text to stderr for each sample (debug).",
    )
    parser.add_argument(
        "--print-llm-output-max-chars",
        type=int,
        default=4000,
        help="Max chars to print for --print-llm-output (0 = unlimited).",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="Print validated Minecraft commands (accepted/rejected counts) to stderr (debug).",
    )
    parser.add_argument(
        "--stream-llm-output",
        action="store_true",
        help="Stream decoded tokens to stderr during generation (debug; can be very verbose).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(config_path)

    run_name = str(cfg.get("run_name") or "baseline")
    seed = int(cfg.get("seed") or 0)

    agents_cfg = cfg.get("agents") or {}
    if not isinstance(agents_cfg, dict):
        raise ValueError("agents must be a mapping")
    num_agents = int(agents_cfg.get("num_agents") or 1)
    if num_agents not in (1, 2):
        raise ValueError("agents.num_agents must be 1 or 2")

    task_cfg = cfg.get("task") or {}
    if not isinstance(task_cfg, dict):
        raise ValueError("task must be a mapping")
    spacing = int(task_cfg.get("spacing") or 1)
    local_z = int(task_cfg.get("local_z") or 0)
    block_even = str(task_cfg.get("block_even") or "white_concrete")
    block_odd = str(task_cfg.get("block_odd") or "black_concrete")
    block_agent2 = str(task_cfg.get("block_agent2") or "red_concrete")
    chamfer_sigma = float(task_cfg.get("chamfer_sigma") or 2.0)

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset must be a mapping")
    csv_path = _resolve_path(config_path, dataset_cfg.get("csv_path")) or (
        config_path.parent / "../../dataset/str_builder/data.csv"
    ).resolve()

    tasks = _load_tasks_from_csv(csv_path, spacing=spacing, local_z=local_z)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        _eprint(f"No tasks found in: {csv_path}")
        return 2

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        raise ValueError("output must be a mapping")
    output_path = _resolve_path(config_path, output_cfg.get("path")) or (config_path.parent / "outputs/output.jsonl")
    output_path = output_path.resolve()
    output_simple_path = _resolve_path(config_path, output_cfg.get("simple_path"))
    if output_simple_path is None:
        output_simple_path = output_path.with_name(output_path.stem + ".simple.jsonl")
    output_simple_path = output_simple_path.resolve()
    if output_simple_path == output_path:
        raise ValueError("output.simple_path must be different from output.path")
    overwrite = bool(output_cfg.get("overwrite", False))
    write_prompt = bool(output_cfg.get("write_prompt", True))
    write_task = bool(output_cfg.get("write_task", True))

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (set output.overwrite=true to overwrite): {output_path}")
    if output_simple_path.exists() and not overwrite:
        raise FileExistsError(f"Simple output exists (set output.overwrite=true to overwrite): {output_simple_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_simple_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt must be a mapping")
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "{task_json}").rstrip()
    user_template_agent1 = str(prompt_cfg.get("user_template_agent1") or user_template).rstrip()
    user_template_agent2 = str(prompt_cfg.get("user_template_agent2") or user_template).rstrip()

    generation_cfg = cfg.get("generation") or {}
    if not isinstance(generation_cfg, dict):
        raise ValueError("generation must be a mapping")
    use_chat_template = bool(generation_cfg.get("use_chat_template", False))

    scoring_cfg = cfg.get("scoring") or {}
    if not isinstance(scoring_cfg, dict):
        scoring_cfg = {}
    best_of = int(scoring_cfg.get("best_of") or 5)
    if best_of < 1:
        best_of = 1
    generation_cfg_effective = dict(generation_cfg)
    num_return_sequences = int(generation_cfg_effective.get("num_return_sequences") or 1)
    if num_return_sequences < best_of:
        generation_cfg_effective["num_return_sequences"] = best_of

    if args.dry_run:
        _eprint(f"[dry-run] tasks={len(tasks)} csv_path={csv_path}")
        for task in tasks[:3]:
            if num_agents == 1:
                user_prompt = user_template.format(
                    task_json=json.dumps(
                        {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    text=task.text,
                    difficulty=task.difficulty,
                    target_rows="\n".join(task.target_rows_topdown),
                    block_even=block_even,
                    block_odd=block_odd,
                    block_agent2=block_agent2,
                    world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
                )
                _eprint(f"\n=== {task.task_id} string={task.text!r} difficulty={task.difficulty} ===\n{user_prompt}\n")
                continue

            user_prompt_1 = user_template_agent1.format(
                task_json=json.dumps(
                    {
                        "task_id": task.task_id,
                        "string": task.text,
                        "difficulty": task.difficulty,
                        "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                task_id=task.task_id,
                text=task.text,
                difficulty=task.difficulty,
                target_rows="\n".join(task.target_rows_topdown),
                block_even=block_even,
                block_odd=block_odd,
                block_agent2=block_agent2,
                world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
            )
            user_prompt_2 = user_template_agent2.format(
                task_json=json.dumps(
                    {
                        "task_id": task.task_id,
                        "string": task.text,
                        "difficulty": task.difficulty,
                        "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                task_id=task.task_id,
                text=task.text,
                difficulty=task.difficulty,
                target_rows="\n".join(task.target_rows_topdown),
                block_even=block_even,
                block_odd=block_odd,
                block_agent2=block_agent2,
                world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
            )
            _eprint(
                f"\n=== {task.task_id} string={task.text!r} difficulty={task.difficulty} ===\n"
                f"[agent1]\n{user_prompt_1}\n\n[agent2]\n{user_prompt_2}\n"
            )
        return 0

    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model must be a mapping")

    backend = str(model_cfg.get("backend") or "transformers").lower().strip()
    if backend != "transformers":
        raise ValueError(f"Unsupported model.backend={backend!r} (only 'transformers' is implemented)")

    # RNG seeds (best-effort; generation can still be nondeterministic on GPU).
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    _eprint(f"Loading model: {model_cfg.get('model_name_or_path')}")
    loaded = _load_transformers_model(model_cfg)
    _eprint(f"Loaded model: {loaded.model_id}")
    _eprint(f"Tasks: {len(tasks)}")
    _eprint(f"Output: {output_path}")
    _eprint(f"Output(simple): {output_simple_path}")
    _eprint(f"Agents: num_agents={num_agents}")

    mc_cfg = cfg.get("minecraft") or {}
    if not isinstance(mc_cfg, dict):
        raise ValueError("minecraft must be a mapping")

    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_host = str(mc_cfg.get("host") or "127.0.0.1")
    mc_port = int(mc_cfg.get("port") or 25565)
    mc_username = str(mc_cfg.get("username") or "executor_bot")
    mc_username2 = str(mc_cfg.get("username2") or f"{mc_username}_2")
    if num_agents == 2 and mc_username2 == mc_username:
        raise ValueError("minecraft.username2 must be different from minecraft.username when agents.num_agents=2")
    mc_version = mc_cfg.get("version")
    mc_version = str(mc_version) if mc_version not in (None, "null") else None
    mc_origin_mode = str(mc_cfg.get("origin_mode") or "fixed").strip().lower()
    mc_spawn_offset = mc_cfg.get("spawn_offset") or [0, 0, 0]
    if not (isinstance(mc_spawn_offset, list) and len(mc_spawn_offset) == 3):
        raise ValueError("minecraft.spawn_offset must be a 3-element list")
    mc_spawn_offset_vec = [int(mc_spawn_offset[0]), int(mc_spawn_offset[1]), int(mc_spawn_offset[2])]
    mc_world_origin_cfg = mc_cfg.get("world_origin")
    mc_world_origin_vec: list[int] | None = None
    if mc_world_origin_cfg is not None:
        if not (isinstance(mc_world_origin_cfg, list) and len(mc_world_origin_cfg) == 3):
            raise ValueError("minecraft.world_origin must be a 3-element list or null")
        mc_world_origin_vec = [int(mc_world_origin_cfg[0]), int(mc_world_origin_cfg[1]), int(mc_world_origin_cfg[2])]

    mc_reset_before_each = bool(mc_cfg.get("reset_before_each", True))
    mc_teleport_before_each = bool(mc_cfg.get("teleport_before_each", True))
    mc_step_delay_ms = int(mc_cfg.get("step_delay_ms") or 200)
    mc_scan_delay_ms = int(mc_cfg.get("scan_delay_ms") or 250)
    mc_timeout_ms = int(mc_cfg.get("timeout_ms") or 120_000)
    mc_max_commands = int(mc_cfg.get("max_commands") or 600)

    node_script = Path(__file__).resolve().parent / "mc_executor.cjs"
    if mc_enabled and not node_script.exists():
        raise FileNotFoundError(f"Missing node executor script: {node_script}")

    world_origin: list[int] | None = None
    if mc_enabled:
        if mc_origin_mode == "fixed":
            if mc_world_origin_vec is None:
                raise ValueError("minecraft.world_origin is required when origin_mode=fixed")
            world_origin = mc_world_origin_vec
        elif mc_origin_mode == "spawn_offset":
            probe = _run_mc_executor(
                node_script=node_script,
                host=mc_host,
                port=mc_port,
                username=mc_username,
                version=mc_version,
                step_delay_ms=mc_step_delay_ms,
                scan_delay_ms=mc_scan_delay_ms,
                timeout_ms=mc_timeout_ms,
                pre_commands=[],
                commands=[],
                post_commands=[],
                scan_from=None,
                scan_to=None,
            )
            if not bool(probe.get("ok", False)):
                raise RuntimeError(
                    "Failed to connect to Minecraft server for origin probe (origin_mode=spawn_offset). "
                    f"host={mc_host} port={mc_port} username={mc_username} errors={probe.get('errors')}"
                )
            spawn = probe.get("spawn_floored")
            if not (isinstance(spawn, list) and len(spawn) == 3):
                raise RuntimeError(
                    "minecraft.origin_mode=spawn_offset requires mc_executor to return spawn_floored=[x,y,z]. "
                    "Update baselines/str_builder/mc_executor.cjs accordingly."
                )
            world_origin = [
                int(spawn[0]) + mc_spawn_offset_vec[0],
                int(spawn[1]) + mc_spawn_offset_vec[1],
                int(spawn[2]) + mc_spawn_offset_vec[2],
            ]
        else:
            raise ValueError("minecraft.origin_mode must be one of: fixed, spawn_offset")

        if num_agents == 1:
            _eprint(f"Minecraft enabled: host={mc_host} port={mc_port} username={mc_username} origin={world_origin}")
        else:
            _eprint(
                f"Minecraft enabled: host={mc_host} port={mc_port} username1={mc_username} username2={mc_username2} "
                f"origin={world_origin}"
            )

    def _open_output_writer(path: Path):
        if path.suffix.lower() == ".jsonl":
            f = path.open("w", encoding="utf-8")

            def write_record(rec: dict[str, Any]) -> None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            def close() -> None:
                f.close()

            return write_record, close

        f = path.open("w", encoding="utf-8")
        f.write("[\n")
        first = True

        def write_record(rec: dict[str, Any]) -> None:
            nonlocal first
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(rec, ensure_ascii=False))
            f.flush()

        def close() -> None:
            f.write("\n]\n")
            f.close()

        return write_record, close

    def _open_jsonl_writer(path: Path):
        f = path.open("w", encoding="utf-8")

        def write_record(rec: dict[str, Any]) -> None:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        def close() -> None:
            f.close()

        return write_record, close

    def _pick_primary_metric(metrics: dict[str, Any], key: str) -> float | None:
        mc_key = f"mc_{key}"
        v = metrics.get(mc_key)
        if isinstance(v, (int, float)):
            return float(v)
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _score_for_selection(metrics: dict[str, Any]) -> float:
        v = _pick_primary_metric(metrics, "score_mean")
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    allowed_blocks_agent1 = [block_even, block_odd]
    allowed_blocks_agent2 = [block_agent2]
    write_record, close_writer = _open_output_writer(output_path)
    write_simple_record, close_simple_writer = _open_jsonl_writer(output_simple_path)
    try:
        for idx, task in enumerate(tasks, start=1):
            local_from = task.local_bbox_from
            local_to = task.local_bbox_to
            w_from = local_from
            w_to = local_to
            if world_origin is not None:
                w_from, w_to = _compute_world_bbox(local_from=local_from, local_to=local_to, world_origin=world_origin)

            max_commands_per_agent = mc_max_commands if num_agents == 1 else max(1, mc_max_commands // num_agents)
            max_commands_agent1 = max_commands_per_agent + (mc_max_commands % num_agents)
            max_commands_agent2 = max_commands_per_agent

            if num_agents == 1:
                user_prompt = user_template.format(
                    task_json=json.dumps(
                        {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": local_from, "to": local_to},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    text=task.text,
                    difficulty=task.difficulty,
                    target_rows="\n".join(task.target_rows_topdown),
                    block_even=block_even,
                    block_odd=block_odd,
                    block_agent2=block_agent2,
                    world_bbox_from=json.dumps(w_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(w_to, separators=(",", ":")),
                )

                prompt_text, chat_messages = _render_prompt(
                    tokenizer=loaded.tokenizer,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    use_chat_template=use_chat_template,
                )

                if args.verbose:
                    _eprint(f"[{idx}/{len(tasks)}] {task.task_id} generating...")
                t0 = time.time()
                if args.stream_llm_output and int(generation_cfg_effective.get("num_return_sequences") or 1) != 1:
                    _eprint("--stream-llm-output only supports num_return_sequences=1; disabling streaming.")
                    stream_output = False
                else:
                    stream_output = bool(args.stream_llm_output)
                if stream_output:
                    _eprint(f"--- streaming LLM output task_id={task.task_id} ---")
                outputs = _generate_with_transformers(
                    loaded,
                    prompt_text,
                    generation_cfg_effective,
                    stream_output=stream_output,
                )
                dt = time.time() - t0
                if args.verbose:
                    _eprint(f"[{idx}/{len(tasks)}] {task.task_id} generated samples={len(outputs)} dt={dt:.2f}s")

                best_metrics: dict[str, Any] | None = None
                best_score = float("-inf")
                for sample_id, output_text in enumerate(outputs):
                    if args.print_llm_output:
                        max_chars = int(args.print_llm_output_max_chars or 0)
                        out = output_text if max_chars <= 0 else output_text[:max_chars]
                        suffix = "" if max_chars <= 0 or len(output_text) <= max_chars else "\n... [truncated]"
                        _eprint(f"\n=== LLM output task_id={task.task_id} sample_id={sample_id} ===\n{out}{suffix}\n")

                    lines = _extract_command_lines(output_text)
                    accepted_cmds, rejected_cmds = _validate_and_normalize_mc_commands(
                        lines=lines,
                        allowed_blocks=allowed_blocks_agent1,
                        world_bbox_from=w_from,
                        world_bbox_to=w_to,
                        max_commands=max_commands_agent1,
                    )

                    if args.print_commands:
                        _eprint(
                            f"[{task.task_id} sample={sample_id}] accepted={len(accepted_cmds)} rejected={len(rejected_cmds)}"
                        )
                        if accepted_cmds:
                            _eprint(
                                "\n".join(accepted_cmds[:50])
                                + ("\n... [truncated]" if len(accepted_cmds) > 50 else "")
                            )

                    mc_result: dict[str, Any] | None = None
                    metrics: dict[str, Any] = {"latency_s": dt}

                    score_origin = world_origin if world_origin is not None else [0, 0, 0]
                    try:
                        sim_blocks = _simulate_commands_to_scan_blocks(
                            commands=accepted_cmds,
                            world_bbox_from=w_from,
                            world_bbox_to=w_to,
                        )
                        metrics.update(
                            _score_str_builder(
                                task=task,
                                world_origin=score_origin,
                                world_scan_blocks=sim_blocks,
                                chamfer_sigma=chamfer_sigma,
                            )
                        )
                    except Exception as e:
                        metrics.update({"score_mean": 0.0, "sim_error": str(e)})

                    if mc_enabled and world_origin is not None:
                        min_x = min(w_from[0], w_to[0])
                        max_x = max(w_from[0], w_to[0])
                        min_y = min(w_from[1], w_to[1])
                        max_y = max(w_from[1], w_to[1])
                        min_z = min(w_from[2], w_to[2])
                        max_z = max(w_from[2], w_to[2])
                        tp_x = (min_x + max_x) // 2
                        tp_y = min_y
                        tp_z = max_z + 2

                        pre_cmds: list[str] = []
                        if mc_teleport_before_each:
                            pre_cmds.append(f"/tp {mc_username} {tp_x} {tp_y} {tp_z}")
                        if mc_reset_before_each:
                            pre_cmds.append(f"/fill {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} air")

                        try:
                            mc_result = _run_mc_executor(
                                node_script=node_script,
                                host=mc_host,
                                port=mc_port,
                                username=mc_username,
                                version=mc_version,
                                step_delay_ms=mc_step_delay_ms,
                                scan_delay_ms=mc_scan_delay_ms,
                                timeout_ms=mc_timeout_ms,
                                pre_commands=pre_cmds,
                                commands=accepted_cmds,
                                post_commands=[],
                                scan_from=[min_x, min_y, min_z],
                                scan_to=[max_x, max_y, max_z],
                            )
                            scan = (mc_result.get("scan") or {}) if isinstance(mc_result, dict) else {}
                            scan_blocks = scan.get("blocks") if isinstance(scan, dict) else None
                            if isinstance(scan_blocks, list):
                                mc_metrics = _score_str_builder(
                                    task=task,
                                    world_origin=world_origin,
                                    world_scan_blocks=scan_blocks,
                                    chamfer_sigma=chamfer_sigma,
                                )
                                metrics.update({f"mc_{k}": v for k, v in mc_metrics.items()})
                            else:
                                metrics.update({"mc_error": "missing scan blocks"})
                        except Exception as e:
                            metrics.update({"mc_error": str(e)})

                    score_val = _score_for_selection(metrics)
                    if score_val > best_score:
                        best_score = score_val
                        best_metrics = dict(metrics)

                    record: dict[str, Any] = {
                        "run_name": run_name,
                        "task_id": task.task_id,
                        "dataset_path": str(csv_path),
                        "dataset_row": task.csv_row_index,
                        "sample_id": sample_id,
                        "num_agents": num_agents,
                        "model": {
                            "backend": backend,
                            "model_id": loaded.model_id,
                        },
                        "generation": generation_cfg_effective,
                        "metrics": metrics,
                        "output_text": output_text,
                        "commands": accepted_cmds,
                        "rejected_commands": rejected_cmds,
                        "mc_result": mc_result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if write_task:
                        record["task"] = {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                            "target_rows_topdown": task.target_rows_topdown,
                            "allowed_blocks": allowed_blocks_agent1,
                        }
                    if write_prompt:
                        record["prompt"] = {
                            "system": system_prompt,
                            "user": user_prompt,
                            "rendered": prompt_text,
                            "chat_messages": chat_messages,
                        }

                    write_record(record)
                if best_metrics is None:
                    best_metrics = {}
                write_simple_record(
                    {
                        "task_id": task.task_id,
                        "string": task.text,
                        "difficulty": task.difficulty,
                        "model_id": loaded.model_id,
                        "score_shape_overlap": _pick_primary_metric(best_metrics, "score_shape_overlap"),
                        "score_components": _pick_primary_metric(best_metrics, "score_components"),
                    }
                )
                _eprint(f"[{idx}/{len(tasks)}] {task.task_id} samples={len(outputs)} {dt:.2f}s")
            else:
                user_prompt_1 = user_template_agent1.format(
                    task_json=json.dumps(
                        {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": local_from, "to": local_to},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    text=task.text,
                    difficulty=task.difficulty,
                    target_rows="\n".join(task.target_rows_topdown),
                    block_even=block_even,
                    block_odd=block_odd,
                    block_agent2=block_agent2,
                    world_bbox_from=json.dumps(w_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(w_to, separators=(",", ":")),
                )
                user_prompt_2 = user_template_agent2.format(
                    task_json=json.dumps(
                        {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": local_from, "to": local_to},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    text=task.text,
                    difficulty=task.difficulty,
                    target_rows="\n".join(task.target_rows_topdown),
                    block_even=block_even,
                    block_odd=block_odd,
                    block_agent2=block_agent2,
                    world_bbox_from=json.dumps(w_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(w_to, separators=(",", ":")),
                )

                prompt_text_1, chat_messages_1 = _render_prompt(
                    tokenizer=loaded.tokenizer,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt_1,
                    use_chat_template=use_chat_template,
                )
                prompt_text_2, chat_messages_2 = _render_prompt(
                    tokenizer=loaded.tokenizer,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt_2,
                    use_chat_template=use_chat_template,
                )

                if args.verbose:
                    _eprint(f"[{idx}/{len(tasks)}] {task.task_id} generating agent1...")
                t0 = time.time()
                if args.stream_llm_output and int(generation_cfg_effective.get("num_return_sequences") or 1) != 1:
                    _eprint("--stream-llm-output only supports num_return_sequences=1; disabling streaming.")
                    stream_output = False
                else:
                    stream_output = bool(args.stream_llm_output)
                if stream_output:
                    _eprint(f"--- streaming LLM output task_id={task.task_id} agent=1 ---")
                outputs_1 = _generate_with_transformers(
                    loaded,
                    prompt_text_1,
                    generation_cfg_effective,
                    stream_output=stream_output,
                )
                dt1 = time.time() - t0

                if args.verbose:
                    _eprint(f"[{idx}/{len(tasks)}] {task.task_id} generating agent2...")
                t1 = time.time()
                if stream_output:
                    _eprint(f"--- streaming LLM output task_id={task.task_id} agent=2 ---")
                outputs_2 = _generate_with_transformers(
                    loaded,
                    prompt_text_2,
                    generation_cfg_effective,
                    stream_output=stream_output,
                )
                dt2 = time.time() - t1
                dt = dt1 + dt2

                n = min(len(outputs_1), len(outputs_2))
                if args.verbose:
                    _eprint(f"[{idx}/{len(tasks)}] {task.task_id} generated samples={n} dt={dt:.2f}s")

                best_metrics: dict[str, Any] | None = None
                best_score = float("-inf")
                for sample_id in range(n):
                    output_text_1 = outputs_1[sample_id]
                    output_text_2 = outputs_2[sample_id]
                    output_text = f"[agent1]\n{output_text_1}\n\n[agent2]\n{output_text_2}\n"

                    if args.print_llm_output:
                        max_chars = int(args.print_llm_output_max_chars or 0)
                        out1 = output_text_1 if max_chars <= 0 else output_text_1[:max_chars]
                        out2 = output_text_2 if max_chars <= 0 else output_text_2[:max_chars]
                        suffix1 = "" if max_chars <= 0 or len(output_text_1) <= max_chars else "\n... [truncated]"
                        suffix2 = "" if max_chars <= 0 or len(output_text_2) <= max_chars else "\n... [truncated]"
                        _eprint(
                            f"\n=== LLM output task_id={task.task_id} sample_id={sample_id} agent=1 ===\n{out1}{suffix1}\n"
                        )
                        _eprint(
                            f"\n=== LLM output task_id={task.task_id} sample_id={sample_id} agent=2 ===\n{out2}{suffix2}\n"
                        )

                    lines_1 = _extract_command_lines(output_text_1)
                    lines_2 = _extract_command_lines(output_text_2)
                    accepted_1, rejected_1 = _validate_and_normalize_mc_commands(
                        lines=lines_1,
                        allowed_blocks=allowed_blocks_agent1,
                        world_bbox_from=w_from,
                        world_bbox_to=w_to,
                        max_commands=max_commands_agent1,
                    )
                    accepted_2, rejected_2 = _validate_and_normalize_mc_commands(
                        lines=lines_2,
                        allowed_blocks=allowed_blocks_agent2,
                        world_bbox_from=w_from,
                        world_bbox_to=w_to,
                        max_commands=max_commands_agent2,
                    )

                    merged_cmds = [*accepted_1, *accepted_2]
                    merged_rejected: list[dict[str, Any]] = [
                        {"agent_id": 1, **r} for r in rejected_1
                    ] + [{"agent_id": 2, **r} for r in rejected_2]

                    if args.print_commands:
                        _eprint(
                            f"[{task.task_id} sample={sample_id}] "
                            f"agent1_accepted={len(accepted_1)} agent1_rejected={len(rejected_1)} "
                            f"agent2_accepted={len(accepted_2)} agent2_rejected={len(rejected_2)} "
                            f"merged={len(merged_cmds)}"
                        )

                    mc_result: dict[str, Any] | None = None
                    metrics: dict[str, Any] = {"latency_s": dt, "latency_s_agent1": dt1, "latency_s_agent2": dt2}

                    score_origin = world_origin if world_origin is not None else [0, 0, 0]
                    try:
                        sim_blocks = _simulate_commands_to_scan_blocks(
                            commands=merged_cmds,
                            world_bbox_from=w_from,
                            world_bbox_to=w_to,
                        )
                        metrics.update(
                            _score_str_builder(
                                task=task,
                                world_origin=score_origin,
                                world_scan_blocks=sim_blocks,
                                chamfer_sigma=chamfer_sigma,
                            )
                        )
                    except Exception as e:
                        metrics.update({"score_mean": 0.0, "sim_error": str(e)})

                    if mc_enabled and world_origin is not None:
                        min_x = min(w_from[0], w_to[0])
                        max_x = max(w_from[0], w_to[0])
                        min_y = min(w_from[1], w_to[1])
                        max_y = max(w_from[1], w_to[1])
                        min_z = min(w_from[2], w_to[2])
                        max_z = max(w_from[2], w_to[2])
                        tp_x = (min_x + max_x) // 2
                        tp_y = min_y
                        tp_z1 = max_z + 2
                        tp_z2 = max_z + 4

                        pre_cmds_1: list[str] = []
                        if mc_teleport_before_each:
                            pre_cmds_1.append(f"/tp {mc_username} {tp_x} {tp_y} {tp_z1}")
                            pre_cmds_1.append(f"/tp {mc_username2} {tp_x} {tp_y} {tp_z2}")
                        if mc_reset_before_each:
                            pre_cmds_1.append(f"/fill {min_x} {min_y} {min_z} {max_x} {max_y} {max_z} air")

                        try:
                            mc_result = _run_mc_executor_multi(
                                node_script=node_script,
                                host=mc_host,
                                port=mc_port,
                                username1=mc_username,
                                username2=mc_username2,
                                version=mc_version,
                                step_delay_ms=mc_step_delay_ms,
                                scan_delay_ms=mc_scan_delay_ms,
                                timeout_ms=mc_timeout_ms,
                                pre_commands_1=pre_cmds_1,
                                commands_1=accepted_1,
                                post_commands_1=[],
                                pre_commands_2=[],
                                commands_2=accepted_2,
                                post_commands_2=[],
                                scan_from=[min_x, min_y, min_z],
                                scan_to=[max_x, max_y, max_z],
                            )
                            scan = (mc_result.get("scan") or {}) if isinstance(mc_result, dict) else {}
                            scan_blocks = scan.get("blocks") if isinstance(scan, dict) else None
                            if isinstance(scan_blocks, list):
                                mc_metrics = _score_str_builder(
                                    task=task,
                                    world_origin=world_origin,
                                    world_scan_blocks=scan_blocks,
                                    chamfer_sigma=chamfer_sigma,
                                )
                                metrics.update({f"mc_{k}": v for k, v in mc_metrics.items()})
                            else:
                                metrics.update({"mc_error": "missing scan blocks"})
                        except Exception as e:
                            metrics.update({"mc_error": str(e)})

                    score_val = _score_for_selection(metrics)
                    if score_val > best_score:
                        best_score = score_val
                        best_metrics = dict(metrics)

                    record: dict[str, Any] = {
                        "run_name": run_name,
                        "task_id": task.task_id,
                        "dataset_path": str(csv_path),
                        "dataset_row": task.csv_row_index,
                        "sample_id": sample_id,
                        "num_agents": num_agents,
                        "model": {
                            "backend": backend,
                            "model_id": loaded.model_id,
                        },
                        "generation": generation_cfg_effective,
                        "metrics": metrics,
                        "output_text": output_text,
                        "agents": [
                            {
                                "agent_id": 1,
                                "model_id": loaded.model_id,
                                "allowed_blocks": allowed_blocks_agent1,
                                "output_text": output_text_1,
                                "commands": accepted_1,
                                "rejected_commands": rejected_1,
                                "prompt": {
                                    "user": user_prompt_1,
                                    "rendered": prompt_text_1,
                                    "chat_messages": chat_messages_1,
                                }
                                if write_prompt
                                else None,
                            },
                            {
                                "agent_id": 2,
                                "model_id": loaded.model_id,
                                "allowed_blocks": allowed_blocks_agent2,
                                "output_text": output_text_2,
                                "commands": accepted_2,
                                "rejected_commands": rejected_2,
                                "prompt": {
                                    "user": user_prompt_2,
                                    "rendered": prompt_text_2,
                                    "chat_messages": chat_messages_2,
                                }
                                if write_prompt
                                else None,
                            },
                        ],
                        "commands": merged_cmds,
                        "rejected_commands": merged_rejected,
                        "mc_result": mc_result,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if write_task:
                        record["task"] = {
                            "task_id": task.task_id,
                            "string": task.text,
                            "difficulty": task.difficulty,
                            "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                            "target_rows_topdown": task.target_rows_topdown,
                            "allowed_blocks_by_agent": {
                                "agent1": allowed_blocks_agent1,
                                "agent2": allowed_blocks_agent2,
                            },
                        }

                    write_record(record)
                if best_metrics is None:
                    best_metrics = {}
                write_simple_record(
                    {
                        "task_id": task.task_id,
                        "string": task.text,
                        "difficulty": task.difficulty,
                        "model_id": loaded.model_id,
                        "score_shape_overlap": _pick_primary_metric(best_metrics, "score_shape_overlap"),
                        "score_components": _pick_primary_metric(best_metrics, "score_components"),
                    }
                )
                _eprint(f"[{idx}/{len(tasks)}] {task.task_id} samples={n} {dt:.2f}s")

    finally:
        close_writer()
        close_simple_writer()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
