#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z0-9_.-]+)\}")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "model"


def _build_template_context(model_name_or_path: str) -> dict[str, str]:
    base = Path(model_name_or_path).name
    return {
        "model_name_or_path": model_name_or_path,
        "model_basename": base,
        "model_slug": _slugify(base),
        "model_slug_full": _slugify(model_name_or_path),
    }


def _expand_templates(value: str, context: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise ValueError(f"Unknown template key: {key}")
        return context[key]

    return _TEMPLATE_RE.sub(repl, value)


def _maybe_expand_templates(value: str, context: dict[str, str]) -> str:
    if "${" not in value:
        return value
    if not context:
        raise ValueError("Template placeholders require model.model_name_or_path to be set")
    return _expand_templates(value, context)


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
    allowed_commands: set[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    allowed = {_normalize_block_id(b) for b in allowed_blocks}
    allowed_cmds = allowed_commands or {"setblock", "fill"}
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
            if "setblock" not in allowed_cmds:
                rejected.append({"line": line, "reason": "unsupported command: setblock"})
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
            if "fill" not in allowed_cmds:
                rejected.append({"line": line, "reason": "unsupported command: fill"})
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bbox(bbox_obj: Any) -> tuple[list[int], list[int]]:
    if not isinstance(bbox_obj, dict):
        raise ValueError("task.bbox must be an object with from/to")
    bfrom = bbox_obj.get("from")
    bto = bbox_obj.get("to")
    if not (isinstance(bfrom, list) and isinstance(bto, list) and len(bfrom) == 3 and len(bto) == 3):
        raise ValueError("task.bbox.from/to must be 3-element lists")
    return [int(bfrom[0]), int(bfrom[1]), int(bfrom[2])], [int(bto[0]), int(bto[1]), int(bto[2])]


def _parse_layers(target_spec: dict[str, Any]) -> dict[int, list[str]]:
    layers_obj = target_spec.get("layers")
    if layers_obj is None:
        raise ValueError("target_spec.layers is required")

    layers: dict[int, list[str]] = {}

    def _normalize_rows(rows_obj: Any, *, y: int) -> list[str]:
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
    local_bbox_from: list[int]
    local_bbox_to: list[int]
    palette: dict[str, str]
    layers_by_y: dict[int, list[str]]


def _load_tasks_from_json(json_path: Path) -> list[TaskSpec]:
    if not json_path.exists():
        raise FileNotFoundError(f"dataset.json_path not found: {json_path}")

    raw = _load_json(json_path)
    if isinstance(raw, list):
        task_objs = raw
    elif isinstance(raw, dict):
        task_objs = [raw]
    else:
        raise ValueError(f"dataset.json_path must contain an object or list, got {type(raw)}")

    tasks: list[TaskSpec] = []
    for idx, task_obj in enumerate(task_objs, start=1):
        if not isinstance(task_obj, dict):
            raise ValueError(f"Task entry {idx} must be an object")
        task_id = str(task_obj.get("task_id") or f"box_builder_{idx:04d}")

        palette_raw = task_obj.get("palette")
        if not isinstance(palette_raw, dict) or not palette_raw:
            raise ValueError(f"{task_id}: palette must be a non-empty mapping")
        palette: dict[str, str] = {}
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
            normalized_rows: list[str] = []
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


def _rows_to_rects(
    *,
    rows: list[str],
    palette: dict[str, str],
    min_x: int,
    min_z: int,
) -> list[tuple[int, int, int, int, str]]:
    if not rows:
        return []
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError("row width mismatch while building rectangles")

    rects: list[tuple[int, int, int, int, str]] = []
    active: dict[tuple[int, int, str], list[int | str]] = {}

    for z_idx, row in enumerate(rows):
        runs: list[tuple[int, int, str]] = []
        start = 0
        cur = row[0]
        for x in range(1, width + 1):
            if x == width or row[x] != cur:
                runs.append((start, x - 1, cur))
                start = x
                if x < width:
                    cur = row[x]

        new_active: dict[tuple[int, int, str], list[int | str]] = {}
        for x1, x2, ch in runs:
            block = palette.get(ch)
            if block is None:
                raise ValueError(f"unknown palette key {ch!r} in layer")
            key = (x1, x2, block)
            if key in active:
                rect = active.pop(key)
                rect[3] = min_z + z_idx  # extend z2
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


def _format_layers_text(
    task: TaskSpec, *, world_from: list[int] | None = None, include_air: bool = True
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
    lines: list[str] = []
    for y in range(min_y, max_y + 1):
        rows = task.layers_by_y.get(y)
        if rows is None:
            raise ValueError(f"{task.task_id}: missing layer y={y}")
        rects = _rows_to_rects(rows=rows, palette=task.palette, min_x=min_x, min_z=min_z)
        if not include_air:
            rects = [
                (x1, z1, x2, z2, block)
                for x1, z1, x2, z2, block in rects
                if _normalize_block_id(str(block)) not in ("air", "cave_air", "void_air")
            ]
        y_abs = y + offset_y
        rect_parts = [
            f"({x1 + offset_x}, {y_abs}, {z1 + offset_z}, {x2 + offset_x}, {y_abs}, {z2 + offset_z} {block})"
            for x1, z1, x2, z2, block in rects
        ]
        lines.append(f"y={y_abs}: {{{', '.join(rect_parts)}}}")
    return "\n".join(lines)


def _score_box_builder(
    *,
    task: TaskSpec,
    world_origin: list[int],
    world_scan_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    palette_norm = {k: _normalize_block_id(v) for k, v in task.palette.items()}

    observed: dict[tuple[int, int, int], str] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        observed[(x, y, z)] = _normalize_block_id(str(name)) if name is not None else "air"

    min_lx = min(task.local_bbox_from[0], task.local_bbox_to[0])
    max_lx = max(task.local_bbox_from[0], task.local_bbox_to[0])
    min_ly = min(task.local_bbox_from[1], task.local_bbox_to[1])
    max_ly = max(task.local_bbox_from[1], task.local_bbox_to[1])
    min_lz = min(task.local_bbox_from[2], task.local_bbox_to[2])
    max_lz = max(task.local_bbox_from[2], task.local_bbox_to[2])

    correct = 0
    total = 0
    for ly in range(min_ly, max_ly + 1):
        rows = task.layers_by_y.get(ly)
        if rows is None:
            raise ValueError(f"{task.task_id}: missing layer y={ly}")
        for rz, row in enumerate(rows):
            lz = min_lz + rz
            for rx, ch in enumerate(row):
                lx = min_lx + rx
                expected = palette_norm.get(ch)
                if expected is None:
                    raise ValueError(f"{task.task_id}: unknown palette key {ch!r}")
                wx = world_origin[0] + lx
                wy = world_origin[1] + ly
                wz = world_origin[2] + lz
                got = observed.get((wx, wy, wz), "air")
                total += 1
                if got == expected:
                    correct += 1

    score_match = (correct / total) if total else 0.0
    return {
        "match_correct": correct,
        "match_total": total,
        "score_match": score_match,
        "score_mean": score_match,
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

    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model must be a mapping")
    model_name_or_path = model_cfg.get("model_name_or_path")
    template_context: dict[str, str] = {}
    if isinstance(model_name_or_path, str) and model_name_or_path:
        template_context = _build_template_context(model_name_or_path)

    run_name = str(cfg.get("run_name") or "baseline")
    run_name = _maybe_expand_templates(run_name, template_context)
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

    def _as_block_list(v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            out: list[str] = []
            for x in v:
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    block_agent1_override = _as_block_list(task_cfg.get("block_agent1"))
    block_agent2_override = _as_block_list(task_cfg.get("block_agent2"))

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset must be a mapping")
    json_path = _resolve_path(config_path, dataset_cfg.get("json_path")) or (
        config_path.parent / "../../dataset/box_builder/one.json"
    ).resolve()

    tasks = _load_tasks_from_json(json_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        _eprint(f"No tasks found in: {json_path}")
        return 2

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        raise ValueError("output must be a mapping")
    output_path_value = output_cfg.get("path")
    if isinstance(output_path_value, str):
        output_path_value = _maybe_expand_templates(output_path_value, template_context)
    output_path = _resolve_path(config_path, output_path_value) or (config_path.parent / "outputs/output.jsonl")
    output_path = output_path.resolve()
    output_simple_value = output_cfg.get("simple_path")
    if isinstance(output_simple_value, str):
        output_simple_value = _maybe_expand_templates(output_simple_value, template_context)
    output_simple_path = _resolve_path(config_path, output_simple_value)
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
    inform_air = bool(prompt_cfg.get("inform_air", True))

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

    def _unique_block_list(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _build_allowed_blocks(palette: dict[str, str], overrides: list[str]) -> list[str]:
        palette_blocks = _unique_block_list(palette.values())
        palette_norm = {_normalize_block_id(b) for b in palette_blocks}
        if overrides:
            out = _unique_block_list(overrides)
            for b in out:
                if _normalize_block_id(b) not in palette_norm:
                    raise ValueError(f"Override block not in palette: {b}")
        else:
            out = list(palette_blocks)
        if "air" in palette_norm and not any(_normalize_block_id(b) == "air" for b in out):
            out.append("air")
        return out

    def _format_legend_lines(palette: dict[str, str]) -> str:
        return "\n".join(f"{k} = {v}" for k, v in palette.items())

    if args.dry_run:
        _eprint(f"[dry-run] tasks={len(tasks)} json_path={json_path}")
        for task in tasks[:3]:
            legend_lines = _format_legend_lines(task.palette)
            layers_text = _format_layers_text(
                task, world_from=task.local_bbox_from, include_air=inform_air
            )
            block_agent1_blocks = _build_allowed_blocks(task.palette, block_agent1_override)
            block_agent2_blocks = _build_allowed_blocks(task.palette, block_agent2_override)
            block_agent1_lines = "\n".join(f"- {b}" for b in block_agent1_blocks)
            block_agent2_lines = "\n".join(f"- {b}" for b in block_agent2_blocks)
            task_json_obj = {
                "task_id": task.task_id,
                "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                "palette": task.palette,
                "target_spec": {"layers": task.layers_by_y},
            }
            if num_agents == 1:
                user_prompt = user_template.format(
                    task_json=json.dumps(
                        task_json_obj,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    legend_lines=legend_lines,
                    layers_text=layers_text,
                    block_agent1_lines=block_agent1_lines,
                    block_agent2_lines=block_agent2_lines,
                    world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
                )
                _eprint(f"\n=== {task.task_id} ===\n{user_prompt}\n")
                continue

            user_prompt_1 = user_template_agent1.format(
                task_json=json.dumps(
                    task_json_obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                task_id=task.task_id,
                legend_lines=legend_lines,
                layers_text=layers_text,
                block_agent1_lines=block_agent1_lines,
                block_agent2_lines=block_agent2_lines,
                world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
            )
            user_prompt_2 = user_template_agent2.format(
                task_json=json.dumps(
                    task_json_obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                task_id=task.task_id,
                legend_lines=legend_lines,
                layers_text=layers_text,
                block_agent1_lines=block_agent1_lines,
                block_agent2_lines=block_agent2_lines,
                world_bbox_from=json.dumps(task.local_bbox_from, separators=(",", ":")),
                world_bbox_to=json.dumps(task.local_bbox_to, separators=(",", ":")),
            )
            _eprint(
                f"\n=== {task.task_id} ===\n"
                f"[agent1]\n{user_prompt_1}\n\n[agent2]\n{user_prompt_2}\n"
            )
        return 0

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
                    "Update baselines/box_builder/mc_executor.cjs accordingly."
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
        v = _pick_primary_metric(metrics, "score_match")
        return float(v) if isinstance(v, (int, float)) else float("-inf")
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

            legend_lines = _format_legend_lines(task.palette)
            layers_text = _format_layers_text(task, world_from=w_from, include_air=inform_air)
            allowed_blocks_agent1 = _build_allowed_blocks(task.palette, block_agent1_override)
            allowed_blocks_agent2 = _build_allowed_blocks(task.palette, block_agent2_override)
            block_agent1_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent1)
            block_agent2_lines = "\n".join(f"- {b}" for b in allowed_blocks_agent2)
            allowed_commands = {"fill"}
            task_json_obj = {
                "task_id": task.task_id,
                "bbox": {"from": local_from, "to": local_to},
                "palette": task.palette,
                "target_spec": {"layers": task.layers_by_y},
            }

            max_commands_per_agent = mc_max_commands if num_agents == 1 else max(1, mc_max_commands // num_agents)
            max_commands_agent1 = max_commands_per_agent + (mc_max_commands % num_agents)
            max_commands_agent2 = max_commands_per_agent

            if num_agents == 1:
                user_prompt = user_template.format(
                    task_json=json.dumps(
                        task_json_obj,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    legend_lines=legend_lines,
                    layers_text=layers_text,
                    block_agent1_lines=block_agent1_lines,
                    block_agent2_lines=block_agent2_lines,
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
                        allowed_commands=allowed_commands,
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
                            _score_box_builder(
                                task=task,
                                world_origin=score_origin,
                                world_scan_blocks=sim_blocks,
                            )
                        )
                    except Exception as e:
                        metrics.update({"score_match": 0.0, "score_mean": 0.0, "sim_error": str(e)})

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
                                mc_metrics = _score_box_builder(
                                    task=task,
                                    world_origin=world_origin,
                                    world_scan_blocks=scan_blocks,
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
                        "dataset_path": str(json_path),
                        "dataset_index": idx,
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
                            "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                            "palette": task.palette,
                            "target_spec": {"layers": task.layers_by_y},
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
                        "model_id": loaded.model_id,
                        "score_match": _pick_primary_metric(best_metrics, "score_match"),
                    }
                )
                _eprint(f"[{idx}/{len(tasks)}] {task.task_id} samples={len(outputs)} {dt:.2f}s")
            else:
                user_prompt_1 = user_template_agent1.format(
                    task_json=json.dumps(
                        task_json_obj,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    legend_lines=legend_lines,
                    layers_text=layers_text,
                    block_agent1_lines=block_agent1_lines,
                    block_agent2_lines=block_agent2_lines,
                    world_bbox_from=json.dumps(w_from, separators=(",", ":")),
                    world_bbox_to=json.dumps(w_to, separators=(",", ":")),
                )
                user_prompt_2 = user_template_agent2.format(
                    task_json=json.dumps(
                        task_json_obj,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    task_id=task.task_id,
                    legend_lines=legend_lines,
                    layers_text=layers_text,
                    block_agent1_lines=block_agent1_lines,
                    block_agent2_lines=block_agent2_lines,
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
                        allowed_commands=allowed_commands,
                    )
                    accepted_2, rejected_2 = _validate_and_normalize_mc_commands(
                        lines=lines_2,
                        allowed_blocks=allowed_blocks_agent2,
                        world_bbox_from=w_from,
                        world_bbox_to=w_to,
                        max_commands=max_commands_agent2,
                        allowed_commands=allowed_commands,
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
                            _score_box_builder(
                                task=task,
                                world_origin=score_origin,
                                world_scan_blocks=sim_blocks,
                            )
                        )
                    except Exception as e:
                        metrics.update({"score_match": 0.0, "score_mean": 0.0, "sim_error": str(e)})

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
                                mc_metrics = _score_box_builder(
                                    task=task,
                                    world_origin=world_origin,
                                    world_scan_blocks=scan_blocks,
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
                        "dataset_path": str(json_path),
                        "dataset_index": idx,
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
                            "bbox": {"from": task.local_bbox_from, "to": task.local_bbox_to},
                            "palette": task.palette,
                            "target_spec": {"layers": task.layers_by_y},
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
                        "model_id": loaded.model_id,
                        "score_match": _pick_primary_metric(best_metrics, "score_match"),
                    }
                )
                _eprint(f"[{idx}/{len(tasks)}] {task.task_id} samples={n} {dt:.2f}s")

    finally:
        close_writer()
        close_simple_writer()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
