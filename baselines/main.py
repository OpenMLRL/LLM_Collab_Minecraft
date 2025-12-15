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


def _iter_dataset_task_paths(
    dataset_root: Path,
    include_glob: str = "**/*.json",
    exclude_globs: Iterable[str] = (),
) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset.root not found: {dataset_root}")

    task_paths: list[Path] = []
    for path in dataset_root.glob(include_glob):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(dataset_root)
        except ValueError:
            rel = path
        if any(rel.match(pattern) for pattern in exclude_globs):
            continue
        task_paths.append(path)
    return sorted(task_paths)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_palette_map(palette: Any) -> str:
    if isinstance(palette, list):
        lines = [f"{i}: {b}" for i, b in enumerate(palette)]
        return "\n".join(lines)
    return json.dumps(palette, ensure_ascii=False)


def _format_grid_rows(task: dict[str, Any]) -> str:
    target = task.get("target_spec") or {}
    if not isinstance(target, dict):
        return ""
    rows = target.get("grid_rows")
    if isinstance(rows, list):
        return "\n".join(str(r) for r in rows)
    grid = target.get("grid_yx")
    if isinstance(grid, list):
        # Fallback for older format: list[list[int]]
        try:
            return "\n".join("".join(str(int(v)) for v in row) for row in grid)
        except Exception:
            return json.dumps(grid, ensure_ascii=False)
    return ""


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
            # Some tokenizers don't support add_generation_prompt.
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            return text, messages

    parts = []
    if system_prompt:
        parts.append(f"[SYSTEM]\n{system_prompt}")
    parts.append(f"[USER]\n{user_prompt}")
    parts.append("[ASSISTANT]\n")
    return "\n\n".join(parts), None


def _extract_json(text: str) -> tuple[Any | None, str | None]:
    raw = text.strip()
    if not raw:
        return None, "empty output"

    if "```" in raw:
        # Prefer the first fenced block content if present.
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
            raw = raw.lstrip().removeprefix("json").lstrip()

    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass

    # Try a best-effort substring parse.
    candidates: list[str] = []
    lbrace = raw.find("{")
    rbrace = raw.rfind("}")
    if 0 <= lbrace < rbrace:
        candidates.append(raw[lbrace : rbrace + 1])

    lbrack = raw.find("[")
    rbrack = raw.rfind("]")
    if 0 <= lbrack < rbrack:
        candidates.append(raw[lbrack : rbrack + 1])

    for cand in candidates:
        try:
            return json.loads(cand), None
        except json.JSONDecodeError:
            continue

    return None, "failed to parse JSON"


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
    # If the fence language is present, drop it.
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
    palette: list[str],
    world_bbox_from: list[int],
    world_bbox_to: list[int],
    max_commands: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    allowed_blocks = {_normalize_block_id(b) for b in palette}
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
            if block not in allowed_blocks:
                rejected.append({"line": line, "reason": f"block not in palette: {block}"})
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
            if block not in allowed_blocks:
                rejected.append({"line": line, "reason": f"block not in palette: {block}"})
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
        check=False,
    )

    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError(f"mc_executor produced no stdout. stderr:\n{proc.stderr}")

    # mineflayer (or its deps) may print stack traces to stdout; extract the last JSON object line.
    for line in reversed(stdout.splitlines()):
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


def _parse_bbox(bbox_obj: Any) -> tuple[list[int], list[int]]:
    if not isinstance(bbox_obj, dict):
        raise ValueError("task.bbox must be an object with from/to")
    bfrom = bbox_obj.get("from")
    bto = bbox_obj.get("to")
    if not (isinstance(bfrom, list) and isinstance(bto, list) and len(bfrom) == 3 and len(bto) == 3):
        raise ValueError("task.bbox.from/to must be 3-element lists")
    return [int(bfrom[0]), int(bfrom[1]), int(bfrom[2])], [int(bto[0]), int(bto[1]), int(bto[2])]


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


def _score_2d_painting(
    *,
    task_obj: dict[str, Any],
    world_origin: list[int],
    world_scan_blocks: list[dict[str, Any]],
) -> tuple[int, int, float]:
    palette = task_obj.get("palette")
    if not isinstance(palette, list) or not palette:
        raise ValueError("task.palette must be a non-empty list")
    palette_norm = [_normalize_block_id(str(b)) for b in palette]

    target = task_obj.get("target_spec") or {}
    if not isinstance(target, dict):
        raise ValueError("task.target_spec must be an object")
    grid_rows = target.get("grid_rows")
    if not isinstance(grid_rows, list) or not grid_rows:
        raise ValueError("task.target_spec.grid_rows must be a list of strings")
    grid_rows_str = [str(r) for r in grid_rows]

    local_from, local_to = _parse_bbox(task_obj.get("bbox"))
    min_lx = min(local_from[0], local_to[0])
    max_lx = max(local_from[0], local_to[0])
    min_ly = min(local_from[1], local_to[1])
    max_ly = max(local_from[1], local_to[1])
    min_lz = min(local_from[2], local_to[2])
    max_lz = max(local_from[2], local_to[2])
    width = max_lx - min_lx + 1
    height = max_ly - min_ly + 1
    depth = max_lz - min_lz + 1

    if len(grid_rows_str) != height:
        raise ValueError(f"grid_rows height mismatch: got {len(grid_rows_str)}, want {height}")
    for r in grid_rows_str:
        if len(r) != width:
            raise ValueError(f"grid_rows width mismatch: got {len(r)}, want {width}")

    # Build a lookup for observed blocks.
    observed: dict[tuple[int, int, int], str | None] = {}
    for b in world_scan_blocks:
        pos = b.get("pos")
        name = b.get("name")
        if not (isinstance(pos, list) and len(pos) == 3):
            continue
        x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
        observed[(x, y, z)] = _normalize_block_id(str(name)) if name is not None else None

    correct = 0
    total = 0
    for wy, row in enumerate(grid_rows_str):
        for wx, ch in enumerate(row):
            try:
                idx = int(ch)
            except ValueError:
                continue
            if idx < 0 or idx >= len(palette_norm):
                continue
            expected_block = palette_norm[idx]
            lx = min_lx + wx
            ly = min_ly + wy
            for lz in range(min_lz, max_lz + 1):
                wx_abs = world_origin[0] + lx
                wy_abs = world_origin[1] + ly
                wz_abs = world_origin[2] + lz
                got = observed.get((wx_abs, wy_abs, wz_abs))
                total += 1
                if got == expected_block:
                    correct += 1

    acc = (correct / total) if total else 0.0
    return correct, total, acc


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
        err = str(e)
        if "model type `qwen3`" in err or "model type `qwen3`" in repr(e) or "KeyError: 'qwen3'" in err:
            raise RuntimeError(
                "Transformers does not recognize the 'qwen3' architecture for this checkpoint. "
                "Fix: upgrade your environment, e.g. `pip install -U transformers tokenizers huggingface-hub` "
                "(Qwen3 support requires a newer Transformers version)."
            ) from e
        raise RuntimeError(
            "Failed to load the model/tokenizer from Hugging Face. "
            "If you see 401/403/404 errors, the repo may be gated/private; "
            "make sure you have access (and HF_TOKEN is set), or switch to a public model id "
            "like 'Qwen/Qwen3-4B-Instruct-2507' or a local path."
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

    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        raise ValueError("dataset must be a mapping")
    dataset_root = _resolve_path(config_path, dataset_cfg.get("root")) or (config_path.parent / "../dataset").resolve()
    include_glob = str(dataset_cfg.get("include_glob") or "**/*.json")
    exclude_globs = dataset_cfg.get("exclude_glob") or []
    if isinstance(exclude_globs, str):
        exclude_globs = [exclude_globs]

    output_cfg = cfg.get("output") or {}
    if not isinstance(output_cfg, dict):
        raise ValueError("output must be a mapping")
    output_path = _resolve_path(config_path, output_cfg.get("path")) or (config_path.parent / "outputs/output.jsonl")
    output_path = output_path.resolve()
    overwrite = bool(output_cfg.get("overwrite", False))
    write_prompt = bool(output_cfg.get("write_prompt", True))
    write_task = bool(output_cfg.get("write_task", True))

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (set output.overwrite=true to overwrite): {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_paths = _iter_dataset_task_paths(dataset_root, include_glob=include_glob, exclude_globs=exclude_globs)
    if args.limit is not None:
        task_paths = task_paths[: args.limit]
    if not task_paths:
        _eprint(f"No tasks found under: {dataset_root} (glob={include_glob})")
        return 2

    prompt_cfg = cfg.get("prompt") or {}
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt must be a mapping")
    system_prompt = str(prompt_cfg.get("system") or "").rstrip()
    user_template = str(prompt_cfg.get("user_template") or "{task_json}").rstrip()

    generation_cfg = cfg.get("generation") or {}
    if not isinstance(generation_cfg, dict):
        raise ValueError("generation must be a mapping")
    use_chat_template = bool(generation_cfg.get("use_chat_template", False))

    if args.dry_run:
        _eprint(f"[dry-run] tasks={len(task_paths)} dataset_root={dataset_root}")
        for path in task_paths[:3]:
            task = _load_json(path)
            if not isinstance(task, dict):
                continue
            bbox_from, bbox_to = _parse_bbox(task.get("bbox"))
            # For dry-run we assume local coords only.
            world_bbox_from = bbox_from
            world_bbox_to = bbox_to
            user_prompt = user_template.format(
                task_json=json.dumps(task, ensure_ascii=False, separators=(",", ":")),
                task_id=task.get("task_id", path.stem),
                palette_map=_format_palette_map(task.get("palette")),
                grid_rows=_format_grid_rows(task),
                world_bbox_from=json.dumps(world_bbox_from, separators=(",", ":")),
                world_bbox_to=json.dumps(world_bbox_to, separators=(",", ":")),
            )
            _eprint(f"\n=== {task.get('task_id', path.stem)} ===\n{user_prompt}\n")
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
    _eprint(f"Tasks: {len(task_paths)}")
    _eprint(f"Output: {output_path}")
    if args.verbose:
        try:
            import torch  # type: ignore

            _eprint(
                "Torch: "
                f"version={torch.__version__} "
                f"cuda_version={getattr(torch.version, 'cuda', None)} "
                f"cuda_available={torch.cuda.is_available()} "
                f"device_count={torch.cuda.device_count()}"
            )
            if getattr(torch.version, "cuda", None) in (None, "None") and "+cpu" in str(torch.__version__):
                _eprint(
                    "WARNING: CPU-only PyTorch detected (torch.version.cuda is None). "
                    "Generation will be extremely slow; install a CUDA-enabled PyTorch build."
                )
            if torch.cuda.is_available():
                _eprint(f"CUDA: device0={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        try:
            from collections import Counter

            dm = getattr(loaded.model, "hf_device_map", None)
            if isinstance(dm, dict):
                counts = Counter(str(v) for v in dm.values())
                _eprint(f"HF device map: {dict(counts)}")
            else:
                _eprint(f"Model device: {getattr(loaded.model, 'device', None)}")
        except Exception:
            pass

    mc_cfg = cfg.get("minecraft") or {}
    if not isinstance(mc_cfg, dict):
        raise ValueError("minecraft must be a mapping")

    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_host = str(mc_cfg.get("host") or "127.0.0.1")
    mc_port = int(mc_cfg.get("port") or 25565)
    mc_username = str(mc_cfg.get("username") or "executor_bot")
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
                    "Update baselines/mc_executor.cjs accordingly."
                )
            world_origin = [
                int(spawn[0]) + mc_spawn_offset_vec[0],
                int(spawn[1]) + mc_spawn_offset_vec[1],
                int(spawn[2]) + mc_spawn_offset_vec[2],
            ]
        else:
            raise ValueError("minecraft.origin_mode must be one of: fixed, spawn_offset")

        _eprint(f"Minecraft enabled: host={mc_host} port={mc_port} username={mc_username} origin={world_origin}")

    def _open_output_writer(path: Path):
        if path.suffix.lower() == ".jsonl":
            f = path.open("w", encoding="utf-8")

            def write_record(rec: dict[str, Any]) -> None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            def close() -> None:
                f.close()

            return write_record, close

        # Default: write a JSON array incrementally.
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

    write_record, close_writer = _open_output_writer(output_path)
    try:
        for idx, path in enumerate(task_paths, start=1):
            task_obj = _load_json(path)
            if not isinstance(task_obj, dict):
                _eprint(f"Skipping non-object json: {path}")
                continue

            task_id = str(task_obj.get("task_id") or path.stem)
            palette_map = _format_palette_map(task_obj.get("palette"))
            grid_rows = _format_grid_rows(task_obj)

            local_from, local_to = _parse_bbox(task_obj.get("bbox"))
            w_from = local_from
            w_to = local_to
            if world_origin is not None:
                w_from, w_to = _compute_world_bbox(local_from=local_from, local_to=local_to, world_origin=world_origin)
            user_prompt = user_template.format(
                task_json=json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")),
                task_id=task_id,
                palette_map=palette_map,
                grid_rows=grid_rows,
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
                _eprint(f"[{idx}/{len(task_paths)}] {task_id} generating...")
            t0 = time.time()
            if args.stream_llm_output and int(generation_cfg.get("num_return_sequences") or 1) != 1:
                _eprint("--stream-llm-output only supports num_return_sequences=1; disabling streaming.")
                stream_output = False
            else:
                stream_output = bool(args.stream_llm_output)
            if stream_output:
                _eprint(f"--- streaming LLM output task_id={task_id} ---")
            outputs = _generate_with_transformers(loaded, prompt_text, generation_cfg, stream_output=stream_output)
            dt = time.time() - t0
            if args.verbose:
                _eprint(f"[{idx}/{len(task_paths)}] {task_id} generated samples={len(outputs)} dt={dt:.2f}s")

            for sample_id, output_text in enumerate(outputs):
                if args.print_llm_output:
                    max_chars = int(args.print_llm_output_max_chars or 0)
                    out = output_text if max_chars <= 0 else output_text[:max_chars]
                    suffix = "" if max_chars <= 0 or len(output_text) <= max_chars else "\n... [truncated]"
                    _eprint(f"\n=== LLM output task_id={task_id} sample_id={sample_id} ===\n{out}{suffix}\n")

                lines = _extract_command_lines(output_text)
                palette = task_obj.get("palette")
                if not isinstance(palette, list):
                    palette = []
                accepted_cmds, rejected_cmds = _validate_and_normalize_mc_commands(
                    lines=lines,
                    palette=[str(b) for b in palette],
                    world_bbox_from=w_from,
                    world_bbox_to=w_to,
                    max_commands=mc_max_commands,
                )

                if args.print_commands:
                    _eprint(
                        f"[{task_id} sample={sample_id}] accepted={len(accepted_cmds)} rejected={len(rejected_cmds)}"
                    )
                    if accepted_cmds:
                        _eprint("\n".join(accepted_cmds[:50]) + ("\n... [truncated]" if len(accepted_cmds) > 50 else ""))

                mc_result: dict[str, Any] | None = None
                metrics: dict[str, Any] = {"latency_s": dt}

                # Offline eval: simulate /fill + /setblock and score within bbox.
                score_origin = world_origin if world_origin is not None else [0, 0, 0]
                try:
                    sim_blocks = _simulate_commands_to_scan_blocks(
                        commands=accepted_cmds,
                        world_bbox_from=w_from,
                        world_bbox_to=w_to,
                    )
                    correct, total, acc = _score_2d_painting(
                        task_obj=task_obj,
                        world_origin=score_origin,
                        world_scan_blocks=sim_blocks,
                    )
                    metrics.update({"correct": correct, "total": total, "accuracy": acc})
                except Exception as e:
                    metrics.update({"correct": 0, "total": 0, "accuracy": 0.0, "sim_error": str(e)})
                if mc_enabled and world_origin is not None:
                    # Pre-commands: teleport near the target area and clear it.
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
                            mc_correct, mc_total, mc_acc = _score_2d_painting(
                                task_obj=task_obj,
                                world_origin=world_origin,
                                world_scan_blocks=scan_blocks,
                            )
                            metrics.update({"mc_correct": mc_correct, "mc_total": mc_total, "mc_accuracy": mc_acc})
                        else:
                            metrics.update({"mc_error": "missing scan blocks"})
                    except Exception as e:
                        metrics.update({"mc_error": str(e)})

                record: dict[str, Any] = {
                    "run_name": run_name,
                    "task_id": task_id,
                    "task_path": str(path),
                    "sample_id": sample_id,
                    "model": {
                        "backend": backend,
                        "model_id": loaded.model_id,
                    },
                    "generation": generation_cfg,
                    "metrics": metrics,
                    "output_text": output_text,
                    "commands": accepted_cmds,
                    "rejected_commands": rejected_cmds,
                    "mc_result": mc_result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if "correct" in metrics and "total" in metrics:
                    record["correct"] = metrics.get("correct")
                    record["total"] = metrics.get("total")
                    try:
                        record["correct_over_total"] = f"{int(metrics.get('correct'))}/{int(metrics.get('total'))}"
                    except Exception:
                        record["correct_over_total"] = None
                if write_task:
                    record["task"] = task_obj
                if write_prompt:
                    record["prompt"] = {
                        "system": system_prompt,
                        "user": user_prompt,
                        "rendered": prompt_text,
                        "chat_messages": chat_messages,
                    }

                write_record(record)

            _eprint(f"[{idx}/{len(task_paths)}] {task_id} samples={len(outputs)} {dt:.2f}s")
    finally:
        close_writer()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
