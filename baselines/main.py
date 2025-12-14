#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
            user_prompt = user_template.format(
                task_json=json.dumps(task, ensure_ascii=False, separators=(",", ":")),
                task_id=task.get("task_id", path.stem),
                palette_map=_format_palette_map(task.get("palette")),
                grid_rows=_format_grid_rows(task),
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

    open_mode = "w" if overwrite else "w"
    with output_path.open(open_mode, encoding="utf-8") as out_f:
        for idx, path in enumerate(task_paths, start=1):
            task_obj = _load_json(path)
            if not isinstance(task_obj, dict):
                _eprint(f"Skipping non-object json: {path}")
                continue

            task_id = str(task_obj.get("task_id") or path.stem)
            palette_map = _format_palette_map(task_obj.get("palette"))
            grid_rows = _format_grid_rows(task_obj)
            user_prompt = user_template.format(
                task_json=json.dumps(task_obj, ensure_ascii=False, separators=(",", ":")),
                task_id=task_id,
                palette_map=palette_map,
                grid_rows=grid_rows,
            )

            prompt_text, chat_messages = _render_prompt(
                tokenizer=loaded.tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_chat_template=use_chat_template,
            )

            t0 = time.time()
            outputs = _generate_with_transformers(loaded, prompt_text, generation_cfg)
            dt = time.time() - t0

            for sample_id, output_text in enumerate(outputs):
                parsed, parse_error = _extract_json(output_text)
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
                    "metrics": {
                        "latency_s": dt,
                    },
                    "output_text": output_text,
                    "parsed_json": parsed,
                    "parse_error": parse_error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if write_task:
                    record["task"] = task_obj
                if write_prompt:
                    record["prompt"] = {
                        "system": system_prompt,
                        "user": user_prompt,
                        "rendered": prompt_text,
                        "chat_messages": chat_messages,
                    }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            _eprint(f"[{idx}/{len(task_paths)}] {task_id} samples={len(outputs)} {dt:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
