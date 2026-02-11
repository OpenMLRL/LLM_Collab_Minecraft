from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_overrides(
    cfg: Dict[str, Any], overrides: Sequence[str] | str
) -> Dict[str, Any]:
    """Apply `a.b.c=value` overrides into a config dict."""
    if not overrides:
        return cfg
    items: Iterable[str]
    if isinstance(overrides, str):
        items = [overrides]
    else:
        items = overrides
    for item in items:
        if item is None:
            continue
        kv = str(item).strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Invalid override format: {kv}. Use key=value format.")
        key, value = kv.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        parsed = _parse_override_value(value)
        cur: Dict[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = parsed
    return cfg


def resolve_path(config_path: str, maybe_rel: Any) -> str:
    """Resolve a config path value relative to the YAML file location."""
    if maybe_rel is None:
        raise ValueError("path is required")
    p = str(maybe_rel)
    if not p:
        raise ValueError("path is required")
    path = Path(p).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    base = Path(config_path).expanduser().resolve().parent
    return str((base / path).resolve())
