from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: Dict[str, Any], override: str) -> Dict[str, Any]:
    """Apply comma-separated `a.b.c=value` overrides into a config dict."""
    if not override:
        return cfg
    for kv in str(override).split(","):
        if not kv.strip() or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue

        if v.lower() in ("true", "false"):
            vv: Any = v.lower() == "true"
        else:
            try:
                vv = int(v) if v.isdigit() else float(v)
            except Exception:
                vv = v

        cur: Dict[str, Any] = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = vv
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


def resolve_job_id() -> str:
    jid = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    if jid:
        return str(jid)
    return time.strftime("nojid-%Y%m%d-%H%M%S")


def expand_jobid_placeholder(path: str) -> str:
    if not isinstance(path, str) or not path:
        return path
    if "[jobid]" in path:
        return path.replace("[jobid]", resolve_job_id())
    return path

