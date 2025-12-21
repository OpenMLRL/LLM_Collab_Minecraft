from __future__ import annotations

from typing import Any, Dict, Optional
import inspect
import os

from comlrl.trainers.magrpo import MAGRPOConfig  # type: ignore

from LLM_Collab_MC.str_painter.utils.config import expand_jobid_placeholder


def _as_int(x: Any, default: int) -> int:
    if x is None or isinstance(x, bool):
        return int(default)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            if s.lower().startswith("0x"):
                return int(s, 16)
            return int(float(s))
        except Exception:
            return int(default)
    return int(default)


def _as_float(x: Any, default: float) -> float:
    if x is None or isinstance(x, bool):
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return float(default)
    return float(default)


def _as_opt_float(x: Any, default: Optional[float]) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("none", "null", ""):
            return None
        try:
            return float(s)
        except Exception:
            return default
    return default


def get_trainer_args(cfg: Dict[str, Any]) -> MAGRPOConfig:
    tr = cfg.get("trainer") or {}
    if not isinstance(tr, dict):
        tr = {}

    output_dir_cfg = tr.get("output_dir", os.path.join(os.getcwd(), "runs"))
    output_dir_resolved = expand_jobid_placeholder(str(output_dir_cfg))

    lr_val = tr.get("learning_rate", tr.get("lr", 3e-5))

    joint_mode = tr.get("joint_mode", tr.get("joint_action_mode", None))
    joint_mode_str = str(joint_mode or "aligned").strip().lower()
    if joint_mode_str in ("align", "aligned"):
        joint_mode_str = "aligned"
    elif joint_mode_str in ("cross", "crossed"):
        joint_mode_str = "cross"

    candidate = {
        "output_dir": output_dir_resolved,
        "num_train_epochs": _as_int(tr.get("num_train_epochs", 3), 3),
        "per_device_train_batch_size": _as_int(tr.get("per_device_train_batch_size", 1), 1),
        "learning_rate": _as_float(lr_val, 3e-5),
        "logging_steps": _as_int(tr.get("logging_steps", 50), 50),
        "save_steps": _as_int(tr.get("save_steps", 200), 200),
        "num_generations": _as_int(tr.get("num_generations", 4), 4),
        "max_new_tokens": _as_int(tr.get("max_new_tokens", 512), 512),
        "temperature": _as_float(tr.get("temperature", 0.2), 0.2),
        "top_p": _as_float(tr.get("top_p", 0.95), 0.95),
        "num_turns": _as_int(tr.get("num_turns", 1), 1),
        "joint_mode": joint_mode_str,
        "normalize_advantage": bool(tr.get("normalize_advantage", False)),
        "epsilon_clip": _as_opt_float(tr.get("epsilon_clip", None), None),
    }

    try:
        params = set(inspect.signature(MAGRPOConfig.__init__).parameters.keys())
    except Exception:
        params = set()
    params.discard("self")
    params.discard("args")
    params.discard("kwargs")
    filtered = {k: v for k, v in candidate.items() if k in params}

    cfg_obj = MAGRPOConfig(**filtered)

    try:
        if not hasattr(cfg_obj, "learning_rate"):
            setattr(cfg_obj, "learning_rate", _as_float(lr_val, 3e-5))
    except Exception:
        pass

    return cfg_obj
