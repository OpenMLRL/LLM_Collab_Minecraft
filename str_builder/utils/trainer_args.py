from __future__ import annotations

from typing import Any, Dict, Optional
import inspect

from comlrl.trainers.actor_critic import IACConfig  # type: ignore
from comlrl.trainers.actor_critic import MAACConfig  # type: ignore
from comlrl.trainers.reinforce import MAGRPOConfig  # type: ignore


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


def _as_opt_int(x: Any, default: Optional[int]) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("none", "null", ""):
            return None
        try:
            return int(float(s))
        except Exception:
            return default
    return default


def _as_bool(x: Any, default: bool) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
    return bool(x)


def get_trainer_args(cfg: Dict[str, Any]) -> MAGRPOConfig:
    tr = cfg.get("magrpo") or {}
    if not isinstance(tr, dict):
        tr = {}

    lr_val = tr.get("learning_rate", tr.get("lr", 3e-5))

    joint_mode = tr.get("joint_mode", tr.get("joint_action_mode", None))
    joint_mode_str = str(joint_mode or "aligned").strip().lower()
    if joint_mode_str in ("align", "aligned"):
        joint_mode_str = "aligned"
    elif joint_mode_str in ("cross", "crossed"):
        joint_mode_str = "cross"

    candidate = {
        "num_turns": _as_int(tr.get("num_turns", 1), 1),
        "num_train_epochs": _as_int(tr.get("num_train_epochs", 3), 3),
        "learning_rate": _as_float(lr_val, 3e-5),
        "logging_steps": _as_int(tr.get("logging_steps", 50), 50),
        "num_generations": _as_int(tr.get("num_generations", 4), 4),
        "max_new_tokens": _as_int(tr.get("max_new_tokens", 512), 512),
        "temperature": _as_float(tr.get("temperature", 0.2), 0.2),
        "top_p": _as_float(tr.get("top_p", 0.95), 0.95),
    }
    if "top_k" in tr:
        candidate["top_k"] = _as_opt_int(tr.get("top_k", None), None)
    candidate.update(
        {
            "discount": _as_float(tr.get("discount", 0.9), 0.9),
            "joint_mode": joint_mode_str,
        }
    )
    if "termination_threshold" in tr:
        candidate["termination_threshold"] = _as_opt_float(
            tr.get("termination_threshold", None), None
        )
    candidate.update(
        {
            "rollout_buffer_size": _as_int(tr.get("rollout_buffer_size", 2), 2),
            "eval_interval": _as_int(tr.get("eval_interval", 16), 16),
            "eval_num_samples": _as_int(tr.get("eval_num_samples", 4), 4),
            "eval_batch_size": _as_int(tr.get("eval_batch_size", 1), 1),
        }
    )

    try:
        params = set(inspect.signature(MAGRPOConfig.__init__).parameters.keys())
    except Exception:
        params = set()
    params.discard("self")
    params.discard("args")
    params.discard("kwargs")
    filtered = {k: v for k, v in candidate.items() if k in params}

    cfg_obj = MAGRPOConfig(**filtered)

    if not hasattr(cfg_obj, "learning_rate"):
        setattr(cfg_obj, "learning_rate", _as_float(lr_val, 3e-5))
    return cfg_obj


def get_maac_args(cfg: Dict[str, Any], *, model_name: Optional[str] = None) -> MAACConfig:
    tr = cfg.get("maac") or {}
    if not isinstance(tr, dict):
        tr = {}

    critic_model = tr.get("critic_model") or tr.get("critic_model_name_or_path") or model_name
    if critic_model is None:
        raise ValueError("maac.critic_model_name_or_path must be provided")

    adv_norm = tr.get("advantage_normalization", tr.get("normalize_advantage", True))

    candidate = {
        "num_turns": _as_int(tr.get("num_turns", 1), 1),
        "num_train_epochs": _as_int(tr.get("num_train_epochs", 40), 40),
        "actor_learning_rate": _as_float(tr.get("actor_learning_rate", 5e-6), 5e-6),
        "critic_learning_rate": _as_float(
            tr.get("critic_learning_rate", 5e-6), 5e-6
        ),
        "weight_decay": _as_float(tr.get("weight_decay", 0.0), 0.0),
        "adam_beta1": _as_float(tr.get("adam_beta1", 0.9), 0.9),
        "adam_beta2": _as_float(tr.get("adam_beta2", 0.999), 0.999),
        "adam_epsilon": _as_float(tr.get("adam_epsilon", 1e-8), 1e-8),
        "max_grad_norm": _as_float(tr.get("max_grad_norm", 0.5), 0.5),
        "rollout_buffer_size": _as_int(tr.get("rollout_buffer_size", 8), 8),
        "value_loss_coef": _as_float(tr.get("value_loss_coef", 0.6), 0.6),
        "advantage_normalization": _as_bool(adv_norm, True),
        "max_new_tokens": _as_int(tr.get("max_new_tokens", 256), 256),
        "temperature": _as_float(tr.get("temperature", 0.6), 0.6),
        "top_p": _as_float(tr.get("top_p", 0.6), 0.6),
        "top_k": _as_opt_int(tr.get("top_k", None), None),
        "do_sample": _as_bool(tr.get("do_sample", True), True),
        "num_agents": _as_int(tr.get("num_agents", 2), 2),
        "num_generations": _as_int(tr.get("num_generations", 1), 1),
        "critic_model_name_or_path": critic_model,
        "discount": _as_float(tr.get("discount", 0.9), 0.9),
        "critic_type": str(tr.get("critic_type", "v")),
        "early_termination_threshold": _as_opt_float(
            tr.get("early_termination_threshold", tr.get("termination_threshold", None)),
            None,
        ),
        "eval_interval": _as_int(tr.get("eval_interval", 16), 16),
        "eval_num_samples": _as_int(tr.get("eval_num_samples", 4), 4),
        "eval_batch_size": _as_int(tr.get("eval_batch_size", 1), 1),
        "logging_steps": _as_int(tr.get("logging_steps", 1), 1),
        "pad_token_id": _as_opt_int(tr.get("pad_token_id", None), None),
    }

    try:
        params = set(inspect.signature(MAACConfig.__init__).parameters.keys())
    except Exception:
        params = set()
    params.discard("self")
    params.discard("args")
    params.discard("kwargs")
    filtered = {k: v for k, v in candidate.items() if k in params}

    return MAACConfig(**filtered)


def get_iac_args(cfg: Dict[str, Any], *, model_name: Optional[str] = None) -> IACConfig:
    tr = cfg.get("iac") or {}
    if not isinstance(tr, dict):
        tr = {}

    use_separate_critic = _as_bool(tr.get("use_separate_critic", True), True)
    critic_model = tr.get("critic_model") or tr.get("critic_model_name_or_path") or model_name
    if use_separate_critic and critic_model is None:
        raise ValueError("iac.critic_model_name_or_path must be provided when use_separate_critic is true")

    adv_norm = tr.get("advantage_normalization", tr.get("normalize_advantage", True))

    candidate = {
        "num_turns": _as_int(tr.get("num_turns", 1), 1),
        "num_train_epochs": _as_int(tr.get("num_train_epochs", 40), 40),
        "actor_learning_rate": _as_float(tr.get("actor_learning_rate", 5e-6), 5e-6),
        "critic_learning_rate": _as_opt_float(
            tr.get("critic_learning_rate", 5e-6), 5e-6
        ),
        "weight_decay": _as_float(tr.get("weight_decay", 0.0), 0.0),
        "adam_beta1": _as_float(tr.get("adam_beta1", 0.9), 0.9),
        "adam_beta2": _as_float(tr.get("adam_beta2", 0.999), 0.999),
        "adam_epsilon": _as_float(tr.get("adam_epsilon", 1e-8), 1e-8),
        "max_grad_norm": _as_float(tr.get("max_grad_norm", 0.5), 0.5),
        "rollout_buffer_size": _as_int(tr.get("rollout_buffer_size", 8), 8),
        "value_loss_coef": _as_float(tr.get("value_loss_coef", 0.6), 0.6),
        "value_clip_range": _as_opt_float(tr.get("value_clip_range", 0.05), 0.05),
        "advantage_normalization": _as_bool(adv_norm, True),
        "max_new_tokens": _as_int(tr.get("max_new_tokens", 256), 256),
        "temperature": _as_float(tr.get("temperature", 0.6), 0.6),
        "top_p": _as_float(tr.get("top_p", 0.6), 0.6),
        "top_k": _as_opt_int(tr.get("top_k", None), None),
        "do_sample": _as_bool(tr.get("do_sample", True), True),
        "num_agents": _as_int(tr.get("num_agents", 2), 2),
        "num_generations": _as_int(tr.get("num_generations", 1), 1),
        "use_separate_critic": use_separate_critic,
        "critic_model_name_or_path": critic_model if use_separate_critic else None,
        "critic_value_head_hidden_dim": _as_opt_int(
            tr.get("critic_value_head_hidden_dim", None), None
        ),
        "value_head_hidden_dim": _as_opt_int(tr.get("value_head_hidden_dim", None), None),
        "discount": _as_float(tr.get("discount", 0.9), 0.9),
        "early_termination_threshold": _as_opt_float(
            tr.get("early_termination_threshold", tr.get("termination_threshold", None)),
            None,
        ),
        "eval_interval": _as_int(tr.get("eval_interval", 16), 16),
        "eval_num_samples": _as_int(tr.get("eval_num_samples", 4), 4),
        "eval_batch_size": _as_int(tr.get("eval_batch_size", 1), 1),
        "logging_steps": _as_int(tr.get("logging_steps", 1), 1),
        "pad_token_id": _as_opt_int(tr.get("pad_token_id", None), None),
    }

    try:
        params = set(inspect.signature(IACConfig.__init__).parameters.keys())
    except Exception:
        params = set()
    params.discard("self")
    params.discard("args")
    params.discard("kwargs")
    filtered = {k: v for k, v in candidate.items() if k in params}

    return IACConfig(**filtered)
