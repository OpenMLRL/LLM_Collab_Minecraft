from __future__ import annotations

from typing import Any, Dict


def build_reward_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    reward_cfg = cfg.get("reward_shaping") or {}
    task_cfg = cfg.get("task") or {}
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    return {
        "path_slots": int(task_cfg.get("max_path_len", 4)),
        "extraction_limit": int(task_cfg.get("extraction_limit", 2)),
        "comm_limit": int(task_cfg.get("comm_limit", 1)),
        "progress_reward_scale": float(reward_cfg.get("progress_reward_scale", 8.0)),
        "terminal_bonus": float(reward_cfg.get("terminal_bonus", 7.0)),
        "move_cost_scale": float(reward_cfg.get("move_cost_scale", 0.0)),
        "comm_cost_scale": float(reward_cfg.get("comm_cost_scale", 0.0)),
        "wasted_extraction_penalty": float(reward_cfg.get("wasted_extraction_penalty", 0.1)),
        "move_to_zone_bonus_scale": float(reward_cfg.get("move_to_zone_bonus_scale", 0.05)),
        "useful_comm_bonus_scale": float(reward_cfg.get("useful_comm_bonus_scale", 0.1)),
        "first_enter_zone_bonus_scale": float(reward_cfg.get("first_enter_zone_bonus_scale", 0.15)),
    }


__all__ = ["build_reward_config"]
