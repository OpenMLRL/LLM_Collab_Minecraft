from __future__ import annotations

from typing import Any, Dict


def build_reward_config(cfg: Dict[str, Any]) -> Dict[str, float]:
    reward_cfg = cfg.get("reward_shaping") or {}
    if not isinstance(reward_cfg, dict):
        reward_cfg = {}
    return {
        "n_adjacent_penalty_scale": float(reward_cfg.get("n_adjacent_penalty_scale", 1.5)),
        "cc_merge_bonus_scale": float(reward_cfg.get("cc_merge_bonus_scale", 0.5)),
        "y_connected_bonus_scale": float(reward_cfg.get("y_connected_bonus_scale", 1.0)),
        "terminal_clean_bonus_scale": float(reward_cfg.get("terminal_clean_bonus_scale", 0.0)),
        "move_progress_bonus_total": float(reward_cfg.get("move_progress_bonus_total", 2.5)),
        "late_quality_turn3_weight": float(reward_cfg.get("late_quality_turn3_weight", 0.0)),
        "late_quality_turn4_weight": float(reward_cfg.get("late_quality_turn4_weight", 0.0)),
        "late_quality_clean_scale": float(reward_cfg.get("late_quality_clean_scale", 0.0)),
        "late_quality_compact_scale": float(reward_cfg.get("late_quality_compact_scale", 0.0)),
        "late_quality_ready_scale": float(reward_cfg.get("late_quality_ready_scale", 0.0)),
    }


__all__ = ["build_reward_config"]
