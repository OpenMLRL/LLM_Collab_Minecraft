from __future__ import annotations

"""
Thin wrappers around shared CoMLRL patch utilities for str_builder.
"""

from typing import Dict

from comlrl.utils.patches import (
    patch_debug_turn_tracking,
    patch_single_agent_returns,
    patch_trainer_generation_for_memory,
)


def apply_default_patches(cfg: Dict[str, Any] | None = None) -> None:
    gates = (cfg or {}).get("patches", {}) if isinstance(cfg, dict) else {}
    use_memory = bool(gates.get("generation_memory", True))
    force_sampling = bool(gates.get("force_sampling", False))
    if use_memory or force_sampling:
        patch_trainer_generation_for_memory(
            use_memory=use_memory, force_sampling=force_sampling
        )
    if gates.get("single_agent_returns", True):
        patch_single_agent_returns()
    if gates.get("debug_turn_tracking", True):
        patch_debug_turn_tracking(turn_key="_str_builder_turn")


__all__ = [
    "apply_default_patches",
    "patch_trainer_generation_for_memory",
    "patch_single_agent_returns",
    "patch_debug_turn_tracking",
]
