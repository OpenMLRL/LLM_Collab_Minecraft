from __future__ import annotations

from typing import Any, Dict


def build_prompt_context(cfg: Dict[str, Any], num_agents: int) -> Dict[str, Any]:
    prompt_cfg = cfg.get("prompt") or {}
    task_cfg = cfg.get("task") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    return {
        "use_chat_template": bool(prompt_cfg.get("use_chat_template", False)),
        "system_prompt": str(prompt_cfg.get("system") or "").rstrip(),
        "user_template_single": str(prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent1": str(prompt_cfg.get("user_template_agent1") or prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent2": str(prompt_cfg.get("user_template_agent2") or prompt_cfg.get("user_template") or "").rstrip(),
        "view": max(0, int(task_cfg.get("view", 2))),
        "extraction_limit": max(0, int(task_cfg.get("extraction_limit", 2))),
        "extraction_range": max(0, int(task_cfg.get("extraction_range", 2))),
        "max_path_len": max(1, int(task_cfg.get("max_path_len", 4))),
        "comm_limit": max(1, int(task_cfg.get("comm_limit", 1))),
        "num_agents": int(num_agents),
    }


__all__ = ["build_prompt_context"]
