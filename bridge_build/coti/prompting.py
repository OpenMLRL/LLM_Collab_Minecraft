from __future__ import annotations

from typing import Any, Dict, List


def _as_block_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def build_prompt_context(cfg: Dict[str, Any], num_agents: int) -> Dict[str, Any]:
    prompt_cfg = cfg.get("prompt") or {}
    task_cfg = cfg.get("task") or {}
    if not isinstance(prompt_cfg, dict):
        prompt_cfg = {}
    if not isinstance(task_cfg, dict):
        task_cfg = {}

    agent1_blocks = _as_block_list(task_cfg.get("block_agent1"))
    if not agent1_blocks:
        raise ValueError("task.block_agent1 must be provided and non-empty")
    agent2_blocks = _as_block_list(task_cfg.get("block_agent2"))
    if num_agents > 1 and not agent2_blocks:
        raise ValueError("task.block_agent2 must be provided when num_agents > 1")
    if not agent2_blocks:
        agent2_blocks = list(agent1_blocks)

    return {
        "use_chat_template": bool(prompt_cfg.get("use_chat_template", False)),
        "system_prompt": str(prompt_cfg.get("system") or "").rstrip(),
        "user_template_single": str(prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent1": str(prompt_cfg.get("user_template_agent1") or prompt_cfg.get("user_template") or "").rstrip(),
        "user_template_agent2": str(prompt_cfg.get("user_template_agent2") or prompt_cfg.get("user_template") or "").rstrip(),
        "view": max(0, int(task_cfg.get("view", 3))),
        "max_probe": max(0, min(3, int(task_cfg.get("max_probe", 2)))),
        "max_commands_total": max(1, int(task_cfg.get("max_commands", 40))),
        "agent1_blocks": agent1_blocks,
        "agent2_blocks": agent2_blocks,
    }


__all__ = ["build_prompt_context"]
