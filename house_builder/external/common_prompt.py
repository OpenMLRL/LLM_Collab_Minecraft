from __future__ import annotations

from typing import Iterable, List


def _normalize_segments(value: str | Iterable[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def apply_common_prompt(
    prompt: str,
    *,
    prefix: str | Iterable[str] | None = None,
    suffix: str | Iterable[str] | None = None,
) -> str:
    segments: List[str] = []
    prefix_parts = _normalize_segments(prefix)
    suffix_parts = _normalize_segments(suffix)
    if prefix_parts:
        segments.append("\n".join(prefix_parts))
    segments.append((prompt or "").strip())
    if suffix_parts:
        segments.append("\n".join(suffix_parts))
    return "\n\n".join([s for s in segments if s]).strip()
