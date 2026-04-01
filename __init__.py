"""LLM_Collab_Minecraft package."""

from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_REPO_ROOT)
_COTI_ROOT = os.path.join(_WORKSPACE_ROOT, "CoTI")
if os.path.isdir(os.path.join(_COTI_ROOT, "coti")) and _COTI_ROOT not in sys.path:
    sys.path.insert(0, _COTI_ROOT)
