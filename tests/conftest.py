"""Make this package and a sibling CoMLRL checkout importable in tests."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLRL_ROOT = REPO_ROOT.parent / "CoMLRL"
# Launchers import this repository as the ``LLM_Collab_Minecraft`` package.
for path in (COMLRL_ROOT, REPO_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
