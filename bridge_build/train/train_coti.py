from __future__ import annotations

import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(REPO_ROOT))

from LLM_Collab_Minecraft.bridge_build.coti.domain import get_domain_spec  # noqa: E402
from coti.train.launcher import run_training_cli  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train bridge_build with CoTI BCMAAC/baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(REPO_ROOT, "bridge_build", "configs", "coti", "bcmaac_meta_data2_dev.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="key.path=value overrides",
    )
    args = parser.parse_args()
    return run_training_cli(
        config_path=args.config,
        overrides=args.override,
        domain_spec=get_domain_spec(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
