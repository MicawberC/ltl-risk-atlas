"""Run the local-update benchmark with IROS CARLA paper default parameters."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ["LTL_RISK_GAMMA"] = "0.8"
os.environ["LTL_RISK_SOFT"] = "1"
os.environ["LTL_RISK_HARD"] = "2"

from local_update_gate import main  # noqa: E402 - parameters must be set first


if "--output" not in sys.argv:
    sys.argv.extend(("--output", str(Path(__file__).with_name(
        "local_update_gate_paper.json"))))

if __name__ == "__main__":
    main()
