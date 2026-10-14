"""Recheck the HiGHS default-mode outlier with sparse interior point."""

from __future__ import annotations

import json
from pathlib import Path

from experiment import build_grid, solve_lp


def main() -> None:
    result = solve_lp(build_grid(32, "terrain", 0), method="highs-ipm")
    output = {"grid": 32, "scenario": "terrain", "seed": 0,
              "method": "highs-ipm", "seconds": result["wall_seconds"],
              "objective": result["objective"]}
    Path(__file__).with_name("terrain_ipm.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
