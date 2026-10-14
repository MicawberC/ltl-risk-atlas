"""Optimistic occupancy-abstraction baseline over several split fractions.

The minimum across fractions is an oracle selection on the test case and
therefore *favours* this baseline; it is not a deployable tuner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import build_grid, run_abstraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--scenario", default="varied")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--fractions", nargs="+", type=float, default=[.02, .04, .08, .16])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("occupancy_oracle.json"))
    args = parser.parse_args()
    output = []
    for seed in args.seeds:
        model = build_grid(args.size, args.scenario, seed)
        runs = []
        for fraction in args.fractions:
            result = run_abstraction(model, "occupancy", fraction=fraction,
                                     max_rounds=20)
            runs.append({"fraction": fraction, "certified": result["certified"],
                         "states": result["final_states"],
                         "seconds": result["total_seconds"],
                         "gap": result["final_gap"],
                         "rounds": len(result["history"])})
            print(args.size, seed, fraction, result["certified"],
                  result["final_states"], round(result["total_seconds"], 3), flush=True)
            output.append({"grid": args.size, "seed": seed,
                           "scenario": args.scenario, "runs": runs.copy()})
            args.output.write_text(json.dumps(output, indent=2))
            output.pop()
        output.append({"grid": args.size, "seed": seed,
                       "scenario": args.scenario, "runs": runs})


if __name__ == "__main__":
    main()
