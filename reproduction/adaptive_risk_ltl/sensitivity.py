"""Sensitivity of state count and certificates to the split fraction."""

from __future__ import annotations

import json
from pathlib import Path

from experiment import build_grid, run_abstraction

CASES = [(16, "near"), (20, "near"), (24, "near"),
         (20, "barrier_hard"), (24, "barrier_hard")]
FRACTIONS = (0.02, 0.04, 0.08)
STRATEGIES = ("guided", "occupancy")


def main() -> None:
    output = Path(__file__).with_name("sensitivity.json")
    results = []
    for n, scenario in CASES:
        model = build_grid(n, scenario)
        for strategy in STRATEGIES:
            for fraction in FRACTIONS:
                run = run_abstraction(model, strategy, fraction=fraction, max_rounds=20)
                results.append({"grid": n, "scenario": scenario,
                                "strategy": strategy, "fraction": fraction,
                                "certified": run["certified"],
                                "states": run["final_states"],
                                "gap": run["final_gap"],
                                "seconds": run["total_seconds"],
                                "rounds": len(run["history"])})
                output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(n, scenario, strategy, fraction, run["certified"],
                      run["final_states"], round(run["total_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
