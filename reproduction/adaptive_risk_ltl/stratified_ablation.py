"""Matched ablation: full-product versus automaton-layer policy iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiment import build_grid
from stratified import solve_cmdp, solve_scalarized, solve_scalarized_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["16:near", "16:middle", "24:near", "24:barrier_hard", "32:near", "32:middle", "32:barrier_hard"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--search-mode", choices=["bisection", "breakpoint"], default="bisection")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stratified_ablation.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(20261012)
    output = []
    for case in args.cases:
        n_text, scenario = case.split(":")
        model = build_grid(int(n_text), scenario)
        samples = {"layered": [], "full": []}
        results = {"layered": [], "full": []}
        for _ in range(args.repeats):
            for mode in rng.permutation(["layered", "full"]):
                result = solve_cmdp(model,
                                    scalarized_solver=(solve_scalarized if mode == "layered" else solve_scalarized_full),
                                    search_mode=args.search_mode)
                samples[mode].append(result["seconds"])
                results[mode].append({"objective": result["objective"],
                                      "risk": result["risk"], "gap": result["gap"],
                                      "evaluations": result["evaluations"]})
        row = {"case": case, "states": model.states, "samples": samples,
               "results": results,
               "layered_median": float(np.median(samples["layered"])),
               "full_median": float(np.median(samples["full"]))}
        output.append(row)
        args.output.write_text(json.dumps(output, indent=2))
        print(case, "full", round(row["full_median"], 3),
              "layered", round(row["layered_median"], 3), flush=True)


if __name__ == "__main__":
    main()
