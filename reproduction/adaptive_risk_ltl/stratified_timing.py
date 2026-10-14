"""Interleaved repeated timing and certificate audit of the final solver."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiment import build_grid, solve_lp
from stratified import solve_cmdp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["16:middle", "24:near", "24:barrier_hard", "24:sequence", "32:near", "32:barrier_hard"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stratified_timing.json"))
    args = parser.parse_args()
    rng = np.random.default_rng(20261012)
    solve_lp(build_grid(4))
    output = []
    for case in args.cases:
        n_text, scenario = case.split(":")
        started = time.perf_counter()
        model = build_grid(int(n_text), scenario)
        build_time = time.perf_counter() - started
        samples = {"sparse_lp": [], "stratified": []}
        objectives = []
        certificates = []
        for _ in range(args.repeats):
            for method in rng.permutation(list(samples)):
                if method == "sparse_lp":
                    result = solve_lp(model)
                    samples[method].append(result["wall_seconds"])
                    objectives.append(result["objective"])
                else:
                    result = solve_cmdp(model, search_mode="breakpoint")
                    samples[method].append(result["seconds"])
                    certificates.append({"objective": result["objective"],
                                         "risk": result["risk"], "upper": result["upper"],
                                         "gap": result["gap"], "evaluations": result["evaluations"]})
        for cert in certificates:
            target = max(1e-4, .01 * abs(cert["objective"]))
            if (cert["risk"] > 5 + 1e-7 or cert["gap"] > target + 1e-7 or
                cert["objective"] > min(objectives) + 1e-6 or
                    cert["upper"] < max(objectives) - 1e-6):
                raise AssertionError(f"Invalid certificate for {case}")
        lp_median = float(np.median(samples["sparse_lp"]))
        method_median = float(np.median(samples["stratified"]))
        row = {"case": case, "states": model.states, "build_seconds": build_time,
               "samples": samples, "certificates": certificates,
               "sparse_lp_median": lp_median, "stratified_median": method_median,
               "median_speedup": lp_median / method_median}
        output.append(row)
        args.output.write_text(json.dumps(output, indent=2))
        print(case, "LP", round(lp_median, 3), "stratified", round(method_median, 3),
              "ratio", round(row["median_speedup"], 2), flush=True)


if __name__ == "__main__":
    main()
