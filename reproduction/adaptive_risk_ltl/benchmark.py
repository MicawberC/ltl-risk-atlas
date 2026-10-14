"""Interleaved repeated timing of sparse LP and certified guided aggregation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiment import build_grid, run_abstraction, solve_lp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=["8:near", "16:middle", "20:near", "24:near"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("timing.json"))
    args = parser.parse_args()
    solve_lp(build_grid(4))  # Warm SciPy/HiGHS once, outside measured calls.
    rng = np.random.default_rng(20261012)
    results = []
    for case in args.cases:
        n_text, scenario = case.split(":", 1)
        n = int(n_text)
        start = time.perf_counter()
        model = build_grid(n, scenario)
        build_seconds = time.perf_counter() - start
        samples = {"sparse_lp": [], "guided": []}
        objectives = []
        for repeat in range(args.repeats):
            for name in rng.permutation(list(samples)):
                if name == "sparse_lp":
                    result = solve_lp(model)
                    elapsed = result["wall_seconds"]
                    objectives.append(result["objective"])
                else:
                    result = run_abstraction(model, "guided", max_rounds=20)
                    elapsed = result["total_seconds"]
                    if not result["certified"]:
                        raise AssertionError(f"Uncertified guided result for {case}")
                samples[name].append(elapsed)
        record = {
            "case": case, "states": model.states, "actions": model.actions,
            "construction_seconds": build_seconds,
            "sparse_lp_seconds": samples["sparse_lp"],
            "guided_seconds": samples["guided"],
            "sparse_lp_median_seconds": float(np.median(samples["sparse_lp"])),
            "guided_median_seconds": float(np.median(samples["guided"])),
            "median_speedup": float(np.median(samples["sparse_lp"]) /
                                    np.median(samples["guided"])),
            "baseline_objective_range": [float(min(objectives)), float(max(objectives))],
        }
        results.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"{case}: sparse={record['sparse_lp_median_seconds']:.3f}s, "
              f"guided={record['guided_median_seconds']:.3f}s, "
              f"speedup={record['median_speedup']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
