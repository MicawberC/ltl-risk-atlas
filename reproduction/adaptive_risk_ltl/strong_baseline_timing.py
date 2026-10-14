"""Interleaved repetitions against selected strong occupancy baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import build_grid, run_abstraction
from stratified import solve_cmdp

CASES = [(24, 1, .16), (32, 1, .08)]
HERE = Path(__file__).resolve().parent


def main() -> None:
    rng = np.random.default_rng(20261012)
    output = []
    for size, seed, fraction in CASES:
        model = build_grid(size, "varied", seed)
        samples = {"occupancy": [], "stratified": []}
        states = []
        for _ in range(3):
            for name in rng.permutation(list(samples)):
                if name == "occupancy":
                    run = run_abstraction(model, "occupancy", fraction=fraction,
                                          max_rounds=20)
                    assert run["certified"]
                    samples[name].append(run["total_seconds"])
                    states.append(run["final_states"])
                else:
                    run = solve_cmdp(model, search_mode="breakpoint")
                    assert run["certified"]
                    samples[name].append(run["seconds"])
        row = {"size": size, "seed": seed, "fraction": fraction,
               "samples": samples, "occupancy_states": states,
               "occupancy_median": float(np.median(samples["occupancy"])),
               "stratified_median": float(np.median(samples["stratified"]))}
        row["speedup"] = row["occupancy_median"] / row["stratified_median"]
        output.append(row)
        (HERE / "strong_baseline_timing.json").write_text(json.dumps(output, indent=2))
        print(size, seed, row["occupancy_median"], row["stratified_median"],
              row["speedup"], flush=True)


if __name__ == "__main__":
    main()
