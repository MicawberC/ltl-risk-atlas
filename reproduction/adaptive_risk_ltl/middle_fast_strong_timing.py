"""Interleaved timing against the best tested occupancy setting on middle goals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import build_grid, run_abstraction
from stratified import solve_cmdp


def main() -> None:
    rng = np.random.default_rng(20261012)
    output = []
    for n in (24, 32):
        model = build_grid(n, "middle_fast")
        samples = {"occupancy": [], "stratified": []}
        states = []
        for _ in range(3):
            for name in rng.permutation(list(samples)):
                if name == "occupancy":
                    result = run_abstraction(model, "occupancy", fraction=.16,
                                             max_rounds=20)
                    assert result["certified"]
                    samples[name].append(result["total_seconds"])
                    states.append(result["final_states"])
                else:
                    result = solve_cmdp(model, search_mode="breakpoint")
                    assert result["certified"]
                    samples[name].append(result["seconds"])
        row = {"case": f"{n}:middle_fast", "fraction": .16,
               "occupancy_states": states, "samples": samples,
               "occupancy_median": float(np.median(samples["occupancy"])),
               "stratified_median": float(np.median(samples["stratified"]))}
        row["speedup"] = row["occupancy_median"] / row["stratified_median"]
        output.append(row)
        Path(__file__).with_name("middle_fast_strong_timing.json").write_text(
            json.dumps(output, indent=2))
        print(row["case"], round(row["speedup"], 2), flush=True)


if __name__ == "__main__":
    main()
