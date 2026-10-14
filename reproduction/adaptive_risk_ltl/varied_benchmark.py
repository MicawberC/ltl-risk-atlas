"""Held-out synthetic label layouts, same motion kernel and DFA product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import build_grid, solve_lp
from stratified import solve_cmdp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[24, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--scenario", choices=["varied", "terrain", "barrier_hard_varied", "barrier_hard_active"], default="varied")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("varied_benchmark.json"))
    args = parser.parse_args()
    output = []
    for n in args.sizes:
        for seed in args.seeds:
            model = build_grid(n, args.scenario, seed)
            lp = solve_lp(model)
            stratified = solve_cmdp(model, search_mode="breakpoint")
            bound_violation = max(0.0, stratified["objective"] - lp["objective"],
                                  lp["objective"] - stratified["upper"])
            target = max(1e-4, .01 * abs(stratified["objective"]))
            certified = (stratified["risk"] <= 5 + 1e-7 and
                         stratified["gap"] <= target + 1e-7 and
                         bound_violation < 1e-6)
            row = {"grid": n, "seed": seed, "scenario": args.scenario,
                   "states": model.states,
                   "sparse_lp_objective": lp["objective"],
                   "sparse_lp_seconds": lp["wall_seconds"],
                   "stratified": stratified, "target_gap": target,
                   "certified": certified, "bound_violation": bound_violation,
                   "speedup": lp["wall_seconds"] / stratified["seconds"]}
            output.append(row)
            args.output.write_text(json.dumps(output, indent=2))
            print(n, seed, "LP", round(lp["wall_seconds"], 3),
                  "stratified", round(stratified["seconds"], 3),
                  "gap", round(stratified["gap"], 6),
                  "certified", certified, flush=True)


if __name__ == "__main__":
    main()
