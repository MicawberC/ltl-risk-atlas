"""Paired offline study on recorded CARLA frames with a changing product model.

The ego start sequence is an explicit counterfactual grid schedule, not a
recorded CARLA ego trajectory.  This script measures planning only.  CARLA
closed-loop task completion is measured separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import benchmark  # sets the paper's gamma/soft/hard parameter environment
from dynamic_certificate import query
from dynamic_product import (frame_parameters, full_rebuild,
                             initial_snapshot, patch)
from experiment import HARD, evaluate_policy, solve_lp


HERE = Path(__file__).resolve().parent


def run(trace: dict, stride: int, frames: int | None = None,
        ego_rows: list[dict] | None = None) -> dict:
    selected = trace["rows"][::stride]
    if frames is not None:
        selected = selected[:frames]
    grid = int(trace["grid_n"])
    snapshot = initial_snapshot(np.asarray(trace["labels"], dtype=np.int8), grid)
    anchor = None
    output = []
    for index, obs in enumerate(selected):
        if ego_rows is None:
            # A transparent synthetic moving-start stress test.
            x = min(int(trace["goal_xy"][0]) - 1,
                    int(trace["crossing_xy"][0]) - 4 + index // 3)
            y = int(trace["crossing_xy"][1])
        else:
            x, y = map(int, ego_rows[index]["ego_xy"])
        occupied = int(min(obs["risk_distance_m"]) <= 2.5)
        start = (x + grid * y) * 8 + occupied * 4
        labels, environment, severity = frame_parameters(trace, obs)
        began = time.perf_counter()
        snapshot = patch(snapshot, labels, environment, severity, start)
        patch_s = time.perf_counter() - began
        began = time.perf_counter()
        rebuilt = full_rebuild(labels, environment, severity, grid, start)
        rebuild_s = time.perf_counter() - began
        difference = snapshot.model.p - rebuilt.model.p
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-11
        assert np.max(np.abs(snapshot.model.reward - rebuilt.model.reward)) < 1e-11
        assert np.max(np.abs(snapshot.model.cost - rebuilt.model.cost)) < 1e-11
        if index % 2:
            exact = solve_lp(rebuilt.model)
            answer, anchor = query(snapshot.model, anchor)
        else:
            answer, anchor = query(snapshot.model, anchor)
            exact = solve_lp(rebuilt.model)
        evaluation = evaluate_policy(snapshot.model, answer["policy"])
        assert abs(evaluation["objective"] - answer["objective"]) < 1e-5
        assert abs(evaluation["risk"] - answer["risk"]) < 1e-5
        assert answer["risk"] <= HARD + 1e-6
        assert answer["objective"] <= exact["objective"] + 1e-4
        assert exact["objective"] <= answer["upper"] + 1e-4
        assert answer["gap"] <= max(1e-4, .01 * abs(answer["objective"])) + 1e-6
        output.append({"index": index, "carla_frame": obs["frame"],
                       "ego_grid_xy": [x, y],
                       "changed_product_rows": snapshot.changed_product_rows,
                       "changed_labels": int(np.count_nonzero(labels != trace["labels"])),
                       "patch_s": patch_s, "rebuild_s": rebuild_s,
                       "method_s": answer["seconds"],
                       "full_lp_s": exact["wall_seconds"],
                       "lp_fallback": answer["lp_fallback"],
                       "objective": answer["objective"],
                       "exact_objective": exact["objective"],
                       "risk": answer["risk"], "upper": answer["upper"],
                       "gap": answer["gap"],
                       "max_residual": (answer.get("max_residual") or
                                        (answer.get("attempt") or {}).get(
                                            "max_residual"))})
        print(index, "rows", snapshot.changed_product_rows,
              "hit", not answer["lp_fallback"],
              "patch", round(patch_s, 3),
              "rebuild", round(rebuild_s, 3),
              "method", round(answer["seconds"], 3),
              "LP", round(exact["wall_seconds"], 3), flush=True)
    return {"scope": ("recorded CARLA pedestrian trace with a measured moving grid start"
                       if ego_rows is not None else
                       "recorded CARLA pedestrian trace with a counterfactual moving grid start"),
            "states": snapshot.model.states,
            "samples": len(output), "stride": stride,
            "rows": output,
            "summary": {
                "incremental_product_and_certificate_s": sum(
                    r["patch_s"] + r["method_s"] for r in output),
                "incremental_product_full_lp_s": sum(
                    r["patch_s"] + r["full_lp_s"] for r in output),
                "full_rebuild_lp_s": sum(
                    r["rebuild_s"] + r["full_lp_s"] for r in output),
                "lp_fallbacks": sum(r["lp_fallback"] for r in output),
                "max_objective_difference": max(abs(r["objective"] -
                                                     r["exact_objective"])
                                                for r in output),
                "max_gap": max(r["gap"] for r in output)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path,
                        default=HERE / "carla_trace_20hz_two_walkers.json")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--ego-from", type=Path,
                        help="Closed-loop CARLA JSON whose first trial supplies ego grid positions")
    parser.add_argument("--output", type=Path,
                        default=HERE / "dynamic_product_benchmark.json")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    ego_rows = None
    if args.ego_from:
        live = json.loads(args.ego_from.read_text(encoding="utf-8"))
        ego_rows = live["trials"][0]["rows"]
        needed = len(trace["rows"][::args.stride])
        if args.frames is not None:
            needed = min(needed, args.frames)
        if len(ego_rows) < needed:
            raise ValueError("Closed-loop CARLA file has too few ego observations")
    result = run(trace, args.stride, args.frames, ego_rows)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
