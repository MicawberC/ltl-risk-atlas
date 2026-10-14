"""Same-stream comparison of dynamic CARLA product replanning methods.

Reads measured ego and walker positions from a completed CARLA trial.  It
replays both planners on exactly the same observations, independent of their
different closed-loop choices.  The replay cannot measure task completion.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import benchmark  # set paper parameters
from dynamic_certificate import (anchor_from_warm_lp, policy_from_occupation,
                                 query, try_certify)
from dynamic_product import (frame_parameters, full_rebuild,
                             initial_snapshot, patch)
from dynamic_warm_lp import prepare as prepare_warm, update_and_query
from experiment import HARD, evaluate_policy, solve_lp


HERE = Path(__file__).resolve().parent


def risk_observation(reference: dict, row: dict) -> dict:
    center = reference["crossing_xyz"]
    forward, right = reference["forward_xy"], reference["right_xy"]
    positions = row["walkers_xyz"]
    distances = []
    for x, y in reference["risk_cells"]:
        cx = center[0] + (x - 10) * 2 * forward[0] + (y - 10) * 2 * right[0]
        cy = center[1] + (x - 10) * 2 * forward[1] + (y - 10) * 2 * right[1]
        distances.append(min(math.hypot(p[0] - cx, p[1] - cy)
                             for p in positions))
    return {"risk_distance_m": distances}


def run(reference: dict, live: dict, trial: int) -> dict:
    measured = live["trials"][trial]["rows"]
    grid = reference["grid_n"]
    snapshot = initial_snapshot(np.asarray(reference["labels"], dtype=np.int8),
                                grid, lane_only=True)
    anchor = None
    warm_cache = None
    hybrid_cache = None
    hybrid_anchor = None
    hybrid_model = None
    rows = []
    for index, measured_row in enumerate(measured):
        obs = risk_observation(reference, measured_row)
        labels, environment, severity = frame_parameters(reference, obs)
        x, y = map(int, measured_row["ego_xy"])
        occupied = int(min(obs["risk_distance_m"]) <= 2.5)
        start = (x + grid * y) * 8 + occupied * 4
        old_model = snapshot.model
        began = time.perf_counter()
        snapshot = patch(snapshot, labels, environment, severity, start)
        patch_s = time.perf_counter() - began
        began = time.perf_counter()
        rebuilt = full_rebuild(labels, environment, severity, grid, start,
                               lane_only=True)
        rebuild_s = time.perf_counter() - began
        difference = snapshot.model.p - rebuilt.model.p
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-11
        assert np.max(np.abs(snapshot.model.cost - rebuilt.model.cost)) < 1e-11

        def warm_call():
            nonlocal warm_cache
            if warm_cache is None:
                warm_cache = prepare_warm(snapshot.model)
                result = update_and_query(warm_cache, snapshot.model,
                                          snapshot.model)
                result["total_seconds"] += warm_cache["prepare_seconds"]
            else:
                result = update_and_query(warm_cache, old_model, snapshot.model)
            return result

        def hybrid_call():
            nonlocal hybrid_cache, hybrid_anchor, hybrid_model
            began = time.perf_counter()
            attempt = try_certify(snapshot.model, hybrid_anchor)
            if attempt["certified"]:
                return attempt
            if hybrid_cache is None:
                hybrid_cache = prepare_warm(snapshot.model)
                solved = update_and_query(hybrid_cache, snapshot.model,
                                          snapshot.model)
            else:
                solved = update_and_query(hybrid_cache, hybrid_model,
                                          snapshot.model)
            hybrid_model = snapshot.model
            hybrid_anchor = anchor_from_warm_lp(
                hybrid_cache, solved["occupation"])
            return {"certified": True, "lp_fallback": True,
                    "objective": solved["objective"], "risk": solved["risk"],
                    "upper": solved["objective"] + 1e-5,
                    "gap": 1e-5, "policy": hybrid_anchor.policy,
                    "seconds": time.perf_counter() - began}

        if index % 2:
            warm = warm_call()
            hybrid = hybrid_call()
            certified, anchor = query(snapshot.model, anchor)
        else:
            certified, anchor = query(snapshot.model, anchor)
            hybrid = hybrid_call()
            warm = warm_call()
        exact = solve_lp(rebuilt.model)
        warm_policy = policy_from_occupation(warm["occupation"])
        warm_eval = evaluate_policy(snapshot.model, warm_policy)
        cert_eval = evaluate_policy(snapshot.model, certified["policy"])
        hybrid_eval = evaluate_policy(snapshot.model, hybrid["policy"])
        assert abs(warm["objective"] - exact["objective"]) < 1e-4
        assert abs(warm_eval["objective"] - exact["objective"]) < 1e-4
        assert abs(cert_eval["objective"] - certified["objective"]) < 1e-4
        assert abs(hybrid_eval["objective"] - hybrid["objective"]) < 1e-4
        assert certified["risk"] <= HARD + 1e-6
        assert certified["objective"] <= exact["objective"] + 1e-4
        assert exact["objective"] <= certified["upper"] + 1e-4
        assert hybrid["risk"] <= HARD + 1e-6
        assert exact["objective"] <= hybrid["upper"] + 1e-4
        assert hybrid["gap"] <= max(1e-4,
                                     .01 * abs(hybrid["objective"])) + 1e-6
        assert certified["gap"] <= max(1e-4,
                                        .01 * abs(certified["objective"])) + 1e-6
        rows.append({"index": index, "tick": measured_row["tick"],
                     "ego_xy": [x, y],
                     "changed_product_rows": snapshot.changed_product_rows,
                     "patch_s": patch_s, "rebuild_s": rebuild_s,
                     "warm_s": warm["total_seconds"],
                     "hybrid_s": hybrid["seconds"],
                     "cert_s": certified["seconds"],
                     "fresh_lp_s": exact["wall_seconds"],
                     "cert_lp_fallback": certified["lp_fallback"],
                     "hybrid_lp_fallback": hybrid["lp_fallback"],
                     "exact_objective": exact["objective"],
                     "cert_objective": certified["objective"],
                     "cert_upper": certified["upper"],
                     "cert_risk": certified["risk"],
                     "warm_forward": float(warm_policy[start, 5]),
                     "cert_forward": float(certified["policy"][start, 5])})
        print(index, "warm", round(warm["total_seconds"], 3),
              "hybrid", round(hybrid["seconds"], 3),
              "cert", round(certified["seconds"], 3),
              "fallback", certified["lp_fallback"], flush=True)
    return {"scope": "same measured CARLA ego/walker trajectory, lane-only product",
            "source_trial": trial, "samples": len(rows), "rows": rows,
            "summary": {
                "warm_lp_total_s": sum(r["patch_s"] + r["warm_s"] for r in rows),
                "hybrid_total_s": sum(r["patch_s"] + r["hybrid_s"] for r in rows),
                "certificate_total_s": sum(r["patch_s"] + r["cert_s"] for r in rows),
                "fresh_full_total_s": sum(r["rebuild_s"] + r["fresh_lp_s"]
                                          for r in rows),
                "certificate_lp_fallbacks": sum(r["cert_lp_fallback"]
                                                 for r in rows),
                "hybrid_lp_fallbacks": sum(r["hybrid_lp_fallback"]
                                            for r in rows),
                "max_objective_difference": max(
                    abs(r["cert_objective"] - r["exact_objective"])
                    for r in rows),
                "max_upper_gap": max(r["cert_upper"] - r["cert_objective"]
                                     for r in rows),
                "max_risk": max(r["cert_risk"] for r in rows)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=HERE / "carla_trace_20hz_two_walkers.json")
    parser.add_argument("--live", type=Path,
                        default=HERE / "closed_loop_two_threeway.json")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=HERE / "dynamic_live_replay.json")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    live = json.loads(args.live.read_text(encoding="utf-8"))
    result = run(reference, live, args.trial)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
