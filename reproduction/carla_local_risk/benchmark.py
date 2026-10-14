"""Replay CARLA pedestrian positions on one fixed LTL product CMDP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ["LTL_RISK_GAMMA"] = "0.8"
os.environ["LTL_RISK_SOFT"] = "1.0"
os.environ["LTL_RISK_HARD"] = "2.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_risk_ltl"))

from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy, solve_lp  # noqa: E402
from highs_reopt import prepare as prepare_lp, query as query_lp  # noqa: E402
from joint_price_atlas import (expand_fast, prepare as prepare_atlas,
                               query as query_atlas)  # noqa: E402
from primal_cache_ablation import best_primal  # noqa: E402


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=HERE / "carla_trace.json")
    parser.add_argument("--output", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--anchor-first", action="store_true",
                        help="Initialize both solvers at the first observed risk map")
    parser.add_argument("--precompute-crossing-step", type=float,
                        help="Precompute a risk-parameter atlas along the known crossing corridor, in metres")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    n = trace["grid_n"]
    labels = np.asarray(trace["labels"], dtype=np.int8)
    model = build_grid(n, "carla_crossing", labels_override=labels)
    start_xy = (trace["crossing_xy"][0] - 4, trace["crossing_xy"][1])
    start = np.flatnonzero(np.all(model.xy == start_xy, axis=1) &
                           (model.mode == 1) & (model.goal == 0) &
                           (model.bad == 0))
    if len(start) != 1:
        raise RuntimeError("Cannot identify the ego approach product state")
    model = replace(model, initial=int(start[0]))
    if labels[start_xy[0] + n * start_xy[1]] == 4:
        raise RuntimeError("Ego start is not mapped to a driving lane")
    directions = np.stack([
        np.where(np.all(model.xy == cell, axis=1), model.cost, 0.0)
        for cell in trace["risk_cells"]])
    if np.any(directions.sum(axis=1) == 0):
        raise RuntimeError("A CARLA risk cell has no positive cost")
    theta_offset = (np.asarray(trace["rows"][0]["theta"], dtype=float)
                    if args.anchor_first else np.zeros(len(directions)))
    if args.anchor_first:
        model = replace(model, cost=model.cost + theta_offset @ directions)

    atlas = prepare_atlas(model, directions)
    nominal = tuple(atlas["policies"])
    precompute_expansions = 0
    precompute_seconds = 0.0
    precompute_points = 0
    if args.precompute_crossing_step is not None:
        if not (0.05 <= args.precompute_crossing_step <= 2.0):
            raise ValueError("precompute crossing step must be between 0.05 and 2 m")
        began_precompute = time.perf_counter()
        half_width = 5.0
        precompute_points = int(np.ceil(2 * half_width /
                                        args.precompute_crossing_step)) + 1
        for lateral_m in np.linspace(half_width, -half_width,
                                     precompute_points):
            future_theta = np.array([
                0.2 + 1.6 * np.exp(-0.5 * ((np.hypot(
                    (cell[0] - trace["crossing_xy"][0]) * trace["cell_m"],
                    lateral_m)) / 3.0) ** 2) - 1.0
                for cell in trace["risk_cells"]]) - theta_offset
            trial = query_atlas(atlas, future_theta, nearest_first=8)
            if not trial["certified"]:
                extension = expand_fast(atlas, future_theta, nearest_first=8)
                precompute_expansions += 1
                trial = query_atlas(atlas, future_theta, nearest_first=8)
            if not trial["certified"]:
                raise RuntimeError("Corridor precomputation failed to certify a grid point")
        precompute_seconds = time.perf_counter() - began_precompute
        atlas["prepare_seconds"] += precompute_seconds
    lp = prepare_lp(model, directions)
    rows = []
    output = {"source_trace": str(args.trace.resolve()),
              "scope": "stationary ego; CARLA pedestrian trace; offline repeated planning",
              "model_assumptions": {
                  "motion_probability": 0.75,
                  "environment_transition": [[0.8, 0.2], [0.2, 0.8]],
                  "risk_multiplier": "0.2 + 1.6 exp(-distance^2 / (2*3^2))",
                  "risk_parameter": "multiplier - 1 per five crossing cells",
                  "risk_labels": "crossing conflict if environment mode is occupied",
                  "offroad_labels": "CARLA driving-lane waypoint mask"},
              "parameters": {"gamma": GAMMA, "soft": SOFT, "hard": HARD},
              "anchor_first": args.anchor_first,
              "theta_offset": theta_offset.tolist(),
              "precompute_crossing_step_m": args.precompute_crossing_step,
              "precompute_points": precompute_points,
              "precompute_expansions": precompute_expansions,
              "precompute_seconds": precompute_seconds,
              "states": model.states, "actions": model.actions,
              "initial_xy": list(start_xy),
              "atlas_setup_s": atlas["prepare_seconds"],
              "lp_setup_s": lp["prepare_seconds"], "rows": rows}
    for i, observation in enumerate(trace["rows"]):
        raw_theta = np.asarray(observation["theta"], dtype=float)
        theta = raw_theta - theta_offset
        if theta.shape != (5,) or np.min(raw_theta) < -0.8 - 1e-9:
            raise ValueError("Invalid CARLA risk parameter")

        def run_atlas():
            began = time.perf_counter()
            answer = query_atlas(atlas, theta, nearest_first=8)
            hit = bool(answer["certified"])
            full_fallback = False
            expansion = None
            if not hit:
                extension = expand_fast(atlas, theta, nearest_first=8)
                full_fallback = bool(extension["fallback"])
                expansion = extension
                answer = query_atlas(atlas, theta, nearest_first=8)
            if not answer["certified"]:
                raise RuntimeError(f"No certificate for CARLA sample {i}: {answer}")
            return answer, hit, full_fallback, time.perf_counter() - began, expansion

        if i % 2:
            exact = query_lp(lp, theta)
            answer, hit, fallback, atlas_s, expansion = run_atlas()
        else:
            answer, hit, fallback, atlas_s, expansion = run_atlas()
            exact = query_lp(lp, theta)
        changed = replace(model, cost=model.cost + theta @ directions)
        evaluated = evaluate_policy(changed, answer["policy"])
        if abs(evaluated["objective"] - answer["objective"]) > 1e-6:
            raise AssertionError("Policy objective evaluation disagrees")
        if abs(evaluated["risk"] - answer["risk"]) > 1e-6:
            raise AssertionError("Policy risk evaluation disagrees")
        if answer["risk"] > HARD + 1e-7 or exact["risk"] > HARD + 1e-6:
            raise AssertionError("Hard risk constraint violated")
        if answer["objective"] > exact["objective"] + 1e-5 or \
           exact["objective"] > answer["upper"] + 1e-5:
            raise AssertionError("Atlas interval does not enclose LP optimum")
        tol = max(1e-4, .01 * abs(answer["objective"])) + 1e-7
        if answer["gap"] > tol:
            raise AssertionError("Certificate is wider than one percent")
        try:
            simple = float(best_primal(nominal, theta))
        except ValueError:
            simple = None
        fresh = None
        if i in (0, len(trace["rows"]) // 2, len(trace["rows"]) - 1):
            fresh = solve_lp(changed, method="highs-ds")
            if abs(fresh["objective"] - exact["objective"]) > 1e-5:
                raise AssertionError("Warm LP disagrees with independent sparse LP")
        row = {"sample": i, "time_s": observation["time_s"],
               "theta": raw_theta.tolist(), "delta_theta": theta.tolist(),
               "walker_xyz": observation["walker_xyz"],
               "atlas_s": atlas_s, "lp_s": exact["seconds"],
               "atlas_objective": float(answer["objective"]),
               "lp_objective": float(exact["objective"]),
               "atlas_upper": float(answer["upper"]),
               "atlas_gap": float(answer["gap"]),
               "certificate_type": answer.get("certificate_type", "exact_region"),
               "residual_info": answer.get("residual_info"),
               "atlas_risk": float(answer["risk"]),
               "lp_risk": float(exact["risk"]),
               "direct_hit": hit, "full_fallback": fallback,
               "expansion": expansion,
               "nominal_policy_cache_objective": simple,
               "nominal_cache_within_one_percent":
                   bool(simple is not None and exact["objective"] - simple <= tol),
               "lp_iterations": exact["iterations"],
               "fresh_sparse_lp_s": None if fresh is None else fresh["wall_seconds"]}
        rows.append(row)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"{i+1}/{len(trace['rows'])}: atlas {1000*atlas_s:.1f} ms, "
              f"LP {1000*exact['seconds']:.1f} ms, "
              f"risk {answer['risk']:.4f}, hit={hit}", flush=True)
    atlas_times = np.array([r["atlas_s"] for r in rows])
    lp_times = np.array([r["lp_s"] for r in rows])
    atlas_total = atlas["prepare_seconds"] + float(atlas_times.sum())
    lp_total = lp["prepare_seconds"] + float(lp_times.sum())
    output["summary"] = {
        "samples": len(rows), "direct_hits": sum(r["direct_hit"] for r in rows),
        "full_fallbacks": sum(r["full_fallback"] for r in rows),
        "certified": len(rows), "hard_risk_satisfied": len(rows),
        "nominal_cache_within_one_percent":
            sum(r["nominal_cache_within_one_percent"] for r in rows),
        "atlas_total_s": atlas_total, "lp_total_s": lp_total,
        "total_speedup_lp_over_atlas": lp_total / atlas_total,
        "atlas_p50_ms": float(1000 * np.percentile(atlas_times, 50)),
        "atlas_p95_ms": float(1000 * np.percentile(atlas_times, 95)),
        "lp_p50_ms": float(1000 * np.percentile(lp_times, 50)),
        "lp_p95_ms": float(1000 * np.percentile(lp_times, 95)),
        "max_objective_difference": max(abs(r["atlas_objective"] -
                                            r["lp_objective"]) for r in rows),
        "max_certificate_gap": max(r["atlas_gap"] for r in rows),
        "max_atlas_risk": max(r["atlas_risk"] for r in rows)}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
