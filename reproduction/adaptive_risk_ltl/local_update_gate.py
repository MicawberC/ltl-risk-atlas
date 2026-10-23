"""Pre-CARLA paired benchmark for spatially local risk-severity updates.

The model, proposition labels, automata, transition kernel, and reward stay
fixed within each trace.  Only five physical risk locations change severity.
This is a synthetic scope test, not a CARLA or full reactive-planning result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy, solve_lp
from five_zone_stress import directions_for_five_cells
from highs_reopt import prepare as prepare_lp, query as query_lp
from joint_price_atlas import expand_fast, prepare as prepare_atlas, query as query_atlas
from primal_cache_ablation import best_primal


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = ("24:middle_fast_varied:0", "24:middle_fast_varied:17",
                 "24:middle_fast_varied:34", "32:middle_fast_varied:0",
                 "32:middle_fast_varied:17", "32:middle_fast_varied:34")
STREAMS = ("walk", "iid")


def risk_trace(kind: str, count: int, seed: int, dimension: int = 5,
               magnitude: float = .8) -> np.ndarray:
    """Predefined local-support traces; seed and trace type never use results."""
    if dimension < 1:
        raise ValueError("Risk trace needs at least one direction")
    rng = np.random.default_rng(seed)
    theta = np.zeros((count, dimension))
    if kind == "iid":
        theta[1:] = rng.uniform(-magnitude, magnitude,
                                size=(count - 1, dimension))
    elif kind == "walk":
        for i in range(1, count):
            theta[i] = theta[i - 1]
            cells = rng.choice(dimension, size=min(dimension, 1 if i % 5 else 2),
                               replace=False)
            theta[i, cells] = np.clip(theta[i, cells] +
                                      rng.uniform(-.65, .65, size=len(cells)),
                                      -magnitude, magnitude)
            if i % 7 == 0:
                cell = int(rng.integers(dimension))
                theta[i, cell] = rng.uniform(-magnitude, magnitude)
    else:
        raise ValueError(kind)
    return theta


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def approach_start(model, scenario: str):
    """Start four cells before the target interaction area, with fresh DFAs."""
    targets = np.unique(model.xy[model.label == 2], axis=0)
    assert len(targets) == 1
    x, y = targets[0]
    start_xy = (np.array((max(0, x - 5), y))
                if scenario == "middle_fast_crossing" else
                np.array((max(0, x - 4), max(0, y - 4))))
    matches = np.flatnonzero(np.all(model.xy == start_xy, axis=1) &
                             (model.mode == 1) & (model.goal == 0) &
                             (model.bad == 0))
    assert len(matches) == 1
    return replace(model, initial=int(matches[0]), layers=model.layers)


def run_stream(model, directions, kind: str, count: int, seed: int,
               spotcheck: bool, progress: bool = False) -> dict:
    thetas = risk_trace(kind, count, seed, len(directions))
    atlas = prepare_atlas(model, directions)
    nominal = tuple(atlas["policies"])
    lp = prepare_lp(model, directions)
    rows = []
    for index, theta in enumerate(thetas):
        # Interleave the two methods and reverse order at every other query.
        def atlas_call():
            began = time.perf_counter()
            answer = query_atlas(atlas, theta, nearest_first=8)
            hit = bool(answer["certified"])
            fallback = False
            if not hit:
                extension = expand_fast(atlas, theta, nearest_first=8)
                fallback = bool(extension["fallback"])
                answer = query_atlas(atlas, theta, nearest_first=8)
            seconds = time.perf_counter() - began
            assert answer["certified"]
            return answer, hit, fallback, seconds

        def lp_call():
            return query_lp(lp, theta)

        if index % 2:
            lp_answer = lp_call()
            answer, hit, fallback, atlas_seconds = atlas_call()
        else:
            answer, hit, fallback, atlas_seconds = atlas_call()
            lp_answer = lp_call()

        assert lp_answer["basis_valid"]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        evaluated = evaluate_policy(changed, answer["policy"])
        assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
        assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
        assert evaluated["risk"] <= HARD + 1e-7
        assert lp_answer["risk"] <= HARD + 1e-6
        assert answer["objective"] <= lp_answer["objective"] + 1e-5
        assert lp_answer["objective"] <= answer["upper"] + 1e-5
        tolerance = max(1e-4, .01 * abs(answer["objective"])) + 1e-7
        assert answer["gap"] <= tolerance
        try:
            simple = best_primal(nominal, theta)
        except ValueError:
            simple = None
        lp_spot = None
        if spotcheck and index in (0, count // 2, count - 1):
            exact = solve_lp(changed, method="highs-ds")
            assert abs(exact["objective"] - lp_answer["objective"]) < 1e-5
            lp_spot = {"objective": float(exact["objective"]),
                       "seconds": float(exact["wall_seconds"])}
        rows.append({"index": index, "theta": theta.tolist(),
                     "atlas_seconds": atlas_seconds,
                     "lp_seconds": lp_answer["seconds"],
                     "atlas_objective": float(answer["objective"]),
                     "lp_objective": float(lp_answer["objective"]),
                     "atlas_upper": float(answer["upper"]),
                     "risk": float(answer["risk"]),
                     "hit": hit, "fallback": fallback,
                     "simple_cache_objective": None if simple is None else float(simple),
                     "simple_cache_within_one_percent": bool(simple is not None and
                         lp_answer["objective"] - simple <=
                         max(1e-4, .01 * abs(lp_answer["objective"])) + 1e-7),
                     "lp_spotcheck": lp_spot})
        if progress:
            print("  update", index, kind, "atlas_ms", round(1000 * atlas_seconds, 1),
                  "lp_ms", round(1000 * lp_answer["seconds"], 1),
                  "spotcheck", lp_spot is not None, flush=True)
    atlas_times = [r["atlas_seconds"] for r in rows]
    lp_times = [r["lp_seconds"] for r in rows]
    atlas_total = atlas["prepare_seconds"] + sum(atlas_times)
    lp_total = lp["prepare_seconds"] + sum(lp_times)
    return {"kind": kind, "seed": seed, "samples": count,
            "atlas_setup_seconds": atlas["prepare_seconds"],
            "lp_setup_seconds": lp["prepare_seconds"],
            "atlas_total_seconds": atlas_total,
            "lp_total_seconds": lp_total,
            "total_speedup": lp_total / atlas_total,
            "atlas_p50_ms": 1000 * percentile(atlas_times, 50),
            "atlas_p95_ms": 1000 * percentile(atlas_times, 95),
            "atlas_max_ms": 1000 * max(atlas_times),
            "lp_p50_ms": 1000 * percentile(lp_times, 50),
            "lp_p95_ms": 1000 * percentile(lp_times, 95),
            "lp_max_ms": 1000 * max(lp_times),
            "direct_hits": sum(r["hit"] for r in rows),
            "full_fallbacks": sum(r["fallback"] for r in rows),
            "simple_cache_within_one_percent": sum(
                r["simple_cache_within_one_percent"] for r in rows),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73041)
    parser.add_argument("--no-spotcheck", action="store_true")
    parser.add_argument("--streams", nargs="+", choices=STREAMS,
                        default=STREAMS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--start", choices=("approach", "origin"),
                        default="approach")
    parser.add_argument("--output", type=Path,
                        default=HERE / "local_update_gate.json")
    args = parser.parse_args()
    if args.samples < 3:
        raise ValueError("Need at least 3 queries for independent LP spot checks")
    results = {"scope": "fixed product MDP, five physical risk locations",
               "parameters": {"gamma": GAMMA, "soft": SOFT, "hard": HARD,
                              "magnitude": .8, "samples_per_stream": args.samples,
                              "stream_seed": args.seed,
                              "start": args.start},
               "cases": []}
    for case in args.cases:
        n, scenario, layout_seed = case.split(":")
        model = build_grid(int(n), scenario, int(layout_seed))
        if args.start == "approach":
            model = approach_start(model, scenario)
        directions = directions_for_five_cells(model)
        locations = np.unique(model.xy[model.cost > 0], axis=0)
        entry = {"case": case, "states": model.states,
                 "initial_xy": model.xy[model.initial].tolist(),
                 "risk_locations": locations.tolist(), "streams": []}
        for kind in args.streams:
            row = run_stream(model, directions, kind, args.samples, args.seed,
                             not args.no_spotcheck, args.progress)
            entry["streams"].append(row)
            print(case, kind, "speedup", round(row["total_speedup"], 2),
                  "p95_ms", round(row["atlas_p95_ms"], 1),
                  round(row["lp_p95_ms"], 1),
                  "hits", row["direct_hits"], "/", args.samples,
                  "simple", row["simple_cache_within_one_percent"],
                  flush=True)
            args.output.write_text(json.dumps({**results, "cases":
                results["cases"] + [entry]}, indent=2))
        results["cases"].append(entry)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
