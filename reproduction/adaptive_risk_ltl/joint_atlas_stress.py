"""Held-out layout and larger perturbation checks for the joint atlas."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand, prepare, query, risk_directions
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def run(n, scene, seed, magnitude, order_seed):
    model = build_grid(n, scene, seed)
    directions = risk_directions(model)
    values = np.array([-magnitude, -magnitude * .4, 0,
                       magnitude * .4, magnitude])
    parameters = np.array([(x, y) for x in values for y in values])
    baseline = []
    for theta in parameters:
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        fresh = solve_cmdp(changed, search_mode="breakpoint",
                           relative_tolerance=.001)
        assert fresh["certified"]
        baseline.append(fresh)
    atlas = prepare(model, directions)
    setup = atlas["prepare_seconds"]
    rng = np.random.default_rng(order_seed)
    hits = misses = 0
    query_seconds = expansion_seconds = 0.0
    worst_gap = 0.0
    for idx in rng.permutation(len(parameters)):
        theta = parameters[idx]
        answer = query(atlas, theta)
        query_seconds += answer["seconds"]
        if not answer["certified"]:
            misses += 1
            e = expand(atlas, theta)
            expansion_seconds += e["seconds"]
            answer = query(atlas, theta)
            query_seconds += answer["seconds"]
        else:
            hits += 1
        assert answer["certified"]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        evaluated = evaluate_policy(changed, answer["policy"])
        assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
        assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
        assert evaluated["risk"] <= HARD + 1e-7
        assert answer["objective"] <= baseline[idx]["upper"] + 1e-6
        assert baseline[idx]["objective"] <= answer["upper"] + 1e-6
        worst_gap = max(worst_gap, answer["gap"])
    fresh_total = sum(x["seconds"] for x in baseline)
    atlas_total = setup + query_seconds + expansion_seconds
    return {"case": f"{n}:{scene}:{seed}:m{magnitude}",
            "order_seed": order_seed, "states": model.states,
            "hits": hits, "expansions": misses,
            "final_policies": len(atlas["policies"]),
            "fresh_seconds": fresh_total,
            "atlas_seconds": atlas_total,
            "speedup": fresh_total / atlas_total,
            "setup_seconds": setup, "query_seconds": query_seconds,
            "expansion_seconds": expansion_seconds,
            "max_gap": worst_gap}


def main():
    path = HERE / "joint_atlas_stress.json"
    output = json.loads(path.read_text()) if path.exists() else []
    cases = [(24, "barrier_hard_active", seed, .05, 100 + seed)
             for seed in (1, 2, 3, 4)]
    cases += [(24, "barrier_hard_active", 0, .2, 201),
              (24, "middle_fast", 0, .2, 202)]
    for case in cases:
        key = f"{case[0]}:{case[1]}:{case[2]}:m{case[3]}"
        if any(x["case"] == key for x in output):
            continue
        started = time.perf_counter()
        record = run(*case)
        output.append(record)
        path.write_text(json.dumps(output, indent=2))
        print(key, "hits", record["hits"], "speed",
              round(record["speedup"], 2), "wall",
              round(time.perf_counter() - started, 1), flush=True)


if __name__ == "__main__":
    main()
