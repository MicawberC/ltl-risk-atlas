"""Fixed-product initial-state and two-zone risk update stress test."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand, prepare, query, rebase_initial, risk_directions
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent
STARTS = ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2))
PERTURBATIONS = ((-.05, .05), (.05, -.05), (0, 0),
                 (-.02, -.02), (.02, .02))


def run(n, scene, seed=0, order_seed=321):
    model = build_grid(n, scene, seed)
    directions = risk_directions(model)
    queries = [(xy, theta) for xy in STARTS for theta in PERTURBATIONS]
    fresh = []
    for xy, theta in queries:
        initial = 8 * (xy[0] + n * xy[1]) + 4
        changed = replace(model, initial=initial,
                          cost=model.cost + np.asarray(theta) @ directions,
                          layers=model.layers)
        result = solve_cmdp(changed, search_mode="breakpoint",
                            relative_tolerance=.001)
        assert result["certified"]
        fresh.append(result)
    atlas = prepare(model, directions)
    setup = atlas["prepare_seconds"]
    rebase_time = query_time = expansion_time = 0.0
    hits = misses = 0
    rows = []
    for index in np.random.default_rng(order_seed).permutation(len(queries)):
        xy, theta = queries[index]
        initial = 8 * (xy[0] + n * xy[1]) + 4
        rebase_time += rebase_initial(atlas, initial)
        answer = query(atlas, theta)
        query_time += answer["seconds"]
        cached = answer["certified"]
        if not answer["certified"]:
            e = expand(atlas, theta)
            expansion_time += e["seconds"]
            misses += 1
            answer = query(atlas, theta)
            query_time += answer["seconds"]
        else:
            hits += 1
        assert answer["certified"], (xy, theta)
        changed = replace(model, initial=initial,
                          cost=model.cost + np.asarray(theta) @ directions,
                          layers=model.layers)
        check = evaluate_policy(changed, answer["policy"])
        assert abs(check["risk"] - answer["risk"]) < 1e-6
        assert abs(check["objective"] - answer["objective"]) < 1e-6
        assert check["risk"] <= HARD + 1e-7
        assert answer["objective"] <= fresh[index]["upper"] + 1e-6
        assert fresh[index]["objective"] <= answer["upper"] + 1e-6
        rows.append({"start": list(xy), "theta": list(theta),
                     "cached": cached,
                     "risk": answer["risk"], "objective": answer["objective"],
                     "upper": answer["upper"], "gap": answer["gap"],
                     "fresh_objective": fresh[index]["objective"],
                     "fresh_upper": fresh[index]["upper"]})
    atlas_total = setup + rebase_time + query_time + expansion_time
    fresh_total = sum(x["seconds"] for x in fresh)
    return {"case": f"{n}:{scene}:{seed}", "states": model.states,
            "queries": len(queries), "hits": hits, "expansions": misses,
            "final_policies": len(atlas["policies"]),
            "fresh_seconds": fresh_total, "atlas_seconds": atlas_total,
            "speedup": fresh_total / atlas_total,
            "setup_seconds": setup, "rebase_seconds": rebase_time,
            "query_seconds": query_time, "expansion_seconds": expansion_time,
            "rows": rows}


def main():
    path = HERE / "joint_atlas_initial.json"
    output = []
    for n, scene in ((24, "barrier_hard_active"), (32, "barrier_hard_active"),
                     (24, "middle_fast"), (32, "middle_fast")):
        result = run(n, scene)
        output.append(result)
        path.write_text(json.dumps(output, indent=2))
        print(result["case"], "hits", result["hits"], "speed",
              round(result["speedup"], 2), flush=True)


if __name__ == "__main__":
    main()
