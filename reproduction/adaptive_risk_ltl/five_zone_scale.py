"""Scale check: 32x32 product and five independent local risk updates."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy, solve_lp
from five_zone_stress import directions_for_five_cells
from highs_reopt import prepare as prepare_highs, query as query_highs
from joint_price_atlas import expand_fast, prepare, query
from primal_cache_ablation import best_primal
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(32, "middle_fast")
    directions = directions_for_five_cells(model)
    thetas = np.random.default_rng(73022).uniform(-.8, .8, size=(25, 5))
    thetas[0] = 0
    fresh = []
    for theta in thetas:
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        result = solve_cmdp(changed, search_mode="breakpoint",
                            relative_tolerance=.001)
        assert result["certified"]
        fresh.append(result)
    atlas = prepare(model, directions)
    simple_within = sum(
        fresh[i]["objective"] - best_primal(atlas["policies"], thetas[i])
        <= max(1e-4, .01 * abs(fresh[i]["objective"])) + 1e-7
        for i in range(len(thetas)))
    hits = fast_saves = full_fallbacks = 0
    query_seconds = expansion_seconds = 0.0
    order = np.random.default_rng(17).permutation(len(thetas))
    for index in order:
        theta = thetas[index]
        answer = query(atlas, theta, nearest_first=8)
        query_seconds += answer["seconds"]
        if not answer["certified"]:
            extension = expand_fast(atlas, theta, nearest_first=8)
            expansion_seconds += extension["seconds"]
            fast_saves += int(not extension["fallback"])
            full_fallbacks += int(extension["fallback"])
            answer = query(atlas, theta, nearest_first=8)
            query_seconds += answer["seconds"]
        else:
            hits += 1
        assert answer["certified"]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        checked = evaluate_policy(changed, answer["policy"])
        assert abs(checked["objective"] - answer["objective"]) < 1e-6
        assert abs(checked["risk"] - answer["risk"]) < 1e-6
        assert checked["risk"] <= HARD + 1e-7
        assert answer["objective"] <= fresh[index]["upper"] + 1e-6
        assert fresh[index]["objective"] <= answer["upper"] + 1e-6
    atlas_seconds = atlas["prepare_seconds"] + query_seconds + expansion_seconds
    warm = prepare_highs(model, directions)
    warm_rows = []
    for index in order:
        answer = query_highs(warm, thetas[index])
        assert answer["basis_valid"] and answer["risk"] <= HARD + 1e-6
        assert fresh[index]["objective"] <= answer["objective"] + 1e-5
        assert answer["objective"] <= fresh[index]["upper"] + 1e-5
        warm_rows.append(answer)
    spot = []
    for index in (0, 12, 24):
        theta = thetas[index]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        exact = solve_lp(changed, method="highs-ds")
        assert fresh[index]["objective"] <= exact["objective"] + 1e-5
        assert exact["objective"] <= fresh[index]["upper"] + 1e-5
        spot.append({"index": index, "objective": exact["objective"],
                     "seconds": exact["wall_seconds"]})
    result = {"case": "32:middle_fast:five_zone:m0.8",
              "states": model.states, "samples": len(thetas),
              "theta": thetas.tolist(),
              "simple_cache_within_one_percent": int(simple_within),
              "fresh_seconds": sum(x["seconds"] for x in fresh),
              "hits": hits, "fast_saves": fast_saves,
              "full_fallbacks": full_fallbacks,
              "prepare_seconds": atlas["prepare_seconds"],
              "query_seconds": query_seconds,
              "expansion_seconds": expansion_seconds,
              "atlas_seconds": atlas_seconds,
              "warm_lp_seconds": warm["prepare_seconds"] +
                                 sum(x["seconds"] for x in warm_rows),
              "spot_lp": spot}
    (HERE / "five_zone_scale.json").write_text(json.dumps(result, indent=2))
    print({k: result[k] for k in ("hits", "fast_saves", "full_fallbacks",
                                   "fresh_seconds", "atlas_seconds",
                                   "warm_lp_seconds")})


if __name__ == "__main__":
    main()
