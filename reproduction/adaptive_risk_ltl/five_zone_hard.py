"""Held-out five-parameter hard-active risk map, with fast atlas expansion."""

from __future__ import annotations

import json
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand_adaptive, expand_fast, prepare, query
from primal_cache_ablation import best_primal
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def spatial_directions(model):
    positions = np.unique(model.xy[model.cost > 0], axis=0)
    positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
    buckets = np.array_split(positions, 5)
    directions = []
    for bucket in buckets:
        selected = np.any(np.all(model.xy[:, None, :] == bucket[None, :, :],
                                 axis=2), axis=1)
        directions.append(np.where(selected, model.cost, 0.0))
    result = np.stack(directions)
    assert np.allclose(result.sum(axis=0), model.cost)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    model = build_grid(24, "barrier_hard_active", seed=2)
    directions = spatial_directions(model)
    rng = np.random.default_rng(874)
    thetas = rng.uniform(-.8, .8, size=(25, 5))
    thetas[0] = 0
    fresh = []
    for theta in thetas:
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        solved = solve_cmdp(changed, search_mode="breakpoint",
                            relative_tolerance=.001)
        assert solved["certified"]
        fresh.append(solved)
    atlas = prepare(model, directions)
    simple_within = sum(
        fresh[i]["objective"] - best_primal(atlas["policies"], thetas[i])
        <= max(1e-4, .01 * abs(fresh[i]["objective"])) + 1e-7
        for i in range(len(thetas)))
    query_seconds = expansion_seconds = 0.0
    hits = fast_saves = full_fallbacks = oracle_calls = 0
    for index in np.random.default_rng(19).permutation(len(thetas)):
        theta = thetas[index]
        answer = query(atlas, theta, nearest_first=8)
        query_seconds += answer["seconds"]
        if not answer["certified"]:
            extension = (expand_fast(atlas, theta, nearest_first=8)
                         if args.fast else
                         expand_adaptive(atlas, theta, nearest_first=8))
            expansion_seconds += extension["seconds"]
            oracle_calls += extension.get("oracles", 1)
            fast_saves += int(not extension["fallback"])
            full_fallbacks += int(extension["fallback"])
            answer = query(atlas, theta, nearest_first=8)
            query_seconds += answer["seconds"]
        else:
            hits += 1
        assert answer["certified"]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        verified = evaluate_policy(changed, answer["policy"])
        assert abs(verified["risk"] - answer["risk"]) < 1e-6
        assert abs(verified["objective"] - answer["objective"]) < 1e-6
        assert answer["risk"] <= HARD + 1e-7
        assert answer["objective"] <= fresh[index]["upper"] + 1e-6
        assert fresh[index]["objective"] <= answer["upper"] + 1e-6
    total = atlas["prepare_seconds"] + query_seconds + expansion_seconds
    result = {"case": "24:barrier_hard_active:2:five_zone:m0.8",
              "theta": thetas.tolist(), "samples": len(thetas),
              "simple_cache_within_one_percent": int(simple_within),
              "fresh_seconds": sum(x["seconds"] for x in fresh),
              "hits": hits, "fast_saves": fast_saves,
              "full_fallbacks": full_fallbacks,
              "oracle_calls": oracle_calls,
              "policies": len(atlas["policies"]),
              "prepare_seconds": atlas["prepare_seconds"],
              "query_seconds": query_seconds,
              "expansion_seconds": expansion_seconds,
              "atlas_seconds": total,
              "speedup": sum(x["seconds"] for x in fresh) / total}
    output = "five_zone_hard_fast.json" if args.fast else "five_zone_hard.json"
    (HERE / output).write_text(json.dumps(result, indent=2))
    print({k: result[k] for k in ("hits", "fast_saves", "full_fallbacks",
                                   "simple_cache_within_one_percent", "speedup")})


if __name__ == "__main__":
    main()
