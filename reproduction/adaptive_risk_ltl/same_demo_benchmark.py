"""Same released IROS demo product: upstream Gurobi LP vs atlas vs warm LP.

The 25 risk-severity updates are our synthetic workload; the original authors
released the fixed initial demo, not their complete CARLA scenario data.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "reproduction"))

from iros_core import Risk_LTL_LP, build_iros_model  # noqa: E402
from experiment import HARD, SOFT, build_grid, evaluate_policy  # noqa: E402
from highs_reopt import prepare as prepare_highs, query as query_highs  # noqa: E402
from joint_price_atlas import expand_fast, prepare, query, risk_directions  # noqa: E402


def upstream_solve(product, cost, initial, reward):
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        values = Risk_LTL_LP().solve(product.prod_transitions, cost, initial,
                                    product.accepting_states, None)
    seconds = time.perf_counter() - started
    n, a = product.prod_transitions.shape[:2]
    occ = np.array([[values[s, action] for action in range(a)]
                    for s in range(n)])
    risk = float(np.sum(occ.sum(axis=1) * cost))
    objective = float(np.sum(occ * reward) - max(0.0, risk - SOFT))
    assert risk <= HARD + 1e-7
    return {"objective": objective, "risk": risk, "seconds": seconds}


def main():
    product, original_cost, initial = build_iros_model()
    model = build_grid(4)
    directions = risk_directions(model)
    assert np.allclose(model.cost[:-1], original_cost)
    values = (-.05, -.02, 0.0, .02, .05)
    thetas = np.asarray([(x, y) for x in values for y in values])
    runs = []
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(thetas))
        atlas = prepare(model, directions)
        warm = prepare_highs(model, directions)
        upstream_seconds = atlas_query_seconds = atlas_expand_seconds = 0.0
        warm_seconds = 0.0
        hits = fast_saves = fallbacks = 0
        rows = []
        for index in order:
            theta = thetas[index]
            changed = replace(model, cost=model.cost + theta @ directions,
                              layers=model.layers)
            exact = upstream_solve(product, changed.cost[:-1], initial,
                                   model.reward[:-1])
            upstream_seconds += exact["seconds"]
            lp = query_highs(warm, theta)
            warm_seconds += lp["seconds"]
            assert abs(lp["objective"] - exact["objective"]) < 1e-6
            answer = query(atlas, theta, nearest_first=8)
            atlas_query_seconds += answer["seconds"]
            if not answer["certified"]:
                extension = expand_fast(atlas, theta, nearest_first=8)
                atlas_expand_seconds += extension["seconds"]
                fast_saves += int(not extension["fallback"])
                fallbacks += int(extension["fallback"])
                answer = query(atlas, theta, nearest_first=8)
                atlas_query_seconds += answer["seconds"]
            else:
                hits += 1
            assert answer["certified"]
            checked = evaluate_policy(changed, answer["policy"])
            assert abs(checked["risk"] - answer["risk"]) < 1e-6
            assert abs(checked["objective"] - answer["objective"]) < 1e-6
            assert answer["risk"] <= HARD + 1e-7
            assert answer["objective"] <= exact["objective"] + 1e-6
            assert exact["objective"] <= answer["upper"] + 1e-6
            rows.append({"index": int(index), "theta": theta.tolist(),
                         "upstream": exact, "warm_lp": lp,
                         "atlas_objective": answer["objective"],
                         "atlas_risk": answer["risk"],
                         "atlas_upper": answer["upper"]})
        atlas_total = (atlas["prepare_seconds"] + atlas_query_seconds +
                       atlas_expand_seconds)
        warm_total = warm["prepare_seconds"] + warm_seconds
        runs.append({"seed": seed, "queries": len(thetas), "hits": hits,
                     "fast_saves": fast_saves, "fallbacks": fallbacks,
                     "upstream_gurobi_seconds": upstream_seconds,
                     "atlas_prepare_seconds": atlas["prepare_seconds"],
                     "atlas_query_seconds": atlas_query_seconds,
                     "atlas_expand_seconds": atlas_expand_seconds,
                     "atlas_total_seconds": atlas_total,
                     "warm_lp_seconds": warm_total,
                     "rows": rows})
        print(seed, "upstream", round(upstream_seconds, 4),
              "atlas", round(atlas_total, 4),
              "warm", round(warm_total, 4),
              "hits", hits, flush=True)
    (HERE / "same_demo_benchmark.json").write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
