"""Ablate atlas cache hits by calling one scalarized MDP on every update."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from five_zone_stress import directions_for_five_cells
from joint_price_atlas import add_policy, expand, prepare, query, risk_directions
from stratified import solve_scalarized

HERE = Path(__file__).resolve().parent


def run_case(case, order_seed):
    model = build_grid(24, "middle_fast")
    if case == "wide2":
        directions = risk_directions(model)
        values = (-.8, -.32, 0.0, .32, .8)
        thetas = np.array([(x, y) for x in values for y in values])
        exact_rows = json.loads((HERE / "wide_middle_warm.json").read_text())["rows"]
        exact = {tuple(row["theta"]): row["objective"] for row in exact_rows}
        reference = [exact[tuple(theta)] for theta in thetas]
        fresh_seconds = json.loads((HERE / "wide_middle_stress.json").read_text())["fresh_seconds"]
    else:
        directions = directions_for_five_cells(model)
        thetas = np.asarray(json.loads((HERE / "five_zone_stress.json").read_text())["theta"])
        reference = json.loads((HERE / "five_zone_reference.json").read_text())
        fresh_seconds = json.loads((HERE / "five_zone_fast.json").read_text())["fresh_total_seconds"]
    atlas = prepare(model, directions)
    forced_seconds = query_seconds = fallback_seconds = 0.0
    oracle_adds = fallbacks = 0
    for idx in np.random.default_rng(order_seed).permutation(len(thetas)):
        theta = thetas[idx]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        neighbors = sorted(atlas["policies"],
                           key=lambda p: float(np.linalg.norm(p["source_theta"] - theta)))
        price = next((p["source_price"] for p in neighbors
                      if p["source_price"] > 1e-10), 1.0)
        started = time.perf_counter()
        result = solve_scalarized(changed, price)
        oracle_adds += int(add_policy(atlas, result, theta))
        forced_seconds += time.perf_counter() - started
        answer = query(atlas, theta, nearest_first=8)
        query_seconds += answer["seconds"]
        if not answer["certified"]:
            fallback = expand(atlas, theta)
            fallback_seconds += fallback["seconds"]
            fallbacks += 1
            answer = query(atlas, theta, nearest_first=8)
            query_seconds += answer["seconds"]
        assert answer["certified"]
        evaluated = evaluate_policy(changed, answer["policy"])
        assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
        assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
        assert answer["risk"] <= HARD + 1e-7
        if case == "wide2":
            assert answer["objective"] <= reference[idx] + 1e-6
            assert reference[idx] <= answer["upper"] + 1e-6
        else:
            assert answer["objective"] <= reference[idx]["upper"] + 1e-6
            assert reference[idx]["objective"] <= answer["upper"] + 1e-6
    total = atlas["prepare_seconds"] + forced_seconds + query_seconds + fallback_seconds
    return {"case": case, "order_seed": order_seed, "queries": len(thetas),
            "oracle_adds": oracle_adds, "fallbacks": fallbacks,
            "policies": len(atlas["policies"]),
            "prepare_seconds": atlas["prepare_seconds"],
            "forced_oracle_seconds": forced_seconds,
            "query_seconds": query_seconds,
            "fallback_seconds": fallback_seconds,
            "total_seconds": total, "fresh_over_forced": fresh_seconds / total}


def main():
    rows = [run_case("wide2", seed) for seed in (404, 405, 406)]
    rows += [run_case("five5", seed) for seed in (0, 1, 2)]
    (HERE / "forced_oracle_baseline.json").write_text(json.dumps(rows, indent=2))
    print([(r["case"], r["order_seed"], r["fallbacks"],
            round(r["total_seconds"], 3)) for r in rows])


if __name__ == "__main__":
    main()
