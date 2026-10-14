"""Certificate-guided one-oracle expansion on the difficult two-zone case."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand_fast, prepare, query, risk_directions

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "middle_fast")
    directions = risk_directions(model)
    warm = json.loads((HERE / "wide_middle_warm.json").read_text())
    exact = {tuple(r["theta"]): r["objective"] for r in warm["rows"]}
    values = (-.8, -.32, 0.0, .32, .8)
    theta = np.array([(x, y) for x in values for y in values])
    fresh_total = json.loads((HERE / "wide_middle_stress.json").read_text())[
        "fresh_seconds"]
    replays = []
    for order_seed in (404, 405, 406):
        atlas = prepare(model, directions)
        hits = fast_saves = full_fallbacks = 0
        query_seconds = expansion_seconds = 0.0
        for idx in np.random.default_rng(order_seed).permutation(len(theta)):
            parameter = theta[idx]
            answer = query(atlas, parameter, nearest_first=8)
            query_seconds += answer["seconds"]
            if not answer["certified"]:
                extension = expand_fast(atlas, parameter, nearest_first=8)
                expansion_seconds += extension["seconds"]
                fast_saves += int(not extension["fallback"])
                full_fallbacks += int(extension["fallback"])
                answer = query(atlas, parameter, nearest_first=8)
                query_seconds += answer["seconds"]
            else:
                hits += 1
            assert answer["certified"]
            changed = replace(model, cost=model.cost + parameter @ directions,
                              layers=model.layers)
            evaluated = evaluate_policy(changed, answer["policy"])
            optimum = exact[tuple(parameter)]
            assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
            assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
            assert answer["risk"] <= HARD + 1e-7
            assert answer["objective"] <= optimum + 1e-6
            assert optimum <= answer["upper"] + 1e-6
            assert optimum - answer["objective"] <= max(
                1e-4, .01 * abs(optimum)) + 1e-6
        total = atlas["prepare_seconds"] + query_seconds + expansion_seconds
        replays.append({"order_seed": order_seed, "hits": hits,
                        "fast_saves": fast_saves,
                        "full_fallbacks": full_fallbacks,
                        "policies": len(atlas["policies"]),
                        "prepare_seconds": atlas["prepare_seconds"],
                        "query_seconds": query_seconds,
                        "expansion_seconds": expansion_seconds,
                        "total_seconds": total,
                        "fresh_over_atlas": fresh_total / total,
                        "warm_lp_over_atlas": warm["total_seconds"] / total})
    result = {"case": "24:middle_fast:0:m0.8",
              "fresh_seconds": fresh_total,
              "warm_lp_seconds": warm["total_seconds"],
              "replays": replays}
    (HERE / "wide_middle_fast.json").write_text(json.dumps(result, indent=2))
    print([(r["hits"], r["fast_saves"], r["full_fallbacks"],
            round(r["fresh_over_atlas"], 2)) for r in replays])


if __name__ == "__main__":
    main()
