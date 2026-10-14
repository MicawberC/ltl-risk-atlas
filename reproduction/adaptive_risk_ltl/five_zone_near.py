"""Nearest-region-first query for five-dimensional risk updates."""

from __future__ import annotations

import json
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from five_zone_stress import directions_for_five_cells
from joint_price_atlas import expand, expand_adaptive, expand_fast, prepare, query
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    assert not (args.fast and args.adaptive)
    old = json.loads((HERE / "five_zone_stress.json").read_text())
    model = build_grid(24, "middle_fast")
    directions = directions_for_five_cells(model)
    theta = np.asarray(old["theta"])
    reference_path = HERE / "five_zone_reference.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
    else:
        reference = []
        for parameter in theta:
            changed = replace(model, cost=model.cost + parameter @ directions,
                              layers=model.layers)
            fresh = solve_cmdp(changed, search_mode="breakpoint",
                               relative_tolerance=.001)
            assert fresh["certified"]
            reference.append({"objective": fresh["objective"],
                              "upper": fresh["upper"],
                              "seconds": fresh["seconds"]})
        reference_path.write_text(json.dumps(reference, indent=2))
    fresh_total = sum(x["seconds"] for x in reference)
    replays = []
    for order_seed in (0, 1, 2):
        atlas = prepare(model, directions)
        hits = 0
        fast_saves = full_fallbacks = 0
        query_seconds = expansion_seconds = 0.0
        for index in np.random.default_rng(order_seed).permutation(len(theta)):
            parameter = theta[index]
            answer = query(atlas, parameter, nearest_first=8)
            query_seconds += answer["seconds"]
            if not answer["certified"]:
                extension = (expand_adaptive(atlas, parameter, nearest_first=8)
                             if args.adaptive else
                             expand_fast(atlas, parameter, nearest_first=8)
                             if args.fast else expand(atlas, parameter))
                expansion_seconds += extension["seconds"]
                if args.fast or args.adaptive:
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
            assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
            assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
            assert evaluated["risk"] <= HARD + 1e-7
            assert answer["objective"] <= reference[index]["upper"] + 1e-6
            assert reference[index]["objective"] <= answer["upper"] + 1e-6
        total = atlas["prepare_seconds"] + query_seconds + expansion_seconds
        replays.append({"order_seed": order_seed, "hits": hits,
                        "expansions": len(theta) - hits,
                        "policies": len(atlas["policies"]),
                        "fast_saves": fast_saves,
                        "full_fallbacks": full_fallbacks,
                        "prepare_seconds": atlas["prepare_seconds"],
                        "query_seconds": query_seconds,
                        "expansion_seconds": expansion_seconds,
                        "total_seconds": total,
                        "fresh_over_atlas": fresh_total / total})
    result = {"case": old["case"], "fresh_total_seconds": fresh_total,
              "replays": replays}
    output_name = ("five_zone_adaptive.json" if args.adaptive else
                   "five_zone_fast.json" if args.fast else "five_zone_near.json")
    (HERE / output_name).write_text(json.dumps(result, indent=2))
    print([(r["hits"], round(r["fresh_over_atlas"], 2),
            round(r["query_seconds"], 2)) for r in replays])


if __name__ == "__main__":
    main()
