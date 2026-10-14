"""Five independent risk-severity parameters on the middle-target product MDP."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand, prepare, query
from primal_cache_ablation import best_primal
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def directions_for_five_cells(model):
    positions = np.unique(model.xy[model.cost > 0], axis=0)
    assert len(positions) == 5
    return np.stack([
        np.where(np.all(model.xy == pos, axis=1), model.cost, 0)
        for pos in positions
    ])


def run(n=24, samples=50, magnitude=.8):
    model = build_grid(n, "middle_fast")
    directions = directions_for_five_cells(model)
    rng = np.random.default_rng(73021)
    thetas = rng.uniform(-magnitude, magnitude, size=(samples, 5))
    thetas[0] = 0
    fresh = []
    for theta in thetas:
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        result = solve_cmdp(changed, search_mode="breakpoint",
                            relative_tolerance=.001)
        assert result["certified"]
        fresh.append(result)
    nominal = prepare(model, directions)
    simple_losses = [max(0, f["objective"] - best_primal(nominal["policies"], theta))
                     for theta, f in zip(thetas, fresh)]
    replays = []
    for order_seed in (0, 1, 2):
        atlas = prepare(model, directions)
        query_seconds = expansion_seconds = 0.0
        hits = 0
        for idx in np.random.default_rng(order_seed).permutation(samples):
            theta = thetas[idx]
            answer = query(atlas, theta)
            query_seconds += answer["seconds"]
            if not answer["certified"]:
                extension = expand(atlas, theta)
                expansion_seconds += extension["seconds"]
                answer = query(atlas, theta)
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
            assert answer["objective"] <= fresh[idx]["upper"] + 1e-6
            assert fresh[idx]["objective"] <= answer["upper"] + 1e-6
        total = atlas["prepare_seconds"] + query_seconds + expansion_seconds
        replays.append({"order_seed": order_seed, "hits": hits,
                        "expansions": samples - hits,
                        "policies": len(atlas["policies"]),
                        "prepare_seconds": atlas["prepare_seconds"],
                        "query_seconds": query_seconds,
                        "expansion_seconds": expansion_seconds,
                        "total_seconds": total,
                        "fresh_over_atlas": sum(f["seconds"] for f in fresh) / total})
    return {"case": f"{n}:middle_fast:five_zone:m{magnitude}",
            "states": model.states, "samples": samples,
            "fresh_total_seconds": sum(f["seconds"] for f in fresh),
            "simple_cache_within_one_percent": int(sum(
                loss <= max(1e-4, .01 * abs(f["objective"])) + 1e-7
                for loss, f in zip(simple_losses, fresh))),
            "simple_cache_max_loss": float(max(simple_losses)),
            "replays": replays,
            "theta": thetas.tolist()}


def main():
    result = run()
    (HERE / "five_zone_stress.json").write_text(json.dumps(result, indent=2))
    print(result["case"], "simple",
          result["simple_cache_within_one_percent"], "/", result["samples"],
          "hits", [r["hits"] for r in result["replays"]],
          "speed", [round(r["fresh_over_atlas"], 2) for r in result["replays"]])


if __name__ == "__main__":
    main()
