"""Repeated-update benchmark for the joint price/risk atlas.

Reports all costs including preparation, failed lookups and expansions.
Independent policy evaluation is used for QA and excluded from production
query latency; a fresh CMDP solve is timed for every parameter point.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy, solve_lp
from joint_price_atlas import expand, prepare, query, risk_directions
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent
VALUES = (-.05, -.02, 0.0, .02, .05)
THETAS = [np.array((x, y)) for x in VALUES for y in VALUES]


def changed_model(model, directions, theta):
    return replace(model, cost=model.cost + theta @ directions,
                   layers=model.layers)


def baseline(model, directions):
    records = []
    for theta in THETAS:
        changed = changed_model(model, directions, theta)
        result = solve_cmdp(changed, search_mode="breakpoint",
                            relative_tolerance=.001)
        assert result["certified"] and result["risk"] <= HARD + 1e-7
        records.append({"theta": theta.tolist(),
                        "objective": result["objective"],
                        "upper": result["upper"],
                        "risk": result["risk"],
                        "seconds": result["seconds"]})
    return records


def replay(model, directions, baseline_rows, order_seed):
    rng = np.random.default_rng(order_seed)
    order = rng.permutation(len(THETAS))
    atlas = prepare(model, directions)
    setup = atlas["prepare_seconds"]
    started = time.perf_counter()
    accepted = expanded = added = 0
    lookup_seconds = expansion_seconds = 0.0
    worst_gap = 0.0
    sequence = []
    for index in order:
        theta = THETAS[index]
        lookup = query(atlas, theta)
        lookup_seconds += lookup["seconds"]
        used_expansion = False
        if not lookup["certified"]:
            extension = expand(atlas, theta)
            expansion_seconds += extension["seconds"]
            expanded += 1
            added += extension["added"]
            used_expansion = True
            lookup = query(atlas, theta)
            lookup_seconds += lookup["seconds"]
        else:
            accepted += 1
        if not lookup["certified"]:
            raise AssertionError(f"Expansion did not certify theta {theta}")
        changed = changed_model(model, directions, theta)
        evaluated = evaluate_policy(changed, lookup["policy"])
        fresh = baseline_rows[index]
        assert abs(evaluated["risk"] - lookup["risk"]) < 1e-6
        assert abs(evaluated["objective"] - lookup["objective"]) < 1e-6
        assert evaluated["risk"] <= HARD + 1e-7
        assert lookup["objective"] <= fresh["upper"] + 1e-6
        assert fresh["objective"] <= lookup["upper"] + 1e-6
        worst_gap = max(worst_gap, lookup["gap"])
        sequence.append({"theta": theta.tolist(),
                         "cached": not used_expansion,
                         "objective": lookup["objective"],
                         "risk": lookup["risk"],
                         "gap": lookup["gap"]})
    # Wall time includes policy QA; timing below is the instrumented online
    # path only, excluding QA on both sides. Keep wall for transparency.
    wall_seconds = time.perf_counter() - started
    online_total = setup + lookup_seconds + expansion_seconds
    return {"order_seed": order_seed, "hits": accepted,
            "expansions": expanded, "added_policies": added,
            "final_policy_count": len(atlas["policies"]),
            "prepare_seconds": setup,
            "lookup_seconds": lookup_seconds,
            "expansion_seconds": expansion_seconds,
            "total_online_seconds": online_total,
            "wall_with_qa_seconds": wall_seconds + setup,
            "fresh_total_seconds": sum(r["seconds"] for r in baseline_rows),
            "fresh_over_atlas": sum(r["seconds"] for r in baseline_rows) / online_total,
            "worst_certificate_gap": worst_gap,
            "sequence": sequence}


def run_case(n, scene, seed=0):
    model = build_grid(n, scene, seed)
    directions = risk_directions(model)
    assert np.count_nonzero(directions[0]) and np.count_nonzero(directions[1])
    fresh = baseline(model, directions)
    replays = [replay(model, directions, fresh, order_seed)
               for order_seed in (0, 1, 2)]
    independent_lp = []
    for index in (0, 12, 24):
        theta = THETAS[index]
        methods = ("highs-ipm", "highs-ds")
        failures = []
        for method in methods:
            try:
                lp = solve_lp(changed_model(model, directions, theta), method=method)
                break
            except RuntimeError as error:
                failures.append(f"{method}: {error}")
        else:
            independent_lp.append({"theta": theta.tolist(),
                                   "failed_methods": failures})
            continue
        assert abs(lp["objective"] - fresh[index]["objective"]) <= max(
            1e-4, .01 * abs(lp["objective"])) + 1e-6
        independent_lp.append({"theta": theta.tolist(),
                               "objective": lp["objective"],
                               "method": method,
                               "seconds": lp["wall_seconds"]})
    return {"case": f"{n}:{scene}:{seed}", "states": model.states,
            "risk_zone_states": np.count_nonzero(directions, axis=1).tolist(),
            "fresh": fresh, "replays": replays,
            "independent_lp": independent_lp}


def main():
    output_path = HERE / "joint_atlas_benchmark.json"
    output = json.loads(output_path.read_text()) if output_path.exists() else []
    for n, scene in ((24, "barrier_hard_active"),
                     (32, "barrier_hard_active"),
                     (24, "middle_fast"),
                     (32, "middle_fast")):
        if any(row["case"] == f"{n}:{scene}:0" for row in output):
            continue
        started = time.perf_counter()
        result = run_case(n, scene)
        output.append(result)
        output_path.write_text(json.dumps(output, indent=2))
        print(result["case"], "hits",
              [r["hits"] for r in result["replays"]], "speed",
              [round(r["fresh_over_atlas"], 2) for r in result["replays"]],
              "wall", round(time.perf_counter() - started, 1), flush=True)


if __name__ == "__main__":
    main()
