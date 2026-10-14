"""Independent cross-checks of archived atlas policies against warm LP optima."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, evaluate_policy
from joint_price_atlas import expand, prepare, query, risk_directions

HERE = Path(__file__).resolve().parent


def key(theta):
    return tuple(round(float(x), 8) for x in theta)


def check_main():
    atlas = json.loads((HERE / "joint_atlas_benchmark.json").read_text())
    warm = json.loads((HERE / "highs_reopt.json").read_text())
    count = 0
    for a, w in zip(atlas, warm):
        assert a["case"] == w["case"]
        exact = {key(row["theta"]): row["objective"] for row in w["rows"]}
        for replay in a["replays"]:
            for row in replay["sequence"]:
                optimum = exact[key(row["theta"])]
                assert row["objective"] <= optimum + 1e-6
                assert optimum - row["objective"] <= max(
                    1e-4, .01 * abs(optimum)) + 1e-6
                assert row["risk"] <= HARD + 1e-7
                count += 1
    return count


def check_initial():
    atlas = json.loads((HERE / "joint_atlas_initial.json").read_text())
    warm = json.loads((HERE / "highs_reopt_initial.json").read_text())
    count = 0
    for a, w in zip(atlas, warm):
        assert a["case"] == w["case"]
        exact = {(tuple(r["start"]), key(r["theta"])): r["objective"]
                 for r in w["rows"]}
        for row in a["rows"]:
            optimum = exact[(tuple(row["start"]), key(row["theta"]))]
            assert row["objective"] <= optimum + 1e-6
            assert optimum <= row["upper"] + 1e-6
            assert row["risk"] <= HARD + 1e-7
            count += 1
    return count


def check_wide():
    warm = json.loads((HERE / "wide_middle_warm.json").read_text())
    exact = {key(row["theta"]): row["objective"] for row in warm["rows"]}
    model = build_grid(24, "middle_fast")
    directions = risk_directions(model)
    atlas = prepare(model, directions)
    requests = [np.asarray(row["theta"]) for row in warm["rows"]]
    result = []
    for theta in requests:
        answer = query(atlas, theta)
        cached = answer["certified"]
        if not cached:
            expand(atlas, theta)
            answer = query(atlas, theta)
        assert answer["certified"]
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        check = evaluate_policy(changed, answer["policy"])
        optimum = exact[key(theta)]
        assert abs(check["objective"] - answer["objective"]) < 1e-6
        assert check["risk"] <= HARD + 1e-7
        assert answer["objective"] <= optimum + 1e-6
        assert optimum <= answer["upper"] + 1e-6
        assert optimum - answer["objective"] <= max(
            1e-4, .01 * abs(optimum)) + 1e-6
        result.append({"theta": theta.tolist(), "cached": cached,
                       "objective": answer["objective"], "lp": optimum,
                       "risk": answer["risk"], "gap": answer["gap"]})
    (HERE / "wide_exact_validation.json").write_text(json.dumps(result, indent=2))
    return len(result), sum(row["cached"] for row in result)


def main():
    main_count = check_main()
    initial_count = check_initial()
    wide_count, wide_hits = check_wide()
    print(f"PASS: {main_count} main, {initial_count} initial, "
          f"{wide_count} wide LP comparisons; wide hits {wide_hits}")


if __name__ == "__main__":
    main()
