"""Two-location risk-map update probe with cached primal-dual certificates.

This is a research probe, not a claim of novel parametric MDP theory.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy
from stratified import solve_cmdp, solve_scalarized

HERE = Path(__file__).resolve().parent


def prepare(model, directions):
    started = time.perf_counter()
    base = solve_cmdp(model, search_mode="breakpoint", relative_tolerance=.001)
    assert base["certified"] and base["mixed"]
    archive = []
    n, a = model.states, model.actions
    for price in base["prices"]:
        result = solve_scalarized(model, price)
        actions = np.argmax(result["policy"], axis=1)
        transition = model.p[np.arange(n) * a + actions].tocsc()
        system = sparse.eye(n, format="csc") - GAMMA * transition
        influences = np.column_stack([spsolve(system, direction)
                                      for direction in directions])
        slack = result["value"][:, None] - result["q"]
        slopes = np.stack([price * (direction[:, None] +
                           GAMMA * (model.p @ influences[:, j]).reshape(n, a) -
                           influences[:, j, None])
                           for j, direction in enumerate(directions)], axis=-1)
        archive.append({"price": price, "slack": slack, "slopes": slopes,
                        "risk": result["risk"],
                        "risk_slopes": np.array([result["occupation"].sum(axis=1) @ h
                                                 for h in directions]),
                        "value": result["initial_value"],
                        "value_slopes": -price * influences[model.initial],
                        "reward": result["reward"],
                        "occupation": result["occupation"]})
    return {"base": base, "archive": archive,
            "directions": directions,
            "prepare_seconds": time.perf_counter() - started}


def query(cache, theta):
    started = time.perf_counter()
    policies = cache["archive"]
    for p in policies:
        if np.min(p["slack"] + np.tensordot(p["slopes"], theta, axes=([2], [0]))) < -1e-8:
            return {"certified": False, "reason": "Bellman region",
                    "seconds": time.perf_counter() - started}
    target = cache["base"]["target"]
    risks = [p["risk"] + p["risk_slopes"] @ theta for p in policies]
    if not (risks[0] >= target - 1e-8 and risks[1] <= target + 1e-8):
        return {"certified": False, "reason": "risk bracket",
                "seconds": time.perf_counter() - started}
    weight = (target - risks[1]) / (risks[0] - risks[1])
    reward = weight * policies[0]["reward"] + (1 - weight) * policies[1]["reward"]
    objective = reward - max(0, target - SOFT)
    upper = min(p["value"] + p["value_slopes"] @ theta +
                SOFT * min(1, p["price"]) + HARD * max(0, p["price"] - 1)
                for p in policies)
    gap = upper - objective
    certified = bool(-1e-7 <= gap <= max(1e-4, .01 * abs(objective)) + 1e-7)
    answer = {"certified": certified, "objective": objective,
              "risk": target, "upper": upper, "gap": gap,
              "weight": weight}
    if certified:
        occupation = (weight * policies[0]["occupation"] +
                      (1 - weight) * policies[1]["occupation"])
        totals = occupation.sum(axis=1)
        policy = np.divide(occupation, totals[:, None], out=np.zeros_like(occupation),
                           where=totals[:, None] > 1e-12)
        policy[totals <= 1e-12, 0] = 1
        answer["policy"] = policy
    answer["seconds"] = time.perf_counter() - started
    return answer


def main():
    output = []
    for n in (24, 32):
        model = build_grid(n, "middle_fast")
        # Partition the five nonzero-risk physical cells into two local zones.
        left = (model.xy[:, 0] <= n // 2) & (model.cost > 0)
        directions = [np.where(left, model.cost, 0),
                      np.where(~left, model.cost, 0)]
        cache = prepare(model, directions)
        rows = []
        for theta1 in (-.05, -.02, 0, .02, .05):
            for theta2 in (-.05, -.02, 0, .02, .05):
                theta = np.array([theta1, theta2])
                changed = replace(model,
                                  cost=model.cost + theta1 * directions[0] +
                                  theta2 * directions[1],
                                  layers=model.layers)
                answer = query(cache, theta)
                fresh = solve_cmdp(changed, search_mode="breakpoint")
                row = {"theta": theta.tolist(), "certified": answer["certified"],
                       "transfer_seconds": answer["seconds"],
                       "fresh_seconds": fresh["seconds"],
                       "fresh_objective": fresh["objective"]}
                if answer["certified"]:
                    evaluated = evaluate_policy(changed, answer["policy"])
                    assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
                    assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
                    assert answer["objective"] <= fresh["upper"] + 1e-6
                    assert fresh["objective"] <= answer["upper"] + 1e-6
                    row.update({"risk": answer["risk"], "objective": answer["objective"],
                                "gap": answer["gap"]})
                else:
                    row["reason"] = answer.get("reason", "gap")
                rows.append(row)
        record = {"case": f"{n}:middle_fast", "prepare_seconds": cache["prepare_seconds"],
                  "rows": rows}
        output.append(record)
        (HERE / "multi_parametric_probe.json").write_text(json.dumps(output, indent=2))
        print(record["case"], "certified", sum(x["certified"] for x in rows),
              "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
