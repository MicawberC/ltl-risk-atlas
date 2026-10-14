"""Dual-price transport for repeated uniform risk-map rescaling queries.

For c_k = k*c_0, the scalarized MDP at new price t is exactly the
original scalarized MDP at price k*t. Cached policies and occupation
measures therefore remain valid; only the optimal soft/hard risk target
and mixing coefficient can change. This is a narrow but exact probe.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy
from stratified import solve_cmdp, solve_scalarized

HERE = Path(__file__).resolve().parent


def prepare(model):
    started = time.perf_counter()
    base = solve_cmdp(model, search_mode="breakpoint", relative_tolerance=.001)
    assert base["certified"] and base["mixed"]
    archive = [solve_scalarized(model, p) for p in base["prices"]]
    return {"archive": archive, "base": base,
            "prepare_seconds": time.perf_counter() - started}


def query(context, scale):
    started = time.perf_counter()
    high, low = context["archive"]
    upper = min(p["initial_value"] + p["bellman_error"] / (1 - GAMMA) +
                SOFT * min(1, p["price"] / scale) +
                HARD * max(0, p["price"] / scale - 1)
                for p in (high, low))
    candidates = []
    for target in (SOFT, HARD):
        c_high, c_low = scale * high["risk"], scale * low["risk"]
        if not (c_high >= target - 1e-8 and c_low <= target + 1e-8):
            continue
        weight = (target - c_low) / (c_high - c_low)
        reward = weight * high["reward"] + (1 - weight) * low["reward"]
        objective = reward - max(0, target - SOFT)
        gap = upper - objective
        if -1e-7 <= gap <= max(1e-4, .01 * abs(objective)) + 1e-7:
            candidates.append((objective, target, weight, gap))
    if not candidates:
        return {"certified": False, "seconds": time.perf_counter() - started}
    objective, target, weight, gap = max(candidates)
    occupation = weight * high["occupation"] + (1 - weight) * low["occupation"]
    totals = occupation.sum(axis=1)
    policy = np.divide(occupation, totals[:, None], out=np.zeros_like(occupation),
                       where=totals[:, None] > 1e-12)
    policy[totals <= 1e-12, 0] = 1
    return {"certified": True, "objective": objective, "risk": target,
            "upper": upper, "gap": gap, "policy": policy,
            "seconds": time.perf_counter() - started}


def main():
    scales = [.8, .9, .95, .99, 1, 1.01, 1.02, 1.05, 1.1, 1.2]
    output = []
    for n, scene, seed in ((24, "barrier_hard_active", 0),
                           (32, "barrier_hard_active", 0),
                           (24, "middle_fast", 0),
                           (32, "middle_fast", 0)):
        model = build_grid(n, scene, seed)
        context = prepare(model)
        rows = []
        for scale in scales:
            changed = replace(model, cost=model.cost * scale, layers=model.layers)
            answer = query(context, scale)
            fresh = solve_cmdp(changed, search_mode="breakpoint")
            row = {"scale": scale, "certified": answer["certified"],
                   "transfer_seconds": answer["seconds"],
                   "fresh_seconds": fresh["seconds"],
                   "fresh_objective": fresh["objective"]}
            if answer["certified"]:
                check = evaluate_policy(changed, answer["policy"])
                assert abs(check["risk"] - answer["risk"]) < 1e-6
                assert abs(check["objective"] - answer["objective"]) < 1e-6
                assert answer["objective"] <= fresh["upper"] + 1e-6
                assert fresh["objective"] <= answer["upper"] + 1e-6
                row.update({"objective": answer["objective"],
                            "risk": answer["risk"], "gap": answer["gap"]})
            rows.append(row)
        record = {"case": f"{n}:{scene}:{seed}",
                  "prepare_seconds": context["prepare_seconds"], "rows": rows}
        output.append(record)
        (HERE / "uniform_scale_transfer.json").write_text(json.dumps(output, indent=2))
        print(record["case"], sum(x["certified"] for x in rows),
              "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
