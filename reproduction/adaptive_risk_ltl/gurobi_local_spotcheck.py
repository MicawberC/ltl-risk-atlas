"""Independently solve selected 32x32 crossing CMDPs as occupancy LPs."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import gurobipy as gp
import numpy as np
from scipy import sparse

from experiment import GAMMA, HARD, SOFT, build_grid
from five_zone_stress import directions_for_five_cells
from local_update_gate import approach_start


HERE = Path(__file__).resolve().parent


def gurobi_lp(model, limit: float = 60.0) -> dict:
    start = time.perf_counter()
    n, a = model.states, model.actions
    count = n * a
    cols = np.arange(count)
    source = sparse.csr_matrix((np.ones(count), (cols // a, cols)),
                               shape=(n, count))
    flow = (source - GAMMA * model.p.T).tocsr()
    alpha = np.zeros(n)
    alpha[model.initial] = 1.0
    risk = np.repeat(model.cost, a)
    opt = gp.Model()
    opt.Params.OutputFlag = 0
    opt.Params.Threads = 1
    opt.Params.TimeLimit = limit
    x = opt.addMVar(count, lb=0)
    slack = opt.addVar(lb=0)
    opt.addMConstr(flow, x, "=", alpha)
    opt.addConstr(risk @ x - slack <= SOFT)
    opt.addConstr(risk @ x <= HARD)
    opt.setObjective(model.reward.ravel() @ x - slack, gp.GRB.MAXIMIZE)
    try:
        opt.optimize()
    except gp.GurobiError as exc:
        if "Model too large for size-limited license" not in str(exc):
            raise
        return {"status": "license_limit", "detail": str(exc),
                "seconds": time.perf_counter() - start}
    if opt.Status != gp.GRB.OPTIMAL:
        return {"status": int(opt.Status),
                "seconds": time.perf_counter() - start}
    return {"status": "optimal", "objective": float(opt.ObjVal),
            "risk": float(risk @ x.X),
            "seconds": time.perf_counter() - start}


def main() -> None:
    saved = json.loads((HERE / "local_update_crossing_iid32.json").read_text())
    rows = saved["cases"][0]["streams"][0]["rows"]
    model = approach_start(build_grid(32, "middle_fast_crossing"),
                           "middle_fast_crossing")
    directions = directions_for_five_cells(model)
    results = []
    for index in (0, 10, 19):
        theta = np.asarray(rows[index]["theta"])
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        checked = gurobi_lp(changed)
        if checked["status"] == "optimal":
            assert abs(checked["objective"] -
                       rows[index]["lp_objective"]) < 1e-5
            assert checked["risk"] <= HARD + 1e-6
        results.append({"index": index, **checked})
        print(index, checked["status"],
              round(checked["seconds"], 3), flush=True)
    (HERE / "gurobi_local_spotcheck.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
