"""Open-source HiGHS LP basis reoptimization baseline for affine risk updates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import highspy
import numpy as np
from scipy import sparse

from experiment import GAMMA, HARD, SOFT, build_grid
from joint_price_atlas import risk_directions

HERE = Path(__file__).resolve().parent


def prepare(model, directions, feasibility_tolerance=None, solver_threads=None):
    started = time.perf_counter()
    n, a = model.states, model.actions
    d = len(directions)
    k = d + 1
    nx = n * a
    columns = np.arange(nx)
    source = sparse.csr_matrix(
        (np.ones(nx), (columns // a, columns)), shape=(n, nx))
    flow = (source - GAMMA * model.p.T).tocsr()
    risk = sparse.csr_matrix(np.stack(
        [np.repeat(model.cost, a)] +
        [np.repeat(direction, a) for direction in directions]))
    zeros = lambda rows, cols: sparse.csr_matrix((rows, cols))
    flow_aug = sparse.hstack((flow, zeros(n, k + 1)), format="csr")
    risk_aug = sparse.hstack((risk, zeros(k, 1), -sparse.eye(k)),
                             format="csr")
    soft = sparse.csr_matrix(
        ([-1.0, 1.0], ([0, 0], [nx, nx + 1])), shape=(1, nx + k + 1))
    hard = sparse.csr_matrix(
        ([1.0], ([0], [nx + 1])), shape=(1, nx + k + 1))
    matrix = sparse.vstack((flow_aug, risk_aug, soft, hard),
                           format="csr")
    lower = np.r_[np.eye(1, n, model.initial).ravel(),
                  np.zeros(k), -highspy.kHighsInf, -highspy.kHighsInf]
    upper = np.r_[np.eye(1, n, model.initial).ravel(),
                  np.zeros(k), SOFT, HARD]
    cost = np.r_[-model.reward.ravel(), 1.0, np.zeros(k)]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    if solver_threads is not None:
        h.setOptionValue("threads", solver_threads)
    if feasibility_tolerance is not None:
        h.setOptionValue("primal_feasibility_tolerance", feasibility_tolerance)
        h.setOptionValue("dual_feasibility_tolerance", feasibility_tolerance)
    status = h.addCols(nx + k + 1, cost,
                       np.r_[np.zeros(nx + 1), np.full(k, -highspy.kHighsInf)],
                       np.full(nx + k + 1, highspy.kHighsInf),
                       0, np.array([0], dtype=np.int32),
                       np.array([], dtype=np.int32), np.array([]))
    assert status == highspy.HighsStatus.kOk
    status = h.addRows(n + k + 2, lower, upper, matrix.nnz,
                       np.asarray(matrix.indptr[:-1], dtype=np.int32),
                       np.asarray(matrix.indices, dtype=np.int32),
                       np.asarray(matrix.data, dtype=np.float64))
    assert status == highspy.HighsStatus.kOk
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
    return {"solver": h, "n": n, "nx": nx, "d": d,
            "initial": model.initial,
            "prepare_seconds": time.perf_counter() - started}


def query(cache, theta, initial=None):
    h = cache["solver"]
    nx = cache["nx"]
    d = cache["d"]
    assert len(theta) == d
    row_soft, row_hard = cache["n"] + d + 1, cache["n"] + d + 2
    started = time.perf_counter()
    if initial is not None and initial != cache["initial"]:
        assert h.changeRowBounds(cache["initial"], 0.0, 0.0) == highspy.HighsStatus.kOk
        assert h.changeRowBounds(initial, 1.0, 1.0) == highspy.HighsStatus.kOk
        cache["initial"] = initial
    for row in (row_soft, row_hard):
        for j in range(d):
            status = h.changeCoeff(row, nx + 2 + j, float(theta[j]))
            assert status == highspy.HighsStatus.kOk
    h.run()
    basis_reset = False
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        # A reused simplex basis can fail numerically on large risk changes.
        # Keep the same LP matrix and pay for a fresh logical basis.
        if h.setBasis() != highspy.HighsStatus.kOk:
            raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
        basis_reset = True
        h.run()
    cold_rebuild = False
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        # setBasis alone may leave stale factorization/model status. Rebuild
        # the same current LP, retaining precision and thread options. Its
        # cost is part of this query's timer.
        current_lp, current_options = h.getLp(), h.getOptions()
        fresh = highspy.Highs()
        fresh.passOptions(current_options)
        fresh.passModel(current_lp)
        fresh.run()
        h = fresh
        cache["solver"] = h
        cold_rebuild = True
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
    sol = h.getSolution()
    rho = np.asarray(sol.col_value[nx + 1:nx + d + 2])
    return {"objective": -h.getObjectiveValue(),
            "risk": float(rho[0] + theta @ rho[1:]),
            "seconds": time.perf_counter() - started,
            "iterations": h.getInfo().simplex_iteration_count,
            "basis_valid": bool(h.getBasis().valid),
            "basis_reset": basis_reset,
            "cold_rebuild": cold_rebuild}


def run_case(n, scene, seed, order_seed=0):
    model = build_grid(n, scene, seed)
    directions = risk_directions(model)
    cache = prepare(model, directions)
    values = (-.05, -.02, 0, .02, .05)
    parameters = np.array([(x, y) for x in values for y in values])
    rows = []
    for idx in np.random.default_rng(order_seed).permutation(len(parameters)):
        theta = parameters[idx]
        result = query(cache, theta)
        assert result["risk"] <= HARD + 1e-7
        rows.append({"theta": theta.tolist(), **result})
    return {"case": f"{n}:{scene}:{seed}",
            "prepare_seconds": cache["prepare_seconds"],
            "total_seconds": cache["prepare_seconds"] +
                             sum(r["seconds"] for r in rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+",
                        default=["24:barrier_hard_active:0",
                                 "32:barrier_hard_active:0",
                                 "24:middle_fast:0",
                                 "32:middle_fast:0"])
    args = parser.parse_args()
    output = []
    for case in args.cases:
        n, scene, seed = case.split(":")
        result = run_case(int(n), scene, int(seed))
        output.append(result)
        (HERE / "highs_reopt.json").write_text(json.dumps(output, indent=2))
        print(case, "prepare", round(result["prepare_seconds"], 3),
              "total", round(result["total_seconds"], 3), flush=True)


if __name__ == "__main__":
    main()
