"""Shared LP utilities for diagnostics of the public IROS example."""

from __future__ import annotations

import time

import gurobipy as gp
import numpy as np
from scipy import sparse

SOFT = 0.8
HARD = 5.0
GAMMA = 0.9


def stochastic_extension(p: np.ndarray) -> np.ndarray:
    """Complete missing boundary mass with a zero-value absorbing state."""
    n, a, _ = p.shape
    result = np.zeros((n + 1, a, n + 1))
    result[:n, :, :n] = p
    missing = 1.0 - p.sum(axis=2)
    if np.min(missing) < -1e-10:
        raise ValueError("Transition row sum exceeds one")
    result[:n, :, n] = np.maximum(missing, 0)
    result[n, :, n] = 1.0
    if not np.allclose(result.sum(axis=2), 1, atol=1e-12):
        raise AssertionError("Extension is not stochastic")
    return result


def metrics(occ: np.ndarray, p: np.ndarray, cost: np.ndarray, initial: int,
            accept: list[int]) -> dict:
    alpha = np.zeros(len(cost))
    alpha[initial] = 1
    flow = occ.sum(axis=1) - GAMMA * np.einsum("sa,san->n", occ, p)
    risk = float(np.sum(occ * cost[:, None]))
    goal = float(np.sum(occ * (2 * p[:, :, accept].sum(axis=2))))
    return {
        "objective": goal - max(0.0, risk - SOFT),
        "risk": risk,
        "goal_reward": goal,
        "max_flow_residual": float(np.max(np.abs(flow - alpha))),
    }


def sparse_lp(p: np.ndarray, cost: np.ndarray, initial: int,
              accept: list[int], hard_bound: float = HARD) -> tuple[np.ndarray, dict]:
    """Solve the IROS occupancy LP with sparse flow constraints."""
    states, actions, _ = p.shape
    started = time.perf_counter()
    reward = 2 * p[:, :, accept].sum(axis=2).reshape(-1)
    risk_coeff = np.repeat(cost, actions)
    columns = np.arange(states * actions)
    src = np.repeat(np.arange(states), actions)
    old_state, action, next_state = np.nonzero(p)
    rows = np.r_[src, next_state]
    cols = np.r_[columns, old_state * actions + action]
    data = np.r_[np.ones(states * actions), -GAMMA * p[old_state, action, next_state]]
    flow = sparse.coo_matrix((data, (rows, cols)), shape=(states, states * actions)).tocsr()
    alpha = np.zeros(states)
    alpha[initial] = 1
    model = gp.Model("iros_sparse_lp")
    model.Params.OutputFlag = 0
    x = model.addMVar(states * actions, lb=0, name="occupation")
    z = model.addVar(lb=0, name="soft_excess")
    model.addConstr(flow @ x == alpha)
    model.addConstr(risk_coeff @ x <= SOFT + z)
    model.addConstr(risk_coeff @ x <= hard_bound)
    model.setObjective(reward @ x - z, gp.GRB.MAXIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Sparse LP status {model.Status}")
    return x.X.reshape(states, actions), {
        "wall_seconds": time.perf_counter() - started,
        "solver_seconds": float(model.Runtime),
        "solver_objective": float(model.ObjVal),
    }
