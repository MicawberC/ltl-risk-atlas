"""Certified adaptive aggregation for the released IROS risk-LTL objective.

This is an experimental extension, not code from the IROS authors.  The
scalable grids retain their nine-action motion kernel, two-state traffic
light, F(target) and safety automata, discounted reward, and risk LP, but are
new synthetic instances rather than unpublished CARLA scenes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import spsolve

GAMMA = float(os.environ.get("LTL_RISK_GAMMA", "0.9"))
SOFT = float(os.environ.get("LTL_RISK_SOFT", "0.8"))
HARD = float(os.environ.get("LTL_RISK_HARD", "5.0"))
if not (0 < GAMMA < 1 and 0 <= SOFT < HARD):
    raise ValueError("Require 0 < gamma < 1 and 0 <= soft < hard")
ACTIONS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


@dataclass
class Model:
    p: sparse.csr_matrix  # (states * actions, states)
    reward: np.ndarray  # (states, actions)
    cost: np.ndarray  # (states,)
    initial: int
    xy: np.ndarray  # (states, 2); sink is (-1,-1)
    mode: np.ndarray  # physical environment mode
    goal: np.ndarray  # F(target) DFA state
    bad: np.ndarray  # safety DFA state
    label: np.ndarray  # label signature
    grid: int
    layers: list[np.ndarray] | None = None  # cached automaton SCC solve order

    @property
    def states(self) -> int:
        return self.p.shape[1]

    @property
    def actions(self) -> int:
        return self.reward.shape[1]


def _labels(n: int, scenario: str, seed: int = 0) -> np.ndarray:
    """Scale the authors' spatial labels; n=4 matches their fixed demo."""
    labels = np.zeros(n * n, dtype=np.int8)
    if n == 4:
        cells = {(2, 2): 1, (3, 2): 1, (2, 3): 1, (3, 3): 2,
                 (2, 1): 4, (1, 1): 8}
    else:
        if scenario == "sequence":
            cells = {(2, 2): 16, (5, 5): 2, (3, 2): 4,
                     (4, 4): 8, (5, 4): 1}
            for (x, y), tag in cells.items():
                labels[x + n * y] = tag
            return labels
        if scenario == "middle_fast_varied":
            # A held-out family of middle-target layouts.  Rotate the same
            # semantic labels and move their centre; keep the fixed product
            # model throughout each risk-update stream.
            shift = ((seed // 4) % 3 - 1, (seed // 12) % 3 - 1)
            cx, cy = n // 2 + 3 * shift[0], n // 2 + 3 * shift[1]
            pattern = {(-1, -1): 8, (0, -1): 4, (0, 0): 1,
                       (1, 0): 1, (0, 1): 1, (1, 1): 2}
            cells = {}
            for (dx, dy), tag in pattern.items():
                for _ in range(seed % 4):
                    dx, dy = -dy, dx
                cells[(cx + dx, cy + dy)] = tag
            for (x, y), tag in cells.items():
                assert 0 <= x < n and 0 <= y < n
                labels[x + n * y] = tag
            return labels
        if scenario == "middle_fast_crossing":
            # A five-cell local sidewalk band between approach and target.
            # Going through it is shorter; going around it has zero risk.
            cx, cy = n // 2, n // 2
            for y in range(cy - 2, cy + 3):
                labels[cx + n * y] = 8
            labels[(cx + 3) + n * cy] = 2
            return labels
        if scenario in ("varied", "terrain"):
            rng = np.random.default_rng(seed)
            ox = int(rng.integers(3, min(9, n - 2)))
            oy = int(rng.integers(3, min(9, n - 2)))
            cells = {}
            if seed % 3 == 1:
                cells.update({(ox - 2, y): 4 for y in range(oy + 2)})
            if seed % 3 == 2:
                cells.update({(x, oy - 2): 8 for x in range(ox + 2)})
            cells.update({(ox, oy): 1, (ox + 1, oy): 1,
                          (ox, oy + 1): 1, (ox + 1, oy + 1): 2,
                          (ox, oy - 1): 4, (ox - 1, oy - 1): 8})
            for (x, y), tag in cells.items():
                labels[x + n * y] = tag
            return labels
        if scenario in ("barrier", "barrier_hard", "barrier_hard_varied", "barrier_hard_active"):
            # Auxiliary risk stress: the only route to the target crosses a
            # two-cell-wide strip of the authors' v1/v0 safety label/cost.
            hazard = 4 if scenario.startswith("barrier_hard") else 8
            start = 2 + seed % 2 if scenario in ("barrier_hard_varied", "barrier_hard_active") else 2
            target_y = 2 + (seed // 2) % 3 if scenario in ("barrier_hard_varied", "barrier_hard_active") else 2
            cells = {(x, y): hazard for x in (start, start + 1) for y in range(n)}
            cells[(start + 3, target_y)] = 2
            for (x, y), tag in cells.items():
                labels[x + n * y] = tag
            return labels
        if scenario == "near":
            ox, oy = 2, 2
        elif scenario in ("middle", "middle_fast"):
            ox, oy = n // 2, n // 2
        else:
            ox, oy = n - 3, n - 3
        cells = {(ox, oy): 1, (ox + 1, oy): 1, (ox, oy + 1): 1,
                 (ox + 1, oy + 1): 2, (ox, oy - 1): 4,
                 (ox - 1, oy - 1): 8}
    for (x, y), tag in cells.items():
        if 0 <= x < n and 0 <= y < n:
            labels[x + n * y] = tag
    return labels


def build_grid(n: int, scenario: str = "near", seed: int = 0,
               labels_override: np.ndarray | None = None,
               lane_only: bool = False) -> Model:
    """Construct sparse product transitions without the dense n^4 tensor."""
    assert n >= 4 and scenario in ("near", "middle", "middle_fast", "middle_fast_varied", "middle_fast_crossing", "carla_crossing", "far", "barrier", "barrier_hard", "barrier_hard_varied", "barrier_hard_active", "varied", "terrain", "sequence")
    assert scenario not in ("barrier", "barrier_hard", "barrier_hard_varied", "barrier_hard_active") or n >= 8
    assert scenario != "sequence" or n >= 8
    assert scenario not in ("middle_fast_varied", "middle_fast_crossing") or n >= 16
    if scenario == "carla_crossing":
        if labels_override is None or np.shape(labels_override) != (n * n,):
            raise ValueError("carla_crossing requires one map-derived label per grid cell")
        labels = np.asarray(labels_override, dtype=np.int8)
    else:
        labels = _labels(n, scenario, seed)
    physical = n * n
    goal_states = 3 if scenario == "sequence" else 2
    stride = 2 * goal_states * 2
    states = physical * stride + 1  # physical x environment x goal/safety DFAs + loss sink
    sink = states - 1
    xy = np.full((states, 2), -1, dtype=np.int16)
    mode = np.full(states, -1, dtype=np.int8)
    goal = np.zeros(states, dtype=np.int8)
    bad = np.zeros(states, dtype=np.int8)
    signature = np.full(states, -1, dtype=np.int8)
    cost = np.zeros(states)
    for phys in range(physical):
        x, y = phys % n, phys // n
        tag = int(labels[phys])
        for env in range(2):
            for qg in range(goal_states):
                for qb in range(2):
                    s = phys * stride + env * (2 * goal_states) + qg * 2 + qb
                    xy[s] = (x, y)
                    mode[s], goal[s], bad[s] = env, qg, qb
                    signature[s] = tag
                    if qb:
                        cost[s] = ((8 if scenario == "carla_crossing" else 5)
                                   if tag & 1 else 0) + (10 if tag & 4 else 0) + (2 if tag & 8 else 0)
    row, col, data = [], [], []
    env_prob = ((0.8, 0.2), (0.2, 0.8))
    for phys in range(physical):
        x, y = phys % n, phys // n
        motion = (0.10 + 0.125 * ((x // 3 + y // 4 + seed) % 3)
                  if scenario == "terrain" else 0.75 if scenario in ("middle_fast", "middle_fast_varied", "middle_fast_crossing", "carla_crossing") else 0.20)
        for env in range(2):
            for qg in range(goal_states):
                for qb in range(2):
                    s = phys * stride + env * (2 * goal_states) + qg * 2 + qb
                    for a, (raw_dx, raw_dy) in enumerate(ACTIONS):
                        dx, dy = ((raw_dx, raw_dy) if not lane_only or a == 5
                                  else (0, 0))
                        rr = s * 9 + a
                        missing = 1.0
                        for mx, px in ((0, 1 - motion), (dx, motion)) if dx else ((0, 1.0),):
                            for my, py in ((0, 1 - motion), (dy, motion)) if dy else ((0, 1.0),):
                                nx, ny = x + mx, y + my
                                if not (0 <= nx < n and 0 <= ny < n):
                                    continue
                                nxt_phys = nx + n * ny
                                tag = int(labels[nxt_phys])
                                for ne in range(2):
                                    probability = px * py * env_prob[env][ne]
                                    if scenario == "sequence":
                                        next_goal = (1 if qg == 0 and (tag & 16) else
                                                     2 if qg == 1 and (tag & 2) else qg)
                                    else:
                                        next_goal = int(qg or bool(tag & 2))
                                    next_bad = qb or bool(tag & 4) or bool(tag & 8) or bool((tag & 1) and ne == 1)
                                    next_s = (nxt_phys * stride + ne * (2 * goal_states)
                                              + int(next_goal) * 2 + int(next_bad))
                                    row.append(rr)
                                    col.append(next_s)
                                    data.append(probability)
                                    missing -= probability
                        if missing > 1e-12:
                            row.append(rr)
                            col.append(sink)
                            data.append(missing)
    for a in range(9):
        row.append(sink * 9 + a)
        col.append(sink)
        data.append(1.0)
    p = sparse.coo_matrix((data, (row, col)), shape=(states * 9, states)).tocsr()
    p.sum_duplicates()
    p.eliminate_zeros()
    if np.max(np.abs(np.asarray(p.sum(axis=1)).ravel() - 1)) > 1e-12:
        raise AssertionError("Product transition row is not stochastic")
    accept = (goal == goal_states - 1).astype(float)
    reward = (2 * (p @ accept)).reshape(states, 9)
    if scenario.startswith("barrier_hard"):
        # Deliberately stronger goal reward makes the unchanged hard risk cap
        # active. This auxiliary stress case is not an IROS parameter setting.
        reward *= 10
    if scenario in ("barrier_hard_varied", "barrier_hard_active"):
        # Vary risk severity as well as barrier/goal placement; otherwise
        # many layouts have the same constrained reward-to-risk ratio.
        cost *= 1 + 0.2 * seed
    if scenario == "barrier_hard_active":
        # Purpose-built stress: increasing reward makes the hard risk cap
        # active despite the simultaneously stronger hazard cost.
        reward *= 1 + 0.3 * seed
    initial = 2 * goal_states  # physical (0,0), red traffic light, both DFA states initial
    return Model(p, reward, cost, initial, xy, mode, goal, bad, signature, n)


def solve_lp(model: Model, method: str = "highs",
             time_limit_seconds: float | None = None) -> dict:
    """Sparse occupancy LP, equivalent to IROS objective including soft slack."""
    started = time.perf_counter()
    n, a = model.states, model.actions
    eye_cols = np.arange(n * a)
    source_rows = eye_cols // a
    source = sparse.csr_matrix((np.ones(n * a), (source_rows, eye_cols)), shape=(n, n * a))
    flow = source - GAMMA * model.p.T
    risk_coeff = np.repeat(model.cost, a)
    a_eq = sparse.hstack([flow, sparse.csr_matrix((n, 1))], format="csr")
    a_ub = sparse.csr_matrix((np.r_[risk_coeff, -1.0],
                             (np.zeros(n * a + 1), np.arange(n * a + 1))),
                            shape=(1, n * a + 1))
    a_ub = sparse.vstack([a_ub, sparse.hstack([sparse.csr_matrix(risk_coeff.reshape(1, -1)),
                                               sparse.csr_matrix((1, 1))])], format="csr")
    alpha = np.zeros(n)
    alpha[model.initial] = 1
    objective = np.r_[-model.reward.ravel(), 1.0]
    options = {"primal_feasibility_tolerance": 1e-8,
               "dual_feasibility_tolerance": 1e-8}
    if time_limit_seconds is not None:
        options["time_limit"] = time_limit_seconds
    result = linprog(objective, A_ub=a_ub, b_ub=[SOFT, HARD],
                     A_eq=a_eq, b_eq=alpha, bounds=(0, None),
                     method=method, options=options)
    if not result.success:
        raise RuntimeError(f"LP status {result.status}: {result.message}")
    return {"occupation": result.x[:-1].reshape(n, a), "objective": -float(result.fun),
            "dual_v": -np.asarray(result.eqlin.marginals),
            "lambda_soft": -float(result.ineqlin.marginals[0]),
            "lambda_hard": -float(result.ineqlin.marginals[1]),
            "wall_seconds": time.perf_counter() - started}


def initial_partition(model: Model, tile: int) -> np.ndarray:
    keys = []
    for s in range(model.states):
        x, y = model.xy[s]
        if x < 0:
            keys.append((-1, -1, -1, -1, -1, -1))
        else:
            keys.append((int(x // tile), int(y // tile), int(model.mode[s]),
                         int(model.goal[s]), int(model.bad[s]), int(model.label[s])))
    mapping = {}
    return np.array([mapping.setdefault(key, len(mapping)) for key in keys], dtype=np.int32)


def aggregate(model: Model, partition: np.ndarray) -> Model:
    n, a = model.states, model.actions
    blocks = int(partition.max()) + 1
    sizes = np.bincount(partition, minlength=blocks)
    source = np.repeat(partition * a, a) + np.tile(np.arange(a), n)
    col = np.arange(n * a)
    weights = np.repeat(1.0 / sizes[partition], a)
    averaging = sparse.csr_matrix((weights, (source, col)), shape=(blocks * a, n * a))
    projection = sparse.csr_matrix((np.ones(n), (np.arange(n), partition)), shape=(n, blocks))
    p = (averaging @ model.p @ projection).tocsr()
    reward = np.asarray(averaging @ model.reward.ravel()).reshape(blocks, a)
    cost = np.bincount(partition, weights=model.cost, minlength=blocks) / sizes
    xy = np.zeros((blocks, 2), dtype=np.int16)
    mode = np.zeros(blocks, dtype=np.int8)
    goal = np.zeros(blocks, dtype=np.int8)
    bad = np.zeros(blocks, dtype=np.int8)
    label = np.zeros(blocks, dtype=np.int8)
    first = np.full(blocks, -1)
    for s, b in enumerate(partition):
        if first[b] < 0:
            first[b] = s
    xy[:] = model.xy[first]
    mode[:] = model.mode[first]
    goal[:] = model.goal[first]
    bad[:] = model.bad[first]
    label[:] = model.label[first]
    return Model(p, reward, cost, int(partition[model.initial]), xy, mode, goal, bad, label, model.grid)


def lift_policy(occupation: np.ndarray, partition: np.ndarray,
                coarse: Model, coarse_v: np.ndarray, lam: float) -> np.ndarray:
    totals = occupation.sum(axis=1)
    policy = np.zeros_like(occupation)
    occupied = totals > 1e-10
    policy[occupied] = occupation[occupied] / totals[occupied, None]
    if np.any(~occupied):
        q = coarse.reward.ravel() - np.repeat(coarse.cost * lam, coarse.actions)
        q += GAMMA * (coarse.p @ coarse_v)
        q = q.reshape(coarse.states, coarse.actions)
        actions = np.argmax(q[~occupied], axis=1)
        policy[np.flatnonzero(~occupied), actions] = 1.0
    return policy[partition]


def evaluate_policy(model: Model, policy: np.ndarray) -> dict:
    n, a = model.states, model.actions
    weighted = sparse.diags(policy.ravel()) @ model.p
    rows = np.repeat(np.arange(n), a)
    selector = sparse.csr_matrix((np.ones(n * a), (rows, np.arange(n * a))), shape=(n, n * a))
    p_pi = (selector @ weighted).tocsr()
    alpha = np.zeros(n)
    alpha[model.initial] = 1
    occupancy = spsolve(sparse.eye(n, format="csc") - GAMMA * p_pi.T.tocsc(), alpha)
    state_reward = np.sum(policy * model.reward, axis=1)
    risk = float(occupancy @ model.cost)
    reward = float(occupancy @ state_reward)
    return {"risk": risk, "reward": reward,
            "objective": reward - max(0.0, risk - SOFT),
            "occupancy": occupancy,
            "flow_residual": float(np.max(np.abs(occupancy - GAMMA * (p_pi.T @ occupancy) - alpha)))}


def upper_certificate(model: Model, partition: np.ndarray, coarse_lp: dict,
                      cache: dict | None = None) -> dict:
    started = time.perf_counter()
    coarse_v = coarse_lp["dual_v"][partition]
    soft, hard = coarse_lp["lambda_soft"], coarse_lp["lambda_hard"]
    adjusted_reward = model.reward - model.cost[:, None] * (soft + hard)
    # The lifted abstract dual can have a large *global* Bellman residual at
    # states barely visited by the candidate policy. Tighten the certificate
    # by solving the scalarized, unconstrained MDP on the original model.
    # This is a sparse policy-evaluation linear system, not a second CMDP LP.
    # Exact floating-point key: reusing a value function for a nearby but
    # different risk price would invalidate the claimed upper bound.
    key = float(soft + hard)
    hit = cache is not None and key in cache
    if hit:
        v, q = cache[key]
        iterations = 0
    else:
        q = adjusted_reward + GAMMA * (model.p @ coarse_v).reshape(model.states, model.actions)
        greedy = np.argmax(q, axis=1)
        v = coarse_v.copy()
        iterations = 0
        for iterations in range(1, 51):
            selected = model.p[np.arange(model.states) * model.actions + greedy]
            rhs = adjusted_reward[np.arange(model.states), greedy]
            v = spsolve(sparse.eye(model.states, format="csc") - GAMMA * selected.tocsc(), rhs)
            q = adjusted_reward + GAMMA * (model.p @ v).reshape(model.states, model.actions)
            improved = np.argmax(q, axis=1)
            if np.array_equal(improved, greedy):
                break
            greedy = improved
        if cache is not None:
            cache[key] = (v, q)
    residuals = np.max(q - v[:, None], axis=1)
    delta = max(0.0, float(np.max(residuals)))
    upper = float(v[model.initial] + delta / (1 - GAMMA) + SOFT * soft + HARD * hard)
    return {"upper": upper, "max_bellman_residual": delta,
            "residuals": residuals, "dual_v": v, "q": q,
            "iterations": iterations, "cache_hit": hit,
            "seconds": time.perf_counter() - started}


def refine(model: Model, partition: np.ndarray, occupancy: np.ndarray,
           residuals: np.ndarray, q: np.ndarray, policy: np.ndarray,
           risk: float, coarse: Model, strategy: str, fraction: float,
           rng: np.random.Generator) -> np.ndarray:
    if strategy == "split_gain":
        return refine_split_gain(model, partition, occupancy, q, policy, fraction)
    blocks = int(partition.max()) + 1
    sizes = np.bincount(partition, minlength=blocks)
    possible = np.flatnonzero(sizes > 1)
    if len(possible) == 0:
        return partition
    rank = np.zeros(blocks)
    if strategy in ("guided", "guided_optocc"):
        # For fixed risk prices, d_pi(s) [max_a Q_lambda(s,a) - E_pi Q_lambda]
        # is an exact local contribution to scalarized policy regret.
        advantage = np.maximum(0, np.max(q, axis=1) - np.sum(policy * q, axis=1))
        if strategy == "guided_optocc":
            greedy = np.argmax(q, axis=1)
            selected = model.p[np.arange(model.states) * model.actions + greedy]
            alpha = np.zeros(model.states)
            alpha[model.initial] = 1
            weighting = spsolve(sparse.eye(model.states, format="csc")
                                - GAMMA * selected.T.tocsc(), alpha)
        else:
            weighting = occupancy
        rank += np.bincount(partition, weights=weighting * advantage, minlength=blocks)
        if risk > HARD + 1e-7:
            rank += np.bincount(partition, weights=occupancy * model.cost,
                                minlength=blocks)
        # All-zero advantage occurs when the abstract policy is optimal for
        # the current risk price but its price is wrong. Split occupied cells.
        if np.max(rank[possible]) < 1e-12:
            rank += np.bincount(partition, weights=occupancy, minlength=blocks)
    elif strategy == "occupancy":
        rank += np.bincount(partition, weights=occupancy, minlength=blocks)
    elif strategy == "model_error":
        # Task-agnostic approximate-bisimulation-style baseline: split where
        # one-step reward, cost, and projected transition distributions differ
        # most from their aggregate block row. No policy/value/risk price used.
        n, a = model.states, model.actions
        projection = sparse.csr_matrix((np.ones(n), (np.arange(n), partition)),
                                       shape=(n, blocks))
        full_rows = (model.p @ projection).tocsr()
        block_rows = coarse.p[np.repeat(partition, a) * a + np.tile(np.arange(a), n)]
        delta_rows = (full_rows - block_rows).tocsr()
        delta_rows.data = np.abs(delta_rows.data)
        tv = 0.5 * np.asarray(delta_rows.sum(axis=1)).ravel().reshape(n, a)
        reward_error = np.max(np.abs(model.reward - coarse.reward[partition]), axis=1)
        cost_error = np.abs(model.cost - coarse.cost[partition])
        state_error = np.max(tv, axis=1) + 0.1 * reward_error + 0.05 * cost_error
        np.maximum.at(rank, partition, state_error)
    elif strategy == "random":
        rank = rng.random(blocks)
    elif strategy == "uniform":
        rank[:] = 1
    else:
        raise ValueError(strategy)
    count = len(possible) if strategy == "uniform" else max(1, int(np.ceil(fraction * len(possible))))
    chosen = possible[np.argsort(-rank[possible], kind="stable")[:count]]
    selected = set(chosen.tolist())
    min_x = np.full(blocks, np.iinfo(np.int16).max, dtype=np.int16)
    max_x = np.full(blocks, -1, dtype=np.int16)
    min_y = np.full(blocks, np.iinfo(np.int16).max, dtype=np.int16)
    max_y = np.full(blocks, -1, dtype=np.int16)
    np.minimum.at(min_x, partition, model.xy[:, 0])
    np.maximum.at(max_x, partition, model.xy[:, 0])
    np.minimum.at(min_y, partition, model.xy[:, 1])
    np.maximum.at(max_y, partition, model.xy[:, 1])
    mid_x = (min_x.astype(np.int32) + max_x.astype(np.int32) + 1) // 2
    mid_y = (min_y.astype(np.int32) + max_y.astype(np.int32) + 1) // 2
    keys = []
    for s, block in enumerate(partition):
        x, y = model.xy[s]
        if block not in selected or x < 0:
            keys.append((int(block), 0, 0))
        else:
            keys.append((int(block), int(x >= mid_x[block]), int(y >= mid_y[block])))
    mapping = {}
    return np.array([mapping.setdefault(key, len(mapping)) for key in keys], dtype=np.int32)


def refine_split_gain(model: Model, partition: np.ndarray, occupancy: np.ndarray,
                      q: np.ndarray, policy: np.ndarray, fraction: float) -> np.ndarray:
    """Choose the spatial cut with greatest predicted action gain per new state.

    The score is a one-step surrogate, not a certificate. Final certification
    remains the full-model upper/lower evaluation in run_abstraction.
    """
    blocks = int(partition.max()) + 1
    sizes = np.bincount(partition, minlength=blocks)
    possible = np.flatnonzero(sizes > 1)
    if len(possible) == 0:
        return partition
    order = np.argsort(partition, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)]
    candidates = []
    for block in possible:
        states = order[starts[block]:starts[block + 1]]
        xy = model.xy[states]
        if np.any(xy < 0):
            continue
        mid_x = (int(xy[:, 0].min()) + int(xy[:, 0].max()) + 1) // 2
        mid_y = (int(xy[:, 1].min()) + int(xy[:, 1].max()) + 1) // 2
        xbit, ybit = (xy[:, 0] >= mid_x).astype(np.int8), (xy[:, 1] >= mid_y).astype(np.int8)
        weighted_q = occupancy[states, None] * q[states]
        current_value = float(np.sum(weighted_q * policy[states]))
        best = None
        for mode, child in (("x", xbit), ("y", ybit), ("xy", 2 * xbit + ybit)):
            groups = np.unique(child)
            if len(groups) <= 1:
                continue
            gain = sum(float(np.max(weighted_q[child == k].sum(axis=0)))
                       for k in groups) - current_value
            added = len(groups) - 1
            efficiency = max(0.0, gain) / added
            value = (efficiency, gain, -added)
            if best is None or value > best[0]:
                best = (value, mode, mid_x, mid_y)
        if best is not None:
            # Fall back to visits for ties caused by scalarized policy ties.
            efficiency = best[0][0]
            candidates.append((efficiency, float(occupancy[states].sum()),
                               int(block), best[1], best[2], best[3]))
    if not candidates:
        return partition
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    count = max(1, int(np.ceil(fraction * len(possible))))
    selected = {c[2]: c[3:] for c in candidates[:count]}
    keys = []
    for state, block in enumerate(partition):
        cut = selected.get(int(block))
        if cut is None:
            keys.append((int(block), 0))
        else:
            mode, mid_x, mid_y = cut
            x, y = model.xy[state]
            child = (int(x >= mid_x) if mode == "x" else
                     int(y >= mid_y) if mode == "y" else
                     2 * int(x >= mid_x) + int(y >= mid_y))
            keys.append((int(block), child))
    mapping = {}
    return np.array([mapping.setdefault(key, len(mapping)) for key in keys], dtype=np.int32)


def run_abstraction(model: Model, strategy: str, tile: int = 4,
                    tolerance: float = 1e-4, relative_tolerance: float = 0.01,
                    max_rounds: int = 10,
                    fraction: float = 0.08, seed: int = 0) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    partition = initial_partition(model, tile)
    history = []
    cert_cache = {}
    for iteration in range(max_rounds):
        coarse = aggregate(model, partition)
        lp = solve_lp(coarse)
        policy = lift_policy(lp["occupation"], partition, coarse, lp["dual_v"],
                             lp["lambda_soft"] + lp["lambda_hard"])
        exact = evaluate_policy(model, policy)
        cert = upper_certificate(model, partition, lp, cert_cache)
        lam = lp["lambda_soft"] + lp["lambda_hard"]
        advantages = np.maximum(0, np.max(cert["q"], axis=1)
                                - np.sum(policy * cert["q"], axis=1))
        local_regret = float(exact["occupancy"] @ advantages)
        scalarized_regret = float(cert["dual_v"][model.initial]
                                  - (exact["reward"] - lam * exact["risk"]))
        if abs(local_regret - scalarized_regret) > 1e-5:
            raise AssertionError("Dual-weighted regret decomposition failed")
        gap = max(0.0, cert["upper"] - exact["objective"])
        safe = exact["risk"] <= HARD + 1e-7
        target_gap = max(tolerance, relative_tolerance * abs(exact["objective"]))
        certified = safe and gap <= target_gap + 1e-7
        history.append({"round": iteration, "states": coarse.states,
                        "lp_seconds": lp["wall_seconds"], "risk": exact["risk"],
                        "objective": exact["objective"], "upper": cert["upper"],
                        "gap": gap, "bellman_residual": cert["max_bellman_residual"],
                        "target_gap": target_gap,
                        "certificate_seconds": cert["seconds"],
                        "certificate_iterations": cert["iterations"],
                        "certificate_cache_hit": cert["cache_hit"],
                        "local_regret": local_regret,
                        "scalarized_regret": scalarized_regret,
                        "lambda_soft": lp["lambda_soft"],
                        "lambda_hard": lp["lambda_hard"],
                        "flow_residual": exact["flow_residual"], "certified": certified})
        if certified or coarse.states == model.states:
            break
        new_partition = refine(model, partition, exact["occupancy"], cert["residuals"],
                               cert["q"], policy, exact["risk"], coarse,
                               strategy, fraction, rng)
        if np.array_equal(new_partition, partition):
            break
        partition = new_partition
    return {"strategy": strategy, "total_seconds": time.perf_counter() - started,
            "certified": history[-1]["certified"], "final_states": history[-1]["states"],
            "final_risk": history[-1]["risk"], "final_objective": history[-1]["objective"],
            "final_gap": history[-1]["gap"], "history": history}


def run_racing(model: Model, tile: int = 4, fraction: float = 0.04,
               tolerance: float = 1e-4, relative_tolerance: float = 0.01,
               max_rounds: int = 20) -> dict:
    """Evaluate two possible refinements and keep the better certified gain.

    This costs extra candidate solves but avoids committing to a brittle
    single ranking rule. Every candidate is evaluated on the original model.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    cert_cache: dict = {}
    evaluated = 0

    def measure(partition: np.ndarray) -> dict:
        nonlocal evaluated
        coarse = aggregate(model, partition)
        lp = solve_lp(coarse)
        policy = lift_policy(lp["occupation"], partition, coarse, lp["dual_v"],
                             lp["lambda_soft"] + lp["lambda_hard"])
        exact = evaluate_policy(model, policy)
        cert = upper_certificate(model, partition, lp, cert_cache)
        gap = max(0.0, cert["upper"] - exact["objective"])
        target = max(tolerance, relative_tolerance * abs(exact["objective"]))
        safe = exact["risk"] <= HARD + 1e-7
        evaluated += 1
        return {"partition": partition, "coarse": coarse, "policy": policy,
                "exact": exact, "cert": cert, "gap": gap, "target": target,
                "certified": safe and gap <= target + 1e-7,
                "record": {"states": coarse.states, "objective": exact["objective"],
                           "risk": exact["risk"], "gap": gap, "target_gap": target,
                           "certified": safe and gap <= target + 1e-7}}

    current = measure(initial_partition(model, tile))
    history = [current["record"]]
    for _ in range(max_rounds - 1):
        if current["certified"] or current["coarse"].states == model.states:
            break
        candidates = []
        for kind in ("guided", "occupancy"):
            new_partition = refine(model, current["partition"],
                                   current["exact"]["occupancy"],
                                   current["cert"]["residuals"],
                                   current["cert"]["q"], current["policy"],
                                   current["exact"]["risk"], current["coarse"],
                                   kind, fraction, rng)
            if np.array_equal(new_partition, current["partition"]):
                continue
            if any(np.array_equal(new_partition, c["partition"]) for c in candidates):
                continue
            candidate = measure(new_partition)
            candidate["source"] = kind
            candidates.append(candidate)
        if not candidates:
            break
        certified = [c for c in candidates if c["certified"]]
        if certified:
            current = min(certified, key=lambda c: (c["coarse"].states, c["gap"]))
        else:
            old_bad = max(0.0, current["exact"]["risk"] - HARD)
            def efficiency(c: dict) -> float:
                new_bad = max(0.0, c["exact"]["risk"] - HARD)
                improvement = (current["gap"] + 100 * old_bad
                               - c["gap"] - 100 * new_bad)
                added = max(1, c["coarse"].states - current["coarse"].states)
                return improvement / added
            current = max(candidates, key=efficiency)
        current["record"]["chosen_from"] = current["source"]
        history.append(current["record"])
    return {"strategy": "racing", "total_seconds": time.perf_counter() - started,
            "candidate_evaluations": evaluated,
            "certified": current["certified"], "final_states": current["coarse"].states,
            "final_risk": current["exact"]["risk"],
            "final_objective": current["exact"]["objective"],
            "final_gap": current["gap"], "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--scenarios", nargs="+", default=["near", "middle", "far"])
    parser.add_argument("--tile", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--fraction", type=float, default=0.08)
    parser.add_argument("--strategies", nargs="+",
                        default=["guided", "occupancy", "model_error", "uniform", "random"],
                        choices=["guided", "guided_optocc", "split_gain", "racing", "occupancy", "model_error", "uniform", "random"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    output = []
    for n in args.sizes:
        for scenario in args.scenarios:
            if n == 4 and scenario != "near":
                continue
            model = build_grid(n, scenario)
            baseline = solve_lp(model)
            baseline_eval = evaluate_policy(model, lift_policy(
                baseline["occupation"], np.arange(model.states), model,
                baseline["dual_v"], baseline["lambda_soft"] + baseline["lambda_hard"]))
            methods = [(run_racing(model, tile=args.tile, fraction=args.fraction,
                                   tolerance=args.tolerance,
                                   relative_tolerance=args.relative_tolerance,
                                   max_rounds=args.rounds)
                        if name == "racing" else
                        run_abstraction(model, name, tile=args.tile,
                                        tolerance=args.tolerance, max_rounds=args.rounds,
                                        relative_tolerance=args.relative_tolerance,
                                        fraction=args.fraction))
                       for name in args.strategies]
            record = {"grid": n, "scenario": scenario, "states": model.states,
                      "actions": model.actions, "nonzero_transitions": model.p.nnz,
                      "sparse_lp_seconds": baseline["wall_seconds"],
                      "sparse_lp_objective": baseline_eval["objective"],
                      "sparse_lp_risk": baseline_eval["risk"], "methods": methods}
            output.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"n={n} {scenario}: LP={baseline['wall_seconds']:.3f}s "
                  + " ".join(f"{m['strategy']}={m['total_seconds']:.3f}s/"
                             f"{m['final_states']}s/{m['final_gap']:.3g}gap/"
                             f"{'ok' if m['certified'] else 'fail'}" for m in methods), flush=True)


if __name__ == "__main__":
    main()
