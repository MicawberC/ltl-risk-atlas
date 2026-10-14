"""Local updates of the fixed-size CARLA LTL product model.

The physical grid and automaton state sets stay fixed.  Changed propositions
affect transitions *into* their cells; changed environment probabilities affect
transitions *from* their cells.  This module patches exactly those product rows.
It does not claim to support adding/removing physical cells or automaton states.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from experiment import Model, build_grid


BASE_ENV = np.array(((0.8, 0.2), (0.2, 0.8)), dtype=float)
ACTIONS = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


@dataclass
class Snapshot:
    model: Model
    labels: np.ndarray
    environment: np.ndarray  # (physical cell, old mode, new mode)
    severity: np.ndarray  # per physical cell, 1 outside updated conflict cells
    changed_product_rows: int = 0
    lane_only: bool = False


def initial_snapshot(labels: np.ndarray, grid: int,
                     lane_only: bool = False) -> Snapshot:
    labels = np.asarray(labels, dtype=np.int8).copy()
    env = np.broadcast_to(BASE_ENV, (grid * grid, 2, 2)).copy()
    return Snapshot(build_grid(grid, "carla_crossing", labels_override=labels,
                               lane_only=lane_only),
                    labels, env, np.ones(grid * grid), lane_only=lane_only)


def _cost(labels: np.ndarray, severity: np.ndarray, grid: int) -> np.ndarray:
    cost = np.zeros(grid * grid * 8 + 1)
    for phys, tag_raw in enumerate(labels):
        tag = int(tag_raw)
        value = (8.0 * severity[phys] if tag & 1 else 0.0)
        value += 10.0 if tag & 4 else 0.0
        value += 2.0 if tag & 8 else 0.0
        cost[phys * 8 + 1:phys * 8 + 8:2] = value
    return cost


def _row(phys: int, env: int, q_goal: int, q_bad: int, action: int,
         labels: np.ndarray, environment: np.ndarray, grid: int,
         lane_only: bool) -> dict[int, float]:
    x, y = phys % grid, phys // grid
    dx, dy = ACTIONS[action]
    if lane_only and action != 5:
        dx, dy = 0, 0
    x_outcomes = ((0, 0.25), (dx, 0.75)) if dx else ((0, 1.0),)
    y_outcomes = ((0, 0.25), (dy, 0.75)) if dy else ((0, 1.0),)
    row: dict[int, float] = {}
    sink = grid * grid * 8
    missing = 1.0
    for mx, px in x_outcomes:
        for my, py in y_outcomes:
            nx, ny = x + mx, y + my
            if not (0 <= nx < grid and 0 <= ny < grid):
                continue
            nxt_phys = nx + grid * ny
            tag = int(labels[nxt_phys])
            for nxt_env in range(2):
                probability = px * py * environment[phys, env, nxt_env]
                nxt_goal = int(q_goal or bool(tag & 2))
                nxt_bad = int(q_bad or bool(tag & 4) or bool(tag & 8)
                              or bool((tag & 1) and nxt_env == 1))
                col = nxt_phys * 8 + nxt_env * 4 + nxt_goal * 2 + nxt_bad
                row[col] = row.get(col, 0.0) + probability
                missing -= probability
    if missing > 1e-12:
        row[sink] = row.get(sink, 0.0) + missing
    return row


def patch(snapshot: Snapshot, labels: np.ndarray, environment: np.ndarray,
          severity: np.ndarray, initial: int | None = None) -> Snapshot:
    """Patch changed transition rows; update reward/cost and optional start."""
    model = snapshot.model
    grid = model.grid
    labels = np.asarray(labels, dtype=np.int8)
    environment = np.asarray(environment, dtype=float)
    severity = np.asarray(severity, dtype=float)
    assert labels.shape == (grid * grid,)
    assert environment.shape == (grid * grid, 2, 2)
    assert severity.shape == (grid * grid,)
    assert np.allclose(environment.sum(axis=2), 1.0)
    changed_label_cells = np.flatnonzero(labels != snapshot.labels)
    changed_env_cells = set(np.flatnonzero(
        np.max(np.abs(environment - snapshot.environment), axis=(1, 2)) > 1e-12))
    affected = set(changed_env_cells)
    for cell in changed_label_cells:
        x, y = int(cell % grid), int(cell // grid)
        for yy in range(max(0, y - 1), min(grid, y + 2)):
            for xx in range(max(0, x - 1), min(grid, x + 2)):
                affected.add(xx + grid * yy)
    delta_r, delta_c, delta_v = [], [], []
    for phys in sorted(affected):
        for env in range(2):
            for q_goal in range(2):
                for q_bad in range(2):
                    state = phys * 8 + env * 4 + q_goal * 2 + q_bad
                    for action in range(9):
                        rr = state * 9 + action
                        old = model.p.getrow(rr)
                        for col, val in zip(old.indices, old.data):
                            delta_r.append(rr)
                            delta_c.append(int(col))
                            delta_v.append(-float(val))
                        for col, val in _row(phys, env, q_goal, q_bad, action,
                                             labels, environment, grid,
                                             snapshot.lane_only).items():
                            delta_r.append(rr)
                            delta_c.append(col)
                            delta_v.append(val)
    if delta_v:
        change = sparse.coo_matrix((delta_v, (delta_r, delta_c)),
                                   shape=model.p.shape).tocsr()
        transition = (model.p + change).tocsr()
        transition.eliminate_zeros()
    else:
        transition = model.p
    accept = (model.goal == 1).astype(float)
    reward = (2.0 * (transition @ accept)).reshape(model.states, 9)
    signature = np.repeat(labels, 8).astype(np.int8)
    signature = np.r_[signature, -1].astype(np.int8)
    changed = replace(model, p=transition, reward=reward,
                      cost=_cost(labels, severity, grid), label=signature,
                      initial=model.initial if initial is None else int(initial))
    if np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1.0)) > 1e-10:
        raise AssertionError("Patched product has non-stochastic row")
    return Snapshot(changed, labels.copy(), environment.copy(), severity.copy(),
                    len(affected) * 8 * 9, snapshot.lane_only)


def full_rebuild(labels: np.ndarray, environment: np.ndarray,
                 severity: np.ndarray, grid: int, initial: int,
                 lane_only: bool = False) -> Snapshot:
    """Independent full product construction used as a reference baseline."""
    base = initial_snapshot(labels, grid, lane_only=lane_only)
    return patch(base, labels, environment, severity, initial)


def frame_parameters(trace: dict, observation: dict,
                     label_radius_m: float = 2.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit sensor-to-model assumptions for a saved CARLA frame."""
    grid = int(trace["grid_n"])
    labels = np.asarray(trace["labels"], dtype=np.int8).copy()
    environment = np.broadcast_to(BASE_ENV, (grid * grid, 2, 2)).copy()
    severity = np.ones(grid * grid)
    for cell, distance in zip(trace["risk_cells"], observation["risk_distance_m"]):
        phys = int(cell[0] + grid * cell[1])
        multiplier = 0.2 + 1.6 * np.exp(-0.5 * (distance / 3.0) ** 2)
        severity[phys] = multiplier
        if distance > label_radius_m:
            labels[phys] &= ~1
        # A local, assumed two-mode transition response; CARLA does not
        # provide these probabilities.  No global environment rows change.
        closeness = np.exp(-0.5 * (distance / 3.0) ** 2)
        environment[phys, 0] = (0.8 - 0.25 * closeness,
                                0.2 + 0.25 * closeness)
        environment[phys, 1] = (0.2 - 0.10 * closeness,
                                0.8 + 0.10 * closeness)
    return labels, environment, severity
