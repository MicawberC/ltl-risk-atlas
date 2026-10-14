"""Repair a few actions of a cached policy with discounted low-rank updates.

Changing m policy rows changes I-gamma P_pi by rank at most m.  Woodbury
updates the scalarized value and discounted state occupancy using the cached
factorization.  A Bellman residual supplies a valid upper bound even when the
repaired policy is not globally optimal at the tested risk price.
"""

from __future__ import annotations

import time

import numpy as np

from experiment import GAMMA, HARD, SOFT
from local_residual import cached_primal
from stream_certificate import materialize, all_primal_policies


def attempt(atlas, theta, max_changes=4, price=None,
            residual_iterations=16, relative_tolerance=.01, max_bases=2,
            max_candidate_states=None):
    if max_changes < 1:
        raise ValueError("max_changes must be positive")
    started = time.perf_counter()
    model = atlas["model"]
    theta = np.asarray(theta, dtype=float)
    if price is None:
        price = atlas.get("last_lp_price", atlas.get("stream_price", 1.0))
    price = max(0.0, float(price))
    risk_cost = model.cost + theta @ atlas["directions"]
    adjusted = model.reward - price * risk_cost[:, None]
    best_result = None
    bases = sorted(atlas["policies"], key=lambda p: float(
        np.linalg.norm(p["source_theta"] - theta)))[:max_bases]
    for base in bases:
        old_action = base["action"]
        old_value = base["values"][:, 0] - price * (
            base["values"][:, 1] + base["values"][:, 2:] @ theta)
        q = adjusted + GAMMA * (model.p @ old_value).reshape(
            model.states, model.actions)
        new_action = np.argmax(q, axis=1)
        advantage = q[np.arange(model.states), new_action] - old_value
        occupancy = base["occupation"].sum(axis=1)
        score = advantage * (occupancy + 1e-7)
        candidates = np.flatnonzero((new_action != old_action) &
                                    (advantage > 1e-8) & (score > 1e-12))
        if not len(candidates):
            continue
        if (max_candidate_states is not None and
                len(candidates) > max_candidate_states):
            best_result = {"certified": False,
                           "reason": "influence_region_too_large",
                           "candidate_states": len(candidates),
                           "seconds": time.perf_counter() - started}
            continue
        changed = candidates[np.argsort(score[candidates])[-max_changes:]]
        m = len(changed)
        excluded = np.setdiff1d(candidates, changed, assume_unique=True)
        excluded_score = float(np.max(score[excluded])) if len(excluded) else 0.0
        actions = old_action.copy()
        actions[changed] = new_action[changed]
        old_rows = changed * model.actions + old_action[changed]
        new_rows = changed * model.actions + actions[changed]
        perturbation = (-GAMMA * (model.p[new_rows] - model.p[old_rows])).tocsr()
        basis_columns = np.zeros((model.states, m))
        basis_columns[changed, np.arange(m)] = 1.0
        forward = base["lu"].solve(basis_columns)
        small = np.eye(m) + perturbation @ forward
        backward = base["lu"].solve(perturbation.T.toarray(), trans="T")
        new_occupancy = occupancy - backward @ np.linalg.solve(
            small.T, occupancy[changed])
        if np.min(new_occupancy) < -1e-7:
            raise AssertionError("Low-rank occupancy became negative")
        new_occupancy = np.maximum(new_occupancy, 0.0)
        occupation = np.zeros((model.states, model.actions))
        occupation[np.arange(model.states), actions] = new_occupancy
        reward = float(np.sum(new_occupancy * model.reward[
            np.arange(model.states), actions]))
        base_risk = float(new_occupancy @ model.cost)
        slopes = atlas["directions"] @ new_occupancy
        risk = float(base_risk + slopes @ theta)
        atom = {"occupation": occupation, "reward": reward,
                "risk_coeff": np.r_[base_risk, slopes]}
        atlas.setdefault("primal_atoms", []).append(atom)
        primal = cached_primal(atlas, theta)
        if primal is None:
            atlas["primal_atoms"].pop()
            continue
        objective, certified_risk, _, _, _ = primal
        tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
        dominated = [upper for old_theta, upper in
                     atlas.get("dominance_archive", ())
                     if np.all(old_theta <= theta + 1e-12)]
        upper = min(dominated, default=float("inf"))
        gap = upper - objective
        bound_source = "reused_monotone_upper"
        largest = None
        if gap > tolerance or gap < -1e-6:
            reward_change = (model.reward[changed, actions[changed]] -
                             model.reward[changed, old_action[changed]])
            rhs_value = old_value + forward @ reward_change
            new_value = rhs_value - forward @ np.linalg.solve(
                small, perturbation @ rhs_value)
            new_q = adjusted + GAMMA * (model.p @ new_value).reshape(
                model.states, model.actions)
            residual = np.maximum(0.0, np.max(
                new_q - new_value[:, None], axis=1))
            largest = float(np.max(residual))
            correction = np.full(model.states, largest / (1.0 - GAMMA))
            for _ in range(residual_iterations):
                correction = residual + GAMMA * (
                    model.p @ correction).reshape(model.states,
                                                   model.actions).max(axis=1)
            residual_upper = float(new_value[model.initial] +
                                   correction[model.initial] +
                                   SOFT * min(1.0, price) +
                                   HARD * max(0.0, price - 1.0) + 1e-7)
            if residual_upper < upper:
                upper = residual_upper
                bound_source = "repaired_bellman_upper"
            gap = upper - objective
        certified = bool(certified_risk <= HARD + 1e-8 and
                         -1e-6 <= gap <= tolerance)
        info = {"certified": certified, "objective": objective,
                "risk": certified_risk, "upper": upper, "gap": gap,
                "certificate_type": "low_rank_local_repair",
                "bound_source": bound_source,
                "changed_states": m,
                "candidate_states": len(candidates),
                "largest_excluded_score": excluded_score,
                "changed_state_indices": changed.tolist(),
                "changed_coordinates": model.xy[changed].tolist(),
                "max_bellman_residual": largest,
                "seconds": time.perf_counter() - started}
        if certified:
            atlas.setdefault("dominance_archive", []).append((theta.copy(), upper))
            info["policy"] = materialize(all_primal_policies(atlas), primal)
            return info
        # A failed local candidate is not reusable evidence: keeping many such
        # atoms makes every later pairwise primal search quadratic in misses.
        atlas["primal_atoms"].pop()
        if best_result is None or gap < best_result["gap"]:
            best_result = info
    return best_result or {"certified": False, "reason": "no_local_change",
                           "seconds": time.perf_counter() - started}
