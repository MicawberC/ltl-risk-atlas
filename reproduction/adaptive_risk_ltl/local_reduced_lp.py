"""Exact restricted CMDP repair using a small occupation-measure LP.

Outside a selected set of product states, the cached deterministic policy is
fixed.  Eliminating the global flow equations with its cached LU leaves one
variable for each allowed alternative action.  The hard/soft risk constraints
and nonnegative base-action occupations form a small LP.  The resulting
occupation is checked against a global upper certificate; on a miss the
caller may use its ordinary full LP fallback.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog

from experiment import GAMMA, HARD, SOFT
from local_residual import cached_primal
from stream_certificate import all_primal_policies, materialize


def attempt(atlas, theta, max_changes=4, residual_iterations=16,
            relative_tolerance=.01):
    if max_changes < 1:
        raise ValueError("max_changes must be positive")
    started = time.perf_counter()
    model = atlas["model"]
    theta = np.asarray(theta, dtype=float)
    cost = model.cost + theta @ atlas["directions"]
    selection_price = max(0.0, float(atlas.get(
        "last_lp_price", atlas.get("stream_price", 1.0))))
    best = None
    bases = sorted(atlas["policies"], key=lambda p: float(
        np.linalg.norm(p["source_theta"] - theta)))[:2]
    for base in bases:
        old_action = base["action"]
        old_occupation = base["occupation"].sum(axis=1)
        value = base["values"][:, 0] - selection_price * (
            base["values"][:, 1] + base["values"][:, 2:] @ theta)
        q = model.reward - selection_price * cost[:, None] + GAMMA * (
            model.p @ value).reshape(model.states, model.actions)
        alternate = np.argmax(q, axis=1)
        advantage = q[np.arange(model.states), alternate] - value
        score = advantage * (old_occupation + 1e-7)
        candidates = np.flatnonzero((alternate != old_action) &
                                    (advantage > 1e-8) & (score > 1e-12))
        if not len(candidates):
            continue
        changed = candidates[np.argsort(score[candidates])[-max_changes:]]
        m = len(changed)
        old_rows = changed * model.actions + old_action[changed]
        new_rows = changed * model.actions + alternate[changed]
        transition_delta = (model.p[new_rows] - model.p[old_rows]).T
        sensitivity = base["lu"].solve(
            (GAMMA * transition_delta).toarray(), trans="T")
        base_reward = float(old_occupation @ model.reward[
            np.arange(model.states), old_action])
        reward_delta = (model.reward[changed, alternate[changed]] -
                        model.reward[changed, old_action[changed]])
        reward_coeff = (model.reward[
            np.arange(model.states), old_action] @ sensitivity + reward_delta)
        risk_coeff = cost @ sensitivity
        base_risk = float(cost @ old_occupation)
        local_capacity = np.eye(m) - sensitivity[changed, :]
        a_ub = np.zeros((m + 2, m + 1))
        a_ub[:m, :m] = local_capacity
        a_ub[m, :m] = risk_coeff
        a_ub[m + 1, :m] = risk_coeff
        a_ub[m + 1, m] = -1.0
        b_ub = np.r_[old_occupation[changed], HARD - base_risk,
                     SOFT - base_risk]
        solved = linprog(np.r_[-reward_coeff, 1.0], A_ub=a_ub, b_ub=b_ub,
                         bounds=(0, None), method="highs")
        if not solved.success:
            continue
        local_occupation = solved.x[:m]
        if float(np.max(local_occupation)) < 1e-9:
            continue
        state_occupation = old_occupation + sensitivity @ local_occupation
        if np.min(state_occupation) < -1e-6 or np.min(
                state_occupation[changed] - local_occupation) < -1e-6:
            raise AssertionError("Reduced flow solution has negative occupation")
        occupation = np.zeros((model.states, model.actions))
        occupation[np.arange(model.states), old_action] = state_occupation
        occupation[changed, old_action[changed]] -= local_occupation
        occupation[changed, alternate[changed]] += local_occupation
        if np.min(occupation) < -1e-6:
            raise AssertionError("Reduced LP occupation is negative")
        occupation = np.maximum(occupation, 0.0)
        reward = float(np.sum(occupation * model.reward))
        risk_base = float(state_occupation @ model.cost)
        slopes = atlas["directions"] @ state_occupation
        atom = {"occupation": occupation, "reward": reward,
                "risk_coeff": np.r_[risk_base, slopes]}
        atlas.setdefault("primal_atoms", []).append(atom)
        primal = cached_primal(atlas, theta)
        if primal is None:
            atlas["primal_atoms"].pop()
            continue
        objective, risk, _, _, _ = primal
        tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
        dominated = [upper for old_theta, upper in
                     atlas.get("dominance_archive", ())
                     if np.all(old_theta <= theta + 1e-12)]
        upper = min(dominated, default=float("inf"))
        bound_source = "reused_monotone_upper"
        residual = None
        if upper - objective > tolerance or upper - objective < -1e-6:
            # Any nonnegative scalar risk price gives a valid global Bellman
            # bound; the action mixture changes at most m rows.
            price = selection_price
            old_value = base["values"][:, 0] - price * (
                base["values"][:, 1] + base["values"][:, 2:] @ theta)
            mixing = local_occupation / np.maximum(
                state_occupation[changed], 1e-15)
            perturbation = (-GAMMA * transition_delta.T.multiply(
                mixing[:, None])).tocsr()
            columns = np.zeros((model.states, m))
            columns[changed, np.arange(m)] = 1.0
            forward = base["lu"].solve(columns)
            small = np.eye(m) + perturbation @ forward
            rhs_value = old_value + forward @ (mixing * reward_delta)
            new_value = rhs_value - forward @ np.linalg.solve(
                small, perturbation @ rhs_value)
            q_new = (model.reward - price * cost[:, None] + GAMMA *
                     (model.p @ new_value).reshape(model.states, model.actions))
            residual_vector = np.maximum(
                0.0, np.max(q_new - new_value[:, None], axis=1))
            residual = float(np.max(residual_vector))
            correction = np.full(model.states, residual / (1.0 - GAMMA))
            for _ in range(residual_iterations):
                correction = residual_vector + GAMMA * (
                    model.p @ correction).reshape(
                        model.states, model.actions).max(axis=1)
            residual_upper = float(
                new_value[model.initial] + correction[model.initial] +
                SOFT * min(1.0, price) +
                HARD * max(0.0, price - 1.0) + 1e-7)
            if residual_upper < upper:
                upper = residual_upper
                bound_source = "repaired_bellman_upper"
        gap = upper - objective
        certified = bool(risk <= HARD + 1e-8 and
                         -1e-6 <= gap <= tolerance)
        info = {"certified": certified, "objective": objective,
                "risk": risk, "upper": upper, "gap": gap,
                "certificate_type": "reduced_lp_local_repair",
                "bound_source": bound_source,
                "changed_states": int(np.sum(local_occupation > 1e-9)),
                "changed_state_indices": changed.tolist(),
                "changed_coordinates": model.xy[changed].tolist(),
                "max_bellman_residual": residual,
                "reduced_lp_variables": m + 1,
                "seconds": time.perf_counter() - started}
        if certified:
            atlas.setdefault("dominance_archive", []).append((theta.copy(), upper))
            info["policy"] = materialize(all_primal_policies(atlas), primal)
            return info
        atlas["primal_atoms"].pop()
        if best is None or gap < best["gap"]:
            best = info
    return best or {"certified": False, "reason": "no_local_change",
                    "seconds": time.perf_counter() - started}
