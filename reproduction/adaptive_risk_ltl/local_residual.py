"""Sound local Bellman-residual upper bounds for cached atlas policies.

For scalarized reward r-t*c, any cached policy value V has residual
delta(s)=max_a[r_t(s,a)+gamma P_a V-V(s)]_+.  If U is a supersolution of
U >= delta+gamma max_a P_a U, then V+U is a Bellman supersolution and
V*(initial) <= V(initial)+U(initial).  This remains true when the cached
policy is no longer optimal at the updated risk map.
"""

from __future__ import annotations

import time

import numpy as np

from experiment import GAMMA, HARD, SOFT
from joint_price_atlas import query as atlas_query


def cached_primal(atlas, theta):
    """Best feasible single cached policy or two-policy risk-bound mixture."""
    policies = atlas["policies"] + atlas.get("primal_atoms", [])
    risks = [float(p["risk_coeff"][0] + p["risk_coeff"][1:] @ theta)
             for p in policies]
    best = None
    for i, p in enumerate(policies):
        if risks[i] <= HARD + 1e-9:
            value = p["reward"] - max(0.0, risks[i] - SOFT)
            if best is None or value > best[0]:
                best = (value, risks[i], i, i, 1.0)
        for j in range(i):
            span = risks[i] - risks[j]
            if abs(span) < 1e-12:
                continue
            for target in (SOFT, HARD):
                w = (target - risks[j]) / span
                if -1e-9 <= w <= 1 + 1e-9:
                    w = float(np.clip(w, 0.0, 1.0))
                    reward = w * p["reward"] + (1.0 - w) * policies[j]["reward"]
                    value = reward - max(0.0, target - SOFT)
                    if best is None or value > best[0]:
                        best = (value, target, i, j, w)
    return best


def residual_upper(atlas, theta, policy, price, max_iterations=80,
                   target_upper=None):
    """Monotone finite-iteration upper bound; every returned value is valid."""
    model = atlas["model"]
    slope = policy["slopes"][:, :, 0] + np.tensordot(
        policy["slopes"][:, :, 1:], theta, axes=([2], [0]))
    bellman_gap = policy["intercept"] + price * slope
    delta = np.maximum(0.0, -np.min(bellman_gap, axis=1))
    maximum = float(np.max(delta))
    risk_value = float(policy["cost_values"][0] +
                       policy["cost_values"][1:] @ theta)
    scalarized_value = policy["reward_value"] - price * risk_value
    constant = (scalarized_value + SOFT * min(1.0, price) +
                HARD * max(0.0, price - 1.0))
    if maximum < 1e-11:
        return constant + 1e-8, 0, maximum
    upper_correction = np.full(model.states, maximum / (1.0 - GAMMA))
    for iteration in range(1, max_iterations + 1):
        upper_correction = delta + GAMMA * (
            model.p @ upper_correction).reshape(model.states,
                                                   model.actions).max(axis=1)
        bound = float(constant + upper_correction[model.initial] + 1e-8)
        if target_upper is not None and bound <= target_upper:
            return bound, iteration, maximum
    return bound, max_iterations, maximum


def query(atlas, theta, max_policies=4, max_iterations=80,
          relative_tolerance=.01, nearest_first=None):
    """Try the ordinary atlas certificate, then a residual certificate."""
    started = time.perf_counter()
    theta = np.asarray(theta, dtype=float)
    baseline = atlas_query(atlas, theta, nearest_first=8)
    if baseline["certified"]:
        baseline["certificate_type"] = "exact_region"
        baseline["seconds"] = time.perf_counter() - started
        return baseline
    primal = cached_primal(atlas, theta)
    if primal is None:
        return {"certified": False, "reason": "no_cached_primal",
                "seconds": time.perf_counter() - started}
    objective, risk, i, j, weight = primal
    tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
    target = objective + tolerance
    best_upper = baseline.get("upper", float("inf"))
    best_info = None
    policies = atlas["policies"]
    if best_upper > target:
        order = sorted(range(len(policies)), key=lambda index: float(
            np.linalg.norm(policies[index]["source_theta"] - theta)))
        for index in order[:max_policies]:
            policy = policies[index]
            for price in sorted(set((0.0, 1.0, float(policy["source_price"])))):
                bound, iterations, residual = residual_upper(
                    atlas, theta, policy, price, max_iterations,
                    target_upper=min(best_upper, target))
                if bound < best_upper:
                    best_upper = bound
                    best_info = (index, price, iterations, residual)
                if best_upper <= target:
                    break
            if best_upper <= target:
                break
    certified = bool(risk <= HARD + 1e-8 and
                     -1e-6 <= best_upper - objective <= tolerance)
    answer = {"certified": certified, "objective": objective,
              "risk": risk, "upper": best_upper,
              "gap": best_upper - objective,
              "certificate_type": "local_residual" if certified else "none",
              "residual_info": best_info,
              "seconds": time.perf_counter() - started}
    if certified:
        occ = weight * policies[i]["occupation"] + (
            1.0 - weight) * policies[j]["occupation"]
        total = occ.sum(axis=1)
        action_probability = np.divide(
            occ, total[:, None], out=np.zeros_like(occ),
            where=total[:, None] > 1e-12)
        action_probability[total <= 1e-12, 0] = 1.0
        answer["policy"] = action_probability
    return answer
