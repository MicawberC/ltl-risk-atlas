"""Certify local risk updates with monotone upper bounds and cached primals.

For nonnegative cost directions, componentwise increases in risk parameters
cannot improve the constrained optimum.  A previous certified upper bound
therefore remains valid.  The lower bound comes from a feasible cached
occupation measure, including optional LP anchors.  The previous Bellman
dual region and a full atlas scan provide additional certificate paths.
"""

from __future__ import annotations

import time

import numpy as np

from experiment import GAMMA, HARD
from joint_price_atlas import (dual_bound, price_interval,
                               query as full_query)
from local_residual import cached_primal, residual_upper


def materialize(policies, primal):
    _, _, i, j, weight = primal
    occ = weight * policies[i]["occupation"] + (
        1.0 - weight) * policies[j]["occupation"]
    totals = occ.sum(axis=1)
    policy = np.divide(occ, totals[:, None], out=np.zeros_like(occ),
                       where=totals[:, None] > 1e-12)
    policy[totals <= 1e-12, 0] = 1.0
    return policy


def remember(atlas, theta, answer):
    """Store a valid upper bound for later componentwise risk increases."""
    if not answer["certified"] or answer.get("certificate_type") in (
            "monotone_dominance", "directional_reach"):
        return answer
    atlas.setdefault("dominance_archive", []).append((
        theta.copy(), float(answer["upper"])))
    if "reach_upper" in atlas and answer.get("price") is not None:
        atlas.setdefault("directional_archive", []).append((
            theta.copy(), float(answer["upper"]), float(answer["price"])))
    return answer


def prepare_reach_upper(atlas, iterations=60):
    """Upper-bound each risk direction's discounted occupancy from the start.

    Value iteration starts from a global Bellman supersolution and descends,
    so every finite iteration is a valid upper bound (up to floating error).
    """
    started = time.perf_counter()
    model = atlas["model"]
    directions = np.asarray(atlas["directions"])
    if np.min(directions) < -1e-12:
        raise ValueError("Directional sensitivity requires nonnegative costs")
    immediate = directions.T
    value = np.broadcast_to(
        np.max(immediate, axis=0) / (1.0 - GAMMA), immediate.shape).copy()
    for _ in range(iterations):
        value = immediate + GAMMA * (
            model.p @ value).reshape(model.states, model.actions,
                                      len(directions)).max(axis=1)
    atlas["reach_upper"] = value[model.initial].copy() + 1e-8
    return time.perf_counter() - started


def add_lp_anchor(atlas, theta, occupation, objective, risk, upper=1e-5,
                  price=None):
    """Cache an exact LP primal and its valid (numerically padded) upper bound.

    The fixed-model occupation remains flow feasible at every future risk map.
    Only its risk and soft penalty change with theta.
    """
    model = atlas["model"]
    directions = atlas["directions"]
    occupation = np.asarray(occupation, dtype=float)
    state_occupation = occupation.sum(axis=1)
    atom = {"occupation": occupation.copy(),
            "reward": float(np.sum(occupation * model.reward)),
            "risk_coeff": np.r_[state_occupation @ model.cost,
                                 directions @ state_occupation]}
    atlas.setdefault("primal_atoms", []).append(atom)
    atlas.setdefault("dominance_archive", []).append((
        np.asarray(theta, dtype=float).copy(), float(objective + upper)))
    if "reach_upper" in atlas and price is not None:
        if price < -1e-8:
            raise AssertionError("LP risk dual price must be nonnegative")
        atlas.setdefault("directional_archive", []).append((
            np.asarray(theta, dtype=float).copy(), float(objective + upper),
            max(0.0, float(price))))
    calculated_risk = float(atom["risk_coeff"][0] +
                            atom["risk_coeff"][1:] @ theta)
    if abs(calculated_risk - risk) > 1e-6:
        raise AssertionError("LP anchor risk does not match its occupation")
    return atom


def all_primal_policies(atlas):
    return atlas["policies"] + atlas.get("primal_atoms", [])


def query(atlas, theta, nearest_first=8, relative_tolerance=.01,
          use_residual=False):
    started = time.perf_counter()
    theta = np.asarray(theta, dtype=float)
    archive = atlas.get("dominance_archive", ())
    if (archive or atlas.get("directional_archive")) and np.min(atlas["directions"]) >= -1e-12:
        primal = cached_primal(atlas, theta)
        if primal is not None:
            objective, risk, _, _, _ = primal
            bounds = [(upper, "monotone_dominance") for anchor, upper in archive
                      if np.all(anchor <= theta + 1e-12)]
            reach = atlas.get("reach_upper")
            if reach is not None:
                bounds.extend((upper + price * float(
                    reach @ np.maximum(anchor - theta, 0.0)),
                    "directional_reach")
                    for anchor, upper, price in atlas.get("directional_archive", ()))
            if bounds:
                upper, certificate_type = min(bounds)
                gap = upper - objective
                tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
                if risk <= HARD + 1e-8 and -1e-6 <= gap <= tolerance:
                    answer = {"certified": True, "objective": objective,
                              "risk": risk, "upper": upper, "gap": gap,
                              "policy": materialize(all_primal_policies(atlas), primal),
                              "certificate_type": certificate_type,
                              "seconds": time.perf_counter() - started}
                    return remember(atlas, theta, answer)
    dual_index = atlas.get("stream_dual_index")
    if dual_index is not None and dual_index < len(atlas["policies"]):
        dual_policy = atlas["policies"][dual_index]
        interval = price_interval(dual_policy, theta)
        primal = cached_primal(atlas, theta)
        if interval is not None:
            lo, hi = interval
            candidate_prices = [lo]
            if np.isfinite(hi):
                candidate_prices.append(hi)
            if lo <= 1.0 <= hi:
                candidate_prices.append(1.0)
            prior = atlas.get("stream_price")
            if prior is not None and lo <= prior <= hi:
                candidate_prices.append(prior)
            bound, price = min((dual_bound(dual_policy, theta, t), t)
                               for t in candidate_prices)
            if primal is not None:
                objective, risk, i, j, weight = primal
                gap = bound - objective
                tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
                if risk <= HARD + 1e-8 and -1e-6 <= gap <= tolerance:
                    atlas["stream_price"] = price
                    answer = {"certified": True, "objective": objective,
                              "risk": risk, "upper": bound, "gap": gap,
                              "price": price, "dual_index": dual_index,
                              "policy": materialize(all_primal_policies(atlas), primal),
                              "certificate_type": "stream_region",
                              "seconds": time.perf_counter() - started}
                    return remember(atlas, theta, answer)
        if use_residual and primal is not None and atlas.get("stream_price") is not None:
            objective, risk, _, _, _ = primal
            tolerance = max(1e-4, relative_tolerance * abs(objective)) + 1e-7
            bound, _, _ = residual_upper(
                atlas, theta, dual_policy, atlas["stream_price"],
                max_iterations=25, target_upper=objective + tolerance)
            gap = bound - objective
            if risk <= HARD + 1e-8 and -1e-6 <= gap <= tolerance:
                answer = {"certified": True, "objective": objective,
                          "risk": risk, "upper": bound, "gap": gap,
                          "price": atlas["stream_price"],
                          "dual_index": dual_index,
                          "policy": materialize(all_primal_policies(atlas), primal),
                          "certificate_type": "stream_residual",
                          "seconds": time.perf_counter() - started}
                return remember(atlas, theta, answer)
    result = full_query(atlas, theta, nearest_first=nearest_first)
    if result["certified"]:
        atlas["stream_dual_index"] = result["dual_index"]
        atlas["stream_price"] = result["price"]
        result["certificate_type"] = "full_atlas_scan"
    result["seconds"] = time.perf_counter() - started
    return remember(atlas, theta, result)
