"""Exploratory joint risk-price / risk-map policy atlas for a fixed product MDP.

For affine risk c(theta), use augmented coordinates z=(t,t*theta). Bellman
optimality of a fixed deterministic policy is a polyhedron in z. Queries
intersect those regions with a ray for their theta, then combine cached
discounted occupations and test a primal-dual certificate.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from experiment import GAMMA, HARD, SOFT, build_grid, evaluate_policy
from stratified import optimal_price_interval, solve_cmdp, solve_scalarized

HERE = Path(__file__).resolve().parent
BELLMAN_TOLERANCE = 1e-8


def risk_directions(model):
    """Split positive-risk physical locations into two spatial zones."""
    mask = model.cost > 0
    x = model.xy[:, 0]
    y = model.xy[:, 1]
    first = mask & (x <= model.grid // 2)
    if not np.any(first) or not np.any(mask & ~first):
        first = mask & (y <= model.grid // 2)
    if not np.any(first) or not np.any(mask & ~first):
        locations = np.unique(model.xy[mask], axis=0)
        if len(locations) < 2:
            raise ValueError("Two-zone test requires two risk locations")
        axis = 0 if len(np.unique(locations[:, 0])) > 1 else 1
        ordered = np.sort(np.unique(locations[:, axis]))
        cutoff = float((ordered[(len(ordered) - 1) // 2] +
                        ordered[len(ordered) // 2]) / 2)
        first = mask & ((x if axis == 0 else y) <= cutoff)
    assert np.any(first) and np.any(mask & ~first), "Need two populated risk zones"
    return np.stack((np.where(first, model.cost, 0),
                     np.where(mask & ~first, model.cost, 0)))


def add_policy(atlas, result, source_theta=None):
    model = atlas["model"]
    n, a = model.states, model.actions
    action = np.argmax(result["policy"], axis=1)
    key = action.tobytes()
    if key in atlas["keys"]:
        return False
    chosen = model.p[np.arange(n) * a + action].tocsc()
    system = sparse.eye(n, format="csc") - GAMMA * chosen
    lu = splu(system)
    rhs = np.column_stack((model.reward[np.arange(n), action],
                           model.cost, *atlas["directions"]))
    values = lu.solve(rhs)
    q_values = (GAMMA * (model.p @ values)).reshape(n, a, rhs.shape[1])
    intercept = values[:, 0, None] - model.reward - q_values[:, :, 0]
    slopes = np.empty((n, a, rhs.shape[1] - 1))
    for j in range(1, rhs.shape[1]):
        slopes[:, :, j - 1] = (
            model.cost[:, None] if j == 1 else atlas["directions"][j - 2, :, None]
        ) + q_values[:, :, j] - values[:, j, None]
    # Absolute residual; no assumption that an old policy remains optimal.
    if source_theta is None:
        source_theta = np.zeros(len(atlas["directions"]))
    gap_at_source = intercept + result["price"] * (
        slopes[:, :, 0] + np.tensordot(slopes[:, :, 1:], source_theta,
                                       axes=([2], [0])))
    assert np.min(gap_at_source) >= -1e-7
    occ = result["occupation"]
    atlas["policies"].append({
        "intercept": intercept, "slopes": slopes,
        "lu": lu, "action": action, "values": values,
        "source_theta": np.asarray(source_theta).copy(),
        "reward_value": float(values[model.initial, 0]),
        "cost_values": values[model.initial, 1:],
        "occupation": occ, "reward": result["reward"],
        "risk_coeff": np.r_[np.sum(occ.sum(axis=1) * model.cost),
                            atlas["directions"] @ occ.sum(axis=1)],
        "source_price": result["price"],
    })
    atlas["keys"].add(key)
    return True


def rebase_initial(atlas, initial):
    """Update all cached occupations for a new initial product state, fixed P."""
    started = time.perf_counter()
    model = atlas["model"]
    if initial == model.initial:
        return 0.0
    atlas["model"] = replace(model, initial=initial, layers=model.layers)
    alpha = np.zeros(model.states)
    alpha[initial] = 1.0
    for p in atlas["policies"]:
        state_occ = p["lu"].solve(alpha, trans="T")
        occupation = np.zeros((model.states, model.actions))
        occupation[np.arange(model.states), p["action"]] = state_occ
        p["occupation"] = occupation
        p["reward"] = float(p["values"][initial, 0])
        p["reward_value"] = p["reward"]
        p["cost_values"] = p["values"][initial, 1:]
        p["risk_coeff"] = p["cost_values"].copy()
    return time.perf_counter() - started


def add_dual_neighbor_policies(atlas, theta, price, epsilon=1e-6):
    """Cache deterministic Bellman optima on both sides of an LP risk price.

    At a constrained optimum the LP dual risk price is a scalarization
    breakpoint.  Its two neighboring deterministic policies can supply the
    primal mixture and Bellman dual regions for nearby risk parameters.
    """
    theta = np.asarray(theta)
    model = atlas["model"]
    changed = replace(model, cost=model.cost + theta @ atlas["directions"],
                      layers=model.layers)
    delta = max(epsilon, epsilon * abs(price))
    added = 0
    for candidate in (max(0.0, price - delta), price + delta):
        result = solve_scalarized(changed, candidate)
        added += int(add_policy(atlas, result, theta))
    return added


def prepare_empty(model, directions):
    """Defer the first policy solve until an observed risk map needs one."""
    return {"model": model, "directions": np.asarray(directions),
            "source_cost": model.cost.copy(), "keys": set(),
            "policies": [], "base": None, "prepare_seconds": 0.0}


def expand(atlas, theta, relative_tolerance=.001,
           reuse_oracles=True):
    """Pay one complete CMDP solve at an uncovered risk-map point."""
    model = atlas["model"]
    theta = np.asarray(theta)
    changed = replace(model, cost=model.cost + theta @ atlas["directions"],
                      layers=model.layers)
    started = time.perf_counter()
    fresh = solve_cmdp(changed, search_mode="breakpoint",
                       relative_tolerance=relative_tolerance,
                       return_oracles=reuse_oracles)
    added = 0
    for price in fresh["prices"]:
        result = (fresh["oracle_results"][price] if reuse_oracles
                  else solve_scalarized(changed, price))
        added += add_policy(atlas, result, theta)
    fresh.pop("oracle_results", None)
    return {"fresh": fresh, "added": added,
            "seconds": time.perf_counter() - started}


def expand_fast(atlas, theta, nearest_first=8,
                relative_tolerance=.001, certifier=None,
                reuse_oracles=True):
    """Try one nearby scalarized price before a full CMDP expansion."""
    started = time.perf_counter()
    theta = np.asarray(theta)
    model = atlas["model"]
    changed = replace(model, cost=model.cost + theta @ atlas["directions"],
                      layers=model.layers)
    neighbors = sorted(atlas["policies"],
                       key=lambda p: float(np.linalg.norm(
                           p["source_theta"] - theta)))
    # A positive price is more informative than a zero-price reward policy.
    candidate = next((p["source_price"] for p in neighbors
                      if p["source_price"] > 1e-10), 1.0)
    result = solve_scalarized(changed, candidate)
    added = int(add_policy(atlas, result, theta))
    if certifier is None:
        trial = query(atlas, theta, materialize=False,
                      nearest_first=nearest_first)
    else:
        trial = certifier(atlas, theta, nearest_first=nearest_first)
    if trial["certified"]:
        return {"fallback": False, "added": added,
                "seconds": time.perf_counter() - started}
    fallback = expand(atlas, theta, relative_tolerance,
                      reuse_oracles=reuse_oracles)
    return {"fallback": True, "added": added + fallback["added"],
            "seconds": time.perf_counter() - started}


def expand_adaptive(atlas, theta, nearest_first=8, max_oracles=6,
                    relative_tolerance=.001):
    """Follow exact Bellman price breakpoints before a full CMDP solve."""
    started = time.perf_counter()
    theta = np.asarray(theta)
    model = atlas["model"]
    changed = replace(model, cost=model.cost + theta @ atlas["directions"],
                      layers=model.layers)
    neighbors = sorted(atlas["policies"],
                       key=lambda p: float(np.linalg.norm(
                           p["source_theta"] - theta)))
    price = next((p["source_price"] for p in neighbors
                  if p["source_price"] > 1e-10), 1.0)
    current = query(atlas, theta, materialize=False,
                    nearest_first=nearest_first)
    if current["certified"]:
        return {"fallback": False, "added": 0, "oracles": 0,
                "seconds": time.perf_counter() - started}
    if current.get("price", 0.0) > 1e-10:
        price = current["price"]
    seed_action = neighbors[0]["action"]
    seen = set()
    added = 0
    count = 0
    for _ in range(max_oracles):
        price = max(0.0, float(price))
        key = round(price, 12)
        if key in seen:
            break
        seen.add(key)
        solved = solve_scalarized(changed, price,
                                  initial_action=seed_action)
        seed_action = np.argmax(solved["policy"], axis=1)
        count += 1
        added += int(add_policy(atlas, solved, theta))
        trial = query(atlas, theta, materialize=False,
                      nearest_first=nearest_first)
        if trial["certified"]:
            return {"fallback": False, "added": added,
                    "oracles": count,
                    "seconds": time.perf_counter() - started}
        low, high = optimal_price_interval(changed, solved)
        risk = solved["risk"]
        if risk > HARD + 1e-8:
            edge = high
            price = edge + max(1e-5, 1e-4 * max(1.0, edge)) if np.isfinite(edge) else price * 2
        elif risk < SOFT - 1e-8:
            if price > 1.0 and low > 1.0:
                price = 1.0
            else:
                price = max(0.0, low - max(1e-5, 1e-4 * max(1.0, low)))
        elif price > 1.0 + 1e-8:
            price = max(1.0, low - max(1e-5, 1e-4 * max(1.0, low)))
        elif price < 1.0 - 1e-8:
            price = 1.0
        else:
            break
    fallback = expand(atlas, theta, relative_tolerance)
    return {"fallback": True, "added": added + fallback["added"],
            "oracles": count,
            "seconds": time.perf_counter() - started}


def prepare(model, directions, relative_tolerance=.001,
            reuse_oracles=True, observed_theta=None):
    started = time.perf_counter()
    directions = np.asarray(directions)
    theta = (np.zeros(len(directions)) if observed_theta is None else
             np.asarray(observed_theta, dtype=float))
    if theta.shape != (len(directions),):
        raise ValueError("Observed risk parameter has the wrong dimension")
    changed = (model if not np.any(theta) else
               replace(model, cost=model.cost + theta @ directions,
                       layers=model.layers))
    base = solve_cmdp(changed, search_mode="breakpoint",
                      relative_tolerance=relative_tolerance,
                      return_oracles=reuse_oracles)
    atlas = {"model": model, "directions": directions,
             "source_cost": model.cost.copy(),
             "keys": set(), "policies": [], "base": base}
    for price in base["prices"]:
        result = (base["oracle_results"][price] if reuse_oracles
                  else solve_scalarized(changed, price))
        add_policy(atlas, result, theta)
    base.pop("oracle_results", None)
    atlas["prepare_seconds"] = time.perf_counter() - started
    return atlas


def price_interval(policy, theta):
    """Return exact t interval where Bellman inequalities hold on this ray."""
    intercept = policy["intercept"]
    slope = policy["slopes"][:, :, 0] + np.tensordot(
        policy["slopes"][:, :, 1:], theta, axes=([2], [0]))
    lo, hi = 0.0, np.inf
    positive = slope > 1e-12
    negative = slope < -1e-12
    zero = ~(positive | negative)
    if np.any(intercept[zero] < -BELLMAN_TOLERANCE):
        return None
    if np.any(positive):
        lo = max(lo, float(np.max((-BELLMAN_TOLERANCE - intercept[positive]) /
                                  slope[positive])))
    if np.any(negative):
        hi = min(hi, float(np.min((-BELLMAN_TOLERANCE - intercept[negative]) /
                                  slope[negative])))
    if lo > hi + 1e-8:
        return None
    return max(0.0, lo), hi


def dual_bound(policy, theta, t):
    risk_value = policy["cost_values"][0] + policy["cost_values"][1:] @ theta
    value = policy["reward_value"] - t * risk_value
    return float(value + BELLMAN_TOLERANCE / (1 - GAMMA) +
                 SOFT * min(1.0, t) + HARD * max(0.0, t - 1.0))


def query(atlas, theta, materialize=True, nearest_first=None):
    started = time.perf_counter()
    theta = np.asarray(theta)
    prices = []
    policies = atlas["policies"]
    if nearest_first is None or len(policies) <= nearest_first:
        first = list(range(len(policies)))
        rest = []
    else:
        order = sorted(range(len(policies)),
                       key=lambda i: float(np.linalg.norm(
                           policies[i]["source_theta"] - theta)))
        first, rest = order[:nearest_first], order[nearest_first:]

    def scan(indices):
        for index in indices:
            policy = policies[index]
            interval = price_interval(policy, theta)
            if interval is None:
                continue
            lo, hi = interval
            for t in (lo, hi, 1.0):
                if np.isfinite(t) and lo - 1e-8 <= t <= hi + 1e-8:
                    prices.append((dual_bound(policy, theta, t), t, index))

    scan(first)
    if not prices and rest:
        scan(rest)
        rest = []
    if not prices:
        return {"certified": False, "reason": "no_dual_region",
                "seconds": time.perf_counter() - started}
    upper, price, dual_index = min(prices)
    candidates = []
    risks = [float(p["risk_coeff"][0] + p["risk_coeff"][1:] @ theta)
             for p in policies]
    for i, p in enumerate(policies):
        if risks[i] <= HARD + 1e-9:
            objective = p["reward"] - max(0.0, risks[i] - SOFT)
            candidates.append((objective, risks[i], i, i, 1.0))
        for j in range(i):
            if abs(risks[i] - risks[j]) < 1e-12:
                continue
            for target in (SOFT, HARD):
                w = (target - risks[j]) / (risks[i] - risks[j])
                if -1e-9 <= w <= 1 + 1e-9:
                    w = float(np.clip(w, 0, 1))
                    reward = w * p["reward"] + (1 - w) * policies[j]["reward"]
                    candidates.append((reward - max(0.0, target - SOFT),
                                       target, i, j, w))
    if not candidates:
        return {"certified": False, "reason": "no_primal_policy",
                "seconds": time.perf_counter() - started}
    objective, risk, i, j, weight = max(candidates)
    gap = upper - objective
    if rest and gap > max(1e-4, .01 * abs(objective)) + 1e-7:
        scan(rest)
        if prices:
            upper, price, dual_index = min(prices)
            gap = upper - objective
    certified = bool(risk <= HARD + 1e-8 and
                     -1e-6 <= gap <= max(1e-4, .01 * abs(objective)) + 1e-7)
    answer = {"certified": certified, "objective": objective, "risk": risk,
              "upper": upper, "gap": gap, "price": price,
              "dual_index": dual_index, "primal_i": i, "primal_j": j,
              "primal_weight": weight,
              "policies": len(policies), "dual_regions": len(prices)}
    if certified and materialize:
        occ = weight * policies[i]["occupation"] + (
            1 - weight) * policies[j]["occupation"]
        total = occ.sum(axis=1)
        policy = np.divide(occ, total[:, None], out=np.zeros_like(occ),
                           where=total[:, None] > 1e-12)
        policy[total <= 1e-12, 0] = 1
        answer["policy"] = policy
    answer["seconds"] = time.perf_counter() - started
    return answer


def run_case(n, scenario, seed=0):
    model = build_grid(n, scenario, seed)
    directions = risk_directions(model)
    atlas = prepare(model, directions)
    rows = []
    for t0 in (-.05, -.02, 0, .02, .05):
        for t1 in (-.05, -.02, 0, .02, .05):
            theta = np.array([t0, t1])
            changed = replace(model, cost=model.cost + theta @ directions,
                              layers=model.layers)
            ans = query(atlas, theta)
            fresh = solve_cmdp(changed, search_mode="breakpoint",
                               relative_tolerance=.001)
            row = {"theta": theta.tolist(), "certified": ans["certified"],
                   "query_seconds": ans["seconds"],
                   "fresh_seconds": fresh["seconds"],
                   "fresh_objective": fresh["objective"],
                   "fresh_upper": fresh["upper"],
                   "risk": ans.get("risk"), "objective": ans.get("objective"),
                   "gap": ans.get("gap"), "reason": ans.get("reason")}
            if ans["certified"]:
                evaluated = evaluate_policy(changed, ans["policy"])
                assert abs(evaluated["risk"] - ans["risk"]) < 1e-6
                assert abs(evaluated["objective"] - ans["objective"]) < 1e-6
                assert ans["objective"] <= fresh["upper"] + 1e-6
                assert fresh["objective"] <= ans["upper"] + 1e-6
            rows.append(row)
    return {"case": f"{n}:{scenario}:{seed}",
            "prepare_seconds": atlas["prepare_seconds"],
            "policies": len(atlas["policies"]), "rows": rows}


def main():
    output = []
    for n, scenario in ((24, "barrier_hard_active"),
                        (32, "barrier_hard_active"),
                        (24, "middle_fast"),
                        (32, "middle_fast")):
        result = run_case(n, scenario)
        output.append(result)
        (HERE / "joint_price_atlas.json").write_text(json.dumps(output, indent=2))
        print(result["case"], sum(r["certified"] for r in result["rows"]),
              "/", len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
