"""Exploratory scalarized solver using monotone LTL automaton layers.

The goal/safety DFA combinations form a partial order: goal progress and
the safety-failure bit only advance. Solve Bellman equations in reverse
topological order, one physical-state layer at a time. This is exploratory code; the
occupancy LP remains the independent reference.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from experiment import GAMMA, HARD, SOFT, Model, build_grid, evaluate_policy, solve_lp


def automaton_scc_layers(model: Model) -> list[np.ndarray]:
    """Discover a reverse-topological SCC schedule from the product graph."""
    if model.layers is not None:
        return model.layers
    valid = np.flatnonzero(model.mode >= 0)
    modes, inverse = np.unique(np.stack((model.goal[valid], model.bad[valid]), axis=1),
                               axis=0, return_inverse=True)
    mode_of_state = np.full(model.states, -1, dtype=np.int32)
    mode_of_state[valid] = inverse
    row_ids = np.repeat(np.arange(model.p.shape[0], dtype=np.int32),
                        np.diff(model.p.indptr))
    source = mode_of_state[row_ids // model.actions]
    target = mode_of_state[model.p.indices]
    keep = (source >= 0) & (target >= 0)
    graph = sparse.coo_matrix((np.ones(int(keep.sum())),
                               (source[keep], target[keep])),
                              shape=(len(modes), len(modes))).tocsr()
    count, component = connected_components(graph, directed=True, connection="strong")
    edges = {(int(component[s]), int(component[t]))
             for s, t in zip(*graph.nonzero()) if component[s] != component[t]}
    successors = [set() for _ in range(count)]
    indegree = [0] * count
    for source_component, target_component in edges:
        successors[source_component].add(target_component)
        indegree[target_component] += 1
    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    order = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != count:
        raise AssertionError("Automaton SCC condensation must be acyclic")
    state_component = component[inverse]
    model.layers = [valid[state_component == block] for block in reversed(order)]
    return model.layers


def solve_scalarized(model: Model, price: float,
                     initial_action: np.ndarray | None = None) -> dict:
    started = time.perf_counter()
    n, a = model.states, model.actions
    value = np.zeros(n)
    iterations = []
    adjusted = model.reward - price * model.cost[:, None]
    for states in automaton_scc_layers(model):
        rows = (states[:, None] * a + np.arange(a)).ravel()
        transition = model.p[rows]
        internal = transition[:, states].tocsr()
        base = adjusted[states].ravel() + GAMMA * (transition @ value)
        base = base.reshape(len(states), a)
        action = (np.asarray(initial_action[states], dtype=np.int64).copy()
                  if initial_action is not None else np.argmax(base, axis=1))
        identity = sparse.eye(len(states), format="csc")
        for iteration in range(1, 101):
            chosen = internal[np.arange(len(states)) * a + action]
            local = spsolve(identity - GAMMA * chosen.tocsc(),
                            base[np.arange(len(states)), action])
            q_local = base + GAMMA * (internal @ local).reshape(len(states), a)
            residual = float(np.max(q_local - local[:, None]))
            if residual < 1e-10:
                break
            new_action = np.argmax(q_local, axis=1)
            if np.array_equal(new_action, action):
                break
            action = new_action
        else:
            raise RuntimeError("Layer policy iteration did not converge")
        value[states] = local
        iterations.append(iteration)
    q = adjusted + GAMMA * (model.p @ value).reshape(n, a)
    bellman_error = float(np.max(np.abs(value - np.max(q, axis=1))))
    if bellman_error > 1e-7:
        raise AssertionError(f"Full scalarized Bellman error {bellman_error}")
    policy = np.eye(a)[np.argmax(q, axis=1)]
    exact = evaluate_policy(model, policy)
    return {"price": price, "value": value, "initial_value": float(value[model.initial]),
            "q": q, "policy": policy,
            "risk": exact["risk"], "reward": exact["reward"],
            "occupation": exact["occupancy"][:, None] * policy,
            "bellman_error": bellman_error,
            "iterations": iterations, "seconds": time.perf_counter() - started}


def solve_scalarized_full(model: Model, price: float) -> dict:
    """Matched full-product policy iteration for the decomposition ablation."""
    started = time.perf_counter()
    n, a = model.states, model.actions
    adjusted = model.reward - price * model.cost[:, None]
    action = np.argmax(adjusted, axis=1)
    identity = sparse.eye(n, format="csc")
    for iteration in range(1, 101):
        chosen = model.p[np.arange(n) * a + action]
        value = spsolve(identity - GAMMA * chosen.tocsc(),
                        adjusted[np.arange(n), action])
        q = adjusted + GAMMA * (model.p @ value).reshape(n, a)
        residual = float(np.max(q - value[:, None]))
        if residual < 1e-10:
            break
        new_action = np.argmax(q, axis=1)
        if np.array_equal(new_action, action):
            break
        action = new_action
    else:
        raise RuntimeError("Full policy iteration did not converge")
    bellman_error = float(np.max(np.abs(value - np.max(q, axis=1))))
    if bellman_error > 1e-7:
        raise AssertionError(f"Full scalarized Bellman error {bellman_error}")
    policy = np.eye(a)[np.argmax(q, axis=1)]
    exact = evaluate_policy(model, policy)
    return {"price": price, "value": value, "initial_value": float(value[model.initial]),
            "q": q, "policy": policy, "risk": exact["risk"],
            "reward": exact["reward"],
            "occupation": exact["occupancy"][:, None] * policy,
            "bellman_error": bellman_error,
            "iterations": [iteration], "seconds": time.perf_counter() - started}


def dual_bound(result: dict) -> float:
    t = result["price"]
    return float(result["initial_value"] + result["bellman_error"] / (1 - GAMMA)
                 + SOFT * min(1.0, t) + HARD * max(0.0, t - 1.0))


def optimal_price_interval(model: Model, result: dict) -> tuple[float, float]:
    """Exact price interval where the returned deterministic policy is optimal.

    Under a fixed policy, its scalarized value is affine in the risk price.
    Bellman action inequalities therefore give linear interval endpoints.
    """
    n, a = model.states, model.actions
    action = np.argmax(result["policy"], axis=1)
    chosen = model.p[np.arange(n) * a + action]
    risk_to_go = spsolve(sparse.eye(n, format="csc") - GAMMA * chosen.tocsc(),
                         model.cost)
    reward_to_go = result["value"] + result["price"] * risk_to_go
    reward_action = model.reward + GAMMA * (model.p @ reward_to_go).reshape(n, a)
    risk_action = model.cost[:, None] + GAMMA * (model.p @ risk_to_go).reshape(n, a)
    intercept = reward_to_go[:, None] - reward_action
    slope = risk_action - risk_to_go[:, None]
    lower = -np.inf
    upper = np.inf
    positive = slope > 1e-10
    negative = slope < -1e-10
    if np.any(positive):
        lower = float(np.max(-intercept[positive] / slope[positive]))
    if np.any(negative):
        upper = float(np.min(-intercept[negative] / slope[negative]))
    if lower > result["price"] + 1e-6 or upper < result["price"] - 1e-6:
        raise AssertionError("Computed policy-optimal interval excludes its price")
    return lower, upper


def solve_cmdp(model: Model, tolerance: float = 1e-4,
               relative_tolerance: float = 0.01,
               max_evaluations: int = 35,
               scalarized_solver=solve_scalarized,
               search_mode: str = "bisection",
               return_oracles: bool = False) -> dict:
    """One-dimensional risk-price search with a primal-dual certificate.

    The soft slack dual has 0<=lambda_soft<=1. For fixed total price t,
    its cheapest feasible decomposition is lambda_soft=min(1,t), so the
    dual search is one-dimensional despite having soft and hard budgets.
    """
    started = time.perf_counter()
    if search_mode not in ("bisection", "breakpoint"):
        raise ValueError(search_mode)
    cache: dict[float, dict] = {}

    def get(t: float) -> dict:
        key = float(t)
        if key not in cache:
            cache[key] = scalarized_solver(model, key)
        return cache[key]

    def finish(answer: dict) -> dict:
        if return_oracles:
            answer["oracle_results"] = {price: cache[price]
                                        for price in answer["prices"]}
        return answer

    zero = get(0.0)
    if zero["risk"] <= SOFT:
        policy = zero["policy"]
        primal = evaluate_policy(model, policy)
        return finish({"objective": primal["objective"], "risk": primal["risk"],
                "upper": dual_bound(zero), "gap": dual_bound(zero) - primal["objective"],
                "evaluations": len(cache), "seconds": time.perf_counter() - started,
                "prices": [0.0], "mixed": False,
                "certified": (primal["risk"] <= HARD + 1e-7 and
                              dual_bound(zero) - primal["objective"] <=
                              max(tolerance, relative_tolerance * abs(primal["objective"])) + 1e-7)})

    # First check the kink at t=1, which separates the soft and hard caps.
    one = get(1.0)
    if one["risk"] >= SOFT and one["risk"] <= HARD:
        primal = evaluate_policy(model, one["policy"])
        return finish({"objective": primal["objective"], "risk": primal["risk"],
                "upper": dual_bound(one), "gap": dual_bound(one) - primal["objective"],
                "evaluations": len(cache), "seconds": time.perf_counter() - started,
                "prices": [0.0, 1.0], "mixed": False,
                "certified": (primal["risk"] <= HARD + 1e-7 and
                              dual_bound(one) - primal["objective"] <=
                              max(tolerance, relative_tolerance * abs(primal["objective"])) + 1e-7)})
    if one["risk"] < SOFT:
        lo, hi, target = zero, one, SOFT
    else:
        lo, target = one, HARD
        price = 2.0
        hi = get(price)
        while hi["risk"] > HARD and len(cache) < max_evaluations:
            price *= 2
            hi = get(price)
    extra_upper_bounds = []
    if search_mode == "breakpoint" and target == HARD:
        # A fixed deterministic policy is optimal on a price interval.
        # Jump to its first Bellman-action breakpoint instead of bisecting
        # through a region where the policy and risk do not change.
        for _ in range(3):
            _, edge_price = optimal_price_interval(model, lo)
            if not (lo["price"] + 1e-7 < edge_price < hi["price"] - 1e-7):
                break
            edge = get(edge_price)
            extra_upper_bounds.append(dual_bound(edge))
            if edge["risk"] <= HARD:
                hi = edge
                break
            right = get(edge_price + 1e-7)
            if right["risk"] < HARD:
                lo, hi = edge, right
                break
            lo = right
    if lo["risk"] <= target or hi["risk"] >= target:
        raise RuntimeError("Could not bracket the risk target")

    best = None
    for _ in range(max_evaluations - len(cache) + 1):
        # Mix discounted occupation measures and normalize back to a
        # stationary randomized policy. The original constraint is expected
        # discounted risk, hence such mixtures are feasible.
        weight_hi_risk = (target - hi["risk"]) / (lo["risk"] - hi["risk"])
        occupation = weight_hi_risk * lo["occupation"] + (1 - weight_hi_risk) * hi["occupation"]
        totals = occupation.sum(axis=1)
        policy = np.divide(occupation, totals[:, None],
                           out=np.zeros_like(occupation), where=totals[:, None] > 1e-12)
        policy[totals <= 1e-12, 0] = 1.0
        primal = evaluate_policy(model, policy)
        upper = min(dual_bound(lo), dual_bound(hi), *extra_upper_bounds)
        if upper + 1e-7 < primal["objective"]:
            raise AssertionError("Dual upper bound is below the feasible objective")
        if np.max(np.abs(primal["occupancy"] - totals)) > 1e-6:
            raise AssertionError("Mixed occupation is inconsistent with its policy")
        gap = max(0.0, upper - primal["objective"])
        best = {"objective": primal["objective"], "risk": primal["risk"],
                "upper": upper, "gap": gap, "evaluations": len(cache),
                "seconds": time.perf_counter() - started,
                "prices": [lo["price"], hi["price"]], "mixed": True,
                "target": target,
                "certified": (primal["risk"] <= HARD + 1e-7 and
                              gap <= max(tolerance, relative_tolerance * abs(primal["objective"])) + 1e-7)}
        if gap <= max(tolerance, relative_tolerance * abs(primal["objective"])) + 1e-7:
            break
        if len(cache) >= max_evaluations:
            break
        mid = get((lo["price"] + hi["price"]) / 2)
        if mid["risk"] > target:
            lo = mid
        else:
            hi = mid
    return finish(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["4:near", "16:near", "16:middle", "32:near", "32:barrier_hard"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stratified_pilot.json"))
    args = parser.parse_args()
    output = []
    for name in args.cases:
        n_text, scenario = name.split(":")
        model = build_grid(int(n_text), scenario)
        baseline = solve_lp(model)
        result = solve_cmdp(model)
        row = {"case": name, "states": model.states,
               "sparse_lp_seconds": baseline["wall_seconds"],
               "sparse_lp_objective": baseline["objective"], **result}
        row["objective_difference"] = result["objective"] - baseline["objective"]
        output.append(row)
        args.output.write_text(json.dumps(output, indent=2))
        print(name, "LP", round(row["sparse_lp_seconds"], 3),
              "stratified", round(result["seconds"], 3),
              "gap", round(result["gap"], 6),
              "evals", result["evaluations"], flush=True)


if __name__ == "__main__":
    main()
