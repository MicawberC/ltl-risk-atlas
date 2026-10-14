"""Certified no-solve queries for an affine update to the LTL risk map.

The transition matrix and reward stay fixed; c(theta) = c(0) + theta*h.
For each cached scalarized deterministic policy, its value is affine in
theta. Bellman action inequalities give an exact interval of policy
optimality. Discounted occupations stay fixed, so rewards and risks of
the two policies are also affine. Inside the intersection, the original
soft/hard CMDP certificate can be evaluated without another LP or MDP solve.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from experiment import GAMMA, HARD, SOFT, Model, build_grid, evaluate_policy, solve_lp
from stratified import solve_cmdp, solve_scalarized


@dataclass
class CachedPolicy:
    price: float
    occupation: np.ndarray
    initial_value: float
    value_slope: float
    risk: float
    risk_slope: float
    reward: float
    interval: tuple[float, float]


def _optimality_interval(model: Model, result: dict,
                         perturbation: np.ndarray) -> tuple[tuple[float, float], float]:
    n, a = model.states, model.actions
    price = result["price"]
    chosen_action = np.argmax(result["policy"], axis=1)
    selected = model.p[np.arange(n) * a + chosen_action].tocsc()
    influence = spsolve(sparse.eye(n, format="csc") - GAMMA * selected,
                        perturbation)
    intercept = result["value"][:, None] - result["q"]
    slope = price * (perturbation[:, None] +
                     GAMMA * (model.p @ influence).reshape(n, a) -
                     influence[:, None])
    lower, upper = -np.inf, np.inf
    positive = slope > 1e-11
    negative = slope < -1e-11
    if np.any(positive):
        lower = max(lower, float(np.max((-intercept[positive] - 1e-9) /
                                         slope[positive])))
    if np.any(negative):
        upper = min(upper, float(np.min((-intercept[negative] - 1e-9) /
                                         slope[negative])))
    if lower > 1e-8 or upper < -1e-8:
        raise AssertionError("Cached policy is not optimal at base parameter")
    return (lower, upper), -price * float(influence[model.initial])


def prepare(model: Model, perturbation: np.ndarray,
            relative_tolerance: float = .001) -> dict:
    if perturbation.shape != model.cost.shape:
        raise ValueError("Risk perturbation must be one coefficient per state")
    started = time.perf_counter()
    base = solve_cmdp(model, search_mode="breakpoint",
                      relative_tolerance=relative_tolerance)
    if not base["certified"] or not base["mixed"]:
        raise RuntimeError("This pilot requires a certified mixed base policy")
    policies = []
    for price in base["prices"]:
        result = solve_scalarized(model, price)
        interval, value_slope = _optimality_interval(model, result, perturbation)
        policies.append(CachedPolicy(
            price, result["occupation"], result["initial_value"], value_slope,
            result["risk"], float(result["occupation"].sum(axis=1) @ perturbation),
            result["reward"], interval))
    target = base["target"]
    lower = max(-1.0, *(p.interval[0] for p in policies))
    upper = min(*(p.interval[1] for p in policies))
    lo, hi = policies
    # Preserve risk_lo(theta) >= target >= risk_hi(theta).
    for value, slope, is_lower in ((lo.risk - target, lo.risk_slope, True),
                                   (hi.risk - target, hi.risk_slope, False)):
        if abs(slope) < 1e-12:
            if (is_lower and value < -1e-9) or (not is_lower and value > 1e-9):
                raise AssertionError("Base risks do not bracket the target")
            continue
        boundary = -value / slope
        if (is_lower and slope > 0) or (not is_lower and slope < 0):
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
    if lower > 1e-8 or upper < -1e-8:
        raise AssertionError("Cached risk bracket excludes base parameter")
    return {"base": base, "policies": policies, "interval": (lower, upper),
            "target": target, "prepare_seconds": time.perf_counter() - started}


def query(context: dict, theta: float, materialize_policy: bool = True) -> dict:
    started = time.perf_counter()
    lo_bound, hi_bound = context["interval"]
    if not (lo_bound + 1e-9 <= theta <= hi_bound - 1e-9):
        return {"certified": False, "reason": "outside_parameter_region",
                "seconds": time.perf_counter() - started}
    high_risk, low_risk = context["policies"]
    target = context["target"]
    c_high = high_risk.risk + theta * high_risk.risk_slope
    c_low = low_risk.risk + theta * low_risk.risk_slope
    if not (c_high >= target - 1e-8 and c_low <= target + 1e-8):
        raise AssertionError("Risk bracket is invalid")
    weight = (target - c_low) / (c_high - c_low)
    reward = weight * high_risk.reward + (1 - weight) * low_risk.reward
    objective = reward - max(0.0, target - SOFT)
    upper = min(p.initial_value + theta * p.value_slope +
                SOFT * min(1.0, p.price) + HARD * max(0.0, p.price - 1.0)
                for p in context["policies"])
    gap = upper - objective
    if gap < -1e-7:
        raise AssertionError("Transferred dual bound is below the primal")
    certified = gap <= max(1e-4, .01 * abs(objective)) + 1e-7
    answer = {"certified": certified, "objective": objective,
              "risk": target, "upper": upper, "gap": gap, "weight": weight}
    if materialize_policy:
        occupation = (weight * high_risk.occupation +
                      (1 - weight) * low_risk.occupation)
        totals = occupation.sum(axis=1)
        policy = np.divide(occupation, totals[:, None],
                           out=np.zeros_like(occupation),
                           where=totals[:, None] > 1e-12)
        policy[totals <= 1e-12, 0] = 1
        answer["policy"] = policy
    answer["seconds"] = time.perf_counter() - started
    return answer


def run_case(n: int, scenario: str, thetas: list[float]) -> dict:
    model = build_grid(n, scenario)
    perturbation = model.cost.copy()
    context = prepare(model, perturbation)
    rows = []
    for theta in thetas:
        changed = replace(model, cost=model.cost + theta * perturbation,
                          layers=model.layers)
        transferred = query(context, theta)
        fresh = solve_cmdp(changed, search_mode="breakpoint")
        row = {"theta": theta, "transfer_certified": transferred["certified"],
               "transfer_seconds": transferred["seconds"],
               "fresh_seconds": fresh["seconds"],
               "fresh_objective": fresh["objective"], "fresh_risk": fresh["risk"]}
        if transferred["certified"]:
            check = evaluate_policy(changed, transferred["policy"])
            if (abs(check["risk"] - transferred["risk"]) > 1e-6 or
                abs(check["objective"] - transferred["objective"]) > 1e-6 or
                check["risk"] > HARD + 1e-7 or
                transferred["objective"] > fresh["upper"] + 1e-6 or
                    fresh["objective"] > transferred["upper"] + 1e-6):
                raise AssertionError("Transferred policy/certificate failed independent evaluation")
            row.update({"objective": transferred["objective"],
                        "risk": transferred["risk"], "gap": transferred["gap"],
                        "verified_objective": check["objective"],
                        "verified_risk": check["risk"]})
        else:
            row["reason"] = transferred.get("reason", "gap_exceeded")
        rows.append(row)
    return {"case": f"{n}:{scenario}",
            "interval": list(context["interval"]),
            "prepare_seconds": context["prepare_seconds"],
            "base_prices": context["base"]["prices"],
            "base_gap": context["base"]["gap"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["24:middle_fast", "32:middle_fast"])
    parser.add_argument("--thetas", nargs="+", type=float,
                        default=[-.02, -.01, -.005, .005, .01, .02, .03, .05])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("parametric_transfer.json"))
    args = parser.parse_args()
    results = []
    for case in args.cases:
        n, scenario = case.split(":")
        result = run_case(int(n), scenario, args.thetas)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2))
        print(case, "region", result["interval"], "certified",
              sum(row["transfer_certified"] for row in result["rows"]),
              "/", len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
