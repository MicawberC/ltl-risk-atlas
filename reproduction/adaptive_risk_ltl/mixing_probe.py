"""Check whether hard-risk certificate slack can be repaired by policy mixing.

This is a diagnostic use of a standard CMDP occupancy-measure idea, not a new
abstraction algorithm or a claim of real-world pathwise safety.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import HARD, SOFT, build_grid, evaluate_policy, upper_certificate

HERE = Path(__file__).resolve().parent


def main() -> None:
    record = next(x for x in json.loads((HERE / "large32.json").read_text())
                  if x["scenario"] == "barrier_hard")
    model = build_grid(32, "barrier_hard")
    guided = next(m for m in record["methods"] if m["strategy"] == "guided")
    dual = guided["history"][-1]
    price = dual["lambda_soft"] + dual["lambda_hard"]
    policies = []
    for offset in (-1e-5, 1e-5):
        lam = price + offset
        # The full-model scalarized MDP is solved only to diagnose the missing
        # risk-budget allocation. The constructed dual fields are valid here.
        cert = upper_certificate(
            model, np.arange(model.states),
            {"dual_v": np.zeros(model.states), "lambda_soft": 1.0,
             "lambda_hard": lam - 1.0},
        )
        action = np.argmax(cert["q"], axis=1)
        policy = np.eye(model.actions)[action]
        exact = evaluate_policy(model, policy)
        policies.append((policy, exact, cert))
    high = max(policies, key=lambda item: item[1]["risk"])
    low = min(policies, key=lambda item: item[1]["risk"])
    if not (low[1]["risk"] < HARD < high[1]["risk"]):
        raise AssertionError("The two policies do not bracket the hard risk budget")
    high_weight = (HARD - low[1]["risk"]) / (high[1]["risk"] - low[1]["risk"])
    # Occupancy convex combination can be normalized into a stationary
    # randomized policy with the same expected reward and discounted cost.
    x = (high_weight * high[1]["occupancy"][:, None] * high[0]
         + (1 - high_weight) * low[1]["occupancy"][:, None] * low[0])
    totals = x.sum(axis=1)
    mixed_policy = np.divide(x, totals[:, None], out=np.zeros_like(x),
                             where=totals[:, None] > 1e-12)
    mixed_policy[totals <= 1e-12, 0] = 1.0
    mixed = evaluate_policy(model, mixed_policy)
    output = {
        "grid": 32, "scenario": "barrier_hard", "full_states": model.states,
        "soft_budget": SOFT, "hard_budget": HARD,
        "dual_price": price,
        "high_risk": {"risk": high[1]["risk"], "reward": high[1]["reward"]},
        "low_risk": {"risk": low[1]["risk"], "reward": low[1]["reward"]},
        "high_risk_policy_weight": high_weight,
        "stationary_mixture": {"risk": mixed["risk"],
                               "objective": mixed["objective"],
                               "flow_residual": mixed["flow_residual"]},
        "sparse_lp_objective": record["sparse_lp_objective"],
        "objective_difference": mixed["objective"] - record["sparse_lp_objective"],
    }
    if abs(mixed["risk"] - HARD) > 1e-7 or abs(output["objective_difference"]) > 1e-7:
        raise AssertionError("Mixture does not match the saved full LP optimum")
    (HERE / "mixing_probe.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
