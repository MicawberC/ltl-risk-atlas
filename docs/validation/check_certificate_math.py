"""Small independent checks for the proof-to-code formulas.

Run from the repository root with a Python environment containing NumPy,
SciPy and the project's reproduction dependencies.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reproduction"
                       / "adaptive_risk_ltl"))
from experiment import HARD, SOFT, build_grid  # noqa: E402
from joint_price_atlas import dual_bound, prepare, risk_directions  # noqa: E402
from local_residual import cached_primal  # noqa: E402
from stratified import solve_scalarized  # noqa: E402


def check_residual_upper() -> int:
    model = build_grid(4, "near")
    directions = risk_directions(model)
    atlas = prepare(model, directions)
    count = 0
    for theta in (np.array([-0.8, 0.5]), np.array([0.3, -0.6]),
                  np.array([0.8, 0.8])):
        changed = replace(model, cost=model.cost + theta @ directions)
        for price in (0.0, 0.3, 1.0, 2.0, 5.0):
            exact = solve_scalarized(changed, price)["initial_value"]
            exact += SOFT * min(1.0, price) + HARD * max(0.0, price - 1.0)
            for policy in atlas["policies"]:
                assert exact <= dual_bound(policy, theta, price) + 1e-7
                count += 1
    return count


def check_two_policy_enumeration() -> int:
    rng = np.random.default_rng(314159)
    for _ in range(50):
        risks = np.r_[0.0, rng.uniform(0.0, 8.0, size=5)]
        rewards = rng.uniform(-2.0, 10.0, size=len(risks))
        atoms = [{"risk_coeff": np.array([risk, 0.0]), "reward": reward}
                 for risk, reward in zip(risks, rewards)]
        candidate = cached_primal({"policies": atoms}, np.array([0.0]))
        n = len(risks)
        lp = linprog(
            np.r_[-rewards, 1.0],
            A_ub=np.r_[np.r_[risks, 0.0][None, :],
                       np.r_[risks, -1.0][None, :]],
            b_ub=[HARD, SOFT],
            A_eq=np.r_[np.ones(n), 0.0][None, :],
            b_eq=[1.0],
            bounds=[(0.0, None)] * (n + 1),
            method="highs",
        )
        assert lp.success
        assert candidate is not None
        assert candidate[1] <= HARD + 1e-8
        assert abs(candidate[0] + lp.fun) < 1e-7
    return 50


if __name__ == "__main__":
    print("PASS:", check_residual_upper(), "scalarized upper checks and",
          check_two_policy_enumeration(), "independent cache-hull LP checks")
