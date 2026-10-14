"""Check whether the released deterministic extraction keeps the LP risk bound."""

from __future__ import annotations

import json

import numpy as np

from iros_core import ROOT, Risk_LTL_LP, build_iros_model
from iros_lp_utils import GAMMA, SOFT, metrics, sparse_lp, stochastic_extension


def policy_values(policy: np.ndarray, p: np.ndarray, cost: np.ndarray,
                  reward: np.ndarray, initial: int) -> dict:
    transition = np.einsum("sa,san->sn", policy, p)
    system = np.eye(len(cost)) - GAMMA * transition
    risk = float(np.linalg.solve(system, cost)[initial])
    goal = float(np.linalg.solve(system, np.sum(policy * reward, axis=1))[initial])
    return {"risk": risk, "goal_reward": goal,
            "objective": goal - max(0.0, risk - SOFT)}


def main() -> None:
    product, cost, initial = build_iros_model()
    p = stochastic_extension(product.prod_transitions)
    cost = np.r_[cost, 0.0]
    accept = list(product.accepting_states)
    reward = 2 * p[:, :, accept].sum(axis=2)
    cases = {}
    for hard_bound in (5.0, 0.5):
        occ, _ = sparse_lp(p, cost, initial, accept, hard_bound=hard_bound)
        solver = Risk_LTL_LP()
        solver.state_num, solver.action_num = occ.shape
        actions, _ = solver.extract({(s, a): float(occ[s, a])
                                     for s in range(occ.shape[0])
                                     for a in range(occ.shape[1])})
        deterministic = np.eye(occ.shape[1])[actions.astype(int)]
        state_mass = occ.sum(axis=1)
        randomized = np.divide(occ, state_mass[:, None],
                               out=np.zeros_like(occ),
                               where=state_mass[:, None] > 1e-12)
        randomized[state_mass <= 1e-12, 0] = 1.0
        cases[str(hard_bound)] = {
            "hard_bound": hard_bound,
            "lp_occupation": metrics(occ, p, cost, initial, accept),
            "paper_normalized_randomized_policy": policy_values(
                randomized, p, cost, reward, initial),
            "released_code_deterministic_policy": policy_values(
                deterministic, p, cost, reward, initial),
            "states_with_multiple_positive_actions": np.flatnonzero(
                (occ > 1e-8).sum(axis=1) > 1).tolist(),
        }
    tight = cases["0.5"]
    if (abs(tight["paper_normalized_randomized_policy"]["risk"] - 0.5) > 1e-8
            or tight["released_code_deterministic_policy"]["risk"] <= 0.5 + 1e-4):
        raise AssertionError("Expected tight-bound extraction counterexample missing")
    output = ROOT / "reproduction" / "policy_extraction_probe_results.json"
    output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cases, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
