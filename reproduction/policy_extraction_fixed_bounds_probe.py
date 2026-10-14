"""Probe policy extraction with the released demo's unchanged risk bounds.

The second case changes only the starting state within the same product model.
It is a diagnostic scenario, not a reproduction of a published CARLA run.
"""

from __future__ import annotations

import json

import numpy as np

from iros_core import ROOT, Risk_LTL_LP, build_iros_model
from iros_lp_utils import GAMMA, HARD, SOFT, metrics, sparse_lp, stochastic_extension
from policy_extraction_probe import policy_values


def evaluate_case(product, p: np.ndarray, cost: np.ndarray,
                  accept: list[int], initial: int) -> dict:
    occupancy, _ = sparse_lp(p, cost, initial, accept, hard_bound=HARD)
    state_mass = occupancy.sum(axis=1)
    randomized = np.divide(
        occupancy, state_mass[:, None], out=np.zeros_like(occupancy),
        where=state_mass[:, None] > 1e-12,
    )
    randomized[state_mass <= 1e-12, 0] = 1.0

    extractor = Risk_LTL_LP()
    extractor.state_num, extractor.action_num = occupancy.shape
    actions, _ = extractor.extract({
        (s, a): float(occupancy[s, a])
        for s in range(occupancy.shape[0])
        for a in range(occupancy.shape[1])
    })
    deterministic = np.eye(occupancy.shape[1])[actions.astype(int)]
    reward = 2 * p[:, :, accept].sum(axis=2)
    mixed_states = np.flatnonzero((occupancy > 1e-8).sum(axis=1) > 1)
    return {
        "initial_product_state": initial,
        "product_state": [int(x) for x in product.prod_state_set[initial]],
        "lp_occupation": metrics(occupancy, p, cost, initial, accept),
        "paper_normalized_randomized_policy": policy_values(
            randomized, p, cost, reward, initial),
        "released_code_deterministic_policy": policy_values(
            deterministic, p, cost, reward, initial),
        "mixed_states": [
            {
                "product_state": int(s),
                "occupancy": {str(a): float(value) for a, value in enumerate(occupancy[s])
                              if value > 1e-8},
                "normalized_probabilities": {
                    str(a): float(value / state_mass[s])
                    for a, value in enumerate(occupancy[s]) if value > 1e-8
                },
                "released_code_action": int(actions[s]),
            }
            for s in mixed_states
        ],
    }


def main() -> None:
    product, base_cost, default_initial = build_iros_model()
    p = stochastic_extension(product.prod_transitions)
    cost = np.r_[base_cost, 0.0]
    accept = list(product.accepting_states)
    # System/environment state 22 is an alternative valid starting state.
    alternate_initial, _ = product.update_prod_state(22, (0, 1, 1))
    cases = {
        "parameters": {
            "soft_risk_threshold": SOFT,
            "hard_risk_bound": HARD,
            "discount": GAMMA,
        },
        "default_initial": evaluate_case(
            product, p, cost, accept, default_initial),
        "alternate_initial": evaluate_case(
            product, p, cost, accept, alternate_initial),
    }
    default = cases["default_initial"]
    alternate = cases["alternate_initial"]
    if (default["released_code_deterministic_policy"]["risk"] > HARD + 1e-8
            or abs(alternate["paper_normalized_randomized_policy"]["risk"] - HARD) > 1e-8
            or alternate["released_code_deterministic_policy"]["risk"] <= HARD + 1e-4):
        raise AssertionError("Unexpected fixed-bound extraction result")
    output = ROOT / "reproduction" / "policy_extraction_fixed_bounds_results.json"
    output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cases, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
