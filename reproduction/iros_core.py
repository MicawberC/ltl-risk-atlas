"""Headless reproduction of the released IROS risk-LTL planning example.

The upstream LTLf2DFA package invokes MONA through a POSIX-only subprocess.
For the two fixed formulas in reactive_LTL_main.py, use their exact two-state
DFAs and otherwise keep the authors' abstraction, product and LP solver.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "papers" / "Risk_LTL_Planning-main"
sys.path.insert(0, str(SOURCE))

from abstraction.MDP import MDP  # noqa: E402
from abstraction.abstraction import Abstraction  # noqa: E402
from abstraction.prod_MDP import Prod_MDP  # noqa: E402
from risk_LP.ltl_risk_LP import Risk_LTL_LP  # noqa: E402
from specification.DFA import DFA  # noqa: E402
from specification.prod_auto import Product  # noqa: E402


def fixed_formula_dfa(kind: str) -> DFA:
    """Exact DFA for F(t) or G(~g -> ~c) & G(~v)."""
    if kind == "goal":
        aps = ["t"]
        bad_or_done = lambda letter: letter[0]
    elif kind == "safety":
        aps = ["c", "g", "v"]
        bad_or_done = lambda letter: (letter[0] and not letter[1]) or letter[2]
    else:
        raise ValueError(kind)

    alphabet = list(__import__("itertools").product([True, False], repeat=len(aps)))
    transitions = {}
    for letter in alphabet:
        transitions[("1", letter)] = "2" if bad_or_done(letter) else "1"
        transitions[("2", letter)] = "2"
    return DFA([1, 2], alphabet, transitions, "1", ["2"], aps)


def build_iros_model():
    """Return the authors' fixed demo product, state cost, and initial index."""
    traffic_light = ["g", "r"]
    env = MDP(
        range(2), [0], np.array([[[0.8, 0.2], [0.2, 0.8]]]),
        traffic_light, 0,
    )
    # Same initial map and vehicle positions as the upstream main script.
    static_labels = {
        (10, 20, 10, 15): "c",
        (10, 15, 15, 20): "c",
        (15, 20, 15, 20): "t",
    }
    opponent_position = (12, 7.5)
    gx = int(np.floor(opponent_position[0] / 5))
    gy = int(np.floor(opponent_position[1] / 5))
    labels = dict(static_labels)
    labels[(gx * 5, (gx + 1) * 5, gy * 5, (gy + 1) * 5)] = "v0"
    labels[((gx - 1) * 5, gx * 5, gy * 5, (gy + 1) * 5)] = "v1"

    abstraction = Abstraction((20, 20), (5, 5), (2.5, 2.5), labels)
    system_env = Prod_MDP(abstraction.MDP, env)
    product = Product(
        system_env.MDP,
        fixed_formula_dfa("goal"),
        fixed_formula_dfa("safety"),
    )
    cost = product.gen_cost_map({"c": 5, "v0": 10, "v1": 2})
    initial_system_env = system_env.get_prod_state_index((0, 1))
    initial, _ = product.update_prod_state(initial_system_env, (0, 1, 1))

    return product, cost, initial


def main() -> None:
    product, cost, initial = build_iros_model()
    lp = Risk_LTL_LP()
    occupancy = lp.solve(product.prod_transitions, cost, initial, product.accepting_states)
    occ = np.array([
        [occupancy[s, a] for a in range(lp.action_num)]
        for s in range(lp.state_num)
    ])
    gamma = 0.9  # Upstream solver's fixed value.
    flow = occ.sum(axis=1) - gamma * np.einsum("sa,san->n", occ, product.prod_transitions)
    alpha = np.zeros(lp.state_num)
    alpha[initial] = 1
    risk = float(np.sum(occ * cost[:, None]))
    goal_flow = float(np.einsum("sa,san->", occ, product.prod_transitions[:, :, product.accepting_states]))
    objective = 2 * goal_flow - max(0, risk - 0.8)
    row_sums = product.prod_transitions.sum(axis=2)
    if risk > 5 + 1e-8 or np.max(np.abs(flow - alpha)) > 1e-8:
        raise AssertionError("Risk bound or occupation flow check failed")
    result = {
        "source": "papers/Risk_LTL_Planning-main",
        "states": int(lp.state_num),
        "actions": int(lp.action_num),
        "initial_state": int(initial),
        "accepting_states": len(product.accepting_states),
        "trap_states": len(product.trap_states),
        "risk": risk,
        "hard_risk_bound": 5,
        "objective": objective,
        "max_flow_residual": float(np.max(np.abs(flow - alpha))),
        "transition_row_sum_min": float(np.min(row_sums)),
        "transition_row_sum_max": float(np.max(row_sums)),
    }
    output = ROOT / "reproduction" / "iros_core_results.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
