"""Meaningful correctness checks for the synthetic extension and certificates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reproduction"))

from iros_core import build_iros_model  # noqa: E402
from adaptive_risk_ltl.experiment import (  # noqa: E402
    build_grid, initial_partition, aggregate, solve_lp, run_abstraction,
)


def main() -> None:
    original, original_cost, original_initial = build_iros_model()
    copied = build_grid(4)
    matrix = copied.p.toarray().reshape(copied.states, copied.actions, copied.states)
    assert copied.initial == original_initial
    assert np.allclose(matrix[:-1, :, :-1], original.prod_transitions, atol=1e-14)
    assert np.allclose(copied.cost[:-1], original_cost, atol=1e-14)
    assert np.allclose(
        copied.reward[:-1],
        2 * original.prod_transitions[:, :, original.accepting_states].sum(axis=2),
        atol=1e-14,
    )
    baseline = solve_lp(copied)
    assert abs(baseline["objective"] - 1.3620227941214877) < 1e-8
    partition = initial_partition(copied, 4)
    coarse = aggregate(copied, partition)
    assert coarse.states < copied.states
    assert np.max(np.abs(np.asarray(coarse.p.sum(axis=1)).ravel() - 1)) < 1e-12
    trial = run_abstraction(copied, "guided", max_rounds=20)
    assert trial["certified"]
    assert trial["final_risk"] <= 5 + 1e-7
    assert trial["final_objective"] <= baseline["objective"] + 1e-7
    assert trial["history"][-1]["upper"] >= baseline["objective"] - 1e-7
    print("PASS: exact IROS demo match, LP objective, stochastic aggregation, certificates")


if __name__ == "__main__":
    main()
