"""Independent correctness checks for scalarized SCC solving and CMDP bounds."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiment import build_grid, solve_lp
from stratified import (automaton_scc_layers, solve_cmdp,
                        solve_scalarized, solve_scalarized_full)


def main() -> None:
    for n, scenario in ((4, "near"), (8, "near"), (8, "sequence"),
                        (8, "barrier_hard")):
        model = build_grid(n, scenario)
        lp = solve_lp(model)
        answer = solve_cmdp(model, search_mode="breakpoint")
        assert answer["certified"]
        assert answer["risk"] <= 5 + 1e-7
        assert answer["objective"] <= lp["objective"] + 1e-6
        assert lp["objective"] <= answer["upper"] + 1e-6
        for price in (0.0, .4, 1.0):
            layered = solve_scalarized(model, price)
            full = solve_scalarized_full(model, price)
            assert np.max(np.abs(layered["value"] - full["value"])) < 1e-7
    model = build_grid(8, "near")
    cyclic = replace(model, goal=(np.arange(model.states) % 3).astype(np.int8), layers=None)
    assert len(automaton_scc_layers(cyclic)) < 6
    layered = solve_scalarized(cyclic, .4)
    full = solve_scalarized_full(cyclic, .4)
    assert np.max(np.abs(layered["value"] - full["value"])) < 1e-7
    print("PASS: LP brackets, automaton SCC and full-product Bellman values")


if __name__ == "__main__":
    main()
