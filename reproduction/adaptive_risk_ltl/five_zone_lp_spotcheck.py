"""Independent occupancy-LP checks for selected five-parameter risk maps."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, solve_lp
from five_zone_stress import directions_for_five_cells

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "middle_fast")
    directions = directions_for_five_cells(model)
    stress = json.loads((HERE / "five_zone_stress.json").read_text())
    reference = json.loads((HERE / "five_zone_reference.json").read_text())
    rows = []
    for index in (0, 12, 25, 37, 49):
        theta = np.asarray(stress["theta"][index])
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        exact = solve_lp(changed, method="highs-ds")
        risk = float(np.sum(exact["occupation"].sum(axis=1) * changed.cost))
        estimated = reference[index]
        assert risk <= HARD + 1e-6
        assert estimated["objective"] <= exact["objective"] + 1e-5
        assert exact["objective"] <= estimated["upper"] + 1e-5
        rows.append({"index": index, "theta": theta.tolist(),
                     "lp_objective": exact["objective"],
                     "lp_risk": risk, "lp_seconds": exact["wall_seconds"],
                     "fresh_lower": estimated["objective"],
                     "fresh_upper": estimated["upper"]})
    (HERE / "five_zone_lp_spotcheck.json").write_text(
        json.dumps(rows, indent=2))
    print([(r["index"], round(r["lp_objective"], 6),
            round(r["lp_seconds"], 2)) for r in rows])


if __name__ == "__main__":
    main()
