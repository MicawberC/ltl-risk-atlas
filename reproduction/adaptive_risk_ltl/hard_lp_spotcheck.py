"""Independently verify selected held-out hard-active warm-LP values."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid, solve_lp
from five_zone_hard import spatial_directions

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "barrier_hard_active", 2)
    directions = spatial_directions(model)
    theta = np.asarray(json.loads((HERE / "five_zone_hard.json").read_text())["theta"])
    warm = json.loads((HERE / "highs_hard_five.json").read_text())
    exact = {row["index"]: row["objective"] for row in warm["rows"]}
    rows = []
    for index in (0, 12, 24):
        changed = replace(model, cost=model.cost + theta[index] @ directions,
                          layers=model.layers)
        result = solve_lp(changed, method="highs-ds")
        risk = float(np.sum(result["occupation"].sum(axis=1) * changed.cost))
        assert risk <= HARD + 1e-6
        assert abs(result["objective"] - exact[index]) < 1e-6
        rows.append({"index": index, "objective": result["objective"],
                     "risk": risk, "seconds": result["wall_seconds"]})
    (HERE / "hard_lp_spotcheck.json").write_text(json.dumps(rows, indent=2))
    print([(r["index"], round(r["objective"], 6)) for r in rows])


if __name__ == "__main__":
    main()
