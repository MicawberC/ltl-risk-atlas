"""Warm LP comparison for large two-zone risk changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import build_grid
from highs_reopt import prepare, query
from joint_price_atlas import risk_directions

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "middle_fast")
    cache = prepare(model, risk_directions(model))
    values = (-.8, -.32, 0.0, .32, .8)
    parameters = np.array([(x, y) for x in values for y in values])
    rows = []
    for index in np.random.default_rng(404).permutation(len(parameters)):
        theta = parameters[index]
        rows.append({"theta": theta.tolist(), **query(cache, theta)})
    result = {"case": "24:middle_fast:0:m0.8",
              "prepare_seconds": cache["prepare_seconds"],
              "total_seconds": cache["prepare_seconds"] +
                               sum(r["seconds"] for r in rows),
              "rows": rows}
    (HERE / "wide_middle_warm.json").write_text(json.dumps(result, indent=2))
    print(result["prepare_seconds"], result["total_seconds"])


if __name__ == "__main__":
    main()
