"""Retained-basis LP baseline for the held-out hard-active five-zone map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid
from five_zone_hard import spatial_directions
from highs_reopt import prepare, query

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "barrier_hard_active", 2)
    directions = spatial_directions(model)
    thetas = np.asarray(json.loads((HERE / "five_zone_hard.json").read_text())["theta"])
    cache = prepare(model, directions)
    rows = []
    for index in np.random.default_rng(19).permutation(len(thetas)):
        answer = query(cache, thetas[index])
        assert answer["basis_valid"]
        assert answer["risk"] <= HARD + 1e-6
        rows.append({"index": int(index), **answer})
    result = {"case": "24:barrier_hard_active:2:five_zone:m0.8",
              "prepare_seconds": cache["prepare_seconds"],
              "query_seconds": sum(r["seconds"] for r in rows),
              "total_seconds": cache["prepare_seconds"] +
                               sum(r["seconds"] for r in rows),
              "rows": rows}
    (HERE / "highs_hard_five.json").write_text(json.dumps(result, indent=2))
    print("warm LP", round(result["total_seconds"], 3))


if __name__ == "__main__":
    main()
