"""Five-dimensional risk-update benchmark with retained HiGHS simplex basis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import HARD, build_grid
from five_zone_stress import directions_for_five_cells
from highs_reopt import prepare, query

HERE = Path(__file__).resolve().parent


def main():
    model = build_grid(24, "middle_fast")
    directions = directions_for_five_cells(model)
    thetas = np.asarray(json.loads((HERE / "five_zone_stress.json").read_text())["theta"])
    reference = json.loads((HERE / "five_zone_reference.json").read_text())
    output = []
    for seed in (0, 1, 2):
        cache = prepare(model, directions)
        rows = []
        for index in np.random.default_rng(seed).permutation(len(thetas)):
            answer = query(cache, thetas[index])
            assert answer["basis_valid"]
            assert answer["risk"] <= HARD + 1e-6
            assert reference[index]["objective"] <= answer["objective"] + 1e-5
            assert answer["objective"] <= reference[index]["upper"] + 1e-5
            rows.append({"index": int(index), **answer})
        output.append({"order_seed": seed, "prepare_seconds": cache["prepare_seconds"],
                       "query_seconds": sum(r["seconds"] for r in rows),
                       "total_seconds": cache["prepare_seconds"] +
                                        sum(r["seconds"] for r in rows),
                       "rows": rows})
        print(seed, round(output[-1]["total_seconds"], 3), flush=True)
        (HERE / "highs_five_zone.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
