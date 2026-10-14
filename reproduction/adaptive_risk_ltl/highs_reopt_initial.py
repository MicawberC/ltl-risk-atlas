"""Warm LP control for the fixed-product changing-initial-state benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import build_grid
from highs_reopt import prepare, query
from joint_atlas_initial import PERTURBATIONS, STARTS
from joint_price_atlas import risk_directions

HERE = Path(__file__).resolve().parent


def run(n, scene, seed=0, order_seed=321):
    model = build_grid(n, scene, seed)
    cache = prepare(model, risk_directions(model))
    requests = [(xy, theta) for xy in STARTS for theta in PERTURBATIONS]
    rows = []
    for idx in np.random.default_rng(order_seed).permutation(len(requests)):
        xy, theta = requests[idx]
        initial = 8 * (xy[0] + n * xy[1]) + 4
        answer = query(cache, np.asarray(theta), initial)
        rows.append({"start": list(xy), "theta": list(theta), **answer})
    return {"case": f"{n}:{scene}:{seed}",
            "prepare_seconds": cache["prepare_seconds"],
            "total_seconds": cache["prepare_seconds"] +
                             sum(r["seconds"] for r in rows),
            "rows": rows}


def main():
    path = HERE / "highs_reopt_initial.json"
    output = []
    for n, scene in ((24, "barrier_hard_active"),
                     (32, "barrier_hard_active"),
                     (24, "middle_fast"),
                     (32, "middle_fast")):
        result = run(n, scene)
        output.append(result)
        path.write_text(json.dumps(output, indent=2))
        print(result["case"], "total", round(result["total_seconds"], 3),
              flush=True)


if __name__ == "__main__":
    main()
