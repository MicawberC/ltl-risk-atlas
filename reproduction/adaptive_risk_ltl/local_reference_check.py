"""Independent checks for the slow 32x32 crossing IID reference points."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiment import build_grid, solve_lp
from five_zone_stress import directions_for_five_cells
from local_update_gate import approach_start
from stratified import solve_cmdp


HERE = Path(__file__).resolve().parent


def main() -> None:
    saved = json.loads((HERE / "local_update_crossing_iid32.json").read_text())
    stream = saved["cases"][0]["streams"][0]
    assert saved["cases"][0]["case"] == "32:middle_fast_crossing:0"
    assert stream["kind"] == "iid"
    model = approach_start(build_grid(32, "middle_fast_crossing"),
                           "middle_fast_crossing")
    directions = directions_for_five_cells(model)
    checks = []
    for index in (0, 10, 19):
        measured = stream["rows"][index]
        theta = np.asarray(measured["theta"])
        changed = replace(model, cost=model.cost + theta @ directions,
                          layers=model.layers)
        fresh = solve_cmdp(changed, search_mode="breakpoint",
                           relative_tolerance=.001)
        assert fresh["certified"]
        exact = measured["lp_objective"]
        assert fresh["objective"] <= exact + 1e-5
        assert exact <= fresh["upper"] + 1e-5
        sparse_lp = None
        try:
            result = solve_lp(changed, method="highs-ds",
                              time_limit_seconds=15.0)
            assert abs(result["objective"] - exact) <= 1e-5
            sparse_lp = {"status": "optimal", "objective": result["objective"],
                         "seconds": result["wall_seconds"]}
        except RuntimeError as exc:
            if "Time limit reached" not in str(exc):
                raise
            sparse_lp = {"status": "time_limit_15s", "detail": str(exc)}
        checks.append({"index": index, "theta": theta.tolist(),
                       "warm_lp_objective": exact,
                       "fresh_cmdp_objective": fresh["objective"],
                       "fresh_cmdp_upper": fresh["upper"],
                       "fresh_cmdp_seconds": fresh["seconds"],
                       "sparse_lp": sparse_lp})
        print(index, sparse_lp["status"], "fresh_sec",
              round(fresh["seconds"], 3), flush=True)
    (HERE / "local_update_crossing_iid32_reference.json").write_text(
        json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
