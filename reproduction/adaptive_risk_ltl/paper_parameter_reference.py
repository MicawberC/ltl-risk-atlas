"""Fresh CMDP cross-checks for the IROS-parameter local-update streams."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ["LTL_RISK_GAMMA"] = "0.8"
os.environ["LTL_RISK_SOFT"] = "1"
os.environ["LTL_RISK_HARD"] = "2"

from experiment import build_grid  # noqa: E402
from five_zone_stress import directions_for_five_cells  # noqa: E402
from local_update_gate import approach_start  # noqa: E402
from stratified import solve_cmdp  # noqa: E402


HERE = Path(__file__).resolve().parent


def main() -> None:
    source = json.loads((HERE / "local_update_gate_paper.json").read_text())
    assert source["parameters"]["gamma"] == .8
    assert source["parameters"]["soft"] == 1
    assert source["parameters"]["hard"] == 2
    checked = []
    for case in source["cases"]:
        n, scenario, seed = case["case"].split(":")
        model = approach_start(build_grid(int(n), scenario, int(seed)), scenario)
        directions = directions_for_five_cells(model)
        for stream in case["streams"]:
            for index in (0, len(stream["rows"]) // 2, len(stream["rows"]) - 1):
                row = stream["rows"][index]
                theta = np.asarray(row["theta"])
                changed = replace(model, cost=model.cost + theta @ directions,
                                  layers=model.layers)
                fresh = solve_cmdp(changed, search_mode="breakpoint",
                                   relative_tolerance=.001)
                assert fresh["certified"]
                assert fresh["objective"] <= row["lp_objective"] + 1e-5
                assert row["lp_objective"] <= fresh["upper"] + 1e-5
                checked.append({"case": case["case"], "stream": stream["kind"],
                                "index": index,
                                "fresh_objective": fresh["objective"],
                                "warm_lp_objective": row["lp_objective"],
                                "fresh_upper": fresh["upper"],
                                "seconds": fresh["seconds"]})
                print(case["case"], stream["kind"], index, "PASS", flush=True)
    (HERE / "paper_parameter_reference.json").write_text(
        json.dumps(checked, indent=2))


if __name__ == "__main__":
    main()
