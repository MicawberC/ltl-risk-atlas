"""Recheck the final breakpoint/SCC solver against saved independent LP optima."""

from __future__ import annotations

import json
from pathlib import Path

from experiment import build_grid
from stratified import solve_cmdp

HERE = Path(__file__).resolve().parent


def main() -> None:
    checked = []
    for filename in ("varied_benchmark.json", "terrain_benchmark.json",
                     "hard_varied_benchmark.json", "hard_active_benchmark.json"):
        if not (HERE / filename).exists():
            continue
        for saved in json.loads((HERE / filename).read_text()):
            scenario = saved.get("scenario", "varied")
            model = build_grid(saved["grid"], scenario, saved["seed"])
            result = solve_cmdp(model, search_mode="breakpoint")
            lp = saved["sparse_lp_objective"]
            limit = max(1e-4, .01 * abs(result["objective"]))
            assert result["certified"]
            assert result["risk"] <= 5 + 1e-7
            assert result["gap"] <= limit + 1e-7
            assert result["objective"] <= lp + 1e-6
            assert lp <= result["upper"] + 1e-6
            checked.append({"scenario": scenario, "grid": saved["grid"],
                            "seed": saved["seed"], "lp_objective": lp,
                            "objective": result["objective"], "risk": result["risk"],
                            "upper": result["upper"], "gap": result["gap"],
                            "evaluations": result["evaluations"],
                            "seconds": result["seconds"]})
            print(scenario, saved["grid"], saved["seed"],
                  "certified", round(result["seconds"], 3), flush=True)
            (HERE / "heldout_final.json").write_text(json.dumps(checked, indent=2))
    print(f"PASS: {len(checked)} cases bound independent LP optima")


if __name__ == "__main__":
    main()
