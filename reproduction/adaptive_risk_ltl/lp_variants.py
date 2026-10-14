"""Compare supported sparse HiGHS variants on identical product LPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment import build_grid, solve_lp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["24:near", "24:barrier_hard", "24:sequence", "32:near", "32:barrier_hard"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("lp_variants.json"))
    args = parser.parse_args()
    output = []
    for case in args.cases:
        n_text, scenario = case.split(":")
        model = build_grid(int(n_text), scenario)
        results = []
        for method in ("highs", "highs-ds", "highs-ipm"):
            try:
                result = solve_lp(model, method=method)
                results.append({"method": method, "status": "ok",
                                "seconds": result["wall_seconds"],
                                "objective": result["objective"]})
            except RuntimeError as error:
                results.append({"method": method, "status": "failed",
                                "error": str(error)})
            print(case, results[-1], flush=True)
        output.append({"case": case, "states": model.states, "results": results})
        args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
