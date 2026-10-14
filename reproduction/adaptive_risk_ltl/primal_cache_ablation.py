"""Ablation: reuse nominal policies and mix them without any dual atlas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment import HARD, SOFT, build_grid
from joint_price_atlas import prepare, risk_directions

HERE = Path(__file__).resolve().parent


def best_primal(policies, theta):
    risk = [float(p["risk_coeff"][0] + p["risk_coeff"][1:] @ theta)
            for p in policies]
    candidates = [p["reward"] - max(0, risk[i] - SOFT)
                  for i, p in enumerate(policies) if risk[i] <= HARD]
    for i in range(len(policies)):
        for j in range(i):
            if abs(risk[i] - risk[j]) < 1e-12:
                continue
            for target in (SOFT, HARD):
                w = (target - risk[j]) / (risk[i] - risk[j])
                if -1e-9 <= w <= 1 + 1e-9:
                    w = np.clip(w, 0, 1)
                    reward = w * policies[i]["reward"] + (
                        1 - w) * policies[j]["reward"]
                    candidates.append(reward - max(0, target - SOFT))
    return max(candidates)


def main():
    warm = json.loads((HERE / "highs_reopt.json").read_text())
    warm.append(json.loads((HERE / "wide_middle_warm.json").read_text()))
    output = []
    for case in warm:
        n, scene, seed = case["case"].split(":")[:3]
        model = build_grid(int(n), scene, int(seed))
        atlas = prepare(model, risk_directions(model))
        rows = []
        for baseline in case["rows"]:
            theta = np.asarray(baseline["theta"])
            candidate = best_primal(atlas["policies"], theta)
            loss = baseline["objective"] - candidate
            assert loss >= -1e-6
            rows.append({"theta": theta.tolist(), "lp_objective": baseline["objective"],
                         "cached_objective": candidate, "loss": loss,
                         "within_one_percent": bool(loss <=
                             max(1e-4, .01 * abs(baseline["objective"])) + 1e-7)})
        output.append({"case": case["case"], "base_policies": len(atlas["policies"]),
                       "within_one_percent": sum(r["within_one_percent"] for r in rows),
                       "max_loss": max(r["loss"] for r in rows), "rows": rows})
    (HERE / "primal_cache_ablation.json").write_text(json.dumps(output, indent=2))
    for row in output:
        print(row["case"], row["within_one_percent"], "/25",
              "max_loss", row["max_loss"], flush=True)


if __name__ == "__main__":
    main()
