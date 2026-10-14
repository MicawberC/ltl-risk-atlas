"""Summarize paired offline CARLA two-walker planning replays."""

import json
from pathlib import Path

folder = Path(__file__).resolve().parent
for path in sorted(folder.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in data:
        continue
    rows = data["rows"]
    gaps = [max(0.0, r["upper"] - r["objective"])
            / max(1e-4, abs(r["objective"])) for r in rows]
    objective_loss = [max(0.0, r["optimal_objective"] - r["objective"])
                      / max(1e-4, abs(r["optimal_objective"])) for r in rows]
    print(path.name,
          "count", data["count"],
          "atlas", round(data["atlas_total_s"], 4),
          "warm_lp", round(data["lp_total_s"], 4),
          "ratio", round(data["lp_total_s"] / data["atlas_total_s"], 3),
          "misses", data["count"] - data["direct_hits"],
          "max_risk", round(max(r["risk"] for r in rows), 6),
          "max_gap_ratio", round(max(gaps), 6),
          "max_objective_loss", round(max(objective_loss), 6),
          "p95_ms", round(data["atlas_p95_ms"], 2),
          "max_ms", round(data["atlas_max_ms"], 2))
