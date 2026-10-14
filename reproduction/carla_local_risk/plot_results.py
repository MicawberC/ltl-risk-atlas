"""Plot the recorded CARLA risk trace and paired planning latency."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


here = Path(__file__).resolve().parent
data = json.loads((here / "benchmark.json").read_text(encoding="utf-8"))
rows = data["rows"]
t = np.array([r["time_s"] for r in rows])
theta = np.array([r["theta"] for r in rows])
atlas = np.array([1000 * r["atlas_s"] for r in rows])
lp = np.array([1000 * r["lp_s"] for r in rows])

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.1]})
for j in range(theta.shape[1]):
    axes[0].plot(t, 1 + theta[:, j], marker=".", lw=1.2,
                 label=f"crossing cell {j+1}")
axes[0].set_ylabel("Risk severity multiplier")
axes[0].legend(ncol=5, loc="upper center", fontsize=8)
axes[0].grid(alpha=0.25)
axes[1].plot(t, atlas, "o-", label="Policy atlas + expansion", color="#c04a25")
axes[1].plot(t, lp, "s-", label="Warm HiGHS LP", color="#245ba6")
axes[1].set_yscale("log")
axes[1].set_ylabel("Planning wall time (ms, log scale)")
axes[1].set_xlabel("CARLA simulation time (s)")
axes[1].legend(loc="upper center")
axes[1].grid(alpha=0.25, which="both")
fig.suptitle("Town01 pedestrian crossing: one observed risk-update stream")
fig.tight_layout()
fig.savefig(here / "carla_benchmark.png", dpi=170)
print(here / "carla_benchmark.png")
