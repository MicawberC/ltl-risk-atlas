"""Render the saved benchmark data; does not rerun any solver."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> None:
    timing = read("timing_strict.json") + read("timing_large32.json")
    large = {record["scenario"]: record for record in read("large32.json")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    ax = axes[0]
    for scenario, marker, color in [("near", "o", "#1f77b4"),
                                    ("barrier_hard", "s", "#c44e52")]:
        rows = sorted((r for r in timing if r["case"].endswith(":" + scenario)),
                      key=lambda r: r["states"])
        states = [r["states"] for r in rows]
        ax.plot(states, [r["sparse_lp_median_seconds"] for r in rows],
                marker=marker, color=color, linestyle="--", label=scenario + " / sparse LP")
        ax.plot(states, [r["guided_median_seconds"] for r in rows],
                marker=marker, color=color, linestyle="-", label=scenario + " / guided")
    ax.set_xlabel("Full product states")
    ax.set_ylabel("Median solve time (s)")
    ax.set_title("Repeated, interleaved timings")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    row = large["near"]
    methods = {m["strategy"]: m for m in row["methods"]}
    names = ["guided", "occupancy", "model_error"]
    counts = [methods[name]["final_states"] for name in names]
    ax.bar(names, counts, color=["#1f77b4", "#55a868", "#8172b3"])
    for x, count in enumerate(counts):
        ax.text(x, count + 35, str(count), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set_ylabel("Abstract states at certificate")
    ax.set_title("32×32 near, 4% split fraction")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(HERE / "scaling.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
