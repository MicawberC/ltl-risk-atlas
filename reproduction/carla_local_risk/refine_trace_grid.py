"""Project captured CARLA risk streams onto a finer fixed product grid.

This is a discretization stress test of the same captured observations, not a
new CARLA capture.  Nearest-neighbor labels keep the captured road geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def refine(trace, n, max_frames=None):
    old_n = int(trace["grid_n"])
    if n <= old_n:
        raise ValueError("new grid must be finer")
    old_labels = np.asarray(trace["labels"], dtype=np.int8).reshape(old_n, old_n)
    axes = np.minimum(old_n - 1, np.floor(
        (np.arange(n) + .5) * old_n / n).astype(int))
    labels = old_labels[np.ix_(axes, axes)]

    def center_map(cell):
        return [min(n - 1, int(round((float(v) + .5) * n / old_n - .5)))
                for v in cell]

    out = trace.copy()
    out["grid_n"] = n
    out["cell_m"] = float(trace["cell_m"]) * old_n / n
    out["crossing_xy"] = center_map(trace["crossing_xy"])
    out["goal_xy"] = center_map(trace["goal_xy"])
    out["risk_cells"] = [center_map(cell) for cell in trace["risk_cells"]]
    if len({tuple(cell) for cell in out["risk_cells"]}) != len(out["risk_cells"]):
        raise ValueError("risk cells collide after refinement")
    out["labels"] = labels.ravel().tolist()
    out["rows"] = trace["rows"][:max_frames]
    out["provenance"] = {"kind": "grid_refinement_of_captured_CARLA_trace",
                         "source_grid_n": old_n,
                         "source_cell_m": trace["cell_m"],
                         "new_CARLA_capture": False}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("traces", type=Path, nargs="+")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for source in a.traces:
        data = json.loads(source.read_text(encoding="utf-8"))
        result = refine(data, a.n, a.max_frames)
        target = a.output_dir / f"{source.stem}_grid{a.n}.json"
        target.write_text(json.dumps(result, separators=(",", ":")),
                          encoding="utf-8")
        print(target, len(result["rows"]))


if __name__ == "__main__":
    main()
