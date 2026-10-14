"""Cross-check the two-sided directional upper bound against independent LPs."""

from __future__ import annotations

import numpy as np

from experiment import build_grid
from highs_reopt import prepare as prepare_lp, query as query_lp
from joint_price_atlas import prepare as prepare_atlas, risk_directions
from stream_certificate import prepare_reach_upper


def main():
    model = build_grid(4, "near", 0)
    directions = risk_directions(model)
    atlas = prepare_atlas(model, directions)
    prepare_reach_upper(atlas)
    lp = prepare_lp(model, directions)
    anchor = np.zeros(len(directions))
    solved = query_lp(lp, anchor)
    row_dual = lp["solver"].getSolution().row_dual
    soft_row = model.states + len(directions) + 1
    price = -(row_dual[soft_row] + row_dual[soft_row + 1])
    assert price >= -1e-8
    upper_at_anchor = solved["objective"] + 1e-5
    for theta in np.random.default_rng(5).uniform(
            -.1, .1, size=(40, len(directions))):
        upper = upper_at_anchor + max(0.0, price) * (
            atlas["reach_upper"] @ np.maximum(anchor - theta, 0.0))
        exact = query_lp(lp, theta)["objective"]
        assert exact <= upper + 1e-6, (theta, exact, upper)
    print("40 independent LP upper-bound checks passed")


if __name__ == "__main__":
    main()
