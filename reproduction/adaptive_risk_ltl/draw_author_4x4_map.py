"""Draw the initial 4x4 map from the released IROS minimal example.

Source: papers/Risk_LTL_Planning-main/reactive_LTL_main.py.  This is an
explanatory reconstruction of the code labels, not a screenshot from CARLA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("white")

    # Static proposition c: intersection cells controlled by the light.
    intersection = [(2, 2), (3, 2), (2, 3)]
    for x, y in intersection:
        ax.add_patch(Rectangle((5 * x, 5 * y), 5, 5,
                               facecolor="#f5a7a2", edgecolor="none", alpha=.7))

    # Dynamic labels at the initial opposing-car position (12, 7.5).
    # v1 marks the anticipated occupancy band; v0 is the current car cell.
    ax.add_patch(Rectangle((5, 5), 10, 5, facecolor="#ffd56e",
                           edgecolor="none", alpha=.72))
    ax.add_patch(Rectangle((10, 5), 5, 5, facecolor="#ed665f",
                           edgecolor="#b73832", linewidth=2, alpha=.8))

    # Target area t and initial ego position (2.5, 2.5).
    ax.add_patch(Rectangle((15, 15), 5, 5, facecolor="#7ebce8",
                           edgecolor="none", alpha=.85))
    ax.scatter([2.5], [2.5], s=180, marker="o", color="#248755", zorder=5)

    for coord in range(0, 21, 5):
        ax.plot([coord, coord], [0, 20], color="#4f5963", linewidth=1)
        ax.plot([0, 20], [coord, coord], color="#4f5963", linewidth=1)

    labels = {
        (2, 3): "C", (3, 3): "T", (2, 2): "C", (3, 2): "C",
        (1, 1): "V1", (2, 1): "V0 + V1",
    }
    for (x, y), label in labels.items():
        ax.text(5 * x + 2.5, 5 * y + 2.5, label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if label == "V0 + V1" else "#183047", zorder=6)
    ax.text(2.5, 3.5, "Ego", ha="center", va="bottom", fontsize=11,
            color="#17613a", fontweight="bold", zorder=6)

    ax.set_xticks([2.5, 7.5, 12.5, 17.5], labels=["x=0", "x=1", "x=2", "x=3"])
    ax.set_yticks([2.5, 7.5, 12.5, 17.5], labels=["y=0", "y=1", "y=2", "y=3"])
    ax.tick_params(length=0, pad=8)
    ax.set_xlim(-.25, 20.25)
    ax.set_ylim(-.25, 20.25)
    ax.set_aspect("equal")
    ax.set_title("Released 4x4 IROS demo: initial labeled map", fontsize=15, pad=18)
    ax.text(22, 18.5, "Traffic light\ninitially RED", fontsize=11,
            color="#ad2929", fontweight="bold", va="top")
    ax.text(22, 14.2, "Ego: starting cell\nT: destination\nC: intersection\nV0: other car now\nV1: anticipated car area",
            fontsize=10.5, color="#243442", va="top", linespacing=1.6)
    ax.text(0, -2.3, "Each cell is 5 m x 5 m. Labels show an initial snapshot, not an optimal route.",
            fontsize=10, color="#53616d")
    ax.set_xlim(-.25, 35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.08, right=.97, top=.9, bottom=.13)
    out = Path(__file__).with_name("author_4x4_initial_map.png")
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
