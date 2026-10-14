"""Run a bounded, headless slice of the upstream IROS simulation on Windows.

The repository's LTLf2DFA/MONA bridge is POSIX-only. This wrapper supplies
exact DFAs for the two formulas hard-coded by the authors, then executes their
unchanged main script, including the product LP and MPC controller.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt  # noqa: E402

from iros_core import SOURCE, fixed_formula_dfa  # noqa: E402

sys.path.insert(0, str(SOURCE))
from specification.specification import LTL_Spec  # noqa: E402


def translate_fixed(self: LTL_Spec, formula: str):
    if formula == "F(t)":
        return fixed_formula_dfa("goal")
    if formula == "G(~g -> ~c) & G(~v)":
        return fixed_formula_dfa("safety")
    raise ValueError(f"No Windows fixture for {formula!r}")


class BoundedDemoFinished(Exception):
    pass


def main() -> None:
    LTL_Spec.translate = translate_fixed
    frames = 0
    max_frames = int(os.environ.get("IROS_FRAMES", "20"))

    def bounded_pause(_seconds: float) -> None:
        nonlocal frames
        frames += 1
        if frames >= max_frames:
            raise BoundedDemoFinished

    plt.pause = bounded_pause
    try:
        runpy.run_path(str(SOURCE / "reactive_LTL_main.py"), run_name="__main__")
    except BoundedDemoFinished:
        print(f"HEADLESS_DEMO_OK frames={frames}")


if __name__ == "__main__":
    main()
