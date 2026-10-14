"""Run the prespecified fixed-product atlas comparison in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASES = (
    ("24:middle_fast_crossing:0", "walk", 73044),
    ("24:middle_fast_crossing:0", "iid", 73044),
    ("32:middle_fast_crossing:0", "walk", 73044),
    ("32:middle_fast_crossing:0", "iid", 73044),
    ("24:middle_fast_varied:0", "iid", 73045),
    ("24:middle_fast_varied:17", "walk", 73046),
    ("16:barrier_hard_active:0", "iid", 73047),
    ("16:barrier_hard_active:1", "walk", 73048),
    ("24:barrier_hard_active:0", "iid", 73049),
    ("24:barrier_hard_active:1", "walk", 73050),
    ("24:barrier_hard_active:2", "iid", 73051),
    ("32:barrier_hard_active3:0", "walk", 73050),
)
DEFAULT_METHODS = ("warm_lp", "atlas", "monotone_lp", "forced_mdp",
                   "simple_cache", "hybrid_atlas")
METHODS = DEFAULT_METHODS + ("dual_atlas",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", nargs="*", type=int,
                        default=list(range(len(CASES))))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--index-only", action="store_true",
                        help="Rebuild the manifest from existing result files")
    parser.add_argument("--output-dir", type=Path, default=HERE / "paper_suite")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number in ([] if args.index_only else args.select):
        case, stream, seed = CASES[number]
        name = f"{number}_{case.replace(':', '_')}_{stream}_{seed}"
        reference = args.output_dir / f"{name}_warm_lp.json"
        for method in args.methods:
            out = args.output_dir / f"{name}_{method}.json"
            command = [sys.executable, str(HERE / "paper_atlas_comparison.py"),
                       "--case", case, "--stream", stream, "--seed", str(seed),
                       "--count", str(args.count), "--method", method,
                       "--output", str(out)]
            if method != "warm_lp":
                if not reference.exists():
                    raise FileNotFoundError(reference)
                command += ["--reference", str(reference)]
            subprocess.run(command, check=True)
    manifest = {"cases": [], "count": args.count, "methods": list(METHODS)}
    for number, (case, stream, seed) in enumerate(CASES):
        name = f"{number}_{case.replace(':', '_')}_{stream}_{seed}"
        files = {method: f"{name}_{method}.json" for method in METHODS
                 if (args.output_dir / f"{name}_{method}.json").exists()}
        if files:
            manifest["cases"].append({"index": number, "case": case,
                                      "stream": stream, "seed": seed,
                                      "files": files})
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
