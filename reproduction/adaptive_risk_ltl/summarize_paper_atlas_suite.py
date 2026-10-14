"""Build a compact, auditable summary of the fixed-product experiment JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "paper_suite")
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text(
        encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        for method, filename in case["files"].items():
            raw = json.loads((args.results / filename).read_text(
                encoding="utf-8"))
            rows.append({"case_index": case["index"], "case": raw["case"],
                         "stream": raw["stream"], "seed": raw["seed"],
                         "method": method, "states": raw["states"],
                         "updates": len(raw["rows"]),
                         "setup_s": raw["setup_seconds"],
                         "online_s": raw["online_seconds"],
                         "total_s": raw["total_seconds"],
                         "p95_ms": raw["p95_ms"], "max_ms": raw["max_ms"],
                         "peak_rss_mb": raw["peak_rss_mb"],
                         "cache_size": raw["cache_size"],
                         "direct_hits": raw["direct_hits"],
                         "fallbacks": raw["fallbacks"],
                         "within_one_percent": raw["within_one_percent"],
                         "hard_active_updates": raw["hard_active_updates"]})
    if not rows:
        raise ValueError("No result files listed in manifest")
    output = args.results / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} case-method summaries to {output}")


if __name__ == "__main__":
    main()
