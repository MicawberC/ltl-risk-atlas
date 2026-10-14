"""Replay measured moving-CARLA observations under a fixed P and LTL labels.

Only the five risk-cost severities and the ego initial product state vary.
The walker locations come from a prior CARLA closed-loop experiment, so both
planners face identical observations.  This replay does not measure driving.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import benchmark  # paper gamma, soft and hard settings
from benchmark_dynamic_live_replay import risk_observation
from dynamic_certificate import policy_from_occupation
from dynamic_product import frame_parameters
from experiment import HARD, build_grid, evaluate_policy
from highs_reopt import prepare as prepare_lp, query as query_lp
from joint_price_atlas import (add_dual_neighbor_policies, expand_fast,
                               prepare as prepare_atlas, prepare_empty,
                               query as query_atlas, rebase_initial)
from stream_certificate import add_lp_anchor, query as query_hybrid


HERE = Path(__file__).resolve().parent


def run(reference: dict, live: dict, trial: int, hybrid: bool = False,
        legacy_expansion: bool = False, quiet: bool = False,
        lp_fallback: bool = False, lp_seed_policies: bool = False,
        lazy_atlas: bool = False, observed_prepare: bool = False) -> dict:
    measured = live["trials"][trial]["rows"]
    grid = int(reference["grid_n"])
    base = build_grid(grid, "carla_crossing",
                      labels_override=np.asarray(reference["labels"], dtype=np.int8),
                      lane_only=True)
    directions = np.stack([
        np.where(np.all(base.xy == cell, axis=1), base.cost, 0.0)
        for cell in reference["risk_cells"]])
    atlas = None
    lp = None
    fallback_lp = None
    rows = []
    for index, measured_row in enumerate(measured):
        obs = risk_observation(reference, measured_row)
        _, _, severity = frame_parameters(reference, obs)
        theta = np.asarray([
            severity[int(x + grid * y)] - 1.0
            for x, y in reference["risk_cells"]], dtype=float)
        x, y = map(int, measured_row["ego_xy"])
        occupied = int(min(obs["risk_distance_m"]) <= 2.5)
        start = (x + grid * y) * 8 + occupied * 4
        current = replace(base, initial=start,
                          cost=base.cost + theta @ directions)
        if atlas is None:
            atlas = (prepare_empty(replace(base, initial=start), directions)
                     if lazy_atlas else prepare_atlas(
                         replace(base, initial=start), directions,
                         reuse_oracles=not legacy_expansion,
                         observed_theta=theta if observed_prepare else None))
            lp = prepare_lp(replace(base, initial=start), directions)

        def atlas_call():
            nonlocal fallback_lp
            began = time.perf_counter()
            if start != atlas["model"].initial:
                atlas.pop("dominance_archive", None)
                atlas.pop("directional_archive", None)
                atlas.pop("primal_atoms", None)
            rebase = rebase_initial(atlas, start)
            atlas_query = query_hybrid if hybrid else query_atlas
            answer = atlas_query(atlas, theta, nearest_first=8)
            direct = bool(answer["certified"])
            if not direct:
                if lp_fallback:
                    if fallback_lp is None:
                        fallback_lp = prepare_lp(
                            replace(base, initial=start), directions)
                    solved = query_lp(fallback_lp, theta, initial=start)
                    if not solved["basis_valid"]:
                        raise RuntimeError("Fallback LP basis invalid")
                    occupation = np.asarray(
                        fallback_lp["solver"].getSolution().col_value[
                            :fallback_lp["nx"]]).reshape(
                                current.states, current.actions)
                    add_lp_anchor(atlas, theta, occupation,
                                  solved["objective"], solved["risk"])
                    # The LP risk multiplier identifies the scalarized
                    # Bellman tie. Solve on both sides to cache its two
                    # deterministic extreme policies for later queries.
                    if lp_seed_policies:
                        dual = np.asarray(
                            fallback_lp["solver"].getSolution().row_dual)
                        price = max(0.0, -float(dual[current.states]))
                        add_dual_neighbor_policies(atlas, theta, price)
                    # The LP upper anchor stays valid. Limit retained
                    # primals to keep pairwise mixture search bounded.
                    atoms = atlas.get("primal_atoms", [])
                    if len(atoms) > 8:
                        del atoms[:-8]
                    answer = {"certified": True,
                              "policy": policy_from_occupation(occupation),
                              "objective": solved["objective"],
                              "risk": solved["risk"],
                              "upper": solved["objective"] + 1e-5,
                              "gap": 1e-5,
                              "certificate_type": "lp_anchor"}
                else:
                    expand_fast(atlas, theta, nearest_first=8,
                                certifier=None if legacy_expansion or not hybrid
                                else query_hybrid,
                                reuse_oracles=not legacy_expansion)
                    answer = atlas_query(atlas, theta, nearest_first=8)
            if not answer["certified"]:
                raise RuntimeError("Atlas failed to certify moving-start query")
            return answer, direct, rebase, time.perf_counter() - began

        def lp_call():
            answer = query_lp(lp, theta, initial=start)
            if not answer["basis_valid"]:
                raise RuntimeError("LP basis invalid")
            return answer

        if index % 2:
            exact = lp_call()
            answer, direct, rebase_s, atlas_s = atlas_call()
        else:
            answer, direct, rebase_s, atlas_s = atlas_call()
            exact = lp_call()
        evaluated = evaluate_policy(current, answer["policy"])
        assert abs(evaluated["objective"] - answer["objective"]) < 1e-5
        assert abs(evaluated["risk"] - answer["risk"]) < 1e-5
        assert evaluated["risk"] <= HARD + 1e-7
        assert answer["objective"] <= exact["objective"] + 1e-5
        assert exact["objective"] <= answer["upper"] + 1e-5
        assert answer["upper"] - answer["objective"] <= max(
            1e-4, .01 * abs(answer["objective"])) + 1e-6
        rows.append({"index": index, "tick": measured_row["tick"],
                     "ego_xy": [x, y], "theta": theta.tolist(),
                     "start": start, "atlas_s": atlas_s,
                     "rebase_s": rebase_s, "warm_lp_s": exact["seconds"],
                     "direct_hit": direct,
                     "certificate_type": answer.get("certificate_type"),
                     "objective": answer["objective"],
                     "optimal_objective": exact["objective"],
                     "risk": answer["risk"], "upper": answer["upper"],
                     "atlas_forward": float(answer["policy"][start, 5])})
        if not quiet:
            print(index, "atlas", round(atlas_s, 3), "lp",
                  round(exact["seconds"], 3), "hit", direct, flush=True)
    at = np.asarray([r["atlas_s"] for r in rows])
    lp_times = np.asarray([r["warm_lp_s"] for r in rows])
    return {"scope": "fixed P/L/reward, moving ego initial state and five risk costs",
            "hybrid": hybrid,
            "legacy_expansion": legacy_expansion,
            "lp_fallback": lp_fallback,
            "lp_seed_policies": lp_seed_policies,
            "lazy_atlas": lazy_atlas,
            "observed_prepare": observed_prepare,
            "trial": trial, "source_method": live["trials"][trial]["method"],
            "states": base.states, "count": len(rows),
            "atlas_setup_s": atlas["prepare_seconds"],
            "lp_setup_s": lp["prepare_seconds"],
            "atlas_total_s": float(atlas["prepare_seconds"] + at.sum()),
            "lp_total_s": float(lp["prepare_seconds"] + lp_times.sum()),
            "atlas_p95_ms": float(np.percentile(at, 95) * 1000),
            "lp_p95_ms": float(np.percentile(lp_times, 95) * 1000),
            "atlas_max_ms": float(at.max() * 1000),
            "lp_max_ms": float(lp_times.max() * 1000),
            "rebase_total_s": sum(r["rebase_s"] for r in rows),
            "direct_hits": sum(r["direct_hit"] for r in rows),
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path,
                        default=HERE / "carla_trace.json")
    parser.add_argument("--live", type=Path,
                        default=HERE / "closed_loop_single_threeway.json")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--legacy-expansion", action="store_true")
    parser.add_argument("--lp-fallback", action="store_true")
    parser.add_argument("--lp-seed-policies", action="store_true")
    parser.add_argument("--lazy-atlas", action="store_true")
    parser.add_argument("--observed-prepare", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=HERE / "fixed_moving_replay.json")
    args = parser.parse_args()
    if args.lp_seed_policies and not args.lp_fallback:
        parser.error("--lp-seed-policies requires --lp-fallback")
    if args.lazy_atlas and not args.lp_seed_policies:
        parser.error("--lazy-atlas requires --lp-seed-policies")
    if args.lazy_atlas and args.observed_prepare:
        parser.error("--lazy-atlas and --observed-prepare are exclusive")
    result = run(json.loads(args.reference.read_text(encoding="utf-8")),
                 json.loads(args.live.read_text(encoding="utf-8")),
                 args.trial, args.hybrid, args.legacy_expansion,
                 args.quiet, args.lp_fallback, args.lp_seed_policies,
                 args.lazy_atlas, args.observed_prepare)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TOTAL", round(result["atlas_total_s"], 3),
          round(result["lp_total_s"], 3), "HITS", result["direct_hits"],
          flush=True)


if __name__ == "__main__":
    main()
