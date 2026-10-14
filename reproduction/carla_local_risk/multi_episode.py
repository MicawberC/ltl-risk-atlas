"""Paired CARLA replay across independently captured pedestrian crossings.

Both methods keep their data structures across episodes on one fixed road
model.  All setup and all strategy expansions are counted once in cumulative
wall time.  The later traces differ in speed, direction, and crossing offset.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import benchmark  # sets paper parameter environment and module path
from experiment import HARD, build_grid, evaluate_policy
from highs_reopt import prepare as prepare_lp, query as query_lp
from joint_price_atlas import (add_policy, expand_adaptive,
                               prepare as prepare_atlas)
from local_residual import cached_primal
from local_policy_repair import attempt as attempt_local_repair
from local_reduced_lp import attempt as attempt_reduced_lp_repair
from stratified import solve_scalarized
from stream_certificate import (add_lp_anchor, prepare_reach_upper,
                                query as query_atlas)


HERE = Path(__file__).resolve().parent
DEFAULT_TRACES = [
    HERE / "carla_trace_20hz.json",
    HERE / "carla_trace_20hz_speed10_sideplus.json",
    HERE / "carla_trace_20hz_speed16_sideminus.json",
    HERE / "carla_trace_20hz_speed12_sideplus.json",
]


def summarize(rows):
    at = np.array([r["atlas_s"] for r in rows])
    lp = np.array([r["lp_s"] for r in rows])
    return {"samples": len(rows),
            "direct_hits": sum(r["direct_hit"] for r in rows),
            "full_fallbacks": sum(r["full_fallback"] for r in rows),
            "lp_fallbacks": sum(r["lp_fallback"] for r in rows),
            "atlas_online_s": float(at.sum()),
            "lp_online_s": float(lp.sum()),
            "atlas_p50_ms": float(1000 * np.percentile(at, 50)),
            "atlas_p95_ms": float(1000 * np.percentile(at, 95)),
            "atlas_max_ms": float(1000 * at.max()),
            "lp_p50_ms": float(1000 * np.percentile(lp, 50)),
            "lp_p95_ms": float(1000 * np.percentile(lp, 95)),
            "lp_max_ms": float(1000 * lp.max()),
            "max_objective_difference": max(abs(r["atlas_objective"] -
                                                r["lp_objective"]) for r in rows),
            "max_certificate_gap": max(r["gap"] for r in rows),
            "max_risk": max(r["atlas_risk"] for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path,
                        default=HERE / "multi_episode.json")
    parser.add_argument("--freeze-after-first", action="store_true",
                        help="Use an independent warm LP fallback after first episode")
    parser.add_argument("--lp-anchor-fallback", action="store_true",
                        help="Cache exact LP misses as reusable upper and primal anchors")
    parser.add_argument("--floor-anchor", action="store_true",
                        help="Precompute the physical minimum-risk anchor (-0.8 per cell)")
    parser.add_argument("--directional-upper", action="store_true",
                        help="Use a reachable-state risk sensitivity upper bound")
    parser.add_argument("--residual-upper", action="store_true",
                        help="Try the cached Bellman residual upper certificate")
    parser.add_argument("--targeted-policy", action="store_true",
                        help="Try one warm scalarized policy before LP fallback")
    parser.add_argument("--low-rank-repair", action="store_true",
                        help="Try changing a few policy actions with Woodbury updates")
    parser.add_argument("--reduced-lp-repair", action="store_true",
                        help="Optimize occupation over a few action changes")
    parser.add_argument("--repair-max-changes", type=int, default=4)
    parser.add_argument("--repair-max-bases", type=int, default=2)
    parser.add_argument("--repair-max-candidates", type=int)
    parser.add_argument("--repair-gap-factor", type=float, default=float("inf"),
                        help="Only repair when a dominated upper bound is within this multiple of 1% tolerance")
    parser.add_argument("--targeted-gap-factor", type=float,
                        default=float("inf"),
                        help="Only generate a policy if old upper/lower gap is within this multiple of 1% tolerance")
    parser.add_argument("--lookahead-anchor", type=int, default=0,
                        help="On decreasing risk, place a certified LP anchor this many observed steps ahead")
    parser.add_argument("--lookahead-max-risk", type=float,
                        help="Only forecast when every raw risk coefficient is at most this value")
    parser.add_argument("--prewarm-center", action="store_true",
                        help="Precompute one generic pedestrian-at-center risk anchor")
    args = parser.parse_args()
    if args.low_rank_repair and args.reduced_lp_repair:
        raise ValueError("Choose one local repair method")
    traces = [json.loads(path.read_text(encoding="utf-8"))
              for path in args.traces]
    first = traces[0]
    if not all(t["labels"] == first["labels"] and
               t["risk_cells"] == first["risk_cells"] and
               t["grid_n"] == first["grid_n"] for t in traces):
        raise ValueError("Episodes must share one fixed CARLA map/product model")
    n = first["grid_n"]
    model = build_grid(n, "carla_crossing",
                       labels_override=np.asarray(first["labels"], dtype=np.int8))
    x0, y0 = first["crossing_xy"][0] - 4, first["crossing_xy"][1]
    initial = np.flatnonzero(np.all(model.xy == [x0, y0], axis=1) &
                             (model.mode == 1) & (model.goal == 0) &
                             (model.bad == 0))
    assert len(initial) == 1
    model = replace(model, initial=int(initial[0]))
    directions = np.stack([
        np.where(np.all(model.xy == cell, axis=1), model.cost, 0.0)
        for cell in first["risk_cells"]])
    anchor = np.asarray(first["rows"][0]["theta"], dtype=float)
    model = replace(model, cost=model.cost + anchor @ directions)
    atlas = prepare_atlas(model, directions)
    reach_s = 0.0
    if args.directional_upper:
        reach_s = prepare_reach_upper(atlas)
        atlas["prepare_seconds"] += reach_s
    prewarm_s = 0.0
    if args.prewarm_center:
        began = time.perf_counter()
        center_theta = np.array([
            0.2 + 1.6 * np.exp(-0.5 * (
                abs(cell[0] - first["crossing_xy"][0]) * first["cell_m"]
                / 3.0) ** 2) - 1.0
            for cell in first["risk_cells"]]) - anchor
        trial = query_atlas(atlas, center_theta)
        if not trial["certified"]:
            expand_adaptive(atlas, center_theta)
            trial = query_atlas(atlas, center_theta)
        if not trial["certified"]:
            raise RuntimeError("Center prewarming failed")
        prewarm_s = time.perf_counter() - began
        atlas["prepare_seconds"] += prewarm_s
    lp = prepare_lp(model, directions)
    hybrid_lp = (prepare_lp(model, directions)
                 if args.freeze_after_first or args.lp_anchor_fallback else None)
    floor_s = 0.0
    if args.floor_anchor:
        if hybrid_lp is None:
            raise ValueError("--floor-anchor requires --lp-anchor-fallback")
        began_floor = time.perf_counter()
        floor_theta = np.full(len(anchor), -.8) - anchor
        floor = query_lp(hybrid_lp, floor_theta)
        floor_occupation = np.asarray(
            hybrid_lp["solver"].getSolution().col_value[:hybrid_lp["nx"]]
        ).reshape(model.states, model.actions)
        floor_duals = hybrid_lp["solver"].getSolution().row_dual
        floor_price = -(floor_duals[model.states + len(anchor) + 1] +
                        floor_duals[model.states + len(anchor) + 2])
        add_lp_anchor(atlas, floor_theta, floor_occupation,
                      floor["objective"], floor["risk"], price=floor_price)
        floor_s = time.perf_counter() - began_floor
        atlas["prepare_seconds"] += floor_s
    if args.lookahead_anchor < 0 or args.lookahead_anchor > 8:
        raise ValueError("lookahead anchor must be between 0 and 8")
    if args.lookahead_anchor and hybrid_lp is None:
        raise ValueError("lookahead anchor requires --lp-anchor-fallback")
    episodes = []
    all_rows = []
    for episode_index, (path, trace) in enumerate(zip(args.traces, traces)):
        rows = []
        last_theta = None
        for sample, obs in enumerate(trace["rows"]):
            theta = np.asarray(obs["theta"], dtype=float) - anchor

            def atlas_call():
                began = time.perf_counter()
                answer = query_atlas(atlas, theta,
                                     use_residual=args.residual_upper)
                direct = bool(answer["certified"])
                extension = None
                used_lp_fallback = False
                targeted_added = False
                targeted_certified = False
                forecast_solved = False
                repair_attempted = False
                repair_certified = False
                repair_changed = 0
                repair_info = None
                repair_pre_gap_ratio = None
                if not direct:
                    if args.low_rank_repair or args.reduced_lp_repair:
                        should_repair = np.isinf(args.repair_gap_factor)
                        if not should_repair:
                            repair_primal = cached_primal(atlas, theta)
                            repair_uppers = [upper for old_theta, upper in
                                             atlas.get("dominance_archive", ())
                                             if np.all(old_theta <= theta + 1e-12)]
                            if repair_primal is not None and repair_uppers:
                                repair_lower = repair_primal[0]
                                repair_tol = max(1e-4, .01 * abs(repair_lower)) + 1e-7
                                repair_pre_gap_ratio = ((min(repair_uppers) -
                                                         repair_lower) / repair_tol)
                                should_repair = (min(repair_uppers) - repair_lower <=
                                                 args.repair_gap_factor * repair_tol)
                        if should_repair:
                            repair_attempted = True
                            repair_fn = (attempt_reduced_lp_repair if
                                         args.reduced_lp_repair else
                                         attempt_local_repair)
                            repair_kwargs = {"max_changes": args.repair_max_changes}
                            if args.low_rank_repair:
                                repair_kwargs["max_bases"] = args.repair_max_bases
                                repair_kwargs["max_candidate_states"] = (
                                    args.repair_max_candidates)
                            repair = repair_fn(atlas, theta, **repair_kwargs)
                            repair_info = {key: repair.get(key) for key in (
                                "reason", "seconds", "changed_states", "gap",
                                "changed_state_indices", "changed_coordinates",
                                "max_bellman_residual", "bound_source",
                                "reduced_lp_variables", "candidate_states",
                                "largest_excluded_score")}
                            repair_certified = bool(repair["certified"])
                            repair_changed = repair.get("changed_states", 0)
                            if repair_certified:
                                answer = repair
                if not answer["certified"]:
                    near_gap = False
                    if args.targeted_policy and np.isfinite(
                            args.targeted_gap_factor):
                        current_primal = cached_primal(atlas, theta)
                        dominated = [upper for anchor_theta, upper in
                                     atlas.get("dominance_archive", ())
                                     if np.all(anchor_theta <= theta + 1e-12)]
                        if current_primal is not None and dominated:
                            lower = current_primal[0]
                            tolerance = max(1e-4, .01 * abs(lower)) + 1e-7
                            near_gap = (min(dominated) - lower <=
                                        args.targeted_gap_factor * tolerance)
                    if args.targeted_policy and (
                            near_gap or np.isinf(args.targeted_gap_factor)):
                        price = atlas.get("last_lp_price", atlas.get(
                            "stream_price", 1.0))
                        neighbor = min(atlas["policies"], key=lambda p: float(
                            np.linalg.norm(p["source_theta"] - theta)))
                        changed_model = replace(
                            model, cost=model.cost + theta @ directions)
                        solved_policy = solve_scalarized(
                            changed_model, price,
                            initial_action=neighbor["action"])
                        targeted_added = add_policy(atlas, solved_policy, theta)
                        answer = query_atlas(atlas, theta,
                                             use_residual=args.residual_upper)
                        targeted_certified = bool(answer["certified"])
                if not answer["certified"]:
                    if (args.lookahead_anchor and last_theta is not None and
                            np.all(theta <= last_theta + 1e-12) and
                            np.any(theta < last_theta - 1e-8) and
                            (args.lookahead_max_risk is None or
                             np.max(theta + anchor) <= args.lookahead_max_risk)):
                        forecast_theta = np.maximum(
                            np.full(len(anchor), -.8) - anchor,
                            theta + args.lookahead_anchor * (theta - last_theta))
                        forecast = query_lp(hybrid_lp, forecast_theta)
                        forecast_occ = np.asarray(
                            hybrid_lp["solver"].getSolution().col_value[
                                :hybrid_lp["nx"]]).reshape(model.states,
                                                            model.actions)
                        forecast_duals = hybrid_lp["solver"].getSolution().row_dual
                        forecast_price = -(
                            forecast_duals[model.states + len(anchor) + 1] +
                            forecast_duals[model.states + len(anchor) + 2])
                        add_lp_anchor(atlas, forecast_theta, forecast_occ,
                                      forecast["objective"], forecast["risk"],
                                      price=forecast_price)
                        forecast_solved = True
                        answer = query_atlas(atlas, theta,
                                             use_residual=args.residual_upper)
                if not answer["certified"]:
                    if hybrid_lp is not None and (args.lp_anchor_fallback or
                                                  episode_index > 0):
                        solved = query_lp(hybrid_lp, theta)
                        assert solved["basis_valid"]
                        occupation = np.asarray(hybrid_lp["solver"].getSolution().col_value[
                            :hybrid_lp["nx"]]).reshape(model.states, model.actions)
                        row_duals = hybrid_lp["solver"].getSolution().row_dual
                        price = -(row_duals[model.states + len(anchor) + 1] +
                                  row_duals[model.states + len(anchor) + 2])
                        atlas["last_lp_price"] = max(0.0, float(price))
                        totals = occupation.sum(axis=1)
                        policy = np.divide(occupation, totals[:, None],
                                           out=np.zeros_like(occupation),
                                           where=totals[:, None] > 1e-12)
                        policy[totals <= 1e-12, 0] = 1.0
                        answer = {"certified": True,
                                  "objective": solved["objective"],
                                  "risk": solved["risk"],
                                  "upper": solved["objective"] + 1e-5,
                                  "gap": 1e-5, "policy": policy,
                                  "certificate_type": "lp_oracle_anchor" if
                                  args.lp_anchor_fallback else "warm_lp_fallback"}
                        if args.lp_anchor_fallback:
                            add_lp_anchor(atlas, theta, occupation,
                                          solved["objective"], solved["risk"],
                                          price=price)
                        used_lp_fallback = True
                    else:
                        extension = expand_adaptive(atlas, theta)
                        answer = query_atlas(atlas, theta)
                if not answer["certified"]:
                    raise RuntimeError(f"Uncertified episode {episode_index}, sample {sample}")
                return (answer, direct, extension, used_lp_fallback,
                        targeted_added, targeted_certified,
                        forecast_solved, repair_attempted,
                        repair_certified, repair_changed, repair_info,
                        repair_pre_gap_ratio,
                        time.perf_counter() - began)

            if sample % 2:
                exact = query_lp(lp, theta)
                (answer, direct, extension, used_lp_fallback, targeted_added,
                 targeted_certified, forecast_solved, repair_attempted,
                 repair_certified, repair_changed, repair_info,
                 repair_pre_gap_ratio,
                 atlas_s) = atlas_call()
            else:
                (answer, direct, extension, used_lp_fallback, targeted_added,
                 targeted_certified, forecast_solved, repair_attempted,
                 repair_certified, repair_changed, repair_info,
                 repair_pre_gap_ratio,
                 atlas_s) = atlas_call()
                exact = query_lp(lp, theta)
            assert exact["basis_valid"]
            changed = replace(model, cost=model.cost + theta @ directions)
            evaluated = evaluate_policy(changed, answer["policy"])
            assert abs(evaluated["objective"] - answer["objective"]) < 1e-6
            assert abs(evaluated["risk"] - answer["risk"]) < 1e-6
            assert answer["risk"] <= HARD + 1e-7
            assert exact["risk"] <= HARD + 1e-6
            assert answer["objective"] <= exact["objective"] + 1e-5
            assert exact["objective"] <= answer["upper"] + 1e-5
            assert answer["gap"] <= max(1e-4, .01 * abs(answer["objective"])) + 1e-7
            row = {"episode": episode_index, "sample": sample,
                   "time_s": obs["time_s"],
                   "theta": obs["theta"],
                   "atlas_s": atlas_s, "lp_s": exact["seconds"],
                   "atlas_objective": answer["objective"],
                   "lp_objective": exact["objective"],
                   "atlas_risk": answer["risk"], "lp_risk": exact["risk"],
                   "upper": answer["upper"], "gap": answer["gap"],
                   "direct_hit": direct,
                   "lp_fallback": used_lp_fallback,
                   "targeted_added": targeted_added,
                   "targeted_certified": targeted_certified,
                   "forecast_solved": forecast_solved,
                   "repair_attempted": repair_attempted,
                   "repair_certified": repair_certified,
                   "repair_changed_states": repair_changed,
                   "repair_info": repair_info,
                   "repair_pre_gap_ratio": repair_pre_gap_ratio,
                   "full_fallback": bool(extension and extension["fallback"]),
                   "expansion": extension,
                   "certificate_type": answer.get("certificate_type"),
                   "lp_iterations": exact["iterations"],
                   "lp_basis_reset": exact.get("basis_reset", False)}
            rows.append(row)
            all_rows.append(row)
            last_theta = theta.copy()
        episode_result = {"trace": str(path),
                          "walker_speed_mps": trace.get("walker_speed_mps", 1.3),
                          "walker_start_side": trace.get("walker_start_side", -1),
                          "crossing_offset_m": trace.get("crossing_offset_m", 0.0),
                          "summary": summarize(rows), "rows": rows}
        episodes.append(episode_result)
        print("episode", episode_index,
              "hits", episode_result["summary"]["direct_hits"],
              "atlas_online_s", round(episode_result["summary"]["atlas_online_s"], 3),
              "lp_online_s", round(episode_result["summary"]["lp_online_s"], 3),
              flush=True)
        args.output.write_text(json.dumps({"states": model.states,
                                           "atlas_setup_s": atlas["prepare_seconds"] +
                                           (hybrid_lp["prepare_seconds"] if hybrid_lp else 0.0),
                                           "lp_setup_s": lp["prepare_seconds"],
                                           "episodes": episodes}, indent=2),
                               encoding="utf-8")
    overall = summarize(all_rows)
    hybrid_setup = hybrid_lp["prepare_seconds"] if hybrid_lp else 0.0
    overall["atlas_total_s"] = (atlas["prepare_seconds"] + hybrid_setup +
                                 overall["atlas_online_s"])
    overall["lp_total_s"] = lp["prepare_seconds"] + overall["lp_online_s"]
    overall["total_speedup_lp_over_atlas"] = (
        overall["lp_total_s"] / overall["atlas_total_s"])
    result = {"scope": "same fixed Town01 product model, four actual CARLA traces",
              "method": ("monotone risk certificate with LP anchor oracle" if
                         args.lp_anchor_fallback else
                         "monotone risk certificate with adaptive atlas expansion"),
              "variants": {"floor_anchor": args.floor_anchor,
                           "low_rank_repair": args.low_rank_repair,
                           "reduced_lp_repair": args.reduced_lp_repair,
                           "repair_max_changes": args.repair_max_changes,
                           "repair_max_bases": args.repair_max_bases,
                           "repair_max_candidates": args.repair_max_candidates,
                           "repair_gap_factor": (args.repair_gap_factor if
                                                 np.isfinite(args.repair_gap_factor)
                                                 else None),
                           "directional_upper": args.directional_upper,
                           "residual_upper": args.residual_upper,
                           "targeted_policy": args.targeted_policy,
                           "targeted_gap_factor": (args.targeted_gap_factor if
                                                   np.isfinite(args.targeted_gap_factor)
                                                   else None),
                           "lookahead_anchor": args.lookahead_anchor,
                           "lookahead_max_risk": args.lookahead_max_risk},
              "states": model.states,
              "atlas_setup_s": atlas["prepare_seconds"] + hybrid_setup,
              "hybrid_lp_setup_s": hybrid_setup,
              "prewarm_center_s": prewarm_s,
              "floor_anchor_s": floor_s,
              "directional_reach_setup_s": reach_s,
              "reach_upper": atlas.get("reach_upper", np.array([])).tolist(),
              "lp_setup_s": lp["prepare_seconds"],
              "anchor_theta": anchor.tolist(),
              "overall": overall, "episodes": episodes}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
