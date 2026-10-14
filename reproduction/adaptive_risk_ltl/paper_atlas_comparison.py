"""Paired fixed-product risk streams with strong and simple alternatives.

Run each method in a separate process for a comparable process RSS peak.  The
same deterministic risk trace is regenerated from the case and seed.  The LP
reference is run first and its JSON supplied to every other method.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ["LTL_RISK_GAMMA"] = "0.8"
os.environ["LTL_RISK_SOFT"] = "1.0"
os.environ["LTL_RISK_HARD"] = "2.0"

import numpy as np
import psutil

from experiment import HARD, SOFT, build_grid, evaluate_policy
from highs_reopt import prepare as prepare_lp, query as query_lp
from joint_price_atlas import (add_dual_neighbor_policies, add_policy,
                               expand_fast, prepare as prepare_atlas,
                               prepare_empty, query as query_atlas)
from local_update_gate import approach_start, risk_trace
from primal_cache_ablation import best_primal
from stratified import solve_scalarized
from stream_certificate import (add_lp_anchor,
                                query as query_hybrid_atlas)


def make_case(name: str):
    n_text, scenario, seed_text = name.split(":")
    n, seed = int(n_text), int(seed_text)
    base_scenario = ("barrier_hard_active" if scenario ==
                     "barrier_hard_active3" else scenario)
    model = build_grid(n, base_scenario, seed)
    if scenario in ("middle_fast_crossing", "middle_fast_varied"):
        model = approach_start(model, scenario)
        positions = np.unique(model.xy[model.cost > 0], axis=0)
        assert len(positions) == 5
    elif scenario in ("barrier_hard_active", "barrier_hard_active3"):
        # Explicit auxiliary stress: larger goal rewards make H=2 binding
        # on some updates.  Only five nearby risk cells vary.
        scale = 3.0 if scenario.endswith("active3") else 4.0
        model = replace(model, reward=scale * model.reward)
        left = 2 + seed % 2
        positions = np.asarray([(left, 0), (left + 1, 0),
                                (left, 1), (left + 1, 1), (left, 2)])
    else:
        raise ValueError(scenario)
    directions = np.stack([
        np.where(np.all(model.xy == position, axis=1), model.cost, 0.0)
        for position in positions
    ])
    assert np.all(directions >= 0) and np.all(directions.max(axis=1) > 0)
    return model, directions, positions.tolist()


def atom(occupation, model, directions):
    occupancy = np.asarray(occupation).reshape(model.states, model.actions)
    state_occupancy = occupancy.sum(axis=1)
    return {"occupation": occupancy.copy(),
            "reward": float(np.sum(occupancy * model.reward)),
            "risk_coeff": np.r_[state_occupancy @ model.cost,
                                 directions @ state_occupancy]}


def best_atom_mix(atoms, theta):
    risks = [float(a["risk_coeff"][0] + a["risk_coeff"][1:] @ theta)
             for a in atoms]
    candidates = []
    for i, a in enumerate(atoms):
        if risks[i] <= HARD + 1e-8:
            candidates.append((a["reward"] - max(0, risks[i] - SOFT),
                               risks[i], i, i, 1.0))
        for j in range(i):
            if abs(risks[i] - risks[j]) < 1e-12:
                continue
            for target in (SOFT, HARD):
                weight = (target - risks[j]) / (risks[i] - risks[j])
                if -1e-9 <= weight <= 1 + 1e-9:
                    weight = float(np.clip(weight, 0, 1))
                    mixed_risk = weight * risks[i] + (1 - weight) * risks[j]
                    if mixed_risk > HARD + 1e-9:
                        continue
                    reward = weight * a["reward"] + (1 - weight) * atoms[j]["reward"]
                    candidates.append((reward - max(0, mixed_risk - SOFT),
                                       mixed_risk, i, j, weight))
    return max(candidates) if candidates else None


def mixed_policy(atoms, primal):
    _, _, i, j, weight = primal
    occupancy = (weight * atoms[i]["occupation"] +
                 (1 - weight) * atoms[j]["occupation"])
    total = occupancy.sum(axis=1)
    policy = np.divide(occupancy, total[:, None], out=np.zeros_like(occupancy),
                       where=total[:, None] > 1e-12)
    policy[total <= 1e-12, 0] = 1.0
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--stream", choices=("walk", "iid"), required=True)
    parser.add_argument("--seed", type=int, default=73044)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--method", choices=("warm_lp", "atlas", "monotone_lp",
                                              "forced_mdp", "simple_cache",
                                              "hybrid_atlas", "dual_atlas",
                                              "observed_hybrid"),
                        required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, directions, positions = make_case(args.case)
    thetas = risk_trace(args.stream, args.count, args.seed,
                        len(directions))
    reference = (json.loads(args.reference.read_text(encoding="utf-8"))
                 if args.reference else None)
    if args.method != "warm_lp" and reference is None:
        raise ValueError("Supply the warm_lp reference for verification")
    if reference is not None:
        assert (reference["case"], reference["stream"], reference["seed"]) == (
            args.case, args.stream, args.seed)
        assert np.allclose(thetas, reference["thetas"])

    process = psutil.Process()
    rss_peak = process.memory_info().rss
    setup_start = time.perf_counter()
    if args.method in ("warm_lp", "monotone_lp"):
        cache = prepare_lp(model, directions)
        if args.method == "monotone_lp":
            initial_occ = np.asarray(cache["solver"].getSolution().col_value[
                :cache["nx"]]).reshape(model.states, model.actions)
            atoms = [atom(initial_occ, model, directions)]
            anchors = [(np.zeros(len(directions)),
                        -cache["solver"].getObjectiveValue() + 1e-5)]
    elif args.method == "dual_atlas":
        atlas = prepare_empty(model, directions)
        dual_lp = None
    elif args.method == "observed_hybrid":
        atlas = prepare_atlas(model, directions,
                              observed_theta=thetas[0])
    else:
        atlas = prepare_atlas(model, directions)
        nominal = tuple(atlas["policies"])
    setup_seconds = time.perf_counter() - setup_start
    rows = []
    for index, theta in enumerate(thetas):
        started = time.perf_counter()
        fallback = False
        hit = False
        answer = None
        if args.method == "warm_lp":
            answer = query_lp(cache, theta)
            assert answer["basis_valid"]
            objective, risk = float(answer["objective"]), float(answer["risk"])
            upper = objective + 1e-5
        elif args.method == "simple_cache":
            objective = float(best_primal(nominal, theta))
            risk, upper = None, None
        elif args.method == "monotone_lp":
            primal = best_atom_mix(atoms, theta)
            bounds = [ub for old, ub in anchors if np.all(old <= theta + 1e-12)]
            if primal is not None and bounds:
                objective, risk = float(primal[0]), float(primal[1])
                upper = min(bounds)
                tol = max(1e-4, .01 * abs(objective)) + 1e-7
                hit = (risk <= HARD + 1e-8 and
                       -1e-6 <= upper - objective <= tol)
            if not hit:
                fallback = True
                solved = query_lp(cache, theta)
                assert solved["basis_valid"]
                objective, risk = float(solved["objective"]), float(solved["risk"])
                upper = objective + 1e-5
                occupation = np.asarray(cache["solver"].getSolution().col_value[
                    :cache["nx"]]).reshape(model.states, model.actions)
                atoms.append(atom(occupation, model, directions))
                anchors.append((theta.copy(), upper))
            else:
                policy = mixed_policy(atoms, primal)
        else:
            if args.method == "forced_mdp":
                neighbors = sorted(atlas["policies"], key=lambda p: float(
                    np.linalg.norm(p["source_theta"] - theta)))
                price = next((p["source_price"] for p in neighbors
                              if p["source_price"] > 1e-10), 1.0)
                changed = replace(model, cost=model.cost + theta @ directions)
                oracle = solve_scalarized(changed, price)
                add_policy(atlas, oracle, theta)
            atlas_query = (query_hybrid_atlas if args.method in (
                               "hybrid_atlas", "dual_atlas",
                               "observed_hybrid")
                           else query_atlas)
            answer = atlas_query(atlas, theta, nearest_first=8)
            hit = bool(answer["certified"])
            if not hit:
                if args.method == "dual_atlas":
                    fallback = True
                    if dual_lp is None:
                        dual_lp = prepare_lp(model, directions)
                    solved = query_lp(dual_lp, theta)
                    assert solved["basis_valid"]
                    occupation = np.asarray(
                        dual_lp["solver"].getSolution().col_value[
                            :dual_lp["nx"]]).reshape(model.states,
                                                      model.actions)
                    add_lp_anchor(atlas, theta, occupation,
                                  solved["objective"], solved["risk"])
                    dual = np.asarray(
                        dual_lp["solver"].getSolution().row_dual)
                    price = max(0.0, -float(dual[model.states]))
                    add_dual_neighbor_policies(atlas, theta, price)
                    atoms = atlas.get("primal_atoms", [])
                    if len(atoms) > 8:
                        del atoms[:-8]
                    totals = occupation.sum(axis=1)
                    policy = np.divide(occupation, totals[:, None],
                                       out=np.zeros_like(occupation),
                                       where=totals[:, None] > 1e-12)
                    policy[totals <= 1e-12, 0] = 1.0
                    # On the 8,193-state LP, flow/occupation rounding can
                    # change the evaluated stationary policy by ~7e-5.
                    # Include that numerical margin in the LP anchor.
                    answer = {"certified": True, "policy": policy,
                              "objective": solved["objective"],
                              "risk": solved["risk"],
                              "upper": solved["objective"] + 1e-4}
                else:
                    extension = expand_fast(
                        atlas, theta, nearest_first=8,
                        certifier=(query_hybrid_atlas
                                   if args.method == "observed_hybrid"
                                   else None))
                    fallback = bool(extension["fallback"])
                    answer = atlas_query(atlas, theta, nearest_first=8)
            if not answer["certified"]:
                raise RuntimeError(
                    f"No certified strategy for {args.case} query {index}")
            objective, risk, upper = (float(answer["objective"]),
                                      float(answer["risk"]), float(answer["upper"]))
            policy = answer["policy"]
        elapsed = time.perf_counter() - started
        rss_peak = max(rss_peak, process.memory_info().rss)
        exact = reference["rows"][index]["objective"] if reference else objective
        policy_objective_error = None
        policy_risk_error = None
        if args.method != "simple_cache":
            assert risk <= HARD + 1e-6
            assert objective <= exact + 1e-5
            assert exact <= upper + 1e-5
            assert upper - objective <= max(1e-4, .01 * abs(objective)) + 1e-6
            if args.method in ("atlas", "forced_mdp", "hybrid_atlas",
                               "dual_atlas", "observed_hybrid") or (
                    args.method == "monotone_lp" and hit):
                changed = replace(model, cost=model.cost + theta @ directions)
                evaluated = evaluate_policy(changed, policy)
                policy_objective_error = abs(evaluated["objective"] - objective)
                policy_risk_error = abs(evaluated["risk"] - risk)
                numerical_tolerance = (1e-4 if args.method == "dual_atlas"
                                       and not hit else 1e-5)
                assert policy_objective_error < numerical_tolerance, (
                    index, args.method, "objective", evaluated["objective"],
                    objective, hit)
                assert policy_risk_error < numerical_tolerance, (
                    index, args.method, "risk", evaluated["risk"], risk, hit)
        rows.append({"index": index, "theta": theta.tolist(),
                     "seconds": elapsed, "objective": objective,
                     "risk": risk, "upper": upper, "direct_hit": hit,
                     "fallback": fallback,
                     "policy_objective_error": policy_objective_error,
                     "policy_risk_error": policy_risk_error,
                     "within_one_percent": bool(exact - objective <=
                         max(1e-4, .01 * abs(exact)) + 1e-7),
                     "hard_active": bool(risk is not None and
                                         abs(risk - HARD) < 1e-5)})
    times = np.asarray([r["seconds"] for r in rows])
    result = {"case": args.case, "stream": args.stream, "seed": args.seed,
              "method": args.method, "states": model.states,
              "risk_cells": positions, "thetas": thetas.tolist(),
              "setup_seconds": setup_seconds,
              "online_seconds": float(times.sum()),
              "total_seconds": float(setup_seconds + times.sum()),
              "p95_ms": float(np.percentile(times, 95) * 1000),
              "max_ms": float(times.max() * 1000),
              "peak_rss_mb": rss_peak / 2**20,
              "cache_size": len(atlas["policies"]) if args.method in (
                  "atlas", "forced_mdp", "simple_cache", "hybrid_atlas",
                  "dual_atlas", "observed_hybrid") else
                  len(atoms) if args.method == "monotone_lp" else 1,
              "direct_hits": sum(r["direct_hit"] for r in rows),
              "fallbacks": sum(r["fallback"] for r in rows),
              "within_one_percent": sum(r["within_one_percent"] for r in rows),
              "hard_active_updates": sum(r["hard_active"] for r in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.case, args.stream, args.seed, args.method,
          "total", round(result["total_seconds"], 3),
          "hits", result["direct_hits"], "quality",
          result["within_one_percent"], "hard", result["hard_active_updates"],
          flush=True)


if __name__ == "__main__":
    main()
