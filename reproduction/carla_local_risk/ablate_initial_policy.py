"""Check whether one initial LP strategy suffices across the CARLA traces."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import benchmark  # sets the paper parameter environment and module path
from experiment import HARD, build_grid, evaluate_policy
from highs_reopt import prepare, query

HERE = Path(__file__).resolve().parent
FILES = ["carla_trace_20hz.json", "carla_trace_20hz_speed10_sideplus_full.json",
         "carla_trace_20hz_speed16_sideminus_full.json",
         "carla_trace_20hz_speed12_sideplus_full.json"]


def main():
    traces = [json.loads((HERE / file).read_text(encoding="utf-8"))
              for file in FILES]
    reference = json.loads((HERE / "multi_episode_lp_anchor_full.json").read_text(
        encoding="utf-8"))
    first = traces[0]
    model = build_grid(first["grid_n"], "carla_crossing",
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
    solver = prepare(model, directions)
    first_result = query(solver, np.zeros(len(anchor)))
    occupation = np.asarray(solver["solver"].getSolution().col_value[
        :solver["nx"]]).reshape(model.states, model.actions)
    totals = occupation.sum(axis=1)
    policy = np.divide(occupation, totals[:, None], out=np.zeros_like(occupation),
                       where=totals[:, None] > 1e-12)
    policy[totals <= 1e-12, 0] = 1.0
    assert first_result["basis_valid"]
    failures = risk_failures = 0
    maximum_loss = 0.0
    for trace, episode in zip(traces, reference["episodes"]):
        assert len(trace["rows"]) == len(episode["rows"])
        for observation, row in zip(trace["rows"], episode["rows"]):
            theta = np.asarray(observation["theta"], dtype=float) - anchor
            evaluated = evaluate_policy(
                replace(model, cost=model.cost + theta @ directions), policy)
            loss = row["lp_objective"] - evaluated["objective"]
            threshold = max(1e-4, .01 * abs(evaluated["objective"]))
            failures += loss > threshold
            risk_failures += evaluated["risk"] > HARD + 1e-7
            maximum_loss = max(maximum_loss, loss)
    result = {"samples": sum(len(t["rows"]) for t in traces),
              "initial_policy_objective_failures_1pct": int(failures),
              "initial_policy_hard_risk_failures": int(risk_failures),
              "maximum_absolute_objective_loss": maximum_loss}
    (HERE / "initial_policy_ablation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
