"""Small closed-loop CARLA integration test for dynamic LTL-CMDP updates.

This is a lane-only high-level model: action 5 means advance, the other action
indices mean wait.  A simple speed controller maps that choice to CARLA.  The
CMDP certificate concerns the grid model, not continuous vehicle dynamics.
The car is controlled in CARLA, so simulated goal time is measured here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import carla
import numpy as np
import psutil

import benchmark  # configure gamma=0.8, soft=1, hard=2
from capture_crossing import add, choose_road
from dynamic_certificate import policy_from_occupation, query
from dynamic_product import (frame_parameters, full_rebuild,
                             initial_snapshot, patch)
from dynamic_warm_lp import (prepare as prepare_warm_lp,
                             update_and_query as query_warm_lp)
from experiment import HARD, build_grid, evaluate_policy, solve_lp
from highs_reopt import prepare as prepare_fixed_lp, query as query_fixed_lp
from joint_price_atlas import (add_dual_neighbor_policies,
                               expand_fast as expand_fixed_atlas,
                               prepare as prepare_fixed_atlas,
                               prepare_empty as prepare_empty_fixed_atlas,
                               query as query_fixed_atlas,
                               rebase_initial)
from paper_atlas_comparison import atom, best_atom_mix, mixed_policy
from stream_certificate import (add_lp_anchor,
                                query as query_fixed_hybrid)


HERE = Path(__file__).resolve().parent


def road_labels(world: carla.World, center: carla.Location,
                forward: carla.Vector3D, right: carla.Vector3D,
                grid: int = 20) -> tuple[list[int], list[list[int]], int]:
    labels = [0] * (grid * grid)
    road_map = world.get_map()
    drivable = 0
    for y in range(grid):
        for x in range(grid):
            point = add(add(center, forward, (x - 10) * 2.0),
                        right, (y - grid // 2) * 2.0)
            if road_map.get_waypoint(point, project_to_road=False,
                                     lane_type=carla.LaneType.Driving) is None:
                labels[x + grid * y] = 4
            else:
                drivable += 1
    risk_cells = [[x, grid // 2] for x in range(8, 13)]
    for x, y in risk_cells:
        if labels[x + grid * y] == 4:
            raise RuntimeError("Crossing cell is outside CARLA driving lane")
        labels[x + grid * y] = 1
    if labels[14 + grid * (grid // 2)] == 4:
        raise RuntimeError("Goal cell is outside CARLA driving lane")
    labels[14 + grid * (grid // 2)] = 2
    return labels, risk_cells, drivable


def observation(walkers: list[carla.Actor], center: carla.Location,
                forward: carla.Vector3D, right: carla.Vector3D,
                risk_cells: list[list[int]]) -> dict:
    positions = [walker.get_location() for walker in walkers]
    distances = []
    for x, y in risk_cells:
        cell = add(add(center, forward, (x - 10) * 2.0), right, (y - 10) * 2.0)
        distances.append(min(math.hypot(p.x - cell.x, p.y - cell.y)
                             for p in positions))
    return {"risk_distance_m": distances,
            "walkers_xyz": [[p.x, p.y, p.z] for p in positions]}


def start_state(ego: carla.Actor, center: carla.Location,
                forward: carla.Vector3D, right: carla.Vector3D,
                observation: dict) -> tuple[int, list[int], float]:
    location = ego.get_location()
    dx, dy = location.x - center.x, location.y - center.y
    longitudinal = dx * forward.x + dy * forward.y
    lateral = dx * right.x + dy * right.y
    x = int(np.clip(round(10 + longitudinal / 2.0), 0, 19))
    y = int(np.clip(round(10 + lateral / 2.0), 0, 19))
    occupied = int(min(observation["risk_distance_m"]) <= 2.5)
    return (x + 20 * y) * 8 + occupied * 4, [x, y], longitudinal


def run_trial(world: carla.World, method: str, crossing: carla.Waypoint,
              spawn: carla.Transform, labels: list[int],
              risk_cells: list[list[int]], max_ticks: int,
              plan_every: int, second_walker: bool,
              verify: bool = False, sample_policy: bool = False,
              seed: int = 0, action_every: int | None = None,
              reward_scale: float = 1.0,
              latch_cell_action: bool = False) -> dict:
    if action_every is None:
        action_every = plan_every
    if action_every % plan_every:
        raise ValueError("action_every must be a multiple of plan_every")
    wall_started = time.perf_counter()
    center = crossing.transform.location
    forward = crossing.transform.get_forward_vector()
    right = crossing.transform.get_right_vector()
    actors = []
    collision_frames = []
    action_rng = np.random.default_rng(seed)
    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / 2**20
    try:
        bp_lib = world.get_blueprint_library()
        vehicles = sorted(bp_lib.filter("vehicle.*"), key=lambda bp: bp.id)
        ego_bp = next((bp for bp in vehicles if bp.id == "vehicle.tesla.model3"),
                      vehicles[0])
        ego = world.try_spawn_actor(ego_bp, spawn)
        if ego is None:
            raise RuntimeError("Unable to spawn ego")
        actors.append(ego)
        sensor_bp = bp_lib.find("sensor.other.collision")
        sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=ego)
        actors.append(sensor)
        sensor.listen(lambda event: collision_frames.append({
            "frame": event.frame,
            "other_actor": event.other_actor.type_id,
            "impulse": event.normal_impulse.length()}))
        walker_bp = sorted(bp_lib.filter("walker.pedestrian.*"),
                           key=lambda bp: bp.id)[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        first_start = add(center, right, -5.0)
        first_start.z += 1.0
        first = world.try_spawn_actor(
            walker_bp, carla.Transform(first_start, carla.Rotation()))
        if first is None:
            raise RuntimeError("Unable to spawn first walker")
        actors.append(first)
        first.apply_control(carla.WalkerControl(
            direction=carla.Vector3D(right.x, right.y, 0.0), speed=1.3))
        walkers = [first]
        # Newly spawned actors have their final CARLA transforms after a tick.
        world.tick()
        trace_stub = {"grid_n": 20, "labels": labels,
                      "risk_cells": risk_cells}
        fixed_methods = ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                         "fixed_observed_hybrid",
                         "fixed_dual_atlas",
                         "fixed_warm_lp", "fixed_mono")
        snapshot = (None if method in fixed_methods else
                    initial_snapshot(np.asarray(labels, dtype=np.int8), 20,
                                     lane_only=True))
        fixed_product_build_s = 0.0
        fixed_base = None
        fixed_directions = None
        fixed_atlas = None
        fixed_lp = None
        mono_atoms = []
        mono_anchors = []
        mono_start = None
        if method in ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                      "fixed_observed_hybrid",
                      "fixed_dual_atlas",
                      "fixed_warm_lp", "fixed_mono"):
            build_started = time.perf_counter()
            fixed_base = build_grid(20, "carla_crossing",
                                    labels_override=np.asarray(labels,
                                                               dtype=np.int8),
                                    lane_only=True)
            if reward_scale != 1.0:
                fixed_base = replace(fixed_base,
                                     reward=reward_scale * fixed_base.reward)
            fixed_directions = np.stack([
                np.where(np.all(fixed_base.xy == cell, axis=1),
                         fixed_base.cost, 0.0)
                for cell in risk_cells])
            fixed_product_build_s = time.perf_counter() - build_started
        anchor = None
        warm_cache = None
        go = False
        go_target_x = None
        rows = []
        goal_tick = None
        planner_failure = None
        for tick in range(max_ticks):
            if second_walker and tick == 100:
                second_start = add(add(center, forward, 3.0), right, 5.0)
                second_start.z += 1.0
                second = world.try_spawn_actor(
                    walker_bp, carla.Transform(second_start, carla.Rotation()))
                if second is None:
                    raise RuntimeError("Unable to spawn second walker")
                actors.append(second)
                second.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(-right.x, -right.y, 0.0),
                    speed=1.2))
                walkers.append(second)
            if tick % plan_every == 0:
                obs = observation(walkers, center, forward, right, risk_cells)
                state, xy, longitudinal = start_state(
                    ego, center, forward, right, obs)
                labels_now, env, severity = frame_parameters(trace_stub, obs)
                print(method, "tick", tick, "ego", xy,
                      "risk_distance_m", [round(v, 2) for v in obs["risk_distance_m"]],
                      flush=True)
                began = time.perf_counter()
                previous_model = None if snapshot is None else snapshot.model
                if method in ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                              "fixed_observed_hybrid",
                              "fixed_dual_atlas",
                              "fixed_warm_lp", "fixed_mono"):
                    theta = np.asarray([
                        severity[x + 20 * y] - 1.0 for x, y in risk_cells])
                    current_model = replace(
                        fixed_base, initial=state,
                        cost=fixed_base.cost + theta @ fixed_directions)
                    product_s = time.perf_counter() - began
                    try:
                        began_plan = time.perf_counter()
                        if method in ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                                      "fixed_observed_hybrid",
                                      "fixed_dual_atlas"):
                            if fixed_atlas is None:
                                prepare_fn = (prepare_empty_fixed_atlas
                                              if method == "fixed_dual_atlas"
                                              else prepare_fixed_atlas)
                                fixed_atlas = prepare_fn(
                                    replace(fixed_base, initial=state),
                                    fixed_directions,
                                    **({"observed_theta": theta}
                                       if method == "fixed_observed_hybrid"
                                       else {}))
                            else:
                                if state != fixed_atlas["model"].initial:
                                    # Cached upper bounds are start-dependent.
                                    fixed_atlas.pop("dominance_archive", None)
                                    fixed_atlas.pop("directional_archive", None)
                                    fixed_atlas.pop("primal_atoms", None)
                                rebase_initial(fixed_atlas, state)
                            atlas_query = (query_fixed_hybrid
                                           if method in ("fixed_hybrid", "fixed_graph_lp",
                                                         "fixed_observed_hybrid",
                                                         "fixed_dual_atlas")
                                           else query_fixed_atlas)
                            answer = atlas_query(
                                fixed_atlas, theta, nearest_first=8)
                            direct = bool(answer["certified"])
                            if not direct:
                                if method in ("fixed_graph_lp", "fixed_dual_atlas"):
                                    if fixed_lp is None:
                                        fixed_lp = prepare_fixed_lp(
                                            replace(fixed_base, initial=state),
                                            fixed_directions)
                                    solved = query_fixed_lp(fixed_lp, theta,
                                                            initial=state)
                                    if not solved["basis_valid"]:
                                        raise RuntimeError("LP anchor basis invalid")
                                    occupation = np.asarray(
                                        fixed_lp["solver"].getSolution().col_value[
                                            :fixed_lp["nx"]]).reshape(
                                                current_model.states,
                                                current_model.actions)
                                    add_lp_anchor(fixed_atlas, theta, occupation,
                                                  solved["objective"], solved["risk"])
                                    if method == "fixed_dual_atlas":
                                        dual = np.asarray(
                                            fixed_lp["solver"].getSolution().row_dual)
                                        price = max(0.0, -float(dual[current_model.states]))
                                        add_dual_neighbor_policies(
                                            fixed_atlas, theta, price)
                                        atoms = fixed_atlas.get("primal_atoms", [])
                                        if len(atoms) > 8:
                                            del atoms[:-8]
                                    answer = {"certified": True,
                                              "policy": policy_from_occupation(occupation),
                                              "objective": solved["objective"],
                                              "risk": solved["risk"],
                                              "upper": solved["objective"] + 1e-5,
                                              "gap": 1e-5}
                                else:
                                    expand_fixed_atlas(fixed_atlas, theta,
                                                       nearest_first=8,
                                                       certifier=(
                                                           query_fixed_hybrid
                                                           if method in ("fixed_hybrid",
                                                                         "fixed_observed_hybrid")
                                                           else None))
                                    answer = atlas_query(
                                        fixed_atlas, theta, nearest_first=8)
                            if not answer["certified"]:
                                raise RuntimeError("Fixed atlas not certified")
                            policy = answer["policy"]
                            objective, risk = answer["objective"], answer["risk"]
                            gap = answer["gap"]
                            fallback = not direct
                        else:
                            if fixed_lp is None:
                                fixed_lp = prepare_fixed_lp(
                                    replace(fixed_base, initial=state),
                                    fixed_directions)
                            if method == "fixed_mono" and mono_start != state:
                                mono_atoms.clear()
                                mono_anchors.clear()
                                mono_start = state
                            primal = (best_atom_mix(mono_atoms, theta)
                                      if method == "fixed_mono" and mono_atoms
                                      else None)
                            bounds = [upper for old, upper in mono_anchors
                                      if np.all(old <= theta + 1e-12)]
                            hit = False
                            if primal is not None and bounds:
                                upper = min(bounds)
                                objective, risk = float(primal[0]), float(primal[1])
                                gap = upper - objective
                                hit = (risk <= HARD + 1e-8 and
                                       -1e-6 <= gap <=
                                       max(1e-4, .01 * abs(objective)) + 1e-7)
                            if hit:
                                policy = mixed_policy(mono_atoms, primal)
                                fallback = False
                            else:
                                solved = query_fixed_lp(fixed_lp, theta,
                                                        initial=state)
                                if not solved["basis_valid"]:
                                    raise RuntimeError("Fixed LP basis invalid")
                                occupation = np.asarray(
                                    fixed_lp["solver"].getSolution().col_value[
                                        :fixed_lp["nx"]]).reshape(
                                            current_model.states,
                                            current_model.actions)
                                policy = policy_from_occupation(occupation)
                                objective, risk = solved["objective"], solved["risk"]
                                upper = objective + 1e-5
                                gap, fallback = 0.0, True
                                if method == "fixed_mono":
                                    mono_atoms.append(atom(occupation, fixed_base,
                                                           fixed_directions))
                                    mono_anchors.append((theta.copy(), upper))
                        plan_s = time.perf_counter() - began_plan
                    except RuntimeError as exc:
                        planner_failure = f"{type(exc).__name__}: {exc}"
                        rows.append({"tick": tick, "sim_time_s": tick * .05,
                                     "ego_xy": xy,
                                     "risk_distance_m": obs["risk_distance_m"],
                                     "planner_failure": planner_failure})
                        break
                elif method == "full_lp":
                    snapshot = full_rebuild(labels_now, env, severity, 20,
                                            state, lane_only=True)
                else:
                    snapshot = patch(snapshot, labels_now, env, severity,
                                     state)
                if method not in ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                                  "fixed_observed_hybrid",
                                  "fixed_dual_atlas",
                                  "fixed_warm_lp", "fixed_mono"):
                    product_s = time.perf_counter() - began
                    current_model = snapshot.model
                if method in ("fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                              "fixed_observed_hybrid",
                              "fixed_dual_atlas",
                              "fixed_warm_lp", "fixed_mono"):
                    pass  # fixed-product planner and timing were computed above
                elif method in ("full_lp", "patched_lp"):
                    solved = solve_lp(snapshot.model)
                    plan_s = solved["wall_seconds"]
                    policy = policy_from_occupation(solved["occupation"])
                    objective, risk = solved["objective"], float(
                        np.sum(solved["occupation"] *
                               snapshot.model.cost[:, None]))
                    fallback = True
                    gap = 0.0
                elif method == "warm_lp":
                    if warm_cache is None:
                        warm_cache = prepare_warm_lp(snapshot.model)
                        solved = query_warm_lp(warm_cache, snapshot.model,
                                               snapshot.model)
                        plan_s = (warm_cache["prepare_seconds"] +
                                  solved["total_seconds"])
                    else:
                        solved = query_warm_lp(warm_cache, previous_model,
                                               snapshot.model)
                        plan_s = solved["total_seconds"]
                    policy = policy_from_occupation(solved["occupation"])
                    objective, risk = solved["objective"], solved["risk"]
                    fallback = True
                    gap = 0.0
                else:
                    answer, anchor = query(snapshot.model, anchor)
                    plan_s = answer["seconds"]
                    policy = answer["policy"]
                    objective, risk = answer["objective"], answer["risk"]
                    fallback = answer["lp_fallback"]
                    gap = answer["gap"]
                    assert risk <= HARD + 1e-6
                if verify:
                    # Verification is excluded from method timing and pauses
                    # the synchronous CARLA world at this tick.
                    evaluated = evaluate_policy(current_model, policy)
                    assert abs(evaluated["objective"] - objective) < 1e-4, (
                        method, tick, "policy objective", evaluated["objective"],
                        objective)
                    assert abs(evaluated["risk"] - risk) < 1e-4, (
                        method, tick, "policy risk", evaluated["risk"], risk)
                    reference = solve_lp(current_model)
                    assert objective <= reference["objective"] + 1e-4
                    if method in ("incremental", "fixed_atlas", "fixed_hybrid",
                                  "fixed_observed_hybrid",
                                  "fixed_graph_lp", "fixed_dual_atlas",
                                  "fixed_mono"):
                        bound = answer["upper"] if method != "fixed_mono" else upper
                        assert reference["objective"] <= bound + 1e-4
                    else:
                        assert abs(objective - reference["objective"]) < 1e-4
                forward_probability = float(policy[state, 5])
                action_due = (tick % action_every == 0 if not latch_cell_action
                              else (xy[0] >= go_target_x if go_target_x is not None
                                    else tick % action_every == 0))
                if action_due:
                    go = (bool(action_rng.random() < forward_probability)
                          if sample_policy else forward_probability >= 0.5)
                    go_target_x = xy[0] + 1 if latch_cell_action and go else None
                peak_rss_mb = max(peak_rss_mb,
                                  process.memory_info().rss / 2**20)
                rows.append({"tick": tick, "sim_time_s": tick * .05,
                             "ego_xy": xy, "longitudinal_m": longitudinal,
                             "walkers_xyz": obs["walkers_xyz"],
                             "risk_distance_m": obs["risk_distance_m"],
                             "changed_product_rows": (0 if fixed_base is not None
                                                      else snapshot.changed_product_rows),
                             "forward_probability": forward_probability,
                             "go": go, "go_target_x": go_target_x,
                             "product_s": product_s,
                             "plan_s": plan_s, "lp_fallback": fallback,
                             "objective": objective, "risk": risk,
                             "gap": gap})
            speed = ego.get_velocity().length()
            if go:
                throttle = float(np.clip(0.30 * (3.0 - speed), 0.0, 0.5))
                control = carla.VehicleControl(throttle=throttle, brake=0.0,
                                               steer=0.0)
            else:
                control = carla.VehicleControl(throttle=0.0, brake=1.0,
                                               steer=0.0)
            ego.apply_control(control)
            world.tick()
            _, _, longitudinal = start_state(
                ego, center, forward, right,
                observation(walkers, center, forward, right, risk_cells))
            if longitudinal >= 7.5:
                goal_tick = tick + 1
                break
            if collision_frames:
                break
        return {"method": method, "goal": goal_tick is not None,
                "sample_policy": sample_policy, "action_seed": seed,
                "reward_scale": reward_scale,
                "latch_cell_action": latch_cell_action,
                "action_every_ticks": action_every,
                "planner_failure": planner_failure,
                "collision": bool(collision_frames),
                "collision_events": collision_frames,
                "goal_sim_time_s": None if goal_tick is None else goal_tick * .05,
                "elapsed_sim_time_s": (tick + 1) * .05,
                "planning_time_s": (fixed_product_build_s +
                                    sum(r["product_s"] + r["plan_s"]
                                        for r in rows if "plan_s" in r)),
                "fixed_product_build_s": fixed_product_build_s,
                "peak_process_rss_mb": peak_rss_mb,
                "wall_time_s": time.perf_counter() - wall_started,
                "lp_fallbacks": sum(r.get("lp_fallback", False) for r in rows),
                "planning_calls": len(rows), "rows": rows}
    finally:
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        world.tick()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town", default="Town01",
                        help="CARLA map used for the crossing trial")
    parser.add_argument("--route-index", type=int, default=0,
                        help="Index of eligible straight route on the map")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Auxiliary fixed-model goal reward multiplier")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--max-ticks", type=int, default=260)
    parser.add_argument("--plan-every", type=int, default=10)
    parser.add_argument("--action-every", type=int,
                        help="Hold a sampled action this many simulator ticks")
    parser.add_argument("--second-walker", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Check every incremental result by an independent LP")
    parser.add_argument("--sample-policy", action="store_true",
                        help="Sample the forward/wait action from the policy")
    parser.add_argument("--latch-cell-action", action="store_true",
                        help="Complete a sampled grid-advance before resampling")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--methods", nargs="+",
                        choices=("full_lp", "patched_lp", "warm_lp", "incremental",
                                 "fixed_atlas", "fixed_hybrid", "fixed_graph_lp",
                                 "fixed_observed_hybrid",
                                 "fixed_dual_atlas",
                                 "fixed_warm_lp",
                                 "fixed_mono"),
                        default=("full_lp", "patched_lp", "warm_lp", "incremental"))
    parser.add_argument("--output", type=Path,
                        default=HERE / "closed_loop_dynamic.json")
    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if not world.get_map().name.endswith("/" + args.town):
        world = client.load_world(args.town)
    original = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        world.apply_settings(settings)
        spawn, crossing, _ = choose_road(world, args.route_index)
        forward = crossing.transform.get_forward_vector()
        right = crossing.transform.get_right_vector()
        labels, risk_cells, drivable = road_labels(
            world, crossing.transform.location, forward, right)
        result = {"simulator": client.get_server_version(),
                  "town": world.get_map().name,
                  "route_index": args.route_index,
                  "crossing_xyz": [crossing.transform.location.x,
                                   crossing.transform.location.y,
                                   crossing.transform.location.z],
                  "scope": "lane-only high-level LTL product; fixed_* modes keep P and labels constant",
                  "controller": ("sample forward/wait from policy" if args.sample_policy
                                 else "forward if action 5 probability >= 0.5"),
                  "drivable_cells": drivable,
                  "max_ticks": args.max_ticks,
                  "plan_every_ticks": args.plan_every,
                  "action_every_ticks": args.action_every or args.plan_every,
                  "sample_policy": args.sample_policy,
                  "reward_scale": args.reward_scale,
                  "latch_cell_action": args.latch_cell_action,
                  "second_walker": args.second_walker, "trials": []}
        for repetition in range(args.repeats):
            methods = list(args.methods)
            if repetition % 2:
                methods.reverse()
            for method in methods:
                trial = run_trial(world, method, crossing, spawn, labels,
                                  risk_cells, args.max_ticks, args.plan_every,
                                  args.second_walker, verify=args.verify,
                                  sample_policy=args.sample_policy,
                                  seed=5000 + repetition,
                                  action_every=args.action_every,
                                  reward_scale=args.reward_scale,
                                  latch_cell_action=args.latch_cell_action)
                trial["repetition"] = repetition
                result["trials"].append(trial)
                args.output.write_text(json.dumps(result, indent=2),
                                       encoding="utf-8")
                print(method, "rep", repetition, "goal", trial["goal"],
                      "collision", trial["collision"],
                      "planner_failure", trial["planner_failure"],
                      "sim_s", trial["goal_sim_time_s"], "plan_s",
                      round(trial["planning_time_s"], 3), "wall_s",
                      round(trial["wall_time_s"], 3), flush=True)
    finally:
        world.apply_settings(original)


if __name__ == "__main__":
    main()
