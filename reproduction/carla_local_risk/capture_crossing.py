"""Collect a deterministic CARLA pedestrian-crossing risk trace.

The ego car is held at the approach point. This produces simulator-observed
pedestrian positions for a repeated-planning benchmark, not a driving trial.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import carla


HERE = Path(__file__).resolve().parent


def add(a: carla.Location, v: carla.Vector3D, scale: float) -> carla.Location:
    return carla.Location(x=a.x + scale * v.x, y=a.y + scale * v.y,
                          z=a.z + scale * v.z)


def choose_road(world: carla.World, route_index: int = 0):
    if route_index < 0:
        raise ValueError("route_index must be nonnegative")
    road_map = world.get_map()
    found = 0
    for transform in road_map.get_spawn_points():
        start = road_map.get_waypoint(transform.location)
        if start.is_junction:
            continue
        mid_candidates = start.next(8.0)
        if not mid_candidates:
            continue
        mid = mid_candidates[0]
        goals = mid.next(8.0)
        if not goals or mid.is_junction or goals[0].is_junction:
            continue
        heading = abs((goals[0].transform.rotation.yaw -
                       start.transform.rotation.yaw + 180) % 360 - 180)
        if heading < 5.0:
            if found == route_index:
                return transform, mid, goals[0]
            found += 1
    raise RuntimeError(f"Map has only {found} suitable 16 m straight routes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=160)
    parser.add_argument("--sample-every", type=int, default=8)
    parser.add_argument("--walker-speed", type=float, default=1.3)
    parser.add_argument("--start-side", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--crossing-offset-m", type=float, default=0.0,
                        help="Longitudinal offset of pedestrian crossing line")
    parser.add_argument("--second-walker", action="store_true")
    parser.add_argument("--second-start-tick", type=int, default=100)
    parser.add_argument("--second-speed", type=float, default=1.2)
    parser.add_argument("--second-start-side", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--second-crossing-offset-m", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=HERE / "carla_trace.json")
    args = parser.parse_args()
    if args.grid < 20:
        raise ValueError("The crossing layout requires at least 20 cells per side")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith("/Town01"):
        world = client.load_world("Town01")
    original = world.get_settings()
    actors = []
    collisions = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        start, crossing, goal = choose_road(world)
        forward = crossing.transform.get_forward_vector()
        right = crossing.transform.get_right_vector()
        center = crossing.transform.location
        n = args.grid
        cross_x, center_y, goal_x = 10, n // 2, 14
        cell_m = 2.0
        labels = [0] * (n * n)
        road_map = world.get_map()
        drivable = 0
        for y in range(n):
            for x in range(n):
                point = add(add(center, forward, (x - cross_x) * cell_m),
                            right, (y - center_y) * cell_m)
                wp = road_map.get_waypoint(point, project_to_road=False,
                                           lane_type=carla.LaneType.Driving)
                if wp is None:
                    labels[x + n * y] = 4  # off-road safety proposition
                else:
                    drivable += 1
        risk_cells = [[x, center_y] for x in range(cross_x - 2, cross_x + 3)]
        for x, y in risk_cells:
            if labels[x + n * y] == 4:
                raise RuntimeError(f"Risk cell {(x, y)} is not on a driving lane")
            labels[x + n * y] = 1  # crossing conflict when env mode occupied
        if labels[goal_x + n * center_y] == 4:
            raise RuntimeError("Goal cell is not on a driving lane")
        labels[goal_x + n * center_y] = 2

        vehicles = world.get_blueprint_library().filter("vehicle.*")
        vehicles = sorted(vehicles, key=lambda bp: bp.id)
        ego_bp = next((bp for bp in vehicles if bp.id == "vehicle.tesla.model3"), vehicles[0])
        ego = world.try_spawn_actor(ego_bp, start)
        if ego is None:
            raise RuntimeError("Unable to spawn ego vehicle at selected start")
        actors.append(ego)
        ego.apply_control(carla.VehicleControl(hand_brake=True))

        walker_bp = sorted(world.get_blueprint_library().filter("walker.pedestrian.*"),
                           key=lambda bp: bp.id)[0]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker_start = add(add(center, forward, args.crossing_offset_m),
                           right, args.start_side * 5.0)
        walker_start.z += 1.0
        walker = world.try_spawn_actor(walker_bp,
                                       carla.Transform(walker_start, carla.Rotation()))
        if walker is None:
            raise RuntimeError("Unable to spawn crossing pedestrian")
        actors.append(walker)
        sensor_bp = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(sensor_bp, carla.Transform(), attach_to=ego)
        actors.append(sensor)
        sensor.listen(lambda event: collisions.append(event.frame))
        direction = carla.Vector3D(-args.start_side * right.x,
                                    -args.start_side * right.y, 0.0)
        walker.apply_control(carla.WalkerControl(direction=direction,
                                                speed=args.walker_speed))
        rows = []
        second = None
        for tick in range(args.ticks):
            if args.second_walker and tick == args.second_start_tick:
                second_start = add(add(center, forward,
                                       args.second_crossing_offset_m),
                                   right, args.second_start_side * 5.0)
                second_start.z += 1.0
                second = world.try_spawn_actor(
                    walker_bp, carla.Transform(second_start, carla.Rotation()))
                if second is None:
                    raise RuntimeError("Unable to spawn second pedestrian")
                actors.append(second)
                second_direction = carla.Vector3D(
                    -args.second_start_side * right.x,
                    -args.second_start_side * right.y, 0.0)
                second.apply_control(carla.WalkerControl(
                    direction=second_direction, speed=args.second_speed))
            frame = world.tick()
            if tick % args.sample_every:
                continue
            position = walker.get_location()
            positions = [position]
            if second is not None:
                positions.append(second.get_location())
            ego_position = ego.get_location()
            theta = []
            distance_m = []
            for x, y in risk_cells:
                cell = add(add(center, forward, (x - cross_x) * cell_m),
                           right, (y - center_y) * cell_m)
                distance = min(math.hypot(p.x - cell.x, p.y - cell.y)
                               for p in positions)
                # Local risk severity is an explicit model assumption.
                multiplier = 0.2 + 1.6 * math.exp(-0.5 * (distance / 3.0) ** 2)
                theta.append(multiplier - 1.0)
                distance_m.append(distance)
            rows.append({"frame": frame, "time_s": round(tick * 0.05, 3),
                         "walker_xyz": [position.x, position.y, position.z],
                         "walkers_xyz": [[p.x, p.y, p.z] for p in positions],
                         "ego_xyz": [ego_position.x, ego_position.y, ego_position.z],
                         "risk_distance_m": distance_m, "theta": theta})
        result = {"simulator": "CARLA 0.9.16", "town": world.get_map().name,
                  "scenario": ("two scripted pedestrian crossings, stationary ego"
                               if args.second_walker else
                               "scripted pedestrian crossing, stationary ego"),
                  "walker_speed_mps": args.walker_speed,
                  "walker_start_side": args.start_side,
                  "crossing_offset_m": args.crossing_offset_m,
                  "second_walker": args.second_walker,
                  "second_start_tick": (args.second_start_tick if
                                        args.second_walker else None),
                  "second_speed_mps": (args.second_speed if
                                       args.second_walker else None),
                  "second_crossing_offset_m": (
                      args.second_crossing_offset_m if args.second_walker
                      else None),
                  "fixed_delta_seconds": 0.05, "sample_every_ticks": args.sample_every,
                  "grid_n": n, "cell_m": cell_m,
                  "grid_semantics": "forward/right aligned to crossing waypoint",
                  "crossing_xy": [cross_x, center_y], "goal_xy": [goal_x, center_y],
                  "risk_cells": risk_cells, "labels": labels,
                  "drivable_cells": drivable,
                  "ego_spawn_xyz": [start.location.x, start.location.y, start.location.z],
                  "crossing_xyz": [center.x, center.y, center.z],
                  "forward_xy": [forward.x, forward.y],
                  "right_xy": [right.x, right.y],
                  "goal_xyz": [goal.transform.location.x,
                               goal.transform.location.y, goal.transform.location.z],
                  "collision_frames": collisions, "rows": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {len(rows)} CARLA observations to {args.output}")
        print(f"Drivable grid cells: {drivable}/{n*n}; collisions: {len(collisions)}")
    finally:
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        world.apply_settings(original)


if __name__ == "__main__":
    main()
