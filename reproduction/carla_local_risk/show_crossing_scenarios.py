"""Show the recorded single- and two-walker Town01 trials in a visible CARLA window.

This is a visual replay of saved closed-loop observations. It does not solve a
new planning problem or change the saved experiment outcomes.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

import carla

from capture_crossing import add, choose_road


HERE = Path(__file__).resolve().parent
TRACES = (
    ("single_walker", HERE / "observed_hybrid_probe" /
     "single_20hz_900_observed_vs_lp.json", 0),
    ("two_walkers", HERE / "observed_hybrid_probe" /
     "double_20hz_1200_observed_vs_lp.json", 0),
)


def present(world, crossing, spawn, name, trace, trial_index, output, speed):
    rows = trace["trials"][trial_index]["rows"]
    forward = crossing.transform.get_forward_vector()
    right = crossing.transform.get_right_vector()
    center = crossing.transform.location
    camera_location = add(center, forward, -15.0)
    camera_location.z += 18.0
    camera_rotation = carla.Rotation(
        pitch=-55.0, yaw=crossing.transform.rotation.yaw)
    camera_transform = carla.Transform(camera_location, camera_rotation)
    world.get_spectator().set_transform(camera_transform)

    bp = world.get_blueprint_library()
    vehicle_bp = next((item for item in bp.filter("vehicle.*")
                       if item.id == "vehicle.tesla.model3"),
                      sorted(bp.filter("vehicle.*"), key=lambda item: item.id)[0])
    walker_bp = sorted(bp.filter("walker.pedestrian.*"),
                       key=lambda item: item.id)[0]
    car = world.try_spawn_actor(vehicle_bp, spawn)
    if car is None:
        raise RuntimeError("Could not spawn the car for visual replay")
    actors = [car]
    images = queue.Queue(maxsize=1)
    try:
        car.set_simulate_physics(False)
        walkers = []
        camera_bp = bp.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("fov", "80")
        camera = world.spawn_actor(camera_bp, camera_transform)
        actors.append(camera)

        def keep_latest(image):
            try:
                images.put_nowait(image)
            except queue.Full:
                try:
                    images.get_nowait()
                except queue.Empty:
                    pass
                images.put_nowait(image)

        camera.listen(keep_latest)
        saved = False
        screenshot_tick = 70 if name == "single_walker" else 110
        label = ("SINGLE PEDESTRIAN" if name == "single_walker"
                 else "TWO PEDESTRIANS")
        print(f"SHOW {name}: {len(rows)} recorded ticks", flush=True)
        for index, row in enumerate(rows):
            walker_positions = row["walkers_xyz"]
            while len(walkers) < len(walker_positions):
                pos = walker_positions[len(walkers)]
                rotation = carla.Rotation(
                    yaw=crossing.transform.rotation.yaw +
                    (90.0 if not walkers else -90.0))
                actor = world.try_spawn_actor(
                    walker_bp,
                    carla.Transform(carla.Location(*pos), rotation))
                if actor is None:
                    raise RuntimeError("Could not spawn a replay pedestrian")
                actors.append(actor)
                walkers.append(actor)
            for walker, pos in zip(walkers, walker_positions):
                walker.set_transform(carla.Transform(
                    carla.Location(*pos), walker.get_transform().rotation))

            location = add(center, forward, row["longitudinal_m"])
            location.z = spawn.location.z
            car.set_transform(carla.Transform(location, spawn.rotation))
            world.get_spectator().set_transform(camera_transform)
            title_position = add(center, forward, 1.0)
            title_position.z += 3.0
            world.debug.draw_string(title_position, label, draw_shadow=True,
                                    color=carla.Color(0, 255, 255),
                                    life_time=0.12)
            for cell_x in range(8, 13):
                point = add(center, forward, (cell_x - 10) * 2.0)
                point.z += 0.25
                world.debug.draw_point(point, size=0.16,
                                       color=carla.Color(255, 165, 0),
                                       life_time=0.12)
            goal = add(center, forward, 8.0)
            goal.z += 1.2
            world.debug.draw_string(goal, "GOAL", draw_shadow=True,
                                    color=carla.Color(40, 255, 40),
                                    life_time=0.12)
            world.tick()
            if not saved and index >= screenshot_tick:
                try:
                    image = images.get_nowait()
                except queue.Empty:
                    pass
                else:
                    path = output / f"{name}.png"
                    image.save_to_disk(str(path))
                    print(f"SCREENSHOT {path}", flush=True)
                    saved = True
            time.sleep(max(0.0, 0.05 / speed))
        print(f"DONE {name}", flush=True)
    finally:
        for actor in reversed(actors):
            if actor.is_alive:
                actor.destroy()
        world.tick()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=HERE / "observed_hybrid_probe" / "preview")
    args = parser.parse_args()
    if args.cycles < 1 or args.speed <= 0:
        parser.error("cycles and speed must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith("/Town01"):
        world = client.load_world("Town01")
    previous = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        spawn, crossing, _ = choose_road(world)
        for cycle in range(args.cycles):
            print(f"CYCLE {cycle + 1}/{args.cycles}", flush=True)
            for name, path, trial_index in TRACES:
                trace = json.loads(path.read_text(encoding="utf-8"))
                present(world, crossing, spawn, name, trace, trial_index,
                        args.output, args.speed)
    finally:
        world.apply_settings(previous)


if __name__ == "__main__":
    main()
