"""Run isolated CARLA closed-loop trials, preserving all outcomes and attempts.

Only processes launched by this scheduler, and verified descendants, may be
stopped. Existing processes listening on a requested port are never touched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASES = (
    ("Town01_route0_single", "Town01", 0, False, 900),
    ("Town01_route0_double", "Town01", 0, True, 1200),
    ("Town01_route1_single", "Town01", 1, False, 900),
    ("Town02_route0_single", "Town02", 0, False, 900),
)
METHODS = ("optimized", "full", "warm_lp")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_hashes():
    relative = [
        "papers/Risk_LTL_Planning-main/experiments_20261028/run_suite.py",
        "papers/Risk_LTL_Planning-main/experiments_20261028/run_trial.py",
        "papers/Risk_LTL_Planning-main/experiments_20261028/planner.py",
        "papers/Risk_LTL_Planning-main/experiments_20261026/optimized_controller.py",
        "papers/Risk_LTL_Planning-main/experiments_20261023/controlled_experiment.py",
        "reproduction/carla_local_risk/closed_loop_dynamic.py",
        "reproduction/carla_local_risk/capture_crossing.py",
        "reproduction/carla_local_risk/benchmark.py",
        "reproduction/carla_local_risk/dynamic_product.py",
        "reproduction/carla_local_risk/dynamic_certificate.py",
        "reproduction/adaptive_risk_ltl/experiment.py",
        "reproduction/adaptive_risk_ltl/highs_reopt.py",
        "reproduction/adaptive_risk_ltl/joint_price_atlas.py",
        "reproduction/adaptive_risk_ltl/stratified.py",
        "reproduction/adaptive_risk_ltl/paper_atlas_comparison.py",
    ]
    return {name: source_hash(REPO / name) for name in relative}


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


class OwnedServer:
    def __init__(self, executable, town, port, directory, timeout):
        self.executable = executable.resolve()
        self.town, self.port, self.directory = town, port, directory
        self.timeout = timeout
        self.process = None
        self.log = None
        self.owned = {}
        self.record = {"town": town, "port": port, "created_utc": utc(), "owned_processes": []}

    def refresh_owned(self):
        if self.process is None:
            return
        try:
            root = psutil.Process(self.process.pid)
            candidates = [root] + root.children(recursive=True)
        except psutil.Error:
            return
        for proc in candidates:
            try:
                exe = Path(proc.exe()).resolve()
                if not exe.is_relative_to(self.executable.parent):
                    continue
                if not proc.name().lower().startswith("carlaue4"):
                    continue
                self.owned[proc.pid] = {"pid": proc.pid, "create_time": proc.create_time(), "exe": str(exe)}
            except (psutil.Error, OSError):
                continue
        self.record["owned_processes"] = list(self.owned.values())

    def start(self):
        if port_in_use(self.port):
            raise RuntimeError(f"Port {self.port} already in use; unknown processes will not be stopped")
        self.directory.mkdir(parents=True, exist_ok=True)
        command = [str(self.executable), f"/Game/Carla/Maps/{self.town}",
                   "-RenderOffScreen", "-quality-level=Low", "-nosound",
                   f"-carla-rpc-port={self.port}", "-ResX=640", "-ResY=480"]
        startup = None
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        self.log = (self.directory / "server.log").open("wb")
        self.process = subprocess.Popen(command, cwd=self.executable.parent,
                                        stdout=self.log, stderr=subprocess.STDOUT,
                                        startupinfo=startup)
        self.record.update(command=command, pid=self.process.pid)
        self.refresh_owned()
        save(self.directory / "server.json", self.record)
        import carla
        client = carla.Client("127.0.0.1", self.port)
        client.set_timeout(2.)
        deadline = time.monotonic() + self.timeout
        last_error = None
        requested_map = False
        while time.monotonic() < deadline:
            self.refresh_owned()
            try:
                # A client created before the Windows RPC listener is ready can
                # retain a failed connection; use a fresh readiness client.
                client = carla.Client("127.0.0.1", self.port)
                client.set_timeout(5.)
                world = client.get_world()
                name = world.get_map().name
                if name.split("/")[-1] != self.town and not requested_map:
                    # Some Windows launchers ignore the initial map URL.
                    requested_map = True
                    client.set_timeout(120.)
                    world = client.load_world(self.town)
                    client.set_timeout(2.)
                    name = world.get_map().name
                if name.split("/")[-1] == self.town:
                    self.record.update(ready_utc=utc(), map=name,
                                       server_version=client.get_server_version(),
                                       client_version=client.get_client_version())
                    save(self.directory / "server.json", self.record)
                    print("SERVER READY", self.town, self.port, flush=True)
                    return
                last_error = f"Expected {self.town}, got {name}"
            except Exception as exc:
                last_error = repr(exc)
            if self.process.poll() is not None and not any(psutil.pid_exists(pid) for pid in self.owned):
                break
            time.sleep(1.)
        self.record.update(startup_failure=last_error, ended_utc=utc())
        save(self.directory / "server.json", self.record)
        raise RuntimeError(f"CARLA startup failed: {last_error}")

    def stop(self):
        self.refresh_owned()
        victims, errors = [], []
        # Reverse discovery order stops shipping children before their launcher.
        for item in reversed(list(self.owned.values())):
            try:
                proc = psutil.Process(item["pid"])
                if (abs(proc.create_time() - item["create_time"]) > 1e-3 or
                        Path(proc.exe()).resolve() != Path(item["exe"]).resolve()):
                    errors.append({"pid": item["pid"], "reason": "Ownership identity changed; left untouched"})
                    continue
                proc.terminate()
                victims.append(proc)
            except psutil.NoSuchProcess:
                continue
            except psutil.Error as exc:
                errors.append({"pid": item["pid"], "error": repr(exc)})
        _, alive = psutil.wait_procs(victims, timeout=12.)
        for proc in alive:
            try:
                item = self.owned[proc.pid]
                if abs(proc.create_time() - item["create_time"]) < 1e-3 and Path(proc.exe()).resolve() == Path(item["exe"]).resolve():
                    proc.kill()
            except psutil.Error as exc:
                errors.append({"pid": proc.pid, "error": repr(exc)})
        psutil.wait_procs(alive, timeout=5.)
        if self.log is not None:
            self.log.close()
            self.log = None
        self.record.update(stopped_utc=utc(), stop_errors=errors)
        if self.directory.exists():
            save(self.directory / "server.json", self.record)


def load_json(path):
    try:
        return json.loads(path.read_text("utf8"))
    except (OSError, ValueError):
        return {}


def archive_attempt(root, stem, attempt, paths):
    target = root / "diagnostics" / stem / f"attempt_{attempt:02d}_{time.time_ns()}"
    target.mkdir(parents=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target / path.name)
    return target.relative_to(root).as_posix()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, type=Path)
    p.add_argument("--out", type=Path, default=HERE)
    p.add_argument("--port", type=int, default=2060, help="Base RPC port; case i uses base+10*i")
    p.add_argument("--case", action="append", choices=[item[0] for item in CASES], help="Repeat to select cases")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--order-seed", type=int, default=20261028)
    p.add_argument("--startup-timeout", type=float, default=180.)
    p.add_argument("--trial-timeout", type=float, default=600.)
    p.add_argument("--infra-retries", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if not args.server.is_file():
        raise FileNotFoundError(args.server)
    if args.repeats < 1 or args.infra_retries not in (0, 1):
        raise ValueError("repeats>=1 and infra-retries in {0,1} required")
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LTL_RISK_GAMMA="0.8", LTL_RISK_SOFT="1.0", LTL_RISK_HARD="2.0")
    # Parent also uses single-thread numerical libraries if any are imported.
    os.environ.update({key: env[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    selected = [item for item in CASES if args.case is None or item[0] in args.case]
    rng = random.Random(args.order_seed)
    jobs = []
    for name, town, route, two, ticks in selected:
        for repetition in range(args.repeats):
            methods = list(args.methods)
            rng.shuffle(methods)
            for method in methods:
                jobs.append({"case": name, "town": town, "route_index": route,
                             "second_walker": two, "max_ticks": ticks,
                             "method": method, "repetition": repetition,
                             "seed": 5000 + repetition})
    hashes = code_hashes()
    protocol = {"jobs": jobs, "source_hashes_lf": hashes,
                "server": str(args.server.resolve()), "base_port": args.port,
                "order_seed": args.order_seed, "trial_timeout_s": args.trial_timeout,
                "startup_timeout_s": args.startup_timeout, "infra_retries": args.infra_retries,
                "fixed_delta_seconds": .05, "action_every_ticks": 10,
                "scope": "New live CARLA trials; complete non-arrivals are retained and never retried"}
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"protocol_sha256": protocol_sha, **protocol}, indent=2))
        return 0
    for case_index, _ in enumerate(selected):
        port = args.port + case_index * 10
        if port_in_use(port):
            raise RuntimeError(f"Port {port} already in use; stop and identify its owner before running this suite")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("results", "policies", "logs", "references", "server_logs"):
        (out / directory).mkdir(exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("Protocol/source differs from frozen manifest; use a new --out directory")
    else:
        packages = {}
        for package in ("carla", "numpy", "scipy", "highspy", "psutil"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        manifest = {"created_utc": utc(), "protocol_sha256": protocol_sha,
                    "base_commit": revision, "python": sys.version,
                    "python_executable": sys.executable, "versions": packages,
                    "environment": {key: env[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "LTL_RISK_GAMMA", "LTL_RISK_SOFT", "LTL_RISK_HARD")},
                    **protocol}
        save(manifest_path, manifest)
    status = {"started_utc": utc(), "protocol_sha256": protocol_sha,
              "expected_trials": len(jobs), "completed": [], "infrastructure_failures": []}
    server = None
    try:
        for case_index, case in enumerate(selected):
            name, town, _, _, _ = case
            case_jobs = [job for job in jobs if job["case"] == name]
            server_attempt = 0
            for job in case_jobs:
                if code_hashes() != hashes:
                    raise RuntimeError("Source changed during suite; stop without mixing implementations")
                stem = f'{name}_r{job["repetition"]}_{job["method"]}'
                result_path = out / "results" / (stem + ".json")
                policy_path = out / "policies" / (stem + ".zip")
                log_path = out / "logs" / (stem + ".log")
                execution_path = out / "logs" / (stem + "_execution.json")
                previous = load_json(result_path)
                previous_execution = load_json(execution_path)
                if (previous.get("trial_complete") is True and policy_path.exists() and
                        previous_execution.get("returncode") == 0 and
                        previous_execution.get("protocol_sha256") == protocol_sha):
                    status["completed"].append({"trial": stem, "outcome": previous.get("outcome"), "resumed": True})
                    save(out / "suite_status.json", status)
                    continue
                if any(path.exists() for path in (result_path, policy_path, log_path, execution_path)):
                    archive_attempt(out, stem, -1, [result_path, policy_path, log_path, execution_path])
                for attempt in range(args.infra_retries + 1):
                    began = time.perf_counter()
                    command = [sys.executable, str(HERE / "run_trial.py"),
                               "--town", town, "--route-index", str(job["route_index"]),
                               "--max-ticks", str(job["max_ticks"]), "--port", str(args.port + case_index * 10),
                               "--method", job["method"], "--seed", str(job["seed"]),
                               "--repetition", str(job["repetition"]), "--output", str(result_path),
                               "--policies", str(policy_path), "--reference", str(out / "references" / (name + ".json"))]
                    if job["second_walker"]:
                        command.append("--second-walker")
                    execution = {"started_utc": utc(), "command": command, "attempt": attempt,
                                 "protocol_sha256": protocol_sha, "source_hashes_lf": hashes}
                    print("CARLA TRIAL", stem, "attempt", attempt, flush=True)
                    try:
                        if server is None:
                            server = OwnedServer(args.server, town, args.port + case_index * 10,
                                                 out / "server_logs" / f"{name}_launch{server_attempt}_{time.time_ns()}",
                                                 args.startup_timeout)
                            server_attempt += 1
                            server.start()
                        with log_path.open("wb") as logfile:
                            completed = subprocess.run(command, cwd=REPO, env=env,
                                                       stdout=logfile, stderr=subprocess.STDOUT,
                                                       timeout=args.trial_timeout)
                        execution["returncode"] = completed.returncode
                        result = load_json(result_path)
                        successful = completed.returncode == 0 and result.get("trial_complete") is True and policy_path.exists()
                        if not successful:
                            execution["error"] = result.get("infrastructure_failure", "Nonzero exit, incomplete result, or missing policy archive")
                    except Exception as exc:
                        successful = False
                        result = load_json(result_path)
                        execution.update(error=repr(exc), traceback=traceback.format_exc(), returncode=None)
                    execution.update(ended_utc=utc(), wall_seconds=time.perf_counter() - began,
                                     trial_complete=successful, outcome=result.get("outcome"))
                    save(execution_path, execution)
                    if successful:
                        status["completed"].append({"trial": stem, "outcome": result.get("outcome"), "attempt": attempt})
                        print("TRIAL COMPLETE", stem, result.get("outcome"), flush=True)
                        server.stop()
                        server = None
                        break
                    location = archive_attempt(out, stem, attempt, [result_path, policy_path, log_path, execution_path])
                    status["infrastructure_failures"].append({"trial": stem, "attempt": attempt, "diagnostics": location, "error": execution.get("error")})
                    if server is not None:
                        server.stop()
                        server = None
                    if attempt == args.infra_retries:
                        print("TRIAL INFRASTRUCTURE FAILURE", stem, execution.get("error"), flush=True)
                save(out / "suite_status.json", status)
            if server is not None:
                server.stop()
                server = None
    finally:
        if server is not None:
            server.stop()
        status.update(ended_utc=utc(), complete=len(status["completed"]) == len(jobs))
        save(out / "suite_status.json", status)
    print("SUITE FINISHED", len(status["completed"]), "/", len(jobs), flush=True)
    return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
