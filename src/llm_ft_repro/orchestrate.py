"""Plan first; execute each repeat in a fresh process with a shared budget ledger."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .config import canonical_hash, load_config
from .evidence import ROOT, utc_now
from .data import read_jsonl


def make_plan(experiment, config_dir="configs"):
    plans = []
    environments = set()
    identifiers = set()
    for path in sorted(Path(config_dir).rglob("*.yaml")):
        if path.name == "base.yaml" or "diagnostics" in path.parts:
            continue
        config = load_config(path)
        if config["experiment"] != experiment or config["role"] == "equivalence":
            continue
        if config["run_id"] in identifiers:
            raise ValueError(f"Duplicate run_id in matrix: {config['run_id']}")
        identifiers.add(config["run_id"])
        environments.add(canonical_hash(config["environment"]))
        repeats = config["bench"]["repeats"] if config["role"] == "benchmark" else 1
        command = {"benchmark": "bench", "train": "train", "eval": "eval", "inspect": "inspect"}[config["role"]]
        for repeat in range(repeats):
            args = [sys.executable, "-m", "llm_ft_repro", command, "--config", str(path)]
            if command == "bench":
                args.extend(["--repeat", str(repeat)])
            plans.append({"run_id": config["run_id"], "repeat": repeat, "command": args,
                          "config_path": str(path.resolve()), "raw_dir": config["output"]["raw_dir"],
                          "config_sha256": canonical_hash(config), "environment": config["environment"]})
    if not plans:
        raise ValueError(f"No configs for {experiment}")
    if len(environments) != 1:
        raise ValueError("Environment must be identical across every comparison")
    return plans


def result_for_job(raw_dir, job_id):
    """Never infer a child's outcome from an older run's filename or modification time."""
    matches = []
    for path in Path(raw_dir).glob("*.jsonl"):
        if path.name == "budget.jsonl":
            continue
        for row in read_jsonl(path):
            if row.get("job_id") == job_id and row.get("event") == "result":
                matches.append(row)
    if len(matches) > 1:
        raise ValueError(f"Multiple terminal records for job {job_id}")
    return matches[0] if matches else {}


def budget_spent(events):
    starts, finishes = {}, {}
    for event in events:
        kind = event["event"]
        if kind not in {"job_started", "job_finished", "job_skipped"}:
            raise ValueError(f"Unknown budget event: {kind}")
        if kind == "job_skipped":
            continue
        records = starts if kind == "job_started" else finishes
        job_id = event["job_id"]
        if job_id in records:
            raise ValueError(f"Duplicate budget event: {kind}/{job_id}")
        if kind == "job_finished":
            elapsed = event.get("elapsed_s")
            if job_id not in starts or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Invalid elapsed time or unmatched finish in budget ledger")
        records[job_id] = event
    if starts.keys() != finishes.keys():
        raise RuntimeError("Unclosed budget entry: reconcile actual elapsed time before launching")
    return sum(e["elapsed_s"] for e in finishes.values())


def run_matrix(experiment, execute=False, budget_hours=20, acknowledgement_hours=15):
    plans = make_plan(experiment)
    if not execute:
        print(json.dumps({"execute": False, "processes": len(plans), "plan": plans}, ensure_ascii=False, indent=2))
        return 0
    if not all(math.isfinite(n) for n in (budget_hours, acknowledgement_hours)) or not 0 < acknowledgement_hours <= budget_hours:
        raise ValueError("Budget and acknowledged runtime must be positive and ordered")
    from .preflight import preflight
    checked = set()
    for job in plans:
        if job["config_path"] in checked:
            continue
        current = load_config(job["config_path"])
        if canonical_hash(current) != job["config_sha256"]:
            raise ValueError("Configuration changed after planning")
        report = preflight(current)
        if not report["ready"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
        checked.add(job["config_path"])
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Prevent two local launchers from racing the budget ledger. OS releases lock after interruption.
    import fcntl
    with (raw / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = raw / "budget.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        spent = budget_spent(events)

        def append(row):
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": utc_now(), **row}, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        failed = False
        oom_configs = set()
        for job in plans:
            if job["run_id"] in oom_configs:
                append({"event": "job_skipped", "run_id": job["run_id"], "repeat": job["repeat"],
                        "reason": "previous_oom_same_config"})
                continue
            remaining = min(budget_hours, acknowledgement_hours) * 3600 - spent
            if remaining <= 0:
                print("GPU wall-time budget reached. Review ledger before explicitly increasing acknowledgement.")
                return 3
            current = load_config(job["config_path"])
            if canonical_hash(current) != job["config_sha256"]:
                raise ValueError("Configuration changed while matrix was running")
            job_id = uuid.uuid4().hex
            append({"event": "job_started", "job_id": job_id, **job,
                    "budget_hours": budget_hours, "acknowledgement_hours": acknowledgement_hours})
            start = time.monotonic()
            env = dict(os.environ, **job["environment"])
            env["FT_REPRO_JOB_ID"] = job_id
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            child = None
            outcome = "completed"
            try:
                child = subprocess.Popen(job["command"], cwd=ROOT, env=env)
                code = child.wait(timeout=remaining)
            except OSError as exc:
                code, outcome = 127, "launch_failed"
                print(f"Unable to launch {job['run_id']}: {type(exc).__name__}")
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                outcome = "budget_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
                if child is not None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                code = 124 if outcome == "budget_timeout" else 130
            elapsed = time.monotonic() - start
            spent += elapsed
            append({"event": "job_finished", "job_id": job_id, "run_id": job["run_id"],
                    "repeat": job["repeat"], "returncode": code, "outcome": outcome, "elapsed_s": elapsed})
            failed |= code != 0
            if outcome != "completed":
                return code
            last = result_for_job(ROOT / job["raw_dir"], job_id)
            if not last or last.get("status") not in {"ok", "oom", "failed"}:
                print("Child has no valid terminal evidence; stopping matrix")
                return 2
            if last.get("status") == "oom":
                oom_configs.add(job["run_id"])
            elif code != 0 or last.get("status") != "ok":
                return code or 2
            # Preserve the original OOM and schedule the reduced shape as a distinct run.
            if job["run_id"] == "e2_eager_gc_off" and job["repeat"] == 0 and code != 0:
                if last.get("status") == "oom":
                    fallback_path = "configs/diagnostics/e2_eager_fallback.yaml"
                    fallback = load_config(fallback_path)
                    if fallback["environment"] != job["environment"]:
                        raise ValueError("Fallback environment differs from original experiment")
                    fallback_report = preflight(fallback)
                    if not fallback_report["ready"]:
                        print(json.dumps(fallback_report, ensure_ascii=False, indent=2))
                        return 2
                    for repeat in range(fallback["bench"]["repeats"]):
                        plans.append({"run_id": fallback["run_id"], "repeat": repeat,
                                      "config_path": str(Path(fallback_path).resolve()),
                                      "raw_dir": fallback["output"]["raw_dir"],
                                      "config_sha256": canonical_hash(fallback),
                                      "environment": fallback["environment"],
                                      "command": [sys.executable, "-m", "llm_ft_repro", "bench",
                                                  "--config", fallback_path, "--repeat", str(repeat)]})
        return 2 if failed else 0
