"""Command-line interface; planning and aggregation do not import torch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import apply_environment, load_config


def inspect_model(config):
    from .evidence import EvidenceLog
    from .model_factory import build_model, memory_snapshot
    apply_environment(config)
    log = EvidenceLog(config)
    try:
        import inspect
        import shutil
        import torch
        from peft import prepare_model_for_kbit_training
        disk = shutil.disk_usage(".")
        log.write("host", disk_free_bytes=disk.free,
                  kbit_helper_source=inspect.getsource(prepare_model_for_kbit_training))
        model = build_model(config, log)
        # A short model pass verifies LoRA connectivity and the installed 4-bit kernels.
        ids = torch.tensor([[151644, 8948, 198, 151645]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing or not torch.isfinite(loss).item():
            raise RuntimeError(f"Minimal backward failed, missing gradients: {missing[:5]}")
        log.write("result", status="ok", loss=loss.item(), memory=memory_snapshot())
        return 0
    except Exception as exc:
        log.failure(exc)
        return 2


def parser():
    root = argparse.ArgumentParser(description="Qwen1.5-14B single-GPU reproducibility project")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate configuration files without GPU")
    validate.add_argument("paths", nargs="*", default=["configs"])
    preflight = commands.add_parser("preflight", help="Read-only local input checks; no GPU or downloads")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--resume")
    for name in ("inspect", "bench", "train", "eval", "prepare-data", "pack-data"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "bench":
            sub.add_argument("--repeat", type=int, default=0)
        if name == "train":
            sub.add_argument("--resume")
        if name in {"prepare-data", "pack-data"}:
            sub.add_argument("--input", required=True)
            sub.add_argument("--sources", required=True)
            sub.add_argument("--output", required=True)
        if name == "prepare-data":
            sub.add_argument("--test-per-task", type=int, default=500)
            sub.add_argument("--validation-per-task", type=int, default=100)
            sub.add_argument("--train-limit", type=int, default=20000)
        if name == "pack-data":
            sub.add_argument("--min-tokens", type=int, default=327680)
    for name in ("plan", "matrix"):
        sub = commands.add_parser(name, help="Print plan; matrix needs --execute to consume GPU time")
        sub.add_argument("experiment", choices=[f"E{i}" for i in range(7)])
        sub.add_argument("--execute", action="store_true")
        sub.add_argument("--budget-hours", type=float, default=20)
        sub.add_argument("--acknowledged-hours", type=float, default=15)
    summary = commands.add_parser("summarize")
    summary.add_argument("--raw", default="results/raw")
    summary.add_argument("--output", default="results/summary")
    summary.add_argument("--reviews")
    summary.add_argument("--excel", action="store_true")
    summary.add_argument("--figures", action="store_true")
    reviews = commands.add_parser("review-template", help="Create a pending review file bound to summary evidence")
    reviews.add_argument("--summary", default="results/summary/summary.json")
    reviews.add_argument("--output", required=True)
    compare = commands.add_parser("compare-numerics")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--atol", type=float, required=True)
    compare.add_argument("--rtol", type=float, required=True)
    return root


def main():
    args = parser().parse_args()
    if args.command == "validate":
        paths = []
        for path in map(Path, args.paths):
            paths.extend(sorted(path.rglob("*.yaml")) if path.is_dir() else [path])
        if not paths:
            raise ValueError("No configuration files found")
        for path in paths:
            load_config(path)
        print(json.dumps({"validated": len(paths), "paths": [str(p) for p in paths]}, indent=2))
        return
    if args.command in {"plan", "matrix"}:
        from .orchestrate import run_matrix
        raise SystemExit(run_matrix(args.experiment, args.command == "matrix" and args.execute,
                                    args.budget_hours, args.acknowledged_hours))
    if args.command == "summarize":
        from .analysis import summarize
        print(json.dumps(summarize(args.raw, args.output, args.reviews, args.excel, args.figures), ensure_ascii=False))
        return
    if args.command == "review-template":
        from .analysis import create_review_template
        print(json.dumps(create_review_template(args.summary, args.output), ensure_ascii=False))
        return
    if args.command == "compare-numerics":
        from .analysis import compare_numerics
        if args.atol < 0 or args.rtol < 0:
            raise ValueError("Tolerances cannot be negative")
        result = compare_numerics(args.left, args.right, args.atol, args.rtol)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["passed"] else 2)
    config = load_config(args.config)
    if args.command == "preflight":
        from .preflight import preflight
        result = preflight(config, args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ready"] else 2)
    if args.command == "inspect":
        raise SystemExit(inspect_model(config))
    if args.command == "bench":
        from .bench import run_benchmark
        if not 0 <= args.repeat < config["bench"]["repeats"]:
            raise ValueError("Repeat index outside configured range")
        raise SystemExit(run_benchmark(config, args.repeat))
    if args.command == "train":
        from .train import run_training
        raise SystemExit(run_training(config, args.resume))
    if args.command == "eval":
        from .eval import run_evaluation
        raise SystemExit(run_evaluation(config))
    if args.command == "prepare-data":
        from .data import prepare_tasks
        if min(args.test_per_task, args.validation_per_task, args.train_limit) <= 0:
            raise ValueError("Split sizes must be positive")
        print(json.dumps(prepare_tasks(config, args.input, args.sources, args.output, args.test_per_task,
                                       args.validation_per_task, args.train_limit), ensure_ascii=False, indent=2))
    elif args.command == "pack-data":
        from .data import pack_corpus
        print(json.dumps(pack_corpus(config, args.input, args.sources, args.output, args.min_tokens),
                         ensure_ascii=False, indent=2))
