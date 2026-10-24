"""Append-only JSONL evidence and reproducibility metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import canonical_hash

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def redact(text):
    text = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED_HF_TOKEN]", str(text))
    text = re.sub(r"(?i)(authorization|token|password)([=: ]+)[^\s,]+", r"\1\2[REDACTED]", text)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            text = text.replace(os.environ[key], "[REDACTED]")
    return text


def provenance():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "flash-attn",
                "numpy", "nvidia-ml-py", "jieba", "rouge-score"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {}
    for folder in ("src", "scripts", "configs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".jinja"}:
                sources[str(path.relative_to(ROOT))] = sha256_file(path)
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": canonical_hash(sources), "versions": versions,
            "python": platform.python_version(), "platform": platform.platform()}


class EvidenceLog:
    def __init__(self, config, repeat=0):
        self.config = config
        self.attempt_id = uuid.uuid4().hex
        self.meta = {"schema_version": 1, "run_id": config["run_id"],
                     "attempt_id": self.attempt_id, "repeat": repeat,
                     "experiment": config["experiment"], "config": config,
                     "config_sha256": canonical_hash(config), "provenance": provenance(),
                     "gpu": None}
        self.meta["job_id"] = os.environ.get("FT_REPRO_JOB_ID")
        folder = Path(config["output"]["raw_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{config['run_id']}.r{repeat}.{self.attempt_id}.jsonl"
        self.path.touch(exist_ok=False)
        print(f"Raw evidence: {self.path}", flush=True)
        self.write("started", status="running")

    def write(self, event, **payload):
        row = {**self.meta, "timestamp": utc_now(), "event": event, **payload}
        def clean(value):
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            return value
        encoded = json.dumps(clean(row), ensure_ascii=False, allow_nan=False, default=str)
        with self.path.open("a", encoding="utf-8") as out:
            out.write(encoded + "\n")
            out.flush()
            os.fsync(out.fileno())

    def failure(self, exc):
        import traceback
        status = "oom" if type(exc).__name__ == "OutOfMemoryError" else "failed"
        self.write("result", status=status, error=redact(exc),
                   traceback=redact(traceback.format_exc()))
