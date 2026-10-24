"""Checkpoint/adapter receipts. Validation hashes files without deserializing pickle state."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .config import canonical_hash
from .evidence import sha256_file

RECEIPT = "artifact-manifest.json"
REQUIRED = {
    "checkpoint": {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                   "trainer_state.json", "rng_state.pth"},
    "adapter": {"adapter_config.json", "adapter_model.safetensors", "training_identity.json"},
}


def _files(folder):
    return {str(p.relative_to(folder)): p for p in folder.rglob("*") if p.is_file() and p.name != RECEIPT}


def seal_artifact(folder, kind, identity, global_step):
    folder = Path(folder)
    files = _files(folder)
    missing = REQUIRED[kind] - files.keys()
    if missing or any(files[name].stat().st_size == 0 for name in REQUIRED[kind]):
        raise ValueError(f"Incomplete {kind}: missing={sorted(missing)} or empty required file")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Artifact must have a positive completed global_step")
    receipt = {"schema_version": 1, "kind": kind, "global_step": global_step,
               "identity_sha256": canonical_hash(identity), "files": {
                   name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                   for name, path in sorted(files.items())}}
    # Publish only a complete receipt; interruption before publication leaves an unsealed checkpoint.
    target = folder / RECEIPT
    if target.exists():
        raise FileExistsError(f"Artifact already sealed: {target}")
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return {"path": str(target), "sha256": sha256_file(target), "global_step": global_step}


def verify_artifact(folder, kind, identity=None):
    folder = Path(folder)
    receipt = json.loads((folder / RECEIPT).read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 1 or receipt.get("kind") != kind:
        raise ValueError("Artifact receipt type/version mismatch")
    if type(receipt.get("global_step")) is not int or receipt["global_step"] <= 0:
        raise ValueError("Invalid artifact global_step")
    files = _files(folder)
    if REQUIRED[kind] - files.keys() or set(receipt["files"]) != set(files):
        raise ValueError("Artifact file set changed or is incomplete")
    for name, path in files.items():
        expected = receipt["files"][name]
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {path}")
    if identity is not None and canonical_hash(identity) != receipt["identity_sha256"]:
        raise ValueError("Artifact training identity mismatch")
    if kind == "checkpoint":
        state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
        if state.get("global_step") != receipt["global_step"] or folder.name != f"checkpoint-{receipt['global_step']}":
            raise ValueError("Checkpoint directory, trainer state and receipt disagree")
    return {"path": str(folder / RECEIPT), "sha256": sha256_file(folder / RECEIPT),
            "global_step": receipt["global_step"]}


def training_identity(config, data_audit, provenance):
    return {"schema_version": 2, "config": config,
            "data_sha256": data_audit["file_sha256"]["train"],
            "validation_sha256": data_audit["file_sha256"]["validation"],
            "test_sha256": data_audit["file_sha256"]["test"],
            "manifest_sha256": data_audit["manifest_sha256"],
            "runtime": {k: provenance[k] for k in ("source_sha256", "versions", "python", "platform")}}


def verify_resume(path, identity):
    path = Path(path).resolve()
    previous = json.loads((path.parent / "identity.json").read_text(encoding="utf-8"))
    if previous != identity:
        raise ValueError("Checkpoint configuration, data, source or runtime differs; resume refused")
    return verify_artifact(path, "checkpoint", identity)
