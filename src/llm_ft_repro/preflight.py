"""Read-only artifact preflight: no downloads, framework imports or CUDA initialization."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import training_identity, verify_artifact, verify_resume
from .config import require_pinned_model, validate
from .data import audit_task_bundle, verify_data
from .evidence import provenance


def check_adapter(config, audit):
    path = Path(config["evaluation"]["adapter_path"])
    identity = json.loads((path / "training_identity.json").read_text(encoding="utf-8"))
    receipt = verify_artifact(path, "adapter", identity)
    if identity.get("schema_version") != 2:
        raise ValueError("Adapter needs a complete v2 training identity")
    for key in ("model", "quantization", "lora"):
        if identity["config"][key] != config[key]:
            raise ValueError(f"Adapter training/evaluation {key} differs")
    if identity["manifest_sha256"] != audit["manifest_sha256"] or any(
            identity[key] != audit["file_sha256"][split] for key, split in
            (("data_sha256", "train"), ("validation_sha256", "validation"), ("test_sha256", "test"))):
        raise ValueError("Adapter and evaluation use different data bundles")
    return receipt


def preflight(config, resume=None):
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "status": "passed", "detail": detail})
            return detail
        except (ValueError, OSError, KeyError, TypeError) as exc:
            checks.append({"name": name, "status": "blocked", "detail": str(exc)})
            return None

    check("configuration", lambda: validate(config) or "Strict schema valid")

    def model_ref():
        require_pinned_model(config)
        folder = Path(config["model"]["path"])
        if folder.is_dir():
            required = {"config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"}
            index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))
            required.update(index["weight_map"].values())
            missing = [name for name in sorted(required) if not (folder / name).is_file()]
            if missing:
                raise ValueError(f"Local snapshot missing files: {missing}")
            return "Local snapshot layout present; upstream integrity and GPU load remain unverified"
        return "Remote revision pinned; remote availability is not checked"
    check("model_reference", model_ref)
    if config["role"] in {"benchmark", "equivalence"}:
        def packed():
            manifest = verify_data(config, config["data"]["path"])
            required = (config["bench"]["warmup_steps"] + config["bench"]["measure_steps"]) * (
                config["data"]["micro_batch"] * config["data"]["seq_len"])
            size = Path(config["data"]["path"]).stat().st_size
            if manifest.get("dtype") != "<u4" or size % 4 or size // 4 != manifest.get("tokens") or size // 4 < required:
                raise ValueError("Packed data size/dtype/token count does not satisfy this configuration")
            return {"available_tokens": size // 4, "required_tokens": required}
        check("packed_data", packed)
    elif config["role"] in {"train", "eval"}:
        audit = check("task_bundle", lambda: audit_task_bundle(config))
        if config["role"] == "eval" and config["lora"]["enabled"]:
            if audit:
                check("adapter", lambda: check_adapter(config, audit))
            else:
                checks.append({"name": "adapter", "status": "blocked", "detail": "Requires a valid task bundle"})
        if resume:
            if config["role"] != "train" or not audit:
                checks.append({"name": "resume", "status": "blocked", "detail": "Requires training config and valid data"})
            else:
                check("resume", lambda: verify_resume(resume, training_identity(config, audit, provenance())))
    return {"run_id": config["run_id"], "ready": all(c["status"] == "passed" for c in checks),
            "scope": "Local config and artifact checks only; CUDA, dependencies and remote access are unverified",
            "checks": checks}
