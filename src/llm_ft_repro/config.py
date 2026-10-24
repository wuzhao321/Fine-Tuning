"""Explicit configuration inheritance and validation, usable without CUDA."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path


SCHEMA = {
    "run_id": str, "experiment": str, "description": str, "role": str,
    "model": {"path": str, "revision": str, "dtype": str},
    "quantization": {"enabled": bool, "type": str, "double_quant": bool,
                     "compute_dtype": str},
    "lora": {"enabled": bool, "r": int, "alpha": int, "dropout": (int, float),
             "target_modules": list, "expected_trainable": int},
    "attention": {"implementation": str},
    "gc": {"enabled": bool, "use_reentrant": bool},
    "optimizer": {"name": str, "weight_decay": (int, float), "max_grad_norm": (int, float)},
    "data": {"mode": str, "path": str, "validation_path": str, "test_path": str,
             "manifest": str, "seq_len": int, "micro_batch": int, "grad_accum": int,
             "max_target_tokens": int},
    "bench": {"warmup_steps": int, "measure_steps": int, "repeats": int,
              "nvml_interval_s": (int, float)},
    "train": {"learning_rate": (int, float), "scheduler": str,
              "warmup_ratio": (int, float), "epochs": (int, float), "seed": int,
              "save_steps": int, "eval_steps": int, "logging_steps": int,
              "save_total_limit": int},
    "evaluation": {"adapter_path": str, "max_new_tokens": int},
    "environment": dict,
    "hardware": {"peak_flops_tflops": (int, float), "nvml_device": str},
    "output": {"raw_dir": str, "checkpoint_dir": str},
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_mapping(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)  # Checked-in YAML uses the JSON subset: no dependency for planning.
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def load_config(path, _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    value = read_mapping(path)
    parent = value.pop("extends", None)
    if parent:
        value = merge(load_config(path.parent / parent, (*_seen, path)), value)
    validate(value)
    return value


def validate(value):
    def walk(node, schema, prefix=""):
        if set(node) != set(schema):
            raise ValueError(f"{prefix}: missing={set(schema)-set(node)}, unknown={set(node)-set(schema)}")
        for key, expected in schema.items():
            item, name = node[key], f"{prefix}.{key}"
            if isinstance(expected, dict):
                if not isinstance(item, dict):
                    raise ValueError(f"{name}: expected mapping")
                walk(item, expected, name)
            elif not isinstance(item, expected) or (isinstance(item, bool) and expected != bool):
                raise ValueError(f"{name}: wrong type")
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"{name}: expected finite number")
    walk(value, SCHEMA)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value["run_id"]):
        raise ValueError("run_id must be a safe filename component")
    if value["experiment"] not in {f"E{i}" for i in range(7)}:
        raise ValueError("Unknown experiment")
    if value["role"] not in {"benchmark", "equivalence", "train", "eval", "inspect"}:
        raise ValueError("Unknown role")
    if value["attention"]["implementation"] not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("Unknown attention implementation")
    if value["model"]["dtype"] != "bfloat16" or value["quantization"]["compute_dtype"] != "bfloat16":
        raise ValueError("This protocol requires bf16")
    if value["quantization"]["type"] != "nf4" or not value["quantization"]["double_quant"]:
        raise ValueError("This protocol requires NF4 + double quantization")
    if value["optimizer"]["name"] not in {"paged_adamw_8bit", "adamw_8bit"}:
        raise ValueError("Unknown optimizer")
    lora = value["lora"]
    if (lora["r"], lora["alpha"], lora["expected_trainable"]) != (64, 128, 104857600):
        raise ValueError("LoRA structure differs from the reproduction protocol")
    if sorted(lora["target_modules"]) != ["k_proj", "o_proj", "q_proj", "v_proj"]:
        raise ValueError("Exactly q/k/v/o projections are required")
    if not 0 <= lora["dropout"] < 1:
        raise ValueError("Invalid LoRA dropout")
    for section, keys in {"data": ["seq_len", "micro_batch", "grad_accum", "max_target_tokens"],
                          "bench": ["measure_steps", "repeats", "nvml_interval_s"],
                          "train": ["learning_rate", "epochs", "save_steps", "eval_steps",
                                    "logging_steps", "save_total_limit"],
                          "hardware": ["peak_flops_tflops"]}.items():
        if any(value[section][key] <= 0 for key in keys):
            raise ValueError(f"{section}: expected positive values")
    if value["evaluation"]["max_new_tokens"] <= 0 or value["optimizer"]["max_grad_norm"] <= 0:
        raise ValueError("Generation limit and gradient norm must be positive")
    if value["optimizer"]["weight_decay"] < 0 or value["data"]["seq_len"] < 2:
        raise ValueError("Invalid weight decay or causal sequence length")
    if value["role"] == "eval" and bool(value["evaluation"]["adapter_path"]) != lora["enabled"]:
        raise ValueError("Evaluation adapter path and lora.enabled must agree")
    if value["bench"]["warmup_steps"] < 0 or not 0 <= value["train"]["warmup_ratio"] < 1:
        raise ValueError("Invalid warmup")
    if value["train"]["scheduler"] != "cosine":
        raise ValueError("Only the declared cosine schedule is implemented")
    if value["data"]["mode"] not in {"synthetic_packed", "task"}:
        raise ValueError("Unknown data mode")
    if value["role"] in {"benchmark", "equivalence"}:
        if value["data"]["grad_accum"] != 1 or value["data"]["mode"] != "synthetic_packed":
            raise ValueError("Benchmarks require packed data and grad_accum=1")
    if value["role"] in {"train", "eval"} and value["data"]["mode"] != "task":
        raise ValueError("Training and evaluation require task data")
    if value["role"] in {"benchmark", "equivalence", "train"} and not lora["enabled"]:
        raise ValueError("Full parameter training is theoretical only")
    if value["role"] == "benchmark" and tuple(value["bench"][k] for k in
                                               ("warmup_steps", "measure_steps", "repeats")) != (10, 30, 3):
        raise ValueError("Standard benchmark protocol is 10 warmup / 30 measured / 3 repeats")
    if value["role"] == "equivalence" and (lora["dropout"] != 0 or
            value["bench"]["warmup_steps"] + value["bench"]["measure_steps"] < 20):
        raise ValueError("Numerical diagnostics require dropout=0 and at least 20 steps")
    allowed_env = {"CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM",
                   "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS"}
    if set(value["environment"]) - allowed_env or not all(isinstance(v, str) for v in value["environment"].values()):
        raise ValueError("Only declared non-secret environment variables may be logged")
    if value["data"]["mode"] == "task" and value["data"]["max_target_tokens"] >= value["data"]["seq_len"]:
        raise ValueError("Target budget must leave space for the prompt")


def apply_environment(config):
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Only a single process / GPU is supported")
    for key, value in config["environment"].items():
        os.environ[key] = value


def require_pinned_model(config):
    """Planning can use main; actual model/tokenizer loading cannot use a moving remote ref."""
    if not Path(config["model"]["path"]).is_dir() and not re.fullmatch(
            r"[0-9a-fA-F]{40}", config["model"]["revision"]):
        raise ValueError("Pin model.revision to the full Hugging Face commit SHA before loading model/tokenizer")
