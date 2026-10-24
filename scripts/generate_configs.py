"""Generate checked-in YAML (JSON subset) experiment matrix. Never overwrites edits."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "run_id": "base", "experiment": "E0", "description": "Explicit common protocol", "role": "inspect",
    "model": {"path": "Qwen/Qwen1.5-14B", "revision": "main", "dtype": "bfloat16"},
    "quantization": {"enabled": True, "type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
    "lora": {"enabled": True, "r": 64, "alpha": 128, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "expected_trainable": 104857600},
    "attention": {"implementation": "flash_attention_2"}, "gc": {"enabled": True, "use_reentrant": False},
    "optimizer": {"name": "paged_adamw_8bit", "weight_decay": 0.0, "max_grad_norm": 1.0},
    "data": {"mode": "synthetic_packed", "path": "data/processed/bench/tokens.u32",
             "validation_path": "data/processed/tasks/validation.jsonl", "test_path": "data/processed/tasks/test.jsonl",
             "manifest": "data/processed/bench/tokens.u32.manifest.json", "seq_len": 2048,
             "micro_batch": 2, "grad_accum": 1, "max_target_tokens": 512},
    "bench": {"warmup_steps": 10, "measure_steps": 30, "repeats": 3, "nvml_interval_s": 0.1},
    "train": {"learning_rate": 0.0002, "scheduler": "cosine", "warmup_ratio": 0.03, "epochs": 1,
              "seed": 42, "save_steps": 100, "eval_steps": 100, "logging_steps": 10, "save_total_limit": 3},
    "evaluation": {"adapter_path": "", "max_new_tokens": 512},
    "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "OMP_NUM_THREADS": "4"},
    "hardware": {"peak_flops_tflops": 312, "nvml_device": "0"},
    "output": {"raw_dir": "results/raw", "checkpoint_dir": "checkpoints"},
}


def generate():
    files = {"base.yaml": BASE}

    def add(folder, name, experiment, description, **overrides):
        files[f"{folder}/{name}.yaml"] = {
            "extends": "../base.yaml", "run_id": name, "experiment": experiment,
            "description": description, "role": "benchmark", **overrides}

    add("inspect", "e0_model", "E0", "结构、量化和最小前反向检查", role="inspect")
    for mode in ("bf16_lora", "qlora"):
        add("bench", f"e1_{mode}", "E1", f"{mode}: FA2 + GC",
            quantization={"enabled": mode == "qlora"})
    for attn, short in (("eager", "eager"), ("sdpa", "sdpa"), ("flash_attention_2", "fa2")):
        for gc in (False, True):
            add("bench", f"e2_{short}_gc_{'on' if gc else 'off'}", "E2", "Attention × GC ablation",
                attention={"implementation": attn}, gc={"enabled": gc})
        for length in (512, 1024, 2048, 4096, 8192):
            add("bench", f"e3_{short}_s{length}", "E3", "Sequence-length scan; 8192 uses more tokens per step",
                attention={"implementation": attn}, data={"seq_len": length, "micro_batch": max(1, 4096 // length)})
    add("diagnostics", "e2_eager_fallback", "E2", "Only after eager/no-GC OOM; separate evidence",
        attention={"implementation": "eager"}, gc={"enabled": False}, data={"seq_len": 1024, "micro_batch": 1})
    for attn, short in (("eager", "eager"), ("flash_attention_2", "fa2")):
        add("diagnostics", f"e2_equivalence_{short}", "E2", "Paired numerical diagnostic; not throughput evidence",
            role="equivalence", attention={"implementation": attn}, lora={"dropout": 0.0},
            data={"seq_len": 1024, "micro_batch": 1}, bench={"warmup_steps": 0, "measure_steps": 20, "repeats": 1})
    for optimizer in ("paged_adamw_8bit", "adamw_8bit"):
        add("bench", f"e4_{optimizer}", "E4", "Optimizer comparison; paging requires separate profiler evidence",
            optimizer={"name": optimizer})
    task_data = {"mode": "task", "path": "data/processed/tasks/train.jsonl",
                 "manifest": "data/processed/tasks/manifest.json", "grad_accum": 8}
    for mode in ("bf16_lora", "qlora"):
        add("train", f"e5_{mode}", "E5", "Same seed, data and training hyperparameters", role="train",
            quantization={"enabled": mode == "qlora"}, data=task_data)
    for kind in ("base", "finetuned"):
        add("eval", f"e6_{kind}", "E6", "Matched NF4 inference for base and fine-tuned adapter", role="eval",
            lora={"enabled": kind == "finetuned"}, gc={"enabled": False}, data=task_data,
            evaluation={"adapter_path": "TODO_SET_FINAL_ADAPTER_PATH" if kind == "finetuned" else ""})
    for relative, value in files.items():
        path = ROOT / "configs" / relative
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"Refusing to overwrite edited configuration: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Generated {len(files)} configs")


if __name__ == "__main__":
    generate()
