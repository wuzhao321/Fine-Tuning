"""The ONLY model construction entry point for E0–E6."""
from __future__ import annotations

import hashlib
import inspect

from .config import require_pinned_model

EXPECTED_STRUCTURE = {"model_type": "qwen2", "num_hidden_layers": 40, "hidden_size": 5120,
                      "num_attention_heads": 40, "num_key_value_heads": 40,
                      "intermediate_size": 13696, "vocab_size": 152064,
                      "tie_word_embeddings": False}


def memory_snapshot():
    import torch
    torch.cuda.synchronize()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def require_gpu(config, log):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU must support bf16")
    props = torch.cuda.get_device_properties(0)
    log.meta["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                       "uuid": str(getattr(props, "uuid", "unavailable")),
                       "cuda_runtime": torch.version.cuda,
                       "peak_flops_tflops": config["hardware"]["peak_flops_tflops"]}
    log.write("hardware")


def parameter_inventory(model):
    groups = {}
    for name, param in model.named_parameters():
        group = "adapter" if "lora_" in name else (
            "embedding" if "embed_tokens" in name else "lm_head" if "lm_head" in name else "backbone")
        key = f"{group}:{param.dtype}:{type(param).__name__}"
        bucket = groups.setdefault(key, {"stored_numel": 0, "tensor_bytes": 0, "trainable_numel": 0})
        bucket["stored_numel"] += param.numel()
        bucket["tensor_bytes"] += param.numel() * param.element_size()
        bucket["trainable_numel"] += param.numel() if param.requires_grad else 0
    # Params4bit tensor bytes exclude quantization metadata / allocator overhead.
    return groups


def build_model(config, log, training=True, adapter_path=None):
    require_pinned_model(config)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    require_gpu(config, log)
    set_seed(config["train"]["seed"])
    model_config = AutoConfig.from_pretrained(config["model"]["path"],
                                              revision=config["model"]["revision"], trust_remote_code=False)
    actual = {key: getattr(model_config, key, None) for key in EXPECTED_STRUCTURE}
    if actual != EXPECTED_STRUCTURE:
        raise ValueError(f"Model structure mismatch: {actual}")
    log.write("structure", structure=actual, resolved_model_revision=getattr(model_config, "_commit_hash", None))
    kwargs = dict(revision=config["model"]["revision"], dtype=torch.bfloat16,
                  device_map={"": 0}, trust_remote_code=False,
                  attn_implementation=config["attention"]["implementation"])
    if config["quantization"]["enabled"]:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head"])
    model = AutoModelForCausalLM.from_pretrained(config["model"]["path"], **kwargs)
    model.config.use_cache = False
    log.write("memory", stage="after_load", memory=memory_snapshot(), inventory=parameter_inventory(model))
    if any(type(module).__name__ == "Linear4bit" for module in
           (model.get_input_embeddings(), model.get_output_embeddings())):
        raise RuntimeError("Embedding and lm_head must remain unquantized")
    targets = [module for name, module in model.named_modules()
               if name.rsplit(".", 1)[-1] in config["lora"]["target_modules"]]
    if len(targets) != 160 or any((m.in_features, m.out_features) != (5120, 5120) for m in targets):
        raise RuntimeError("Expected 160 square attention projections")
    if training and config["quantization"]["enabled"]:
        source = inspect.getsource(prepare_model_for_kbit_training)
        before = parameter_inventory(model)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        log.write("kbit_preparation", source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   signature=str(inspect.signature(prepare_model_for_kbit_training)),
                   inventory_before=before, inventory_after=parameter_inventory(model),
                   actual_gc=bool(model.is_gradient_checkpointing), memory=memory_snapshot())
        # Explicit policy: restore the large frozen tables to bf16; keep norms fp32.
        model.get_input_embeddings().to(dtype=torch.bfloat16)
        model.get_output_embeddings().to(dtype=torch.bfloat16)
    else:
        model.requires_grad_(False)
    if training:
        for name, module in model.named_modules():
            if name.endswith(("input_layernorm", "post_attention_layernorm", ".norm")):
                module.to(dtype=torch.float32)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=training)
    elif config["lora"]["enabled"]:
        spec = config["lora"]
        model = get_peft_model(model, LoraConfig(r=spec["r"], lora_alpha=spec["alpha"],
                              lora_dropout=spec["dropout"], target_modules=spec["target_modules"],
                              bias="none", task_type="CAUSAL_LM"))
    if training:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if count != config["lora"]["expected_trainable"]:
            raise RuntimeError(f"Trainable parameters: {count}, expected 104857600")
    else:
        model.requires_grad_(False)
        count = 0
    model.gradient_checkpointing_disable()
    if training and config["gc"]["enabled"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": config["gc"]["use_reentrant"]})
        if config["gc"]["use_reentrant"]:
            model.enable_input_require_grads()
    actual_gc = bool(model.is_gradient_checkpointing)
    if actual_gc != bool(training and config["gc"]["enabled"]):
        raise RuntimeError("Actual GC state differs from config")
    actual_attention = model.config._attn_implementation
    if actual_attention != config["attention"]["implementation"]:
        raise RuntimeError(f"Attention changed to {actual_attention}")
    model.train(training)
    log.write("model_ready", trainable_parameters=count, actual_gc=actual_gc,
              actual_attention=actual_attention, memory=memory_snapshot(), inventory=parameter_inventory(model))
    return model


def build_optimizer(model, config):
    import bitsandbytes as bnb
    cls = {"paged_adamw_8bit": bnb.optim.PagedAdamW8bit, "adamw_8bit": bnb.optim.AdamW8bit}[
        config["optimizer"]["name"]]
    return cls([p for p in model.parameters() if p.requires_grad], lr=config["train"]["learning_rate"],
               weight_decay=config["optimizer"]["weight_decay"])
