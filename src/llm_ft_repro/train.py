"""E5: shared model factory, explicit optimizer, Trainer checkpoint/resume support."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import apply_environment
from .data import TaskDataset, audit_task_bundle, load_tokenizer, task_collator
from .evidence import EvidenceLog
from .artifacts import seal_artifact, training_identity, verify_resume
from .model_factory import build_model, build_optimizer, memory_snapshot


def run_training(config, resume=None):
    apply_environment(config)
    log = EvidenceLog(config)
    try:
        import torch
        from transformers import Trainer, TrainerCallback, TrainingArguments
        if config["role"] != "train":
            raise ValueError("Expected training config")
        training = config["train"]
        output = Path(config["output"]["checkpoint_dir"]) / config["run_id"] / log.attempt_id
        output.mkdir(parents=True, exist_ok=False)
        audit = audit_task_bundle(config)
        identity = training_identity(config, audit, log.meta["provenance"])
        resume_receipt = None
        if resume:
            resume_receipt = verify_resume(resume, identity)
        (output / "identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        tokenizer = load_tokenizer(config)
        from .config import canonical_hash, read_mapping
        if canonical_hash(tokenizer.chat_template) != read_mapping(config["data"]["manifest"])["chat_template_sha256"]:
            raise ValueError("Training tokenizer template differs from prepared data")
        train_data = TaskDataset(config, config["data"]["path"])
        validation_data = TaskDataset(config, config["data"]["validation_path"])
        if {r["group_id"] for r in train_data.rows} & {r["group_id"] for r in validation_data.rows}:
            raise ValueError("Training/validation context leakage")
        log.write("training_inputs", identity=identity, data_audit=audit, checkpoint_dir=str(output),
                   resume_from=resume, resume_receipt=resume_receipt,
                   train_examples=len(train_data), validation_examples=len(validation_data))
        model = build_model(config, log)
        optimizer = build_optimizer(model, config)

        class RawCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                import math
                if any(isinstance(value, float) and not math.isfinite(value) for value in (logs or {}).values()):
                    raise FloatingPointError("Trainer emitted non-finite metrics")
                log.write("training_metrics", global_step=state.global_step, epoch=state.epoch,
                           metrics=logs or {}, memory=memory_snapshot())

            def on_save(self, args, state, control, **kwargs):
                path = output / f"checkpoint-{state.global_step}"
                receipt = seal_artifact(path, "checkpoint", identity, state.global_step)
                log.write("checkpoint", global_step=state.global_step,
                           path=str(path), receipt=receipt)

        args = TrainingArguments(
            output_dir=str(output), per_device_train_batch_size=config["data"]["micro_batch"],
            per_device_eval_batch_size=1, gradient_accumulation_steps=config["data"]["grad_accum"],
            learning_rate=training["learning_rate"], lr_scheduler_type=training["scheduler"],
            warmup_ratio=training["warmup_ratio"], num_train_epochs=training["epochs"],
            bf16=True, fp16=False, tf32=False, seed=training["seed"], data_seed=training["seed"],
            # Factory already owns GC. Trainer must not override its explicit non-reentrant policy.
            gradient_checkpointing=False, max_grad_norm=config["optimizer"]["max_grad_norm"],
            weight_decay=config["optimizer"]["weight_decay"], eval_strategy="steps",
            eval_steps=training["eval_steps"], save_strategy="steps", save_steps=training["save_steps"],
            save_total_limit=training["save_total_limit"], logging_steps=training["logging_steps"],
            prediction_loss_only=True, logging_nan_inf_filter=False, report_to=[], remove_unused_columns=False,
            dataloader_num_workers=0, save_safetensors=True,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_data,
                          eval_dataset=validation_data, data_collator=task_collator(tokenizer),
                          processing_class=tokenizer, optimizers=(optimizer, None), callbacks=[RawCallback()])
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        validation = trainer.evaluate()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        final = output / "final_adapter"
        trainer.save_model(str(final))
        tokenizer.save_pretrained(str(final))
        (final / "training_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
        receipt = seal_artifact(final, "adapter", identity, trainer.state.global_step)
        log.write("result", status="ok", metrics=result.metrics, validation_metrics=validation,
                   elapsed_s=elapsed, elapsed_includes_validation=True, final_adapter=str(final),
                   memory=memory_snapshot(), resumed=bool(resume), identity=identity,
                   final_adapter_receipt=receipt, global_step=trainer.state.global_step,
                   resume_receipt=resume_receipt)
        return 0
    except Exception as exc:
        log.failure(exc)
        return 2
