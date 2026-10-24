"""A benchmark invocation executes exactly one configuration in one process."""
from __future__ import annotations

import statistics
import time

from .config import apply_environment
from .data import verify_data
from .evidence import EvidenceLog, sha256_file
from .model_factory import build_model, build_optimizer, memory_snapshot
from .telemetry import NvmlSampler


def mfu(tokens_per_second, gc, peak_tflops):
    return tokens_per_second * (6 if gc else 4) * 13.39e9 / (peak_tflops * 1e12)


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def run_benchmark(config, repeat=0):
    apply_environment(config)
    log = EvidenceLog(config, repeat)
    sampler = None
    try:
        import numpy as np
        import torch
        if config["role"] not in {"benchmark", "equivalence"}:
            raise ValueError("Expected benchmark or equivalence config")
        data, protocol = config["data"], config["bench"]
        manifest = verify_data(config, data["path"])
        log.write("data", manifest=manifest, data_sha256=sha256_file(data["path"]))
        count = protocol["warmup_steps"] + protocol["measure_steps"]
        step_tokens = data["micro_batch"] * data["seq_len"]
        tokens = np.fromfile(data["path"], dtype="<u4", count=count * step_tokens)
        if len(tokens) != count * step_tokens:
            raise ValueError("Packed token stream too short")
        if int(tokens.max()) >= 152064:
            raise ValueError("Packed stream contains invalid token IDs")
        model = build_model(config, log)
        optimizer = build_optimizer(model, config)
        # Preload fixed data once; timed steps exclude H2D copies and Python data processing.
        batches = torch.as_tensor(tokens.astype("int64"), device="cuda").reshape(
            count, data["micro_batch"], data["seq_len"])
        sampler = NvmlSampler(config["hardware"]["nvml_device"], protocol["nvml_interval_s"]).start()
        cuda_uuid = log.meta["gpu"]["uuid"]
        if cuda_uuid != "unavailable" and cuda_uuid != str(sampler.uuid):
            raise RuntimeError("NVML device differs from the CUDA device; configure hardware.nvml_device")
        durations, losses, first_backward_memory = [], [], None
        measured_peak = None
        for step, ids in enumerate(batches):
            if step == protocol["warmup_steps"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                sampler.phase = "measure"
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            # Take this diagnostic during warmup, outside measured steps in standard benchmarks.
            if step == 0:
                first_backward_memory = memory_snapshot()
                trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
                missing = [name for name, p in trainable if p.grad is None]
                if missing:
                    raise RuntimeError(f"Disconnected adapter gradients: {missing[:5]}")
                if any(not torch.isfinite(p.grad).all().item() for _, p in trainable):
                    raise FloatingPointError("Non-finite adapter gradients")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          config["optimizer"]["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            loss_value = loss.detach().float().item()
            if not __import__("math").isfinite(loss_value):
                raise FloatingPointError("Non-finite loss")
            losses.append(loss_value)
            if step >= protocol["warmup_steps"]:
                durations.append(elapsed)
            del loss
        measured_peak = memory_snapshot()
        telemetry = sampler.stop()
        sampler = None
        median = statistics.median(durations)
        throughput = step_tokens / median
        utilization = mfu(throughput, config["gc"]["enabled"], config["hardware"]["peak_flops_tflops"])
        log.write("memory", stage="first_forward_backward", memory=first_backward_memory)
        log.write("telemetry", **telemetry)
        log.write("result", status="ok", step_times_s=durations, losses=losses,
                   median_step_s=median, iqr_step_s=percentile(durations, .75)-percentile(durations, .25),
                   tokens_per_second=throughput, input_tokens_per_step=step_tokens,
                   predicted_tokens_per_step=data["micro_batch"]*(data["seq_len"]-1),
                   mfu=utilization, mfu_flag=utilization > .6, memory=measured_peak,
                   data_sha256=sha256_file(data["path"]), paging_observed=None)
        return 0
    except Exception as exc:
        if sampler is not None:
            log.write("telemetry", **sampler.stop())
        try:
            import torch
            if torch.cuda.is_available():
                log.write("failure_memory", memory=memory_snapshot())
        except Exception:
            pass
        log.failure(exc)
        return 2
