"""Evidence aggregation. Computed values remain quarantined until review is bound to raw files."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .bench import mfu, percentile
from .config import canonical_hash, read_mapping
from .data import read_jsonl_numbered
from .evidence import sha256_file

REVIEW_CHECKS = ("static_memory", "numerical_equivalence", "gc_overhead", "hardware", "provenance")
ROLE_REVIEWS = {
    "benchmark": REVIEW_CHECKS,
    "inspect": ("model_structure", "hardware", "provenance"),
    "train": ("data_integrity", "checkpoint_integrity", "convergence", "hardware", "provenance"),
    "eval": ("data_integrity", "adapter_integrity", "metrics", "hardware", "provenance"),
    "equivalence": ("numerical_equivalence", "hardware", "provenance"),
}


def collect(raw_dir):
    attempts = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        if path.name == "budget.jsonl":
            continue
        before = sha256_file(path)
        rows = list(read_jsonl_numbered(path))
        if not rows:
            continue
        if any(r.get("schema_version") != 1 for _, r in rows):
            raise ValueError(f"Unsupported evidence schema: {path}")
        first = rows[0][1]
        if first.get("event") != "started":
            raise ValueError(f"Missing start event: {path}")
        for _, record in rows:
            if any(record.get(k) != first.get(k) for k in
                   ("attempt_id", "run_id", "repeat", "experiment", "config_sha256", "provenance")):
                raise ValueError(f"Mixed identities in evidence: {path}")
            if canonical_hash(record["config"]) != record["config_sha256"]:
                raise ValueError(f"Config hash mismatch: {path}")
        result = [(line, row) for line, row in rows if row["event"] == "result"]
        if len(result) > 1 or (result and result[0][0] != rows[-1][0]):
            raise ValueError(f"Multiple or non-terminal result events: {path}")
        after = sha256_file(path)
        if before != after:
            raise ValueError(f"Evidence changed during summary; wait for the writer: {path}")
        record = result[-1][1] if result else {**rows[-1][1], "status": "interrupted"}
        attempts.append({"record": record, "rows": rows, "path": str(path),
                         "line": result[-1][0] if result else rows[-1][0], "sha256": after})
    return attempts


def group_identity(attempt):
    row = attempt["record"]
    provenance = row["provenance"]
    return canonical_hash({"config_sha256": row["config_sha256"], "gpu": row["gpu"],
                           "provenance": provenance,
                           "training_identity": row.get("identity") or next(
                               (r.get("identity") for _, r in attempt["rows"] if r["event"] == "training_inputs"), None),
                           "adapter_receipt": row.get("adapter_receipt"),
                           "data_sha256": row.get("data_sha256") or next(
                               (r.get("data_sha256") for _, r in attempt["rows"] if r["event"] == "data"), None)})


def benchmark_consistency(record):
    """Recompute reported statistics from primary samples; never trust summary scalars alone."""
    def finite(value, positive=False):
        return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)
    config = record["config"]
    times, losses = record.get("step_times_s", []), record.get("losses", [])
    valid_times = isinstance(times, list) and len(times) == config["bench"]["measure_steps"] and all(
        finite(t, True) for t in times)
    checks = {"step_samples": valid_times,
              "loss_samples": isinstance(losses, list) and len(losses) == sum(config["bench"][k] for k in
                  ("warmup_steps", "measure_steps")) and all(finite(v) for v in losses)}
    tokens = config["data"]["micro_batch"] * config["data"]["seq_len"]
    checks["token_count"] = record.get("input_tokens_per_step") == tokens and record.get(
        "predicted_tokens_per_step") == config["data"]["micro_batch"] * (config["data"]["seq_len"] - 1)
    checks["recomputed_statistics"] = False
    if valid_times:
        median = statistics.median(times)
        expected = {"median_step_s": median, "iqr_step_s": percentile(times, .75) - percentile(times, .25),
                    "tokens_per_second": tokens / median,
                    "mfu": mfu(tokens / median, config["gc"]["enabled"], config["hardware"]["peak_flops_tflops"])}
        checks["recomputed_statistics"] = all(finite(record.get(k)) and math.isclose(record[k], v,
            rel_tol=1e-8, abs_tol=1e-9) for k, v in expected.items())
    memory = record.get("memory", {})
    checks["memory_samples"] = all(type(memory.get(k)) is int and memory[k] >= 0 for k in
        ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"))
    if checks["memory_samples"]:
        checks["memory_samples"] = (0 < memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"] and
                                    memory["allocated_bytes"] <= memory["peak_allocated_bytes"] and
                                    memory["reserved_bytes"] <= memory["peak_reserved_bytes"])
    return checks


def evaluation_consistency(attempt):
    record = attempt["record"]
    predictions = [r for _, r in attempt["rows"] if r["event"] == "prediction"]
    if not predictions or len({r.get("sample_id") for r in predictions}) != len(predictions):
        return False
    expected_counts = record.get("data_audit", {}).get("counts", {}).get("test")
    if record.get("counts") != expected_counts or not expected_counts:
        return False
    for task, keys in (("qa", {"em", "f1"}), ("summary", {"rouge1", "rouge2", "rougeL"})):
        rows = [r for r in predictions if r.get("task") == task]
        if len(rows) != expected_counts.get(task) or not rows:
            return False
        for row in rows:
            if set(row.get("scores", {})) != keys or any(type(v) not in (int, float) or
                    not math.isfinite(v) or not 0 <= v <= 1 for v in row["scores"].values()):
                return False
        metrics = record.get("metrics", {}).get(task, {})
        if set(metrics) != keys or any(type(metrics[k]) not in (int, float) or not math.isclose(
                metrics[k], statistics.mean(r["scores"][k] for r in rows), rel_tol=1e-8, abs_tol=1e-9) for k in keys):
            return False
    return sum(expected_counts.values()) == len(predictions)


def create_review_template(summary_path, output_path):
    rows = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    template = {"groups": {r["group_key"]: {"evidence_digest": r["evidence_digest"], "decision": "pending",
        "reviewer": "", "rationale": "", "checks": {name: {"status": "pending", "evidence": ""}
            for name in r["required_reviews"]}} for r in rows}}
    with Path(output_path).open("x", encoding="utf-8") as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2)
    return {"groups": len(rows), "output": str(output_path)}


def csv_write(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in fields})


def summarize(raw_dir, output_dir, reviews_path=None, excel=False, figures=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    attempts = collect(raw_dir)
    reviews = read_mapping(reviews_path).get("groups", {}) if reviews_path else {}
    grouped = defaultdict(list)
    for attempt in attempts:
        grouped[group_identity(attempt)].append(attempt)
    summaries, trace = [], []
    for key, items in grouped.items():
        first = items[0]["record"]
        config, role = first["config"], first["config"]["role"]
        evidence_digest = canonical_hash(sorted((item["path"], item["sha256"]) for item in items))
        row = {"group_key": key, "evidence_digest": evidence_digest, "run_id": first["run_id"],
               "experiment": first["experiment"], "role": role, "attempts": len(items),
               "gpu": first["gpu"], "config": config, "status_counts": {}, "checks": {},
               "provenance": first["provenance"],
               "data_sha256": first.get("data_sha256"),
               "raw_refs": [f"{i['path']}:{i['line']}" for i in items],
               "timestamp": first["timestamp"]}
        for item in items:
            state = item["record"]["status"]
            row["status_counts"][state] = row["status_counts"].get(state, 0) + 1
        checks = row["checks"]
        checks["all_success"] = all(i["record"]["status"] == "ok" for i in items)
        checks["git_recorded"] = all(i["record"]["provenance"]["git_commit"] is not None for i in items)
        checks["unique_attempts"] = len({i["record"]["attempt_id"] for i in items}) == len(items)
        if role == "benchmark":
            checks["three_distinct_repeats"] = len(items) == 3 and {i["record"]["repeat"] for i in items} == {0, 1, 2}
            success = [i for i in items if i["record"]["status"] == "ok"]
            validations = [benchmark_consistency(i["record"]) for i in success]
            for name in ("step_samples", "loss_samples", "token_count", "recomputed_statistics", "memory_samples"):
                checks[name] = bool(validations) and all(v[name] for v in validations)
            success = [i for i, v in zip(success, validations) if all(v.values())]
            checks["trainable_parameters"] = all(any(r["event"] == "model_ready" and
                    r.get("trainable_parameters") == 104857600 for _, r in i["rows"]) for i in items)
            checks["nvml_measured_samples"] = all(any(r["event"] == "telemetry" and
                    any(s["phase"] == "measure" and s.get("used_bytes") is not None for s in r["samples"])
                    for _, r in i["rows"]) for i in items)
            if success:
                speeds = [i["record"]["tokens_per_second"] for i in success]
                peaks = [i["record"]["memory"]["peak_allocated_bytes"] / 1e9 for i in success]
                row.update(tokens_per_second=statistics.median(speeds),
                           throughput_min=min(speeds), throughput_max=max(speeds),
                           peak_allocated_gb=statistics.median(peaks),
                           peak_reserved_gb=statistics.median(i["record"]["memory"]["peak_reserved_bytes"] / 1e9 for i in success),
                           median_step_s=statistics.median(i["record"]["median_step_s"] for i in success),
                           mfu=statistics.median(i["record"]["mfu"] for i in success))
                checks["repeat_spread"] = (max(speeds)-min(speeds)) / statistics.median(speeds) <= .1
                checks["mfu"] = all(0 < i["record"]["mfu"] <= .6 and abs(
                    i["record"]["mfu"] - mfu(i["record"]["tokens_per_second"], config["gc"]["enabled"],
                                              config["hardware"]["peak_flops_tflops"])) < 1e-9 for i in success)
                static_values = []
                for item in success:
                    ready = next((r for _, r in item["rows"] if r["event"] == "model_ready"), None)
                    if ready:
                        static_values.append(ready["memory"]["allocated_bytes"] / 1e9)
                row["static_after_lora_gb"] = statistics.median(static_values) if static_values else None
                row["static_estimate_gb"] = 10.7 if config["quantization"]["enabled"] else 30.0
                row["static_deviation_fraction"] = (abs(row["static_after_lora_gb"] - row["static_estimate_gb"]) /
                                                       row["static_estimate_gb"] if static_values else None)
        elif checks["all_success"]:
            checks["single_attempt"] = len(items) == 1
            if role == "train":
                checks["training_identity"] = all(i["record"].get("identity", {}).get("schema_version") == 2 for i in items)
                checks["checkpoint_receipt"] = all(bool(i["record"].get("final_adapter_receipt")) for i in items)
                checks["validation_loss"] = all(type(i["record"].get("validation_metrics", {}).get("eval_loss"))
                    in (int, float) and math.isfinite(i["record"]["validation_metrics"]["eval_loss"]) for i in items)
            if role == "eval":
                checks["test_bundle"] = all(bool(i["record"].get("data_audit")) for i in items)
                checks["predictions"] = all(evaluation_consistency(i) for i in items)
            row["metrics"] = [i["record"].get("metrics", {}) for i in items]
            row["validation_metrics"] = [i["record"].get("validation_metrics", {}) for i in items]
        review = reviews.get(key, {})
        row["required_reviews"] = list(ROLE_REVIEWS[role])
        approved = (review.get("evidence_digest") == evidence_digest and review.get("decision") == "approve"
                    and bool(review.get("reviewer")) and bool(review.get("rationale"))
                    and all(review.get("checks", {}).get(name, {}).get("status") == "passed"
                            and review["checks"][name].get("evidence") for name in row["required_reviews"]))
        # Numerical diagnostic configs are never published as performance results.
        row["eligibility"] = ("accepted" if checks and all(checks.values()) and approved and role != "equivalence"
                              else "blocked" if not all(checks.values()) else "pending_review")
        row["review"] = review
        summaries.append(row)
        for field in ("tokens_per_second", "peak_allocated_gb", "peak_reserved_gb", "mfu", "median_step_s",
                      "static_after_lora_gb", "metrics", "validation_metrics"):
            if field in row:
                refs = row["raw_refs"]
                if field == "static_after_lora_gb":
                    refs = [f"{i['path']}:{line}" for i in items for line, event in i["rows"]
                            if event["event"] == "model_ready"]
                trace.append({"source_key": f"{key}.{field}", "value": row[field],
                              "eligibility": row["eligibility"], "raw_refs": refs,
                              "evidence_digest": evidence_digest})
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["group_key", "run_id", "experiment", "role", "eligibility", "attempts", "status_counts",
              "static_after_lora_gb", "peak_allocated_gb", "peak_reserved_gb", "tokens_per_second",
              "throughput_min", "throughput_max", "median_step_s", "mfu", "metrics", "validation_metrics",
              "checks", "required_reviews", "evidence_digest", "raw_refs"]
    csv_write(output / "audit.csv", summaries, fields)
    csv_write(output / "accepted.csv", [r for r in summaries if r["eligibility"] == "accepted"], fields)
    csv_write(output / "traceability.csv", trace, ["source_key", "value", "eligibility", "raw_refs", "evidence_digest"])
    if excel:
        from openpyxl import Workbook
        book = Workbook()
        sheet = book.active
        sheet.title = "Audit"
        sheet.append(fields)
        for row in summaries:
            sheet.append([json.dumps(row.get(f), ensure_ascii=False) if isinstance(row.get(f), (dict, list))
                          else row.get(f) for f in fields])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        book.save(output / "summary.xlsx")
    if figures:
        plot_results(summaries, output / "figures")
        plot_losses(attempts, summaries, output / "figures")
    write_report(summaries, output)
    return {"groups": len(summaries), "accepted": sum(r["eligibility"] == "accepted" for r in summaries),
            "output": str(output)}


def write_report(rows, output):
    accepted = [r for r in rows if r["eligibility"] == "accepted"]
    lines = ["# 复现实验报告", "", "本报告由原始证据汇总生成；待审查和失败结果仅列入 audit.csv。", "",
             "## 环境", "", "环境、GPU、依赖与代码指纹见 summary.json 中对应分组及原始日志。", "",
             "## 方法", "", "单卡 Qwen1.5-14B，公开替代数据。基准采用预装载、无 padding 的定长 token 流；"
             "10 步预热、30 步测量，每配置 3 个独立进程。GB 为十进制字节 / 1e9。", "",
             "## 结果", ""]
    if not accepted:
        lines.append("【未测或尚无通过全部审查的结果】")
    for row in accepted:
        lines.extend([f"### {row['run_id']}", "", f"来源键：`{row['group_key']}`", "",
                      "```json", json.dumps({k: row[k] for k in ("tokens_per_second", "peak_allocated_gb", "mfu",
                                                                 "metrics", "validation_metrics") if k in row},
                                           ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["", "## 校验", "", "自动校验和人工审查均绑定原始文件哈希。全部尝试（含 OOM/中断）见 audit.csv。",
                  "", "## 局限", "", "MFU 是估算口径：4N / 6N，N=13.39B，忽略 attention 与 LoRA FLOPs；"
                  "长序列时忽略项会增大。NVML 不能证明优化器是否换页。数值比较的阈值必须事先声明。",
                  "", "## 结论", "", "仅能引用 accepted.csv 的已审查结果；本报告不自动推断方法优劣。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidates = [r for r in accepted if r["run_id"] == "e1_qlora"]
    if len(candidates) == 1:
        r = candidates[0]
        key = r["group_key"]
        snippet = (f"在 {r['gpu']['name']} 上完成公开替代数据复现实验（{r['timestamp'][:10]}，UTC），"
                   f"峰值 allocated 显存 {r['peak_allocated_gb']:.2f} GB [{key}.peak_allocated_gb]，"
                   f"吞吐 {r['tokens_per_second']:.1f} tokens/s [{key}.tokens_per_second]，"
                   f"估算 MFU {r['mfu']:.1%} [{key}.mfu]。任务改善与方法对照结论须另附对应证据。")
    else:
        snippet = "公开替代数据复现实验：硬件【未测】，日期【未测】，峰值显存【未测】，吞吐【未测】，MFU【未测】。"
    (output / "resume_snippet.md").write_text(snippet + "\n", encoding="utf-8")


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["eligibility"] == "accepted" and r["role"] == "benchmark"]
    if not rows:
        return
    output.mkdir(parents=True, exist_ok=True)
    for metric, label in (("peak_allocated_gb", "Peak allocated (GB)"), ("tokens_per_second", "Input tokens / s")):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar([f"{r['run_id']} [{r['group_key'][:6]}]" for r in rows], [r[metric] for r in rows])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=80)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        curves = defaultdict(list)
        for row in rows:
            if row["experiment"] != "E3":
                continue
            config = json.loads(json.dumps(row["config"]))
            for key in ("run_id", "description"):
                config.pop(key)
            for key in ("seq_len", "micro_batch"):
                config["data"].pop(key)
            curve = canonical_hash({"config": config, "gpu": row["gpu"], "provenance": row["provenance"],
                                    "data_sha256": row["data_sha256"]})
            curves[curve].append(row)
        for key, points in curves.items():
            points.sort(key=lambda r: r["config"]["data"]["seq_len"])
            attn = points[0]["config"]["attention"]["implementation"]
            ax.plot([r["config"]["data"]["seq_len"] for r in points],
                    [r[metric] for r in points], marker="o", label=f"{attn} [{key[:6]}]")
        ax.set(xlabel="Sequence length", ylabel=label)
        if ax.lines:
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"e3_{metric}.png", dpi=160)
        plt.close(fig)


def plot_losses(attempts, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    accepted = {r["group_key"] for r in summaries if r["role"] == "train" and r["eligibility"] == "accepted"}
    if not accepted:
        return
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for attempt in attempts:
        if group_identity(attempt) not in accepted:
            continue
        for field, linestyle in (("loss", "-"), ("eval_loss", "--")):
            points = [r for _, r in attempt["rows"] if r["event"] == "training_metrics" and field in r["metrics"]]
            if points:
                ax.plot([r["global_step"] for r in points], [r["metrics"][field] for r in points],
                        linestyle, label=f"{attempt['record']['run_id']} {field}")
    ax.set(xlabel="Optimizer step", ylabel="Loss")
    if ax.lines:
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "e5_loss.png", dpi=160)
    plt.close(fig)


def compare_numerics(left_path, right_path, atol, rtol):
    left, right = collect(Path(left_path).parent), collect(Path(right_path).parent)
    a = next(i for i in left if Path(i["path"]).resolve() == Path(left_path).resolve())
    b = next(i for i in right if Path(i["path"]).resolve() == Path(right_path).resolve())
    ra, rb = a["record"], b["record"]
    ca, cb = json.loads(json.dumps(ra["config"])), json.loads(json.dumps(rb["config"]))
    if any(r["status"] != "ok" or r["config"]["role"] != "equivalence" or
           r["config"]["lora"]["dropout"] != 0 for r in (ra, rb)):
        raise ValueError("Requires successful paired equivalence runs with dropout=0")
    if {ca["attention"]["implementation"], cb["attention"]["implementation"]} != {"eager", "flash_attention_2"}:
        raise ValueError("Expected eager and FA2")
    for config in (ca, cb):
        for key in ("run_id", "description", "attention"):
            config.pop(key)
    if ca != cb or ra["data_sha256"] != rb["data_sha256"] or ra["provenance"] != rb["provenance"]:
        raise ValueError("Paired runs differ beyond attention implementation")
    losses_a, losses_b = ra["losses"][:20], rb["losses"][:20]
    if len(losses_a) != 20 or len(losses_b) != 20:
        raise ValueError("Need 20 losses per run")
    differences = [abs(x-y) for x, y in zip(losses_a, losses_b)]
    return {"passed": all(d <= atol + rtol * abs(y) for d, y in zip(differences, losses_b)),
            "atol": atol, "rtol": rtol, "max_absolute_difference": max(differences),
            "left": {"path": a["path"], "sha256": a["sha256"]},
            "right": {"path": b["path"], "sha256": b["sha256"]}, "differences": differences}
