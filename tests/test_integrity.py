"""Focused standard-library regressions; synthetic evidence stays in temporary directories."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_ft_repro.analysis import (benchmark_consistency, collect, create_review_template, summarize)
from llm_ft_repro.artifacts import REQUIRED, seal_artifact, verify_artifact
from llm_ft_repro.bench import mfu
from llm_ft_repro.config import canonical_hash, load_config, validate
from llm_ft_repro.data import audit_task_bundle, normalize
from llm_ft_repro.evidence import sha256_file
from llm_ft_repro.orchestrate import budget_spent, result_for_job
from llm_ft_repro.preflight import preflight

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def task_fixture(folder):
    config = load_config(ROOT / "configs/train/e5_qlora.yaml")
    config["model"]["revision"] = "a" * 40
    config["data"]["manifest"] = str(folder / "manifest.json")
    sources = [{"id": "fixture", "url": "fixture://synthetic", "license": "fixture-only",
                "license_url": "fixture://license", "revision": "fixture", "approved": True,
                "approved_by": "unit-fixture", "approved_at": "2026-10-24"}]
    manifest = {"schema_version": 2, "kind": "task", "model": config["model"], "sources": sources,
                "seq_len": 2048, "max_target_tokens": 512, "files": {},
                "split_policy": {"test_per_task": 1, "validation_per_task": 1, "train_limit": 2}}
    for split, key in (("train", "path"), ("validation", "validation_path"), ("test", "test_path")):
        rows = []
        for task in ("qa", "summary"):
            context, question = f"SYNTHETIC {split} {task}", "fixture question" if task == "qa" else ""
            rows.append({"task": task, "source_id": "fixture", "context": context, "question": question,
                         "answers": ["fixture answer"],
                         "id": canonical_hash([task, normalize(context), normalize(question)]),
                         "group_id": canonical_hash(normalize(context)),
                         "input_ids": [1, 2, 3, 4], "prompt_ids": [1, 2], "labels": [-100, -100, 3, 4]})
        path = folder / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        config["data"][key] = str(path)
        manifest["files"][path.name] = sha256_file(path)
    dump(folder / "manifest.json", manifest)
    return config


def change_rows(config, split, mutate):
    key = {"train": "path", "validation": "validation_path", "test": "test_path"}[split]
    path = Path(config["data"][key])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest_path = Path(config["data"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = sha256_file(path)
    dump(manifest_path, manifest)


def benchmark_record(repeat=0):
    config = load_config(ROOT / "configs/bench/e1_qlora.yaml")
    return {"schema_version": 1, "event": "result", "status": "ok", "run_id": "e1_qlora", "experiment": "E1",
            "repeat": repeat, "attempt_id": f"SYNTHETIC_FIXTURE_{repeat}", "timestamp": "2026-10-24T00:00:00+00:00",
            "config": config, "config_sha256": canonical_hash(config), "gpu": {"name": "SYNTHETIC_FIXTURE"},
            "provenance": {"git_commit": "fixture-only", "git_dirty": False, "source_sha256": "fixture",
                           "versions": {}, "python": "fixture", "platform": "fixture"},
            "data_sha256": "fixture", "step_times_s": [2.0] * 30, "losses": [1.0] * 40,
            "median_step_s": 2.0, "iqr_step_s": 0.0, "tokens_per_second": 2048,
            "input_tokens_per_step": 4096, "predicted_tokens_per_step": 4094,
            "mfu": mfu(2048, True, 312),
            "memory": {"allocated_bytes": 10000000000, "reserved_bytes": 22000000000,
                       "peak_allocated_bytes": 20000000000, "peak_reserved_bytes": 22000000000}}


def write_benchmark(path, record):
    meta = {k: record[k] for k in ("schema_version", "run_id", "experiment", "repeat", "attempt_id", "timestamp",
                                  "config", "config_sha256", "gpu", "provenance")}
    events = [{**meta, "event": "started"}, {**meta, "event": "model_ready", "trainable_parameters": 104857600,
               "memory": {"allocated_bytes": 10000000000}},
              {**meta, "event": "telemetry", "samples": [{"phase": "measure", "used_bytes": 21000000000}]}, record]
    path.write_text("\n" + "\n\n".join(json.dumps(e) for e in events) + "\n")


class IntegrityTests(unittest.TestCase):
    def test_nonfinite_config_rejected(self):
        config = load_config(ROOT / "configs/base.yaml")
        config["train"]["learning_rate"] = float("nan")
        with self.assertRaises(ValueError):
            validate(config)

    def test_preflight_is_read_only_and_reports_missing_inputs(self):
        result = preflight(load_config(ROOT / "configs/train/e5_qlora.yaml"))
        self.assertFalse(result["ready"])
        self.assertIn("model_reference", [r["name"] for r in result["checks"] if r["status"] == "blocked"])

    def test_bundle_validates_without_transformers(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = task_fixture(Path(temporary))
            self.assertEqual(audit_task_bundle(config)["counts"]["test"], {"qa": 1, "summary": 1})

    def test_rehashed_bad_labels_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = task_fixture(Path(temporary))
            change_rows(config, "train", lambda rows: rows[0]["labels"].__setitem__(0, 1))
            with self.assertRaisesRegex(ValueError, "masking"):
                audit_task_bundle(config)

    def test_cross_split_context_leak_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = task_fixture(Path(temporary))
            def leak(rows):
                row = rows[0]
                row["context"] = "SYNTHETIC train qa"
                row["question"] = "another fixture question"
                row["group_id"] = canonical_hash(normalize(row["context"]))
                row["id"] = canonical_hash(["qa", normalize(row["context"]), normalize(row["question"])])
            change_rows(config, "test", leak)
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit_task_bundle(config)

    def test_checkpoint_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "checkpoint-10"
            folder.mkdir()
            for name in REQUIRED["checkpoint"]:
                (folder / name).write_bytes(b"SYNTHETIC not a real training checkpoint")
            dump(folder / "trainer_state.json", {"global_step": 10})
            identity = {"fixture": True}
            seal_artifact(folder, "checkpoint", identity, 10)
            self.assertEqual(verify_artifact(folder, "checkpoint", identity)["global_step"], 10)
            (folder / "optimizer.pt").write_bytes(b"corrupted fixture")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_artifact(folder, "checkpoint", identity)

    def test_incomplete_checkpoint_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                seal_artifact(Path(temporary), "checkpoint", {}, 10)

    def test_benchmark_values_are_recomputed(self):
        row = benchmark_record()
        self.assertTrue(all(benchmark_consistency(row).values()))
        row["tokens_per_second"] *= 1.01
        self.assertFalse(benchmark_consistency(row)["recomputed_statistics"])

    def test_summary_reviews_bind_to_evidence_and_real_line_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw, out = folder / "raw", folder / "out"
            raw.mkdir()
            for repeat in range(3):
                write_benchmark(raw / f"fixture{repeat}.jsonl", benchmark_record(repeat))
            self.assertEqual(collect(raw)[0]["line"], 8)
            self.assertEqual(summarize(raw, out)["accepted"], 0)
            review_path = folder / "reviews.json"
            create_review_template(out / "summary.json", review_path)
            reviews = json.loads(review_path.read_text())
            for review in reviews["groups"].values():
                review.update(decision="approve", reviewer="SYNTHETIC unit fixture", rationale="fixture only")
                for check in review["checks"].values():
                    check.update(status="passed", evidence="SYNTHETIC unit fixture")
            dump(review_path, reviews)
            self.assertEqual(summarize(raw, out, review_path)["accepted"], 1)
            changed = benchmark_record(0)
            changed["tokens_per_second"] += 1
            write_benchmark(raw / "fixture0.jsonl", changed)
            self.assertEqual(summarize(raw, out, review_path)["accepted"], 0)

    def test_old_oom_is_not_a_new_job_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.jsonl"
            path.write_text(json.dumps({"event": "result", "job_id": "old", "status": "oom"}) + "\n")
            self.assertEqual(result_for_job(temporary, "new"), {})

    def test_duplicate_budget_finish_rejected(self):
        events = [{"event": "job_started", "job_id": "one"},
                  {"event": "job_finished", "job_id": "one", "elapsed_s": 10}]
        self.assertEqual(budget_spent(events), 10)
        with self.assertRaises(ValueError):
            budget_spent(events + [events[-1]])

    def test_unclosed_budget_rejected(self):
        with self.assertRaises(RuntimeError):
            budget_spent([{"event": "job_started", "job_id": "one"}])


if __name__ == "__main__":
    unittest.main()
