"""Local, licensed public-data ingestion; no dataset is silently chosen or downloaded."""
from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from .config import canonical_hash, read_mapping, require_pinned_model
from .evidence import sha256_file

SYSTEM = "你是一个可靠的中文助手。请依据用户提供的信息回答，不要编造事实。"


def read_jsonl(path):
    for _, row in read_jsonl_numbered(path):
        yield row


def read_jsonl_numbered(path):
    """Preserve physical line numbers even when the input contains blank lines."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected JSON object: {path}:{line_no}")
                    yield line_no, row
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_no}") from exc


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def check_sources(path):
    manifest = read_mapping(path)
    validate_sources(manifest)
    return manifest


def validate_sources(manifest):
    if not manifest.get("sources"):
        raise ValueError("No licensed data sources declared")
    seen = set()
    for source in manifest["sources"]:
        for key in ("id", "url", "license", "license_url", "revision", "approved_by", "approved_at"):
            if not source.get(key) or str(source[key]).startswith("TODO"):
                raise ValueError(f"Dataset source requires {key}")
        if source.get("approved") is not True:
            raise ValueError("Dataset must be selected and license reviewed before use")
        if source["id"] in seen:
            raise ValueError(f"Duplicate source id: {source['id']}")
        seen.add(source["id"])


def load_tokenizer(config):
    require_pinned_model(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"],
                                               revision=config["model"]["revision"], trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Model tokenizer has no chat template; do not invent one silently")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prompt_messages(row):
    if row["task"] == "qa":
        user = f"请根据以下资料回答问题。\n资料：{row['context']}\n问题：{row['question']}"
    elif row["task"] == "summary":
        user = f"请概括以下对话或文本的主要内容，保留关键信息。\n{row['context']}"
    else:
        raise ValueError(f"Unknown task: {row['task']}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def encode_row(row, tokenizer, config):
    messages = prompt_messages(row)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["answers"][0]}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template is not prefix stable: cannot safely mask prompt labels")
    target = full[len(prompt):]
    if not target:
        raise ValueError("No target tokens")
    target_budget = config["data"]["max_target_tokens"]
    prompt_budget = config["data"]["seq_len"] - target_budget
    used_prompt = prompt[-prompt_budget:]
    # ChatML ends with im_end AND a newline; retaining only the final token loses im_end.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endings = [i for i, token in enumerate(target) if token == end_id]
    if not endings:
        raise ValueError("Expected Qwen ChatML end-of-turn marker in target")
    suffix = target[endings[-1]:]
    if len(suffix) >= target_budget:
        raise ValueError("Target budget is too short for answer and template terminator")
    used_target = target if len(target) <= target_budget else target[:target_budget - len(suffix)] + suffix
    return {**row, "input_ids": used_prompt + used_target,
            "labels": [-100] * len(used_prompt) + used_target,
            "prompt_ids": used_prompt, "original_tokens": len(full),
            "prompt_truncated": len(prompt) > prompt_budget, "target_truncated": len(target) > target_budget}


def length_stats(rows):
    lengths = sorted(row["original_tokens"] for row in rows)
    if not lengths:
        return {"count": 0}
    return {"count": len(rows), "min": lengths[0], "max": lengths[-1],
            "p50": lengths[int((len(lengths)-1)*.5)], "p95": lengths[int((len(lengths)-1)*.95)],
            "truncation_fraction": sum(r["prompt_truncated"] or r["target_truncated"] for r in rows) / len(rows),
            "input_tokens": sum(len(r["input_ids"]) for r in rows),
            "supervised_tokens": sum(sum(t != -100 for t in r["labels"]) for r in rows)}


def prepare_tasks(config, input_path, sources_path, output_dir, test_per_task=500,
                  validation_per_task=100, train_limit=20000):
    sources = check_sources(sources_path)
    source_ids = {s["id"] for s in sources["sources"]}
    tokenizer = load_tokenizer(config)
    unique, seen, groups = [], set(), {}
    duplicate_count = 0
    for row in read_jsonl(input_path):
        if row.get("task") not in {"qa", "summary"} or row.get("source_id") not in source_ids:
            raise ValueError("Every row requires a declared source_id and qa/summary task")
        if not isinstance(row.get("context"), str) or not normalize(row["context"]):
            raise ValueError("Non-empty context required for leakage-safe grouping")
        if row["task"] == "qa" and (not isinstance(row.get("question"), str) or not normalize(row["question"])):
            raise ValueError("QA question required")
        if not isinstance(row.get("answers"), list) or not row["answers"] or any(
                not isinstance(a, str) or not normalize(a) for a in row["answers"]):
            raise ValueError("answers must be a non-empty list of non-empty strings")
        row = {k: row[k] for k in ("task", "source_id", "context", "answers", "question") if k in row}
        key = canonical_hash([row["task"], normalize(row["context"]), normalize(row.get("question", ""))])
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        row["id"] = key
        # Same source text must never occur in different splits, even for different questions/tasks.
        row["group_id"] = canonical_hash(normalize(row["context"]))
        unique.append(row)
        groups.setdefault(row["group_id"], []).append(row)
    rng = random.Random(config["train"]["seed"])
    keys = sorted(groups)
    rng.shuffle(keys)
    buckets = {"train": [], "validation": [], "test": []}
    counts = {"validation": Counter(), "test": Counter()}
    discarded = 0
    for group_id in keys:
        group = groups[group_id]
        split = "train"
        for candidate, quota in (("test", test_per_task), ("validation", validation_per_task)):
            if any(counts[candidate][row["task"]] < quota for row in group):
                split = candidate
                break
        if split == "train":
            buckets[split].extend(group)
        else:
            quota = test_per_task if split == "test" else validation_per_task
            for row in group:
                if counts[split][row["task"]] < quota:
                    buckets[split].append(row)
                    counts[split][row["task"]] += 1
                else:
                    discarded += 1  # Never move surplus rows from this group into training.
    for split, quota in (("test", test_per_task), ("validation", validation_per_task)):
        if any(counts[split][task] != quota for task in ("qa", "summary")):
            raise ValueError(f"Insufficient disjoint groups for {split}: {dict(counts[split])}")
    rng.shuffle(buckets["train"])
    buckets["train"] = buckets["train"][:train_limit]
    if {row["task"] for row in buckets["train"]} != {"qa", "summary"}:
        raise ValueError("Both tasks need training examples after group split and train_limit")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 2, "kind": "task", "sources": sources["sources"],
                "split_policy": {"test_per_task": test_per_task, "validation_per_task": validation_per_task,
                                 "train_limit": train_limit, "grouping": "normalized_context_sha256"},
                "input_sha256": sha256_file(input_path), "seed": config["train"]["seed"],
                "model": config["model"], "chat_template_sha256": canonical_hash(tokenizer.chat_template),
                "seq_len": config["data"]["seq_len"], "max_target_tokens": config["data"]["max_target_tokens"],
                "duplicates_removed": duplicate_count, "group_surplus_discarded": discarded,
                "files": {}, "stats": {}}
    for split, rows in buckets.items():
        encoded = [encode_row(row, tokenizer, config) for row in rows]
        path = output / f"{split}.jsonl"
        write_jsonl(path, encoded)
        manifest["files"][path.name] = sha256_file(path)
        manifest["stats"][split] = {task: length_stats([r for r in encoded if r["task"] == task])
                                   for task in ("qa", "summary")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def pack_corpus(config, input_path, sources_path, output_path, min_tokens=327680):
    import numpy as np
    sources = check_sources(sources_path)
    tokenizer = load_tokenizer(config)
    tokens = []
    with Path(input_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                tokens.extend(tokenizer.encode(line.strip(), add_special_tokens=False))
                tokens.append(tokenizer.eos_token_id)
    if len(tokens) < min_tokens:
        raise ValueError(f"Need at least {min_tokens} tokens, got {len(tokens)}; corpus is never repeated silently")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.asarray(tokens, dtype="<u4").tofile(stream)
    manifest = {"kind": "synthetic_packed", "sources": sources["sources"], "model": config["model"],
                "input_sha256": sha256_file(input_path), "tokens": len(tokens), "dtype": "<u4",
                "files": {path.name: sha256_file(path)}}
    with Path(str(path) + ".manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def verify_data(config, path):
    manifest = read_mapping(config["data"]["manifest"])
    if manifest["model"] != config["model"]:
        raise ValueError("Dataset tokenizer/model revision differs from config")
    if manifest["files"].get(Path(path).name) != sha256_file(path):
        raise ValueError(f"Dataset hash mismatch: {path}")
    if manifest["kind"] != config["data"]["mode"]:
        raise ValueError("Data mode mismatch")
    if manifest["kind"] == "task":
        for key in ("seq_len", "max_target_tokens"):
            if manifest[key] != config["data"][key]:
                raise ValueError(f"Preprocessing {key} differs from config")
    validate_sources(manifest)
    return manifest


def audit_task_bundle(config):
    """Check all three splits, target masking and content identities without a tokenizer or GPU."""
    paths = {"train": config["data"]["path"], "validation": config["data"]["validation_path"],
             "test": config["data"]["test_path"]}
    if len({str(Path(p).resolve()) for p in paths.values()}) != 3:
        raise ValueError("Train, validation and test must be three distinct files")
    groups, sample_ids, counts, hashes = {}, set(), {}, {}
    manifest = None
    for split, path in paths.items():
        manifest = verify_data(config, path)
        counts[split] = Counter()
        hashes[split] = sha256_file(path)
        allowed_sources = {s["id"] for s in manifest["sources"]}
        for line, row in read_jsonl_numbered(path):
            prefix = f"{path}:{line}"
            if row.get("task") not in {"qa", "summary"} or row.get("source_id") not in allowed_sources:
                raise ValueError(f"{prefix}: invalid task or source")
            if not isinstance(row.get("context"), str) or not normalize(row["context"]):
                raise ValueError(f"{prefix}: empty context")
            question = row.get("question", "")
            if not isinstance(question, str) or (row["task"] == "qa" and not normalize(question)):
                raise ValueError(f"{prefix}: invalid question")
            if not isinstance(row.get("answers"), list) or not row["answers"] or any(
                    not isinstance(a, str) or not normalize(a) for a in row["answers"]):
                raise ValueError(f"{prefix}: invalid answers")
            expected_id = canonical_hash([row["task"], normalize(row["context"]), normalize(question)])
            group = canonical_hash(normalize(row["context"]))
            if row.get("id") != expected_id or row.get("group_id") != group:
                raise ValueError(f"{prefix}: content identity mismatch")
            if expected_id in sample_ids or (group in groups and groups[group] != split):
                raise ValueError(f"{prefix}: duplicate sample or cross-split context leakage")
            sample_ids.add(expected_id)
            groups[group] = split
            ids, labels, prompt = (row.get(k) for k in ("input_ids", "labels", "prompt_ids"))
            if any(not isinstance(v, list) or not v for v in (ids, labels, prompt)):
                raise ValueError(f"{prefix}: missing token arrays")
            if len(ids) > config["data"]["seq_len"] or len(ids) != len(labels) or len(prompt) >= len(ids):
                raise ValueError(f"{prefix}: invalid token lengths")
            if any(type(t) is not int or not 0 <= t < 152064 for t in ids + prompt):
                raise ValueError(f"{prefix}: invalid token IDs")
            if any(type(t) is not int for t in labels):
                raise ValueError(f"{prefix}: invalid labels")
            if ids[:len(prompt)] != prompt or labels != [-100] * len(prompt) + ids[len(prompt):]:
                raise ValueError(f"{prefix}: prompt masking or supervised labels differ from input")
            if len(prompt) > config["data"]["seq_len"] - config["data"]["max_target_tokens"] or (
                    len(ids) - len(prompt) > config["data"]["max_target_tokens"]):
                raise ValueError(f"{prefix}: truncation budgets differ")
            counts[split][row["task"]] += 1
        if set(counts[split]) != {"qa", "summary"}:
            raise ValueError(f"{path}: both tasks must be represented")
    policy = manifest.get("split_policy")
    if not policy:
        raise ValueError("Data manifest lacks split_policy; regenerate prepared data")
    for split, quota in (("test", "test_per_task"), ("validation", "validation_per_task")):
        if any(n != policy[quota] for n in counts[split].values()):
            raise ValueError(f"{split}: sample count differs from declared policy")
    if sum(counts["train"].values()) > policy["train_limit"]:
        raise ValueError("Training size exceeds declared limit")
    return {"manifest_sha256": sha256_file(config["data"]["manifest"]),
            "file_sha256": hashes, "counts": counts, "split_policy": policy}


class TaskDataset:
    def __init__(self, config, path):
        verify_data(config, path)
        self.rows = list(read_jsonl(path))
        if not self.rows:
            raise ValueError("Empty dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"input_ids": row["input_ids"], "labels": row["labels"],
                "attention_mask": [1] * len(row["input_ids"])}


def task_collator(tokenizer):
    def collate(rows):
        import torch
        size = max(len(row["input_ids"]) for row in rows)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in rows:
            padding = size - len(row["input_ids"])
            for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                batch[key].append(row[key] + [pad] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    return collate
