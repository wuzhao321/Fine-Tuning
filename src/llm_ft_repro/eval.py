"""E6: deterministic generation; character QA metrics and jieba-tokenized ROUGE."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .config import apply_environment, canonical_hash
from .data import audit_task_bundle, load_tokenizer, read_jsonl, verify_data
from .evidence import EvidenceLog, sha256_file
from .model_factory import build_model


def qa_normalize(text):
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def qa_scores(prediction, references):
    pred = qa_normalize(prediction)
    best_em, best_f1 = 0.0, 0.0
    for reference in references:
        gold = qa_normalize(reference)
        overlap = sum((Counter(pred) & Counter(gold)).values())
        score = 2 * overlap / (len(pred) + len(gold)) if pred or gold else 1.0
        best_f1 = max(best_f1, score)
        best_em = max(best_em, float(pred == gold))
    return {"em": best_em, "f1": best_f1}


class JiebaTokenizer:
    def tokenize(self, text):
        import jieba
        return [token for token in jieba.lcut(unicodedata.normalize("NFKC", text)) if token.strip()]


def summary_scores(prediction, references):
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=JiebaTokenizer(), use_stemmer=False)
    scores = [scorer.score(reference, prediction) for reference in references]
    return {key: max(row[key].fmeasure for row in scores) for key in ("rouge1", "rouge2", "rougeL")}


def run_evaluation(config):
    apply_environment(config)
    log = EvidenceLog(config)
    try:
        import torch
        if config["role"] != "eval":
            raise ValueError("Expected evaluation config")
        path = config["data"]["test_path"]
        audit = audit_task_bundle(config)
        manifest = verify_data(config, path)
        tokenizer = load_tokenizer(config)
        if canonical_hash(tokenizer.chat_template) != manifest["chat_template_sha256"]:
            raise ValueError("Evaluation chat template differs from preprocessing")
        adapter = config["evaluation"]["adapter_path"] or None
        if config["lora"]["enabled"] and not adapter:
            raise ValueError("Fine-tuned evaluation requires an explicit saved adapter path")
        adapter_hashes = None
        adapter_receipt = None
        if adapter:
            from .preflight import check_adapter
            adapter_receipt = check_adapter(config, audit)
            adapter_hashes = {p.name: sha256_file(p) for p in Path(adapter).iterdir() if p.is_file()}
        rows = list(read_jsonl(path))
        if not rows or {row["task"] for row in rows} != {"qa", "summary"}:
            raise ValueError("Both QA and summary test data are required")
        model = build_model(config, log, training=False, adapter_path=adapter)
        log.write("evaluation_inputs", data_sha256=sha256_file(path), manifest=manifest,
                   adapter_path=adapter, adapter_hashes=adapter_hashes, adapter_receipt=adapter_receipt,
                   data_audit=audit,
                   metrics_definition="QA: NFKC, remove whitespace/punctuation, lowercase, char F1; ROUGE: jieba, F-measure",
                   decoding={"do_sample": False, "num_beams": 1,
                             "max_new_tokens": config["evaluation"]["max_new_tokens"]})
        aggregates = defaultdict(list)
        stop_ids = {tokenizer.eos_token_id}
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != tokenizer.unk_token_id:
            stop_ids.add(im_end)
        for row in rows:
            ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        max_new_tokens=config["evaluation"]["max_new_tokens"],
                                        do_sample=False, num_beams=1, use_cache=True,
                                        eos_token_id=sorted(stop_ids), pad_token_id=tokenizer.pad_token_id)
            prediction = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            scores = (qa_scores if row["task"] == "qa" else summary_scores)(prediction, row["answers"])
            aggregates[row["task"]].append(scores)
            log.write("prediction", sample_id=row["id"], task=row["task"], prediction=prediction,
                       references=row["answers"], scores=scores, prompt_truncated=row["prompt_truncated"],
                       generated_tokens=int(output.shape[1] - ids.shape[1]),
                       reached_generation_limit=int(output.shape[1] - ids.shape[1]) >= config["evaluation"]["max_new_tokens"])
        metrics = {task: {key: sum(r[key] for r in items) / len(items) for key in items[0]}
                   for task, items in aggregates.items()}
        log.write("result", status="ok", metrics=metrics,
                   counts={task: len(items) for task, items in aggregates.items()},
                   data_sha256=sha256_file(path), adapter_path=adapter,
                   adapter_receipt=adapter_receipt, data_audit=audit)
        return 0
    except Exception as exc:
        log.failure(exc)
        return 2
