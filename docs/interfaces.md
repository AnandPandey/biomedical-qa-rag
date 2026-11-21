# Shared interfaces

This file is the contract between the four components (data, classifier, LLM baseline, RAG,
evaluation). The same definitions exist in code in [`src/bmqa/schema.py`](../src/bmqa/schema.py),
which is the source of truth if the two ever disagree.

**Changing an interface:** open a PR that updates both this file and `schema.py`, and request
review from every member whose component reads or writes the changed record. Do not change an
interface inside an unrelated feature PR.

All files are UTF-8 JSON Lines (`.jsonl`): one JSON object per line.

## Labels

```
"yes" | "no" | "maybe"
```

Always lowercase. Any system that cannot produce one of these must still emit one (and record
the problem in `raw_output`); an unparseable output is scored as wrong, never dropped.

## 1. Example (`data/processed/{split}.jsonl`)

One PubMedQA question. Produced by the data pipeline (Member 1).

| Field | Type | Notes |
|---|---|---|
| `id` | str | PubMed ID (PMID) of the source article |
| `question` | str | |
| `contexts` | list[str] | Abstract sections **without** the conclusion |
| `long_answer` | str | The article's conclusion. **Never** part of a model input in the main experiments: it states the answer |
| `final_decision` | label | Gold label |

`split` is one of `train`, `dev`, `test`. See [experiment-protocol.md](experiment-protocol.md)
for how splits are made and how the test split is protected.

## 2. Corpus document (retrieval corpus)

Produced by the retrieval component (Member 3).

| Field | Type | Notes |
|---|---|---|
| `doc_id` | str | PMID |
| `title` | str | May be empty |
| `text` | str | Abstract text without the conclusion section |
| `source` | str | `"pqa_a"`, `"pqa_u"` or `"pqa_l"`, the PubMedQA subset it came from |

## 3. Retriever

```
retrieve(query: str, k: int) -> list[Hit]
Hit = {"doc_id": str, "score": float, "rank": int}   # rank starts at 1, best first
```

Returns at most `k` hits, sorted by rank. Scores are only comparable within one retriever.

## 4. LLM client

```
generate(prompt: str, *, max_tokens: int) -> LLMResponse
LLMResponse = {
    "text": str,           # the model's reply
    "model": str,          # exact model ID used
    "input_tokens": int,
    "output_tokens": int,
    "cached": bool,        # True if served from the local response cache
}
```

- The model is fixed when the client is created, not per call, so a run cannot silently mix models.
- Every real response is cached on disk, keyed by a hash of (model, prompt, max_tokens), so
  re-running an experiment costs nothing and reproduces the same text.
- A fake client with the same interface is used in tests; tests never call a real API.
- Provider and model are not decided yet. Sampling parameters such as `temperature` are not part
  of the interface (some current APIs reject them); run-to-run variation is measured with repeated
  runs instead (see the protocol).

## 5. Prediction (`results/predictions/{system}/{split}_run{n}.jsonl`)

Written by **every** system, including the classifier, so the evaluator treats them identically.

| Field | Type | Notes |
|---|---|---|
| `id` | str | Example `id` |
| `system` | str | e.g. `classifier`, `closed_book`, `rag_strict`, `rag_gold`, `oracle` |
| `prediction` | label | |
| `probs` | object \| null | `{"yes": p, "no": p, "maybe": p}` summing to 1, or `null` |
| `rationale` | str \| null | Free-text justification (LLM systems); `null` for the classifier |
| `evidence` | list[Hit] | Documents shown to the model; `[]` for closed-book |
| `raw_output` | str | Unparsed model output, kept for auditing |
| `model` | str | Model ID or checkpoint name |
| `prompt_version` | str \| null | e.g. `v1`; `null` for the classifier |
| `config_hash` | str | Hash of the experiment config that produced this row |
| `seed` | int \| null | Training seed (classifier); `null` if not applicable |
| `run` | int | Repeat number, starting at 1 |

**Parse failures.** A row whose `raw_output` starts with `PARSE_FAILURE:` is scored as
incorrect whatever its `prediction` field says (the field still holds a valid placeholder label),
and is counted in the metrics as a parse failure.

## 6. Evaluator (implemented in Phase 1)

```
python -m bmqa.eval --pred <predictions.jsonl> --split <dev|test> [--allow-test]
```

Writes `results/metrics/{system}_{split}_run{n}.json` with `n`, `accuracy`, `macro_f1`,
per-label precision/recall/F1/support, the confusion matrix (rows gold, columns predicted, label
order yes/no/maybe) and `parse_failures`. The predicted IDs must be exactly the split's IDs.
Macro-F1 matches scikit-learn's `f1_score(average="macro")` used by the official PubMedQA
script. Confidence intervals and significance tests are added in Phase 4. The evaluator depends
only on the prediction format above, never on how a system works internally.

## 7. Python helpers (Phase 1)

| Function | Module | Purpose |
|---|---|---|
| `load_split(split, allow_test=False) -> list[Example]` | `bmqa.data.pubmedqa` | Load a split; `PermissionError` for `test` without `allow_test=True` |
| `gold_labels(split, allow_test=False) -> dict[str, str]` | `bmqa.data.pubmedqa` | Example id -> gold label |
| `read_jsonl`, `write_jsonl` | `bmqa.io` | JSON Lines files |
| `write_predictions`, `load_predictions` | `bmqa.io` | Prediction files (validated on load) |
| `load_config(path) -> dict`, `config_hash(cfg) -> str` | `bmqa.config` | YAML configs; 12-character SHA-256 of the canonical JSON |
| `compute_metrics(gold, predictions) -> dict` | `bmqa.eval.metrics` | Metrics used by the evaluator |
| `accuracy`, `macro_f1`, `per_label`, `confusion_matrix` | `bmqa.eval.metrics` | Metrics on label lists |

## 8. Commands (Phase 1)

| Command | Output |
|---|---|
| `python -m bmqa.data.download` | `data/raw/` (checksums verified) |
| `python -m bmqa.data.splits` | `data/splits/*_ids.txt` (committed, frozen), `data/processed/*.jsonl` |
| `python -m bmqa.baselines --split dev` | `results/predictions/majority/dev_run1.jsonl` |
| `python -m bmqa.classifier.train --config configs/classifier_pubmedbert.yaml` | checkpoints in `outputs/classifier/`, dev predictions, `results/classifier_checkpoints.json` |
| `python -m bmqa.classifier.predict --checkpoint DIR --split dev --run N --seed S` | `results/predictions/classifier/{split}_run{N}.jsonl` |

Every command that can load data has an `--allow-test` flag, off by default. Only the frozen
Phase 4 test run uses it.
