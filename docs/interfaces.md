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

```python
retrieve(query: str, k: int) -> list[Hit]
Hit = {"doc_id": str, "score": float, "rank": int}   # rank starts at 1, best first
```

Returns at most `k` hits, sorted by rank. Scores are only comparable within one retriever.

## 4. LLM client

```python
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

## 6. Evaluator (planned)

```
python -m bmqa.eval --pred <predictions.jsonl> --split <dev|test>
```

Writes `results/metrics/{system}_{split}_run{n}.json` with accuracy, macro-F1, per-class
precision/recall/F1, the confusion matrix and 95% bootstrap confidence intervals. The evaluator
depends only on the prediction format above, never on how a system works internally.
