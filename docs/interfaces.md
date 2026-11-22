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
  Clients also expose `model` and `last_stop_reason` attributes.
- Every real response is cached on disk under `outputs/llm_cache/`; see "Cache key" in section 9.
  Re-running the same run costs nothing and reproduces the same text.
- `FakeLLMClient` has the same interface and is used in tests and `--dry-run`; tests never call a
  real API.
- Provider and model are set in the `llm` block of the configs. Phase 2 uses the open-weights
  `Qwen/Qwen2.5-7B-Instruct` (provider `qwen`); an Anthropic client (provider `anthropic`) also
  exists but is unused. Sampling parameters such as `temperature` are not part of the interface;
  run-to-run variation is measured with repeated runs instead (see the protocol).

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
script. Confidence intervals, significance tests and comparison tables are in
`bmqa.eval.stats` and `bmqa.eval.compare` (section 11). The evaluator depends
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

## 9. LLM layer (Phase 2)

### Creating a client

```python
from bmqa.llm.client import make_client, FakeLLMClient

client = make_client(llm_cfg, run=1)  # real client chosen by llm_cfg["provider"]
client = make_client(llm_cfg, run=1, dry_run=True)  # FakeLLMClient: no GPU, key or network
client = FakeLLMClient()  # same, created directly (tests)
```

`llm_cfg` is the `llm:` block of a config (`configs/closed_book.yaml`). Phase 3 and 4 configs must
copy this block unchanged (including `revision`), so every LLM system uses the same model.

| Client | `provider` | What it does |
|---|---|---|
| `QwenClient` | `qwen` | `Qwen/Qwen2.5-7B-Instruct` with `transformers`, 4-bit NF4 quantization (`bitsandbytes`, float16 compute) on a GPU; greedy decoding (`do_sample=False`); the model's chat template is applied. Needs the `llm` extra and a CUDA GPU (a free Colab T4 works). **Needs no API key.** |
| `FakeLLMClient` | (any, with `dry_run=True`) | Returns a fixed JSON `yes` answer (or a function of the prompt); records every prompt in `.prompts`; never touches the network; used by all tests and `--dry-run`. |
| `AnthropicClient` | `anthropic` | Claude through the Anthropic SDK with structured JSON output. Implemented but not used in the reported results; needs `ANTHROPIC_API_KEY`. |

Config keys read by `QwenClient`: `model`, `revision` (pin to an immutable Hugging Face commit; the
default `main` is not reproducible) and `load_in_4bit` (default `true`); an optional `config_hash`
key is also passed into the cache key if present. `max_tokens` is the maximum number of newly
generated tokens.

### Cache key

`bmqa.llm.cache.cache_key(...)` is the SHA-256 of the canonical JSON (sorted keys, no spaces) of:

| Field | Meaning |
|---|---|
| `model`, `effort`, `max_tokens`, `prompt` | the request |
| `run` | repeat number: run 1, 2, 3 are independent calls; repeating the **same** run is a cache hit |
| `provider` | added when given (`qwen`) |
| `extra` | added when given: for Qwen `revision`, `quantization` (4-bit flag, `nf4`, float16 compute), `generation` (`do_sample: false`, temperature 0) and, if present in the config, `config_hash` |

So changing the model, revision, quantization, generation settings, prompt, `max_tokens`, run or
provider gives a different key and a new generation. One JSON file per key is stored under
`outputs/llm_cache/` (gitignored). To replay a finished run for free, repeat it with the **same**
`--run` number (a different run number is a new generation).

### Prompt (version `v1`) and evidence functions

`build_prompt(question, docs, version="v1")` (`bmqa.llm.prompts`) builds one prompt from the
question and the document texts only, so `long_answer` can never reach the model. All LLM systems
use the same template; only the evidence block differs:

| Evidence function | Module | Documents shown |
|---|---|---|
| `no_evidence(example)` | `bmqa.llm.run` | none: "No documents are provided. Answer from your own knowledge and set "citations" to []." (closed-book) |
| `gold_evidence(example)` | `bmqa.llm.run` | one document `[1]` = the example's own abstract contexts joined by spaces, as `Hit(doc_id=example.id, score=1.0, rank=1)` (oracle) |

```
EvidenceFn = Callable[[Example], list[tuple[Hit, str]]]   # (hit, document text) pairs
```

Documents are numbered `[1]`, `[2]`, ... and PMIDs are never shown. Phase 3 supplies its own
evidence function with the same type. Released prompt versions are never edited; a new version
gets a new name (`v2`) and the dev results of every version tried are reported. Only `v1` exists.

### Parsing answers

The model replies in JSON with `answer` (`yes`/`no`/`maybe`), `rationale` and `citations` (document
numbers). `parse_answer(text, n_docs) -> ParsedAnswer` (`bmqa.llm.parse`) accepts plain JSON, JSON in
a Markdown fence, or the first `{...}` object, lower-cases and strips the answer, and returns:

| Field | Meaning |
|---|---|
| `label` | `yes` / `no` / `maybe`, or `None` if no valid answer was found |
| `rationale` | the explanation ("" if missing) |
| `citations` | valid citation numbers, `1..n_docs` |
| `invalid_citations` | numbers outside `1..n_docs` (including any citation when `n_docs = 0`): these are **fabricated citations**, counted in Phase 4 |

Citations are not a separate prediction field: they stay inside `raw_output` and are recovered with
`parse_answer(raw_output, n_docs=len(evidence))`.

**Parse failures** (`label` is `None`, or the stop reason is `max_tokens`/`refusal`): `prediction` is
the placeholder `"maybe"`, `raw_output` starts with `PARSE_FAILURE:` and `rationale` is `null`. The
scorer counts such rows as wrong; they are never dropped.

### `run_system` and the run command

```python
run_system(examples, client, evidence_fn, *, system, prompt_version, config_hash, run, max_tokens)
    -> (list[Prediction], usage_dict)
execute(cfg, evidence_fn, *, split, run, limit=None, dry_run=False, allow_test=False, ...)
    -> (predictions_path, log_path)      # loads the split, runs, writes both files
```

`run_system` calls, for each example, the evidence function, `build_prompt`, `client.generate` and
`parse_answer`, and returns the predictions in input order plus a usage summary. Retrieval needs no
change to `run.py`: pass your own `evidence_fn`.

```bash
python -m bmqa.llm.run --config configs/closed_book.yaml --split dev --run 1 [--limit N] [--dry-run]
python -m bmqa.llm.run --config configs/oracle.yaml --split dev --run 1
```

| Output | Path |
|---|---|
| Predictions | `results/predictions/{system}/{split}_run{n}.jsonl` |
| Usage log | `results/runs/{system}_{split}_run{n}.json`: model, effort, prompt version, config hash, token totals, `cached_responses`, `uncached_*_tokens`, `parse_failures`, `stop_reasons`, `estimated_cost_usd`, start and end time |

`--dry-run` uses `FakeLLMClient` and writes to `outputs/dry_run/` instead of `results/`.
`--allow-test` is for the frozen Phase 4 run only.

## 10. Retrieval and RAG (Phase 3)

| Name | Module | Purpose |
|---|---|---|
| `build_corpora(pqa_a_rows, pqa_u_rows, pqal)` | `bmqa.retrieval.corpus` | Strict and gold corpora + statistics |
| `DocStore(corpus_path)[doc_id] -> str` | `bmqa.retrieval.store` | Document text by ID |
| `BM25Retriever(index_dir)` | `bmqa.retrieval.bm25` | Implements the `Retriever` protocol (section 3) |
| `build_index(corpus_path, index_dir)` | `bmqa.retrieval.bm25` | BM25 index (`bm25s`, English stop words, Snowball stemmer) |
| `retrieval_metrics(ranks)`, `evaluate_gold(...)`, `leakage_count(...)` | `bmqa.retrieval.evaluate` | Recall@k, MRR, leakage check |
| `make_retrieval_evidence(retriever, store, k) -> EvidenceFn` | `bmqa.rag.evidence` | Retrieval evidence for the Phase 2 runner (query = question text) |

- **Corpora** (`data/corpus/`, gitignored, rebuilt by command): `strict.jsonl` = PQA-A + PQA-U
  minus all 1,000 PQA-L articles; `gold.jsonl` = strict + PQA-L. One `CorpusDoc` per line,
  sorted by `doc_id`; text = abstract contexts without the conclusion; `title` is empty.
  Source: Hugging Face `qiaojin/PubMedQA`, revision `9001f2853fb87cab8d220904e0de81ac6973b318`.
  Sizes and SHA-256 values are committed in `results/corpus_stats.json`.
- **Indexes** (`outputs/indexes/bm25_strict`, `outputs/indexes/bm25_gold`) are rebuilt from the
  corpora and never committed; `doc_ids.json` maps index positions to PMIDs.
- **System names:** `rag_strict`, `rag_gold`; dev ablation runs with `--k K` are named
  `rag_strict_kK` / `rag_gold_kK` and never overwrite the final systems.
- **Final k = 1** for both RAG configs, chosen on dev by the rule fixed before the ablation
  (issue #9): the k in {1, 3, 5, 10} with the highest `rag_strict` macro-F1; ties go to the
  smaller k; the same k for `rag_gold`.
- The RAG configs reuse `prompt_version`, `llm` and the prices of `configs/closed_book.yaml`
  unchanged (tested in `tests/test_rag.py`). `bmqa.rag.run` calls Phase 2's `execute` unchanged.

Commands:

```bash
python -m bmqa.retrieval.corpus
python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl --index-dir outputs/indexes/bm25_strict
python -m bmqa.retrieval.bm25 --corpus data/corpus/gold.jsonl --index-dir outputs/indexes/bm25_gold
python -m bmqa.retrieval.evaluate
python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --run 1 [--k K] [--limit N] [--dry-run]
```

### Running RAG with Qwen on a Colab T4 GPU

Two environment issues appear only when BM25 and Qwen run in the same process on Colab:

- `bm25s` uses JAX when it is installed (Colab preinstalls it), and JAX reserves about 75% of the
  GPU memory at start-up, so the 4-bit Qwen model no longer fits. Set
  `JAX_PLATFORMS=cpu` (and `XLA_PYTHON_CLIENT_PREALLOCATE=false`) before running.
- `QwenClient` uses `device_map="auto"`; Phase 3 loaded the model with `device_map={"": 0}`
  (the whole model on GPU 0). `make_client` does not read `device_map` from the config yet, so
  Phase 3 used a small Colab-only wrapper that sets this default before calling
  `bmqa.rag.run.main`. `device_map` is not part of the cache key, so answers are unaffected.

Phase 3 dev runs used `torch 2.6.0`, `transformers 4.51.3`, `bitsandbytes 0.45.5` and
`accelerate 1.6.0` (installed while debugging, before the JAX cause was found; Colab's defaults at
the time, `torch 2.14` / `transformers 5.22`, were not retested with the JAX fix). Phase 4 should
run every LLM system in one environment with fixed versions.

## 11. Evaluation and analysis (Phase 4)

| Name | Module | Purpose |
|---|---|---|
| `bootstrap_ci(gold, pred, metric, n_boot=10000, seed=0)` | `bmqa.eval.stats` | 95 % percentile bootstrap CI |
| `paired_bootstrap(gold, pred_a, pred_b, metric, ...)` | `bmqa.eval.stats` | Difference A - B and its CI |
| `mcnemar_exact(correct_a, correct_b) -> (b, c, p)` | `bmqa.eval.stats` | Exact two-sided McNemar test |
| `holm(p_values)`, `cohen_kappa(a, b)` | `bmqa.eval.stats` | Multiple-comparison correction, agreement |
| `split_claims`, `decide`, `check_claims`, `summarize`, `fabricated_citations` | `bmqa.eval.claims` | Unsupported claims and fabricated citations |
| `stratified_sample`, `label_report` | `bmqa.eval.annotate` | Human validation of the claim checker on dev |
| `aggregate`, `load_metrics` | `bmqa.eval.compare` | Result tables across runs and seeds |

- **Claim labels:** `supported`, `contradicted`, `not_supported`. A claim is supported if the
  NLI model's most likely label is entailment for any premise; otherwise contradicted if it is
  contradiction for any premise; otherwise not supported. Premises: the documents shown
  (*evidence-grounded*; none for closed-book) and the example's own abstract sections
  (*gold-grounded*; all systems).
- **Outputs:** `results/claims/{system}_{split}_run{n}.jsonl` (one row per claim),
  `results/claims/summary_{split}.json`, `results/annotation/*`, `results/tables/*`,
  `results/figures/*`.

Commands:

```bash
python -m bmqa.eval.annotate sample --strict-corpus data/corpus/strict.jsonl --gold-corpus data/corpus/gold.jsonl
python -m bmqa.eval.annotate agreement
python -m bmqa.eval.claims --pred results/predictions/rag_strict/test_run1.jsonl --split test --allow-test --corpus data/corpus/strict.jsonl
python -m bmqa.eval.compare --split test --allow-test
```
