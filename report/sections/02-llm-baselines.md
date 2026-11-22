# Section 02 — LLM Baselines (Closed-Book and Oracle)

**Author:** Hari Singh (`harisingh294`) — Member 2  
**Branch:** `llm/phase-2-closed-book-baseline`  
**Reviewer:** Ayush Yadav (`AyushYadav-rgb477`) — Member 3  

---

## 1. Objective and Scope

Phase 2 establishes the shared Large Language Model (LLM) execution layer for the project. Its primary objectives are:
1. Constructing a provider-neutral, versioned, and deterministic LLM software infrastructure to be reused across all LLM experimental conditions in Phase 2, Phase 3 (Retrieval-Augmented Generation), and Phase 4 (Final Evaluation).
2. Implementing the **Closed-book baseline**, where the model answers biomedical questions strictly from memory without access to external documents.
3. Implementing the **Oracle baseline**, where the model is provided with the source document's gold abstract contexts, establishing the upper bound for retrieval performance.
4. Ensuring complete data integrity, eliminating answer leakage, and enforcing test-split isolation.

---

## 2. Experimental Design

All LLM conditions in this project follow a strictly controlled experimental design:
- **Model Architecture & Settings:** Both baselines use `Qwen/Qwen2.5-7B-Instruct`, an open-weights model (pretrained inference only, no training or fine-tuning), pinned to Hugging Face commit `a09a35458c702b33eeacc393d103063234e8bc28`. It runs with 4-bit NF4 quantization (`bitsandbytes`, float16 compute) on a Google Colab Tesla T4 GPU, with deterministic greedy decoding (`do_sample=False`) and `max_tokens: 512`. The originally planned Anthropic `claude-haiku-5-5` path (`AnthropicClient`) is implemented and tested with a fake client but was **not used and no paid API call was made**. This model choice differs from the plan's Anthropic default and must be confirmed by the team on the Phase 2 issue.
- **Controlled Comparison:** The prompt template, instructions, requested output format, model identifier, and generation parameters are kept identical across all LLM systems. The *only* variable between systems is the content of the evidence block:
  - **Closed-book:** An explicit directive stating no documents are provided and citations must be empty.
  - **Oracle:** The question's own abstract contexts supplied as document `[1]`.
  - **RAG (Phase 3):** BM25-retrieved document abstracts numbered sequentially `[1]`, `[2]`, ...
- **Non-Determinism & Repetition Protocol:** Because temperature and top-p sampling are restricted by modern reasoning APIs, variance is measured across repeated runs (`run: 1, 2, 3`) rather than sampling parameters, preserving independent reproducibility.

---

## 3. Prompt Design and Versioning

Prompts are centrally defined in `src/bmqa/llm/prompts.py` under immutable template versions. Template `v1` is structured as follows:

```text
Answer the biomedical research question below with "yes", "no" or "maybe".
{evidence_block}

Question: {question}

Reply in JSON with the keys "answer", "rationale" (at most three sentences explaining your answer) and "citations" (the numbers of the documents that support your answer).
```

### Critical Safeguards
- **PMID Suppression:** Source identifiers (e.g., PubMed IDs) are stripped entirely from prompts. Documents are presented strictly as generic numbered blocks (`[1]`, `[2]`). This prevents the language model from retrieving answers directly from memorized PMID associations.
- **Answer Leakage Elimination:** The source article's conclusion (`long_answer`) directly reveals the answer to the research question. The prompt constructor `build_prompt(question, docs, version)` accepts only the question text and abstract context paragraphs. The `long_answer` field is strictly excluded from all model prompts.

---

## 4. Closed-Book Baseline

- **Configuration:** [configs/closed_book.yaml](file:///c:/Users/haris/Desktop/BiomedicalProject/biomedical-qa-rag/configs/closed_book.yaml) (`evidence: none`)
- **Evidence Function:** `no_evidence(example) -> []`
- **Evidence Block Text:**
  ```text
  No documents are provided. Answer from your own knowledge and set "citations" to [].
  ```
- **Role:** Measures the parametric biomedical knowledge acquired by the pre-trained model and establishes the performance baseline without external context.

---

## 5. Oracle Baseline

- **Configuration:** [configs/oracle.yaml](file:///c:/Users/haris/Desktop/BiomedicalProject/biomedical-qa-rag/configs/oracle.yaml) (`evidence: gold`)
- **Evidence Function:** `gold_evidence(example) -> [(Hit(doc_id=example.id, score=1.0, rank=1), " ".join(example.contexts))]`
- **Evidence Block Text:**
  ```text
  Use only the numbered documents below.

  [1] <Abstract background and results paragraphs joined with spaces>
  ```
- **Role:** Demonstrates LLM performance when supplied with the exact relevant literature context (excluding the conclusion), representing the theoretical ceiling that a perfect retriever could achieve.

---

## 6. Development Dataset

Evaluation is conducted exclusively on the 100-question development split (`data/processed/dev.jsonl`), stratified as follows:

| Split | Total Questions | Yes | No | Maybe |
|---|---|---|---|---|
| **Dev** | 100 | 55 (55.0%) | 34 (34.0%) | 11 (11.0%) |

The development set was extracted via strict dev-only preparation (Stage 10). The official 500-question test split (`test.jsonl` / `test_ground_truth.json`) remains completely unmaterialized and untouched.

---

## 7. Evaluation Methodology

Outputs are formatted as `bmqa.schema.Prediction` records and scored using the shared Phase 1 evaluator (`python -m bmqa.eval`):
- **Metrics:** Classification Accuracy and unweighted Macro-F1 across all three classes (`yes`, `no`, `maybe`), matching the official PubMedQA scoring protocol.
- **Parse Failure Protocol:** Output parsing is handled by `src/bmqa/llm/parse.py`. If a model output violates JSON syntax, omits required keys, hallucinates citations outside the provided document range, or terminates unexpectedly (`stop_reason` in `{"max_tokens", "refusal"}`), `raw_output` is tagged with the `PARSE_FAILURE:` prefix and a fallback label (`"maybe"`) is recorded. The evaluator scores all `PARSE_FAILURE:` rows as incorrect (0 accuracy), preventing models from inflating scores by declining difficult questions.

---

## 8. Engineering and Offline Validation

The complete software layer was verified offline using `FakeLLMClient` (zero API cost, zero network traffic):
- **Unit and Integration Tests:** 63 / 63 tests passing in `pytest`, including the dedicated Phase 2 tests for cache hashing, prompt leakage, schema validation, JSON parsing, and runner execution.
- **Code Quality:** `ruff check src tests` and `ruff format --check src tests` clean.
- **Offline Dry-Run Verification:** Executed both closed-book and oracle runners across all 100 dev examples using `FakeLLMClient` (`--dry-run`). Confirmed:
  - Exactly 100 predictions generated per condition with full ID alignment.
  - Zero leakage of `long_answer` or context into closed-book prompts.
  - Correct formatting of document `[1]` in oracle prompts.
  - Evaluator scored FakeLLM outputs (accuracy: 0.5500, macro-F1: 0.2366, reflecting canned `"yes"` responses against the 55% majority class).
  - Evaluator correctly penalized synthetic parse failures.

---

## 9. Dev Results (Qwen2.5-7B-Instruct, 4-bit, prompt `v1`, run 1)

Executed on Google Colab (Tesla T4) from a payload containing only the 100-question dev split. Artifacts were checked locally: both prediction files hold exactly the 100 approved dev IDs, the stored metrics equal a fresh recomputation with the Phase 1 scorer, and all 200 replies ended with `end_turn`.

| System | Prompt Version | Accuracy | Macro-F1 | F1 (yes) | F1 (no) | F1 (maybe) | Parse Failures |
|---|---|---|---|---|---|---|---|
| Majority Baseline (Phase 1) | — | 0.5500 | 0.2366 | 0.7097 | 0.0000 | 0.0000 | — |
| PubMedBERT (Phase 1, 5 seeds) | — | 0.576 ± 0.049 | 0.395 ± 0.048 | *See Sec 01* | *See Sec 01* | *See Sec 01* | — |
| **Closed-Book LLM (Qwen2.5-7B)** | `v1` | 0.2600 | 0.2348 | 0.3636 | 0.1081 | 0.2326 | 0 |
| **Oracle LLM (Qwen2.5-7B)** | `v1` | 0.5400 | 0.4435 | 0.7810 | 0.3830 | 0.1667 | 0 |

Predicted label distribution (gold: 55 yes, 34 no, 11 maybe):

| System | yes | no | maybe |
|---|---|---|---|
| Closed-book | 22 | 3 | 75 |
| Oracle | 50 | 13 | 37 |

**Observations (single run, n = 100, so differences of a few points are within noise):**
- Closed-book accuracy (0.26) is far below the majority baseline (0.55): the model answers "maybe" for 75% of questions while only 11% of gold labels are "maybe".
- Giving the model its own abstract (oracle) raises accuracy from 0.26 to 0.54 and macro-F1 from 0.235 to 0.444, but accuracy only matches the majority baseline; macro-F1 is higher than majority and PubMedBERT's mean.
- Only prompt version `v1` was run; no other prompt versions were tried.

---

## 10. Token Usage and Cost

Measured from `results/runs/`:

| System | Input tokens | Output tokens | Log start → end (UTC, 2026-11-22) | Cached responses |
|---|---|---|---|---|
| Closed-book | 12,483 | 7,176 | 08:42:12 → 08:48:39 | 0 |
| Oracle | 43,622 | 7,861 | 08:49:59 → 08:57:31 | 0 |

- **API cost: $0.00.** No Anthropic or other paid API was called; the configured token prices are 0.
- The Colab account's billing tier was not recorded, so the report does not claim that the Colab session itself was free.
- The log timestamps include loading the model in the runner process; they are not a pure GPU generation time.

### Cache replay
The cache key includes the run number, so a different run number is a new generation. Replay was therefore tested by repeating run 1; this was confirmed in the Colab session (100/100 cached responses, byte-identical predictions). The replay logs are not committed.

---

## 11. Limitations

1. **Single run, single prompt, n = 100:** no confidence intervals; repeated runs and statistics belong to Phase 4.
2. **Small quantized open model:** results are for a 4-bit 7B model and do not represent larger models such as Claude.
3. **Parametric memorization vs. reasoning:** a closed-book model may rely on memorized pre-training data; PubMedQA may have been seen in training.
4. **Degenerate closed-book behaviour:** the model strongly over-predicts "maybe"; this is a prompt/model behaviour, not a parsing artefact (0 parse failures).
5. **Context constraints:** The oracle condition provides only the abstract (excluding the conclusion).
6. **Test isolation:** the Colab payload contained only the dev split; the test split was not used.
