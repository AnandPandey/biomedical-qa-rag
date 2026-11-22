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
- **Model Architecture & Settings:** The planned baseline model is `claude-haiku-5-5` accessed via the official Anthropic Python SDK (`anthropic>=2.12`). Both baselines are configured with identical settings: low reasoning effort (`effort: "low"`), a maximum generation limit of 4,096 tokens (`max_tokens: 4096`), and structured JSON response formatting (`ANSWER_SCHEMA`).
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
- **Unit and Integration Tests:** 59 / 59 tests passing in `pytest` (0.69s), including 27 dedicated Phase 2 tests for cache hashing, prompt leakage, schema validation, JSON parsing, and runner execution.
- **Code Quality:** `ruff check .` (all clean) and `ruff format --check .` (44 files formatted).
- **Offline Dry-Run Verification:** Executed both closed-book and oracle runners across all 100 dev examples using `FakeLLMClient` (`--dry-run`). Confirmed:
  - Exactly 100 predictions generated per condition with full ID alignment.
  - Zero leakage of `long_answer` or context into closed-book prompts.
  - Correct formatting of document `[1]` in oracle prompts.
  - Evaluator scored FakeLLM outputs (accuracy: 0.5500, macro-F1: 0.2366, reflecting canned `"yes"` responses against the 55% majority class).
  - Evaluator correctly penalized synthetic parse failures.

---

## 9. Genuine Model Results — Pending

> [!WARNING]
> Genuine live model inference on the development set has **not yet been executed**. Results in this section are currently **NOT AVAILABLE / PENDING FUNDING**.

| System | Prompt Version | Accuracy | Macro-F1 | F1 (yes) | F1 (no) | F1 (maybe) | Parse Failures |
|---|---|---|---|---|---|---|---|
| **Majority Baseline (Phase 1)** | — | 0.5500 | 0.2366 | 0.7097 | 0.0000 | 0.0000 | — |
| **PubMedBERT (Phase 1, 5 seeds)** | — | 0.576 ± 0.049 | 0.395 ± 0.048 | *See Sec 01* | *See Sec 01* | *See Sec 01* | — |
| **Closed-Book LLM (`claude-haiku-5-5`)** | `v1` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| **Oracle LLM (`claude-haiku-5-5`)** | `v1` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |

*Note: The figures reported in Section 8 reflect software pipeline testing via `FakeLLMClient` and must not be interpreted as genuine biomedical reasoning performance.*

---

## 10. Token Usage and Cost — Pending

### Configured Cost Estimates (Unverified Assumptions)
In both configuration files, pricing assumptions are configured as:
- Input: \$0.10 per million tokens
- Output: \$0.50 per million tokens

Based on measured prompt sizes (Stage 11 dry-runs):
- Closed-book input: ~6,564 tokens per 100 questions (estimated input cost: ~\$0.0007).
- Oracle input: ~26,051 tokens per 100 questions (estimated input cost: ~\$0.0026).
- Expected output tokens: ~5,000–8,000 tokens per run (estimated output cost: ~\$0.0025–\$0.0040).
- Expected total combined run expenditure: **<\$0.01 USD**.

### Actual Measured Expenditure
- **Actual Billed Cost to Date:** **\$0.00 USD**.
- A preliminary one-question live API pilot was attempted in Stage 13; however, the request terminated with an HTTP 401 Authentication Error (`Invalid bearer token`) prior to generation. No billable tokens were consumed, no responses were cached, and zero expenditure was incurred.
- Genuine measured token counts and actual provider invoices will be populated once live inference is funded and executed.

---

## 11. Limitations

1. **Pending Live Model Evaluation:** Baseline accuracy and macro-F1 remain unmeasured until an active, funded Anthropic API key is supplied.
2. **Parametric Memorization vs. Reasoning:** A closed-book model may rely on memorized pre-training data rather than clinical reasoning.
3. **Absence of Real-Time Literature:** Closed-book models cannot access literature published after their training cutoff date.
4. **Context Constraints:** The Oracle condition provides only the abstract (excluding conclusion); full-text clinical methodology is not represented.

---

## 12. Implications for Phase 3 (Retrieval-Augmented Generation)

The Phase 2 software deliverables provide the exact foundation required for Member 3 (Ayush Yadav):
1. **Generic Runner Reuse:** Member 3 can invoke `bmqa.llm.run.run_system` directly by passing a custom retrieval evidence function `retrieved_evidence(example) -> list[tuple[Hit, str]]`. No changes to `run.py` are required.
2. **Experimental Parity:** Phase 3 RAG systems will import `prompt_version: v1` and the identical `llm:` block from `configs/closed_book.yaml`, guaranteeing that any delta in accuracy between Closed-book, RAG, and Oracle is strictly attributable to retrieval quality.
3. **Benchmark Ceiling:** Once genuine Oracle inference is run, it will establish the empirical upper bound against which BM25 retrieval configurations (varying $k$) will be compared.

---

## 13. Reproducibility and Safety

- **Deterministic Caching:** Every real API response is stored under `outputs/llm_cache/` using SHA-256 hashes of `(model, effort, max_tokens, prompt, run)`. Re-running experiments makes 0 repeat API calls and guarantees identical text.
- **Atomic Persistence:** Cache entries are written via temporary `.tmp` files and atomic replacement, eliminating cache corruption from interruptions.
- **Strict Test-Split Isolation:** Test split protection remains unbroken. No code path loads or evaluates the official 500-question test split.
