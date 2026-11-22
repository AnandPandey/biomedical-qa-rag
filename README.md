# Reducing Hallucination in Biomedical Question Answering with Retrieval-Augmented Generation

A four-member university Deep Learning semester project (private repository
[AnandPandey/biomedical-qa-rag](https://github.com/AnandPandey/biomedical-qa-rag)).

**Research question:** on PubMedQA, does retrieval-augmented generation (RAG) over PubMed
abstracts make a large language model's answers more accurate and better supported by evidence
than answering from memory alone? How do both compare with a fine-tuned biomedical classifier?

> **Status:** Phase 1 (data pipeline, scorer, majority baseline and PubMedBERT classifier) is
> merged with development-set results. Phase 2 (shared LLM layer, prompts, parser, and
> closed-book/oracle runner infrastructure) is implemented and verified offline (63/63 tests pass), with dev results for Qwen2.5-7B-Instruct
> (4-bit, pinned commit) run on a Google Colab T4 GPU.
> Phase 3 adds the strict and gold retrieval corpora, a BM25 retriever and the two RAG systems,
> with a dev ablation over the number of retrieved abstracts k (final k = 1). Merging Phase 3
> freezes all systems for the Phase 4 test run.
> The official test set is protected and loaded only once in the frozen Phase 4 test run.
> Phase 4 (frozen test run, statistics, claim analysis, report) is partly done: run 1 of the
> majority baseline and of the four LLM systems was run on the test set at tag
> `v1.0-test-results`. The classifier was not run on the test set and runs 2 and 3 were not done;
> see the results section below and `report/sections/04-evaluation.md`.

### Results on the official test set (n = 500, run 1 of each system)

| System | Accuracy | Macro-F1 | Macro-F1 95 % CI |
|---|---|---|---|
| Majority | 0.552 | 0.237 | [0.225, 0.248] |
| Closed-book LLM | 0.214 | 0.180 | [0.152, 0.210] |
| RAG strict | 0.198 | 0.203 | [0.167, 0.238] |
| RAG gold | 0.510 | 0.452 | [0.412, 0.493] |
| Oracle LLM | 0.588 | 0.503 | [0.462, 0.543] |

Not available: the PubMedBERT classifier on the test set (checkpoints were not available) and
the second and third run of each LLM system, so no mean ± std is reported. RAG strict is not
distinguishable from closed-book (Δ macro-F1 +0.022, 95 % CI [-0.020, +0.064], McNemar
p = 0.54). RAG gold retrieves the question's own article at rank 1 for 363 of 500 questions.
Full tables: [results/tables/main_results.md](results/tables/main_results.md),
[significance](results/tables/significance.md) and
[hallucination](results/tables/hallucination.md). The claim analysis used an NLI checker whose
agreement with our labels was low (kappa 0.133), and those labels came from an AI assistant, not
human annotators (`results/annotation/PROVENANCE.md`); treat the hallucination numbers as
unreliable. Every command of the test run, with incidents, is in
[results/test_run_log.md](results/test_run_log.md).

Regenerate the tables and figures (no API calls needed):

```bash
python -m pip install -e ".[report]"
python -m bmqa.data.download && python -m bmqa.data.splits
python -m bmqa.eval.compare --split test --allow-test
```

### Results so far (development set, 100 questions)

| System | Accuracy | Macro-F1 | Status |
|---|---|---|---|
| Majority (Phase 1) | 0.550 | 0.237 | Merged |
| PubMedBERT classifier (Phase 1, mean ± std over 5 seeds) | 0.576 ± 0.049 | 0.395 ± 0.048 | Merged |
| Closed-book LLM (`Qwen2.5-7B-Instruct`, dev run 1) | 0.260 | 0.235 | Dev run 1, prompt v1, 0 parse failures |
| Oracle LLM (`Qwen2.5-7B-Instruct`, dev run 1) | 0.540 | 0.444 | Dev run 1, prompt v1, 0 parse failures |
| RAG strict (BM25, k = 1, own article removed) | 0.210 | 0.211 | Dev run 1, prompt v1, 0 parse failures |
| RAG gold (BM25, k = 1, own article may be retrieved) | 0.430 | 0.370 | Dev run 1, prompt v1, 0 parse failures |

The classifier's dev scores are optimistic because dev was used to pick the best epoch. It
almost never predicts "maybe". See `report/sections/01-data-and-classifier.md`.
Closed-book Qwen answers "maybe" for 75% of dev questions and falls below the majority baseline; see
`report/sections/02-llm-baselines.md`. Strict RAG does not beat closed-book on dev; gold-corpus RAG
does, because BM25 finds the question's own abstract for 72 of 100 dev questions at k = 1. See
`report/sections/03-retrieval-and-rag.md`. Results are one run on 100 questions.

## Team

| Phase | Member | GitHub | Responsibility |
|---|---|---|---|
| 1 | Anand Kumar Pandey | [@AnandPandey](https://github.com/AnandPandey) | PubMedQA data pipeline, fixed splits, scorer, majority baseline, PubMedBERT classifier |
| 2 | Hari Singh | [@harisingh294](https://github.com/harisingh294) | Shared LLM client, prompt template, closed-book and oracle LLM baselines |
| 3 | Ayush Yadav | [@AyushYadav-rgb477](https://github.com/AyushYadav-rgb477) | PubMed retrieval corpus, BM25 retriever, RAG systems |
| 4 | Guman Singh Rajpoot | [@Guman-Singh-Rajpoot](https://github.com/Guman-Singh-Rajpoot) | Frozen test run, statistics, unsupported-claim analysis, final report |

## Dataset

[PubMedQA](https://pubmedqa.github.io/) (Jin et al., 2019; MIT license): each question is
answered **yes**, **no** or **maybe** from a PubMed abstract.

| Subset | Size | Used for |
|---|---|---|
| PQA-L (expert-labelled) | 1,000 | train 400, dev 100, official test 500 |
| PQA-A (artificially labelled) | 211,269 | retrieval corpus only |
| PQA-U (unlabelled) | 61,249 | retrieval corpus only |

The official test set (the 500 PMIDs in the PubMedQA repository's `test_ground_truth.json`) is
used once, in the final test run. The rules that protect it are in
[docs/experiment-protocol.md](docs/experiment-protocol.md).

## Systems compared

| System | Description | Evidence it sees |
|---|---|---|
| Majority | Always predicts the most frequent training label | none |
| Classifier | PubMedBERT fine-tuned for 3-class (yes / no / maybe) classification | gold abstract |
| Closed-book LLM | An LLM answers from the question alone | none |
| Oracle LLM | The same LLM, given the question's own abstract (upper bound) | gold abstract |
| RAG, strict | The same LLM with BM25-retrieved abstracts; the question's own article is removed from the corpus | retrieved |
| RAG, gold-in-corpus | The same LLM with BM25-retrieved abstracts; the own article may be retrieved | retrieved |

All LLM systems use the same model, prompt template and answer format; only the evidence
differs. Hallucination is measured at three levels: wrong final answer, rationale sentences not
supported by the evidence (checked with an NLI model that is validated against human
annotation), and fabricated citations.

## Development plan

The project is built in four **sequential** phases. A member starts only after the previous
phase's pull request is merged into `main`.

```
Phase 1 (Anand) -> merge -> Phase 2 (Hari) -> merge -> Phase 3 (Ayush) -> merge
    -> freeze tag v1.0-test-results -> Phase 4 (Guman) -> merge -> tag v1.1-report
```

| Phase | Branch | Delivers |
|---|---|---|
| 1 | `data/phase-1-pipeline-and-classifier` | data download and splits, test-set guard, scorer, majority baseline, classifier |
| 2 | `llm/phase-2-closed-book-baseline` | cached LLM client, prompts, parser, closed-book and oracle systems |
| 3 | `rag/phase-3-retrieval-and-rag` | strict and gold corpora, BM25, RAG systems, dev ablation over k |
| 4 | `eval/phase-4-evaluation-and-report` | frozen test run, bootstrap CIs and significance tests, claim analysis, report |

## References

- Jin et al. (2019). *PubMedQA: A Dataset for Biomedical Research Question Answering.*
  [arXiv:1909.06146](https://arxiv.org/abs/1909.06146)
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

## Setup

Requires Python 3.11, 3.12, or 3.13. The base development tools need no PyTorch, no GPU and no API key.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional extras, installed only when needed (CI installs only `dev`):

| Extra | Needed for |
|---|---|
| `classifier` | Training or running the PubMedBERT classifier (PyTorch, Transformers) |
| `llm` | Calling Anthropic Claude API or running Qwen locally/Colab (`anthropic>=2.12`, `transformers>=4.40`, `torch>=2.2`, `accelerate>=0.28`, `bitsandbytes>=0.43`) |
| `corpus` | Building the retrieval corpus from Hugging Face (`datasets`) |
| `retrieval` | BM25 retrieval (`bm25s`, `PyStemmer`); also installed in CI so the BM25 tests run |

Run standard code verification checks:

```bash
ruff check .
ruff format --check .
pytest
```

## Phase 2 Architecture & LLM Baselines

Phase 2 introduces a shared, provider-neutral LLM execution layer reused by downstream Phase 3 RAG systems:

- **Provider-Neutral Client Protocol (`src/bmqa/schema.py`):** `LLMClient` defines a uniform protocol (`model: str` and `generate(prompt, max_tokens) -> LLMResponse`).
- **Offline Client (`FakeLLMClient`):** Fully deterministic client for isolated testing, CI, and local dry-runs (`--dry-run`). Requires zero network access and costs $0.00.
- **Provider Client (`AnthropicClient`):** Integrates Claude models via the official `anthropic` SDK, requesting structured JSON output (`ANSWER_SCHEMA`) with low reasoning effort.
- **Open-Weights Client (`QwenClient`):** Integrates `Qwen/Qwen2.5-7B-Instruct` via `transformers` with 4-bit NormalFloat4 quantization (`bitsandbytes`) for free, GPU-accelerated inference on Google Colab T4.
- **Prompt Templating (`src/bmqa/llm/prompts.py`):** Versioned template `v1` enforcing consistent instructions across all conditions. Suppresses PMIDs, numbers documents sequentially as `[1]`, `[2]`, and strictly excludes `long_answer` (conclusion) to prevent answer leakage.
- **Robust Output Parser (`src/bmqa/llm/parse.py`):** Parses JSON responses, normalizes label variants to `yes`/`no`/`maybe`, validates citations against supplied evidence count, and tags malformed outputs with `PARSE_FAILURE:` so they are scored as incorrect.
- **Deterministic Response Cache (`src/bmqa/llm/cache.py`):** Caches replies on disk under `outputs/llm_cache/` using canonical SHA-256 hashes of `(model, effort, max_tokens, prompt, run)`. Re-running an identical run makes zero API calls. Failed calls are never cached. Cache writes use atomic `.tmp` replacement.
- **Generic Runner (`src/bmqa/llm/run.py`):** Accepts an arbitrary `EvidenceFn` (`no_evidence` for closed-book, `gold_evidence` for oracle, and future retrieval evidence for RAG).

### Baseline Definitions

- **Closed-book LLM:** Answers questions from memory without supporting evidence documents (`evidence: none`, `citations: []`).
- **Oracle LLM:** Answers questions using the permitted gold abstract contexts (`evidence: gold`) formatted as document `[1]`. Establishes the upper bound for retrieval systems.

### Environment Variable Setup

For live Anthropic model execution, copy `.env.example` to `.env` and set your key locally:

```bash
cp .env.example .env
# Edit .env locally: ANTHROPIC_API_KEY=<your_api_key_here>
```

> [!IMPORTANT]
> `.env` is gitignored and must never be committed. Never share or paste API keys in chat or documentation. Tests, Qwen Colab runs, and offline dry-runs do **not** require an API key.

## Reproducing Phase 1 (development set)

```bash
python -m bmqa.data.download          # PQA-L at a pinned commit, checksums verified
python -m bmqa.data.splits            # reproduces data/splits/, writes data/processed/
python -m bmqa.baselines --split dev
python -m bmqa.eval --pred results/predictions/majority/dev_run1.jsonl --split dev
```

Classifier training needs a GPU; see [docs/colab.md](docs/colab.md):

```bash
python -m pip install -e ".[classifier]"
python -m bmqa.classifier.train --config configs/classifier_pubmedbert.yaml
python -m bmqa.eval --pred results/predictions/classifier/dev_run1.jsonl --split dev
```

## Reproducing Phase 2 (Offline & Dry-Runs)

All Phase 2 engineering checks, prompts, parsers, and pipeline dry-runs run completely offline at zero cost:

```bash
# 1. Run all unit and regression tests (62 tests pass):
python -m pytest

# 2. Run Ruff linter and format checks:
python -m ruff check .
python -m ruff format --check .

# 3. Closed-book baseline offline dry-run (processes all 100 dev examples, costs $0.00):
python -m bmqa.llm.run --config configs/closed_book.yaml --split dev --run 1 --dry-run

# 4. Oracle baseline offline dry-run (processes all 100 dev examples, costs $0.00):
python -m bmqa.llm.run --config configs/oracle.yaml --split dev --run 1 --dry-run

# 5. Score dry-run predictions with official evaluator:
python -m bmqa.eval --pred outputs/dry_run/predictions/closed_book/dev_run1.jsonl --split dev
python -m bmqa.eval --pred outputs/dry_run/predictions/oracle/dev_run1.jsonl --split dev
```

### Running Phase 2 on Google Colab (Free T4 GPU)

To execute live closed-book and oracle baselines with `Qwen2.5-7B-Instruct` (4-bit) at zero cost on Google Colab:

```bash
# 1. Install LLM extra with GPU libraries:
!pip install -q -e ".[llm]"

# 2. Run closed-book baseline (100 dev questions):
!python -m bmqa.llm.run --config configs/closed_book.yaml --split dev --run 1

# 3. Run oracle baseline (100 dev questions):
!python -m bmqa.llm.run --config configs/oracle.yaml --split dev --run 1

# 4. Score predictions:
!python -m bmqa.eval --pred results/predictions/closed_book/dev_run1.jsonl --split dev
!python -m bmqa.eval --pred results/predictions/oracle/dev_run1.jsonl --split dev
```

## Reproducing Phase 3 (development set)

Build the corpora and indexes (CPU is enough; the corpus files are about 370 MB each and are
gitignored):

```bash
python -m pip install -e ".[dev,corpus,retrieval]"
python -m bmqa.data.download && python -m bmqa.data.splits
python -m bmqa.retrieval.corpus                      # data/corpus/{strict,gold}.jsonl + results/corpus_stats.json
python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl --index-dir outputs/indexes/bm25_strict
python -m bmqa.retrieval.bm25 --corpus data/corpus/gold.jsonl --index-dir outputs/indexes/bm25_gold
python -m bmqa.retrieval.evaluate                    # results/retrieval/bm25_gold_traindev.json
python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --run 1 --dry-run  # free
```

`git diff results/corpus_stats.json` must be empty after a rebuild (same SHA-256 values).

The RAG runs use Qwen on a GPU (Colab T4). Before running there, keep JAX off the GPU and load the
model fully on GPU 0; see "Running RAG with Qwen on a Colab T4 GPU" in
[docs/interfaces.md](docs/interfaces.md):

```bash
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --run 1 --k 3   # ablation run
python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --run 1         # final (k from config)
python -m bmqa.rag.run --config configs/rag_gold_bm25.yaml --split dev --run 1
python -m bmqa.eval --pred results/predictions/rag_strict/dev_run1.jsonl --split dev
```

The dev ablation used k in {1, 3, 5, 10} for both corpora; k = 1 was chosen by the rule posted on
issue #9 before the ablation.

## Repository layout

```
.github/        CI workflow, PR template, CODEOWNERS
configs/        experiment configs (one YAML file per system)
data/           local datasets (gitignored) and committed split ID lists; see data/README.md
docs/           shared interfaces, experiment protocol, Colab guide
report/         report sections
results/        committed predictions and metrics (development set only until Phase 4)
src/bmqa/       the Python package
tests/          automated tests (run in CI, need no data or credentials)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, pull requests, commit messages and
attribution.

## License

Not yet decided by the group. Until a license is added, all rights are reserved by the authors.
