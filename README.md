# Reducing Hallucination in Biomedical Question Answering with Retrieval-Augmented Generation

A four-member university Deep Learning semester project (private repository
[AnandPandey/biomedical-qa-rag](https://github.com/AnandPandey/biomedical-qa-rag)).

**Research question:** on PubMedQA, does retrieval-augmented generation (RAG) over PubMed
abstracts make a large language model's answers more accurate and better supported by evidence
than answering from memory alone? How do both compare with a fine-tuned biomedical classifier?

> **Status:** Phase 1 (data pipeline, scorer, majority baseline and PubMedBERT classifier) is
> implemented, with development-set results only; the test set is used once, in Phase 4. The
> LLM, retrieval and evaluation phases follow (see [Development plan](#development-plan)).
> This section is updated as each phase is merged.

### Results so far (development set, 100 questions)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Majority | 0.550 | 0.237 |
| PubMedBERT classifier (mean ± std over 5 seeds) | 0.576 ± 0.049 | 0.395 ± 0.048 |

The classifier's dev scores are optimistic because dev was used to pick the best epoch. It
almost never predicts "maybe". See `report/sections/01-data-and-classifier.md`.

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

Requires Python 3.11 or 3.12. The development tools need no PyTorch, no GPU and no API key.

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

Run the same checks as CI:

```bash
ruff check .
ruff format --check .
pytest
```

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
