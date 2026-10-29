# Reducing Hallucination in Biomedical Question Answering with Retrieval-Augmented Generation

A four-member university Deep Learning semester project.

**Question:** on PubMedQA, does retrieval-augmented generation (RAG) over PubMed abstracts make
a large language model's answers more accurate and better supported by evidence than answering
from memory alone? How do both compare with a fine-tuned biomedical classifier?

> **Status: Phase 1 (project skeleton).** No data has been downloaded, no models trained and no
> results produced yet. This section will be updated as each phase lands.

## Systems compared

| System | Description |
|---|---|
| Classifier | BioBERT or PubMedBERT fine-tuned for 3-class (yes / no / maybe) classification |
| Closed-book LLM | An LLM answers from the question alone |
| RAG LLM | The same LLM answers using PubMed abstracts it retrieved |

The exact experimental conditions, splits and test-set rules are in
[docs/experiment-protocol.md](docs/experiment-protocol.md).

## References

- Jin et al. (2019). *PubMedQA: A Dataset for Biomedical Research Question Answering.*
  [arXiv:1909.06146](https://arxiv.org/abs/1909.06146). Dataset: <https://pubmedqa.github.io/>
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

## Setup

Requires Python 3.11 or 3.12. Phase 1 needs only the development tools (no PyTorch, no GPU,
no API key).

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the same checks as CI:

```bash
ruff check .
ruff format --check .
pytest
```

## Repository layout

```
.github/        CI workflow, PR template, CODEOWNERS
data/           local datasets (gitignored); see data/README.md
docs/           shared interfaces and experiment protocol
src/bmqa/       the Python package
tests/          automated tests (run in CI, need no data or credentials)
```

Further folders (`configs/`, `scripts/`, `results/`, `report/`) are added by the phase that
first needs them.

## Team

Roles are proposed and will be confirmed by the group.

| Member | Area | GitHub |
|---|---|---|
| Member 1 (TBD) | PubMedQA data pipeline, biomedical transformer classifier | TBD |
| Member 2 (TBD) | Closed-book LLM baseline, shared LLM client | TBD |
| Member 3 (TBD) | PubMed retrieval and RAG pipeline | TBD |
| Member 4 (TBD) | Evaluation, unsupported-claim analysis, comparison, report coordination | TBD |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branches, pull requests, commit messages and
attribution.

## License

Not yet decided by the group. Until a license is added, all rights are reserved by the authors.
