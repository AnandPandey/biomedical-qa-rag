# Data

Datasets are **not** stored in Git. Everything in this folder is gitignored except this file
and, once created, `splits/` (the committed train/dev/test ID lists).

No data has been downloaded yet. The data pipeline (Phase 2) will document:

- where PubMedQA is downloaded from and which version is used,
- the exact command that recreates `data/raw/` and `data/processed/`,
- how `data/splits/` was generated.

Source: PubMedQA, <https://pubmedqa.github.io/> (Jin et al., 2019, arXiv:1909.06146).

Read [docs/experiment-protocol.md](../docs/experiment-protocol.md) before working with the
test split.
