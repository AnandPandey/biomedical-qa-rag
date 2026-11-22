# Data

Datasets are **not** stored in Git. Everything in this folder is gitignored except this file
and `splits/` (the committed train/dev/test ID lists).

Source: PubMedQA, <https://pubmedqa.github.io/> (Jin et al., 2019, arXiv:1909.06146; MIT
license).

## PQA-L (labelled questions)

Downloaded from the official repository at a pinned commit so the data cannot change:

| File | URL | SHA-256 |
|---|---|---|
| `ori_pqal.json` | `https://raw.githubusercontent.com/pubmedqa/pubmedqa/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/data/ori_pqal.json` | `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d` |
| `test_ground_truth.json` | `https://raw.githubusercontent.com/pubmedqa/pubmedqa/1cbae8e92f72f20c8d3747cbb3bf5bc53554d997/data/test_ground_truth.json` | `939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51` |

```bash
python -m bmqa.data.download    # -> data/raw/, checksums verified
python -m bmqa.data.splits      # -> data/processed/{train,dev,test}.jsonl
```

`python -m bmqa.data.splits` checks that a fresh computation reproduces the committed
`data/splits/*_ids.txt` exactly, then writes `data/processed/`.

| Split | Questions | yes | no | maybe |
|---|---|---|---|---|
| train | 400 | 221 | 135 | 44 |
| dev | 100 | 55 | 34 | 11 |
| test | 500 | 276 | 169 | 55 |

## Retrieval corpus (PQA-A and PQA-U)

Built from the Hugging Face dataset `qiaojin/PubMedQA` (MIT license), configs `pqa_artificial`
(211,269 rows) and `pqa_unlabeled` (61,249 rows), pinned to revision
`9001f2853fb87cab8d220904e0de81ac6973b318`.

```bash
python -m bmqa.retrieval.corpus    # -> data/corpus/strict.jsonl, data/corpus/gold.jsonl
```

- `strict.jsonl`: PQA-A + PQA-U minus every PQA-L article, de-duplicated by PMID.
- `gold.jsonl`: `strict.jsonl` plus the 1,000 PQA-L articles.
- Text = abstract contexts without the conclusion. Files are sorted by PMID, so their SHA-256
  values (in `results/corpus_stats.json`) are reproducible.

Read [docs/experiment-protocol.md](../docs/experiment-protocol.md) before working with the
test split: it is loaded only in the frozen Phase 4 test run.
