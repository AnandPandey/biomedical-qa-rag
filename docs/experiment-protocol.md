# Experiment protocol

These rules exist so our final numbers are honest. They apply to every member and every system.
Breaking one invalidates the affected results; if it happens by accident, say so in the PR and
in the report rather than quietly re-running.

## Data and splits

- We use **PQA-L**, the 1,000 expert-labelled PubMedQA questions, downloaded from
  `github.com/pubmedqa/pubmedqa` at commit `1cbae8e92f72f20c8d3747cbb3bf5bc53554d997` with
  SHA-256 checks (see `data/README.md`).
- **Test:** the official 500-question PubMedQA test set, identified by the official PMID list
  `data/test_ground_truth.json` (276 yes, 169 no, 55 maybe).
- **Train / dev:** the remaining 500 questions, split 400 / 100, stratified by label: for each
  label, the PMIDs are sorted, shuffled with `random.Random(42)` (one generator, labels in the
  order yes, no, maybe) and `round(0.2 * n)` go to dev. Result: train 221 / 135 / 44, dev
  55 / 34 / 11 (yes / no / maybe). The ID lists are committed under `data/splits/` so every member
  uses identical splits. Splits are never regenerated once committed: `python -m bmqa.data.splits`
  refuses to run if a fresh computation differs from the committed lists.
- PQA-A (artificial) and PQA-U (unlabelled) are used only as the retrieval corpus and, optionally,
  as extra classifier training data. They are never used for evaluation.

## Protecting the test set

1. **No one looks at test examples** (questions, contexts or labels) while developing. Error
   analysis, prompt writing and debugging use train and dev only.
2. **All tuning uses dev:** prompts, hyperparameters, retriever choice, `k`, thresholds, parsing
   rules, everything.
3. **Freeze before testing.** When every system is final, the configs and code are merged and
   tagged (`v1.0-test-results`). Only then are systems run on test, once, by the agreed
   procedure.
4. **No changes after seeing test results.** If a bug is found after the test run, fix it, re-run
   *all* affected systems, and report both the original and corrected numbers.
5. Code that loads data requires an explicit opt-in to load the test split, so it cannot be used
   by accident: `bmqa.data.pubmedqa.load_split("test")` raises `PermissionError` unless
   `allow_test=True`, and every command has an `--allow-test` flag that is off by default.

## Model inputs and leakage

- `long_answer` (the article's conclusion) is never part of any model input in the main
  experiments, because it states the answer.
- The retrieval corpus uses abstracts **without** their conclusion sections.
- Experimental conditions (all LLM conditions use the same model, prompt template and output
  format; only the evidence block differs):

  | Condition | Evidence shown |
  |---|---|
  | Classifier | Gold contexts |
  | Closed-book LLM | None |
  | RAG, strict | Retrieved; the question's own article is excluded from the corpus |
  | RAG, gold-in-corpus | Retrieved; the question's own article may be retrieved |
  | Oracle LLM | Gold contexts (upper bound) |

## Reporting

- Primary metrics: **accuracy** and **macro-F1** on test, with 95% bootstrap confidence
  intervals. Also report per-class F1 and the confusion matrix.
- Always report the majority-class baseline alongside model results.
- Compare systems with a paired test (McNemar or paired bootstrap) on the same test items.
  Differences inside the confidence interval are reported as inconclusive, not as wins.
- Classifier: report mean ± std over several training seeds. Dev scores of the classifier are
  optimistic, because dev was used to choose the best epoch.
- LLM systems: outputs are not guaranteed to be deterministic, so each test condition is run
  3 times; report mean ± std. All raw outputs are cached and kept.
- Every number in the report must be reproducible from a committed config with a documented
  command. No hand-edited results.

## Record of the freeze and test run

- Freeze tag `v1.0-test-results` on commit `e860c15` (the Phase 3 merge commit), 2026-11-22.
  **Deviation:** the tag was first created on `1695851` (the Phase 0 close-out, which contains none of
  the Phase 1-3 code) and was deleted and re-created on `e860c15` before any test prediction was
  made. No test prediction exists that predates the corrected tag.
- Test run carried out by Guman Singh Rajpoot on 2026-11-22 (Google Colab and Kaggle, T4 GPUs);
  every command, machine and incident is in `results/test_run_log.md`.
- The pre-declared analysis plan was posted on the Phase 4 issue before any test prediction:
  <https://github.com/AnandPandey/biomedical-qa-rag/issues/3#issuecomment-7066870888>.
- Frozen code and configs: `git diff v1.0-test-results -- src/bmqa/data src/bmqa/classifier
  src/bmqa/llm src/bmqa/retrieval src/bmqa/rag` and the six system configs is empty.
- Deviations from this protocol and from the pre-declared plan:
  - The classifier was not run on the test set (checkpoints not available) and runs 2 and 3 of the
    LLM systems were not done, so there is one run per system and no mean ± std.
  - The fourth pre-declared comparison (classifier vs rag_strict) was not made; Holm correction
    was applied over the three comparisons that exist.
  - The claim-checker validation on dev used labels produced by an AI assistant (two passes and an
    adjudication run), not two human annotators (`results/annotation/PROVENANCE.md`).
  - The LLM runs were made in two sessions after a Colab GPU limit was reached; see the log.
