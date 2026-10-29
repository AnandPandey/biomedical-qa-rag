# Experiment protocol

These rules exist so our final numbers are honest. They apply to every member and every system.
Breaking one invalidates the affected results; if it happens by accident, say so in the PR and
in the report rather than quietly re-running.

## Data and splits

- We use **PQA-L**, the 1,000 expert-labelled PubMedQA questions.
- **Test:** the official 500-question PubMedQA test set, identified by the official PMID list.
- **Train / dev:** the remaining 500 questions, split 400 / 100, stratified by label, with a
  fixed seed. The ID lists are committed under `data/splits/` so every member uses identical
  splits. Splits are never regenerated once committed.
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
5. Code that loads data must require an explicit opt-in to load the test split, so it cannot be
   used by accident. (To be implemented with the data loader in Phase 2.)

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
- Classifier: report mean ± std over several training seeds.
- LLM systems: outputs are not guaranteed to be deterministic, so each test condition is run
  3 times; report mean ± std. All raw outputs are cached and kept.
- Every number in the report must be reproducible from a committed config with a documented
  command. No hand-edited results.
