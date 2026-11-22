# Reducing Hallucination in Biomedical Question Answering with Retrieval-Augmented Generation

Anand Kumar Pandey, Hari Singh, Ayush Yadav, Guman Singh Rajput

*Coordinated by Guman Singh Rajput (Member 4). Sections 1-3 are written by their authors and
edited here only for consistency.*

## Abstract

We ask whether retrieval-augmented generation (RAG) over PubMed abstracts makes a language
model's answers on PubMedQA more accurate and better supported than answering from memory. We use
the official 500-question test set and one open model (Qwen2.5-7B-Instruct, 4-bit), compared as
closed-book, with the question's own abstract (oracle), and with BM25 retrieval over a corpus that
excludes (strict) or includes (gold) the question's own article, against a majority baseline. On
the test set, closed-book reaches accuracy 0.214 and macro-F1 0.180, strict RAG 0.198 and 0.203,
gold RAG 0.510 and 0.452, and the oracle 0.588 and 0.503; the majority baseline has accuracy
0.552. Strict RAG is not distinguishable from closed-book (macro-F1 difference +0.022, 95 % CI
[-0.020, +0.064], McNemar p = 0.54). Gold RAG is much better, but it retrieves the question's own
article for 363 of 500 questions, so it largely recreates the oracle. Fabricated citations are
almost absent. Main limitations: one run per system, no classifier result, and a claim checker
that agreed poorly with our labels (kappa 0.133), which were themselves produced by an AI
assistant rather than human annotators.

## Introduction

Large language models (LLMs) can state biomedical claims that no source supports. In medicine
this matters because a fluent, confident answer may be wrong or unsupported by the literature.
Retrieval-augmented generation (RAG) gives the model retrieved documents to answer from, which
should make answers more accurate and easier to check against evidence. Our research question
is: on PubMedQA, does RAG over PubMed abstracts make an LLM's answers more accurate and better
supported by evidence than answering from memory, and how do both compare with a fine-tuned
biomedical classifier?

## Background

- PubMedQA (Jin et al., 2019): a biomedical question answering dataset in which each question
  is answered **yes**, **no** or **maybe** from a PubMed abstract. We use the 1,000
  expert-labelled questions (PQA-L: train 400, dev 100, official test 500) for training and
  evaluation, and the artificially labelled (PQA-A) and unlabelled (PQA-U) subsets only as the
  retrieval corpus.
- Retrieval-augmented generation (Lewis et al., 2020): a generator conditions its answer on
  documents found by a retriever. Here the retriever is BM25 over PubMed abstracts and the
  generator is the same LLM used in the closed-book and oracle baselines.

## Data and methods

See the sections below, included in this order when the report is assembled:

1. [Data and classifier](sections/01-data-and-classifier.md) (Anand Kumar Pandey)
2. [LLM baselines](sections/02-llm-baselines.md) (Hari Singh)
3. [Retrieval and RAG](sections/03-retrieval-and-rag.md) (Ayush Yadav)
4. [Evaluation, results and hallucination analysis](sections/04-evaluation.md)
   (Guman Singh Rajput)

## Conclusion

On our test data, retrieving from a corpus that excludes the question's own article did not
measurably help the model: strict RAG and closed-book are statistically indistinguishable, and both
fall below the majority baseline in accuracy. The large gain of gold RAG comes from retrieving
the question's own abstract, not from retrieval in general. These results rest on a single run of
each system and one model, and our hallucination measurement is weakened by a claim checker that
agreed poorly with our labels. A fair test of the main hypothesis would need the planned three
runs per system, the classifier comparison, human annotation for validating the checker, and a
retrieval step that finds relevant abstracts more reliably than BM25 with one document.

## Reproducibility

Code and results: tag `v1.1-report`. Frozen systems: tag `v1.0-test-results`. Every command
of the test run is in `results/test_run_log.md`; tables regenerate with
`python -m bmqa.eval.compare --split test --allow-test`.

## Individual contributions

| Member | Contribution |
|---|---|
| Anand Kumar Pandey | Phase 1: PubMedQA data pipeline, fixed splits, scorer, majority baseline and PubMedBERT classifier; project skeleton and team conventions |
| Hari Singh | Phase 2: shared LLM client, prompt template, closed-book and oracle LLM baselines |
| Ayush Yadav | Phase 3: PubMed retrieval corpus, BM25 retriever, RAG systems and their dev ablation |
| Guman Singh Rajput | Phase 4: frozen test run, statistics, unsupported-claim analysis, final report and report coordination |

Annotators for Section 4.4: both annotation passes were done by an AI assistant (Claude), not by
team members; no human annotated (see the deviation noted in Section 4.4 and
`results/annotation/PROVENANCE.md`).

## Use of AI assistants

This section covers the Phase 4 work and the report coordination. Authors of Sections 1-3 should
state their own AI-assistant use for those parts.

- **Tool:** Claude (Anthropic), used through Claude Code.
- **Phase 4 code and plan.** The Phase 4 plan and the initial drafts of `bmqa.eval.stats`,
  `claims`, `annotate`, `compare`, their tests, the annotation guideline and the report skeleton were
  produced with an AI assistant from the written plan. They were copied into the repository step by
  step, run against the existing test suite (87 tests pass; ruff clean) and adapted where they
  conflicted with earlier phases: the package copies of `pyproject.toml`, `README.md` and
  `docs/interfaces.md` were not copied wholesale because they would have removed Phase 2 and 3
  content.
- **Experiment execution.** The AI assistant wrote the step-by-step Colab and Kaggle commands and
  the log helpers, and helped diagnose errors (a torchvision mismatch, a Colab GPU limit, a
  notebook display problem). The experiments themselves were run by the human author on Colab and
  Kaggle, and the outputs were pasted back and checked. The assistant also checked the predictions
  (500 rows, IDs match the test split) and found that rag_gold retrieves the question's own article
  for 363 of 500 questions.
- **Annotation (important).** The 160 dev claims used to validate the claim checker were labelled by
  the AI assistant, not by human annotators: two separate passes and a separate adjudication run
  (`results/annotation/PROVENANCE.md`). This deviates from the pre-declared plan, which required two
  human annotators, and is why the checker's agreement figures are not a human validation.
- **Writing.** The evaluation section (Section 4), the README results and the freeze record in
  `docs/experiment-protocol.md` were drafted by the AI assistant from the files under `results/`;
  every number in them comes from a committed file. The final reviewer (Anand Kumar Pandey) has not
  reviewed this text at the time of writing.
- **What the AI assistant did not do:** it did not run the test set on its own initiative, did not
  change any frozen system, prompt or configuration after the freeze, and did not push to `main`.

## References

- Jin, Q., Dhingra, B., Liu, Z., Cohen, W., and Lu, X. (2019). PubMedQA: A Dataset for
  Biomedical Research Question Answering. EMNLP-IJCNLP. arXiv:1909.06146.
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
  NeurIPS. arXiv:2005.11401.
- Models and libraries: classifier `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`;
  LLM `Qwen/Qwen2.5-7B-Instruct` (4-bit); retrieval `bm25s`; NLI checker
  `cross-encoder/nli-deberta-v3-base`; Hugging Face `transformers` and PyTorch.
