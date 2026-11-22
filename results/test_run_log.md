- 2026-11-22 16:11 | Tesla T4 | pip install -q torch==2.6.0 transformers==4.51.3 bitsandbytes==0.45.5 accelerate==1.6.0
- 2026-11-22 16:14 | Tesla T4 | pip install -q -e ".[llm,corpus,retrieval,classifier]"
- 2026-11-22 16:14 | Tesla T4 | pip freeze | grep -iE "^(torch|transformers|bitsandbytes|accelerate|bm25s|pystemmer)=="
- 2026-11-22 16:14 | Tesla T4 | python -m bmqa.data.download
- 2026-11-22 16:15 | Tesla T4 | python -m bmqa.data.splits
- 2026-11-22 16:15 | Tesla T4 | git status --short
- 2026-11-22 16:15 | Tesla T4 | python -m bmqa.retrieval.corpus
- 2026-11-22 16:15 | Tesla T4 | git diff --ignore-cr-at-eol --stat -- results/corpus_stats.json
- 2026-11-22 16:15 | Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl --index-dir outputs/indexes/bm25_strict
- 2026-11-22 16:17 | Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/gold.jsonl --index-dir outputs/indexes/bm25_gold
- 2026-11-22 16:18 | Tesla T4 | python -m bmqa.baselines --split test --allow-test
- 2026-11-22 16:18 | Tesla T4 | python -m bmqa.eval --pred results/predictions/majority/test_run1.jsonl --split test --allow-test
- 2026-11-22 16:20 | Tesla T4 | python -m bmqa.llm.run --config configs/closed_book.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 16:22 | Tesla T4 | pip install -q torch==2.6.0 transformers==4.51.3 bitsandbytes==0.45.5 accelerate==1.6.0
- 2026-11-22 16:22 | Tesla T4 | pip install -q -e ".[llm,corpus,retrieval,classifier]"
- 2026-11-22 16:22 | Tesla T4 | pip freeze | grep -iE "^(torch|transformers|bitsandbytes|accelerate|bm25s|pystemmer)=="
- 2026-11-22 16:22 | Tesla T4 | python -m bmqa.data.download
- 2026-11-22 16:22 | Tesla T4 | python -m bmqa.data.splits
- 2026-11-22 16:22 | Tesla T4 | git status --short
- 2026-11-22 16:22 | Tesla T4 | python -m bmqa.retrieval.corpus
- 2026-11-22 16:23 | Tesla T4 | git diff --ignore-cr-at-eol --stat -- results/corpus_stats.json
- 2026-11-22 16:23 | Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl --index-dir outputs/indexes/bm25_strict
- 2026-11-22 16:24 | Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/gold.jsonl --index-dir outputs/indexes/bm25_gold
- 2026-11-22 16:26 | Tesla T4 | python -m bmqa.baselines --split test --allow-test
- 2026-11-22 16:26 | Tesla T4 | python -m bmqa.eval --pred results/predictions/majority/test_run1.jsonl --split test --allow-test
- 2026-11-22 16:26 | Tesla T4 | python -m bmqa.llm.run --config configs/closed_book.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 17:02 | Tesla T4 | python -m bmqa.eval --pred results/predictions/closed_book/test_run1.jsonl --split test --allow-test
- 2026-11-22 17:02 | Tesla T4 | python -m bmqa.llm.run --config configs/closed_book.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 17:03 | Tesla T4 | python -m bmqa.eval --pred results/predictions/closed_book/test_run1.jsonl --split test --allow-test
- 2026-11-22 18:47 | Kaggle Tesla T4 / Tesla T4 | pip install -q torch==2.6.0 transformers==4.51.3 bitsandbytes==0.45.5 accelerate==1.6.0
- 2026-11-22 18:50 | Kaggle Tesla T4 / Tesla T4 | pip install -q -e ".[llm,corpus,retrieval,classifier]"
- 2026-11-22 18:50 | Kaggle Tesla T4 / Tesla T4 | pip freeze | grep -iE "^(torch|transformers|bitsandbytes|accelerate|bm25s|pystemmer)=="
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.data.download
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.data.splits
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | git status --short
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.retrieval.corpus
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | git diff --ignore-cr-at-eol --stat -- results/corpus_stats.json
- 2026-11-22 18:51 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl --index-dir outputs/indexes/bm25_strict
- 2026-11-22 18:53 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.retrieval.bm25 --corpus data/corpus/gold.jsonl --index-dir outputs/indexes/bm25_gold
- 2026-11-22 18:54 | Kaggle Tesla T4 / Tesla T4 | INCIDENT: first Kaggle notebook session stuck at 'Starting' and never ran; continued in a new Kaggle notebook after restoring results and llm_cache. No code or config changed.
- 2026-11-22 18:54 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.llm.run --config configs/oracle.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 19:38 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval --pred results/predictions/oracle/test_run1.jsonl --split test --allow-test
- 2026-11-22 19:38 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 20:22 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval --pred results/predictions/rag_strict/test_run1.jsonl --split test --allow-test
- 2026-11-22 20:22 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.rag.run --config configs/rag_gold_bm25.yaml --split test --run 1 --allow-test   [in-process, Qwen device_map={'':0}]
- 2026-11-22 20:23 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval --pred results/predictions/rag_gold/test_run1.jsonl --split test --allow-test
- 2026-11-22 20:27 | Kaggle Tesla T4 / Tesla T4 | NOTE: rag_gold run 1 reported 498 of 500 responses from the response cache. Cause: the cache is keyed by prompt; 364/500 rag_gold prompts were identical to oracle prompts (own article retrieved at k=1, so raw_output identical to oracle) and the remaining cached prompts matched rag_strict prompts (same top-1 document). Only 2 answers were newly generated. No command was executed twice. Predictions are deterministic (greedy) and identical to a fresh generation. Earlier note about 'an earlier execution' (if present above) is retracted.
- 2026-11-22 20:28 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval.claims --pred results/predictions/closed_book/test_run1.jsonl --split test --allow-test
- 2026-11-22 20:30 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval.claims --pred results/predictions/oracle/test_run1.jsonl --split test --allow-test
- 2026-11-22 20:32 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval.claims --pred results/predictions/rag_strict/test_run1.jsonl --split test --allow-test --corpus data/corpus/strict.jsonl
- 2026-11-22 20:34 | Kaggle Tesla T4 / Tesla T4 | python -m bmqa.eval.claims --pred results/predictions/rag_gold/test_run1.jsonl --split test --allow-test --corpus data/corpus/gold.jsonl
- 2026-11-22 20:36 | Kaggle Tesla T4 / Tesla T4 | zip -qr /kaggle/working/phase4_results.zip results/predictions results/metrics results/runs results/claims results/test_run_log.md

## Addendum (written after the run, from the session record; the lines above were logged automatically)

All times above are UTC. The following events happened but were not auto-logged, because the log was only
backed up to Drive by `save()` and these events came after the last backup or were not commands:

1. First `closed_book` run 1 attempt (Colab, ~16:20) failed at import with `RuntimeError: operator
   torchvision::nms does not exist` (Colab's preinstalled torchvision did not match the pinned torch 2.6.0).
   Fix: `pip uninstall torchvision torchaudio` and restart the session; the setup block was then run again
   (log lines 14-26). No code or config was changed.
2. `closed_book` run 1 was executed twice (log lines 26 and 28). The first execution (16:26-17:02) generated
   the 500 answers; the second (17:02-17:03) was served entirely from the response cache, so
   `results/runs/closed_book_test_run1.json` shows 0 uncached tokens. Predictions are identical.
3. The Colab `oracle` run 1 attempt was interrupted when Colab's free GPU usage limit was reached. Its command
   is not in the log (it came after the last backup). The Colab session ended; results and the response cache
   were restored from the Drive backup into a Kaggle notebook (2x T4, same pinned versions), where `oracle`,
   `rag_strict` and `rag_gold` were run. The first Kaggle notebook never started a session (log line 40).
4. During `rag_strict` run 1 on Kaggle the notebook output showed `Error displaying widget: model not found`
   and a BM25 `Split strings 0%` progress bar for a long time. This was a display problem only: the GPU was at
   ~98% utilisation, and the cell finished normally without being re-run.
   Correction to the NOTE at line 47: counted directly from the `evidence` field of the predictions, the
   question's own article is the rank-1 document for 363 of 500 rag_gold questions (0 of 500 for rag_strict);
   364 of 500 rag_gold answers have a raw_output identical to the oracle's (one of them with a different
   retrieved document). The explanation of the cache hits is unchanged.
5. Run times (UTC, from this log): closed_book ~36 min, oracle ~44 min, rag_strict ~44 min; rag_gold took ~1 min
   because 498 of 500 prompts were already in the cache (see the NOTE at line 47).
6. Not done: the classifier (5 seeds) on the test set, because the five checkpoints were not available, and
   runs 2 and 3 of the four LLM systems (not enough time). The analysis therefore uses run 1 of each LLM
   system and has no classifier result. This deviates from the pre-declared plan.
