# 3. Retrieval and retrieval-augmented generation

*Author: Ayush Yadav (Member 3).*

## 3.1 Corpus

The corpus contains PubMed abstracts from PubMedQA's PQA-A and PQA-U subsets (Hugging Face
`qiaojin/PubMedQA`, revision `9001f28`), without their conclusion sections.

| Corpus | Documents | Contains the question's own article? |
|---|---|---|
| strict | 272,518 | Never: all 1,000 PQA-L articles removed |
| gold | 273,518 | Yes: strict + the 1,000 PQA-L articles |

PQA-A contributed 211,269 abstracts and PQA-U 61,249. Neither subset contained any PQA-L article
or any duplicate PMID, so the strict filter removed nothing in practice; it still guarantees that
no question can retrieve its own abstract. Rebuilding the corpora (on Windows and on Colab) gave
identical SHA-256 values. Statistics and checksums: `results/corpus_stats.json`.

## 3.2 Retriever

BM25 (`bm25s` 0.3.16), English stop words, Snowball stemming (`PyStemmer`); the query is the
question text.

Retrieval of the question's own article in the gold corpus (500 train + dev questions):

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@20 |
|---|---|---|---|---|---|
| 0.696 | 0.812 | 0.862 | 0.904 | 0.936 | 0.768 |

Strict-corpus leakage check (PQA-L articles in the top 20): 0.

PubMedQA questions are derived from article titles, so finding the own article is easier here
than in real use; the strict corpus measures the realistic case.

## 3.3 RAG systems and choice of k

Both RAG systems use exactly the closed-book prompt, model and settings (Section 2:
`Qwen/Qwen2.5-7B-Instruct`, 4-bit, pinned revision, prompt `v1`, greedy decoding); only the
evidence block differs. The rule for k was posted on the Phase 3 issue (#9) before any ablation
run: *"k = argmax dev macro-F1 of rag_strict over {1, 3, 5, 10}; ties -> smaller k; same k for
rag_gold."*

| k | rag_strict accuracy | rag_strict macro-F1 | rag_gold accuracy | rag_gold macro-F1 |
|---|---|---|---|---|
| 1 | 0.210 | **0.211** | 0.430 | 0.370 |
| 3 | 0.180 | 0.172 | 0.420 | 0.328 |
| 5 | 0.190 | 0.184 | 0.440 | 0.342 |
| 10 | 0.220 | 0.203 | 0.460 | 0.363 |

Chosen k: 1 (highest rag_strict macro-F1). All eight runs had 0 parse failures. The final
`rag_strict` and `rag_gold` dev runs were served entirely from the response cache of the matching
k = 1 ablation runs, so they are identical to them.

## 3.4 Dev comparison

| System | Accuracy | Macro-F1 | Parse failures |
|---|---|---|---|
| Closed-book (Section 2) | 0.260 | 0.235 | 0 |
| Oracle (Section 2) | 0.540 | 0.444 | 0 |
| RAG strict (k = 1) | 0.210 | 0.211 | 0 |
| RAG gold (k = 1) | 0.430 | 0.370 | 0 |

On the 100 dev questions, strict RAG does not improve on closed-book: it still answers "maybe"
for 72 questions (closed-book: 75; gold label: 11), and its macro-F1 is slightly lower. Abstracts
that are only lexically similar to the question do not give Qwen enough evidence to commit to
"yes" or "no". Gold-corpus RAG is much better (macro-F1 0.370) because BM25 places the question's
own abstract first for 72 of 100 dev questions: on those 72 its accuracy is 0.542, close to the
oracle's 0.540, while on the other 28 it is 0.143. The gain therefore comes almost entirely from
retrieving the source study, not from related literature. These are single dev runs on 100
questions; confidence intervals and the test-set comparison follow in Section 4.

## 3.5 Limitations

- Title-derived questions make retrieval of the own article unusually easy (gold corpus).
- The strict corpus may still contain closely related articles from the same research group.
- BM25 only; the optional dense retriever (MedCPT) was not built.
- k was selected on the same 100 dev questions that are reported here, so dev scores for the
  chosen k are slightly optimistic.
- Environment: the RAG runs used `torch 2.6.0`, `transformers 4.51.3`, `bitsandbytes 0.45.5`
  and `accelerate 1.6.0` on a Colab T4, with JAX kept off the GPU and the model loaded fully on
  GPU 0 (see `docs/interfaces.md`, section 10). Phase 2's dev runs may have used different
  library versions; the Phase 4 test run uses one environment for every LLM system.
