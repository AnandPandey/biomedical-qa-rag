"""Retrieval quality on train + dev questions (never test).

In the gold corpus each question's own article is present, so Recall@k and MRR measure how
often BM25 finds it. In the strict corpus no PQA-L article may ever be retrieved (leakage check).

Usage:
    python -m bmqa.retrieval.evaluate --gold-index outputs/indexes/bm25_gold \
        --strict-index outputs/indexes/bm25_strict
"""

# Lets us write type hints such as "int | None" on every supported Python version.
from __future__ import annotations

# argparse reads command-line options such as --gold-index.
import argparse

# json writes the results file.
import json

# Path is an object-oriented way to work with file and folder paths.
from pathlib import Path

# PROCESSED_DIR = "data/processed"; load_split loads the questions of one split as Examples.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split

# SPLITS_DIR = "data/splits"; read_ids reads a text file with one PubMed ID per line.
from bmqa.data.splits import SPLITS_DIR, read_ids

# Example = one PubMedQA question; Retriever = the interface any retriever must follow.
from bmqa.schema import Example, Retriever

# The cut-offs k at which we report Recall@k. The largest (20) is also how many documents
# we retrieve per question and the cut-off for MRR.
KS = (1, 3, 5, 10, 20)


# Finds where a document appears in a ranked list.
# Inputs: doc_id to look for; ranked = list of doc_ids, best first.
# Output: its 1-based rank (1 = top result), or None if it was not retrieved at all.
def rank_of(doc_id: str, ranked: list[str]) -> int | None:
    # list.index() is 0-based, so add 1. Check "in" first because index() raises if missing.
    return ranked.index(doc_id) + 1 if doc_id in ranked else None


# Computes Recall@k for each k and MRR from the rank of the correct document per question.
# Input: ranks = one entry per question (an int rank, or None if not found); ks = cut-offs.
# Output: a dict such as {"recall@1": 0.4, ..., "mrr@20": 0.5, "n": 800}.
def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict:
    """Recall@k and MRR@max(ks) from the rank of the relevant document (None = not found)."""
    # Number of questions.
    n = len(ranks)
    # Recall@k = fraction of questions whose correct document is in the top k.
    # (r is not None and r <= k) is True/False, and sum() counts the Trues.
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    # MRR (mean reciprocal rank): each question scores 1/rank (rank 1 -> 1.0, rank 2 -> 0.5,
    # ...), or 0 if the document is missing or below the cut-off; then we average.
    out[f"mrr@{max(ks)}"] = sum(1 / r for r in ranks if r is not None and r <= max(ks)) / n
    # Also store how many questions the numbers are based on.
    out["n"] = n
    # Return the metrics dict.
    return out


# Measures retrieval quality in the gold corpus, where each question's own abstract exists.
# Inputs: a retriever, the questions, and the cut-offs ks. Output: the metrics dict.
def evaluate_gold(retriever: Retriever, examples: list[Example], ks=KS) -> dict:
    # One rank (or None) per question.
    ranks = []
    # Go through every question.
    for ex in examples:
        # Search with the question text and keep only the doc_ids, best first.
        ranked = [hit.doc_id for hit in retriever.retrieve(ex.question, max(ks))]
        # The "correct" document is the question's own article; its doc_id equals the
        # question's ID because both are the PubMed ID.
        ranks.append(rank_of(ex.id, ranked))
    # Turn the ranks into Recall@k and MRR.
    return retrieval_metrics(ranks, ks)


# Leakage check for the strict corpus.
# Inputs: retriever, questions, the set of all PQA-L IDs, and k results per question.
# Output: how many retrieved documents are PQA-L articles (should be exactly 0).
def leakage_count(retriever: Retriever, examples: list[Example], pqal_ids: set[str], k: int) -> int:
    """Number of retrieved PQA-L documents in the top k (must be 0 for the strict corpus)."""
    # Count True values: one for every retrieved hit whose doc_id is a PQA-L article.
    # Nested loop: for each question, for each of its top-k hits.
    return sum(
        hit.doc_id in pqal_ids for ex in examples for hit in retriever.retrieve(ex.question, k)
    )


# Command-line entry point: evaluates both indexes and writes the results file.
# Input: argv = list of command-line arguments (None means "use the real command line").
# Output: the results dict (also written to JSON and printed).
def main(argv: list[str] | None = None) -> dict:
    # Imported here because it needs the optional bm25s library; the metric functions above
    # (and their tests) work without it.
    from bmqa.retrieval.bm25 import BM25Retriever

    # Create the command-line parser; the description appears in --help.
    parser = argparse.ArgumentParser(description="Retrieval metrics on train + dev.")
    # --gold-index: folder of the BM25 index built from gold.jsonl.
    parser.add_argument("--gold-index", type=Path, default=Path("outputs/indexes/bm25_gold"))
    # --strict-index: folder of the BM25 index built from strict.jsonl.
    parser.add_argument("--strict-index", type=Path, default=Path("outputs/indexes/bm25_strict"))
    # --processed-dir: where the processed train/dev question files live.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # --out: where the results JSON is written.
    parser.add_argument(
        "--out", type=Path, default=Path("results/retrieval/bm25_gold_traindev.json")
    )
    # Read the options the user typed (or the argv list passed in by a test).
    args = parser.parse_args(argv)

    # Load the train and dev questions and join the two lists with "+".
    # The test split is deliberately NOT used: it is kept untouched for the final evaluation.
    examples = load_split("train", processed_dir=args.processed_dir) + load_split(
        "dev", processed_dir=args.processed_dir
    )
    # Only the ID lists are read here (to exclude every PQA-L article), never test content.
    # Build the set of all 1,000 PQA-L IDs by combining the three split ID files.
    pqal_ids = {
        # Each PubMed ID becomes one member of the set.
        pmid
        # Loop over the three split names ...
        for split in ("train", "dev", "test")
        # ... and over every ID in that split's file, e.g. data/splits/dev_ids.txt.
        for pmid in read_ids(SPLITS_DIR / f"{split}_ids.txt")
    }
    # Collect everything we want to report in one dict.
    result = {
        # Which questions were used.
        "questions": "train+dev",
        # Recall@k and MRR of BM25 on the gold corpus.
        "gold_corpus": evaluate_gold(BM25Retriever(args.gold_index), examples),
        # Leakage check: PQA-L articles found in the top 20 of the strict corpus (must be 0).
        # max(KS) = 20 results per question.
        "strict_corpus_pqal_retrieved_top20": leakage_count(
            BM25Retriever(args.strict_index), examples, pqal_ids, max(KS)
        ),
    }
    # Make sure the output folder exists.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save the results as indented JSON with a final newline.
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Show the results on screen.
    print(json.dumps(result, indent=2))
    # Return them so tests or other code can use them.
    return result


# Runs main() only when this file is executed directly (python -m bmqa.retrieval.evaluate),
# not when it is imported by another module or a test.
if __name__ == "__main__":
    main()
