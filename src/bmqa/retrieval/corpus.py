"""Build the strict and gold retrieval corpora from PubMedQA PQA-A and PQA-U.

strict = PQA-A + PQA-U minus every PQA-L article (no question can retrieve its own abstract)
gold   = strict + the 1,000 PQA-L articles

Usage:  python -m bmqa.retrieval.corpus   (needs the `corpus` extra and data/raw/ori_pqal.json)
"""

# Lets us write type hints such as "list[str] | None" on every supported Python version.
from __future__ import annotations

# argparse reads command-line options such as --out-dir.
import argparse

# json turns Python dicts into text (and back) so we can save the statistics file.
import json

# Iterable is a type hint meaning "anything you can loop over" (a list, a dataset, ...).
from collections.abc import Iterable

# asdict converts a dataclass object (CorpusDoc) into a plain dict that json can write.
from dataclasses import asdict

# Path is an object-oriented way to work with file and folder paths.
from pathlib import Path

# RAW_DIR = "data/raw" (where ori_pqal.json lives); sha256_file computes a file fingerprint.
from bmqa.data.download import RAW_DIR, sha256_file

# load_pqal reads ori_pqal.json: the 1,000 labelled (PQA-L) questions keyed by PubMed ID.
from bmqa.data.pubmedqa import load_pqal

# write_jsonl writes one JSON object per line (the "JSON Lines" format).
from bmqa.io import write_jsonl

# CorpusDoc is the shared record type for one abstract: doc_id, title, text, source.
from bmqa.schema import CorpusDoc

# Name of the dataset on the Hugging Face Hub that holds PQA-A and PQA-U.
HF_DATASET = "qiaojin/PubMedQA"
# Exact commit of that dataset. Pinning it means everyone downloads identical data, even if
# the dataset owner later edits it, so our corpus files (and their SHA-256) are reproducible.
HF_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
# Maps our short source names to the Hugging Face config (subset) names.
# pqa_artificial = ~211k auto-generated questions; pqa_unlabeled = ~61k unlabelled questions.
# We only use their abstracts as documents to search, never their questions or answers.
HF_CONFIGS = {"pqa_a": "pqa_artificial", "pqa_u": "pqa_unlabeled"}
# Default folder where strict.jsonl and gold.jsonl are written.
CORPUS_DIR = Path("data/corpus")


# Turns one Hugging Face row into a CorpusDoc.
# Inputs: row = one dataset record (a dict); source = "pqa_a" or "pqa_u".
# Output: a CorpusDoc whose text is the abstract body without its conclusion.
def doc_from_hf_row(row: dict, source: str) -> CorpusDoc:
    """Abstract text without its conclusion; the question and long_answer are never used."""
    # Build and return the document record.
    return CorpusDoc(
        # The PubMed ID is an int in the dataset; we store it as a string to match PQA-L ids.
        doc_id=str(row["pubid"]),
        # PubMedQA has no titles, so the title is always empty.
        title="",
        # "contexts" is a list of abstract sections (background, methods, results).
        # The conclusion is stored separately as long_answer, so leaving it out means the
        # answer is not handed to the model. We join the sections with spaces into one string.
        text=" ".join(row["context"]["contexts"]),
        # Remember which subset the document came from (useful for statistics and debugging).
        source=source,
    )


# Builds both corpora from the raw rows.
# Inputs: pqa_a_rows and pqa_u_rows = Hugging Face rows; pqal = the 1,000 PQA-L entries.
# Output: (strict docs, gold docs, a dict of counts describing what was kept/removed).
def build_corpora(
    pqa_a_rows: Iterable[dict], pqa_u_rows: Iterable[dict], pqal: dict[str, dict]
) -> tuple[list[CorpusDoc], list[CorpusDoc], dict]:
    """Return (strict, gold, stats); both corpora sorted by doc_id."""
    # The set of all PQA-L PubMed IDs (the keys of the dict). A set makes "in" checks fast.
    pqal_ids = set(pqal)
    # IDs already added to the strict corpus, used to drop duplicate abstracts.
    seen: set[str] = set()
    # The strict corpus being built (a list of CorpusDoc).
    strict: list[CorpusDoc] = []
    # Counters that are saved to results/corpus_stats.json for the report.
    stats: dict = {}
    # Process PQA-A first, then PQA-U. Order matters: if an article appears in both, the
    # PQA-A copy is the one kept and the PQA-U copy is counted as a duplicate.
    for source, rows in (("pqa_a", pqa_a_rows), ("pqa_u", pqa_u_rows)):
        # Reset the three counters for this source (all start at 0).
        read = removed = duplicates = 0
        # Look at every row of this source.
        for row in rows:
            # Count every row we read.
            read += 1
            # Convert the row into a CorpusDoc.
            doc = doc_from_hf_row(row, source)
            # If this abstract is one of the 1,000 labelled PQA-L articles, drop it.
            # We drop ALL PQA-L articles (train, dev and test), not only test ones, so that no
            # question can ever retrieve its own abstract: that would leak the answer source.
            if doc.doc_id in pqal_ids:
                removed += 1
            # If we already kept this ID earlier, skip it so each article appears only once.
            elif doc.doc_id in seen:
                duplicates += 1
            # Otherwise it is a new, allowed article: remember its ID and keep it.
            else:
                seen.add(doc.doc_id)
                strict.append(doc)
        # Save the counts for this source.
        stats[source] = {
            # How many rows the source contained.
            "rows_read": read,
            # How many were removed because they are PQA-L articles.
            "pqal_removed": removed,
            # How many were removed because the same article was already in the corpus.
            "duplicates_removed": duplicates,
        }
    # The gold corpus = everything in strict PLUS the 1,000 PQA-L articles.
    # Here each question's own abstract IS searchable, which gives an upper bound on how
    # much retrieval can help (and lets us measure whether BM25 finds the right article).
    gold = strict + [
        # PQA-L entries use upper-case keys; CONTEXTS again excludes the conclusion.
        CorpusDoc(doc_id=pmid, title="", text=" ".join(entry["CONTEXTS"]), source="pqa_l")
        # One document per PQA-L entry (pmid = the PubMed ID, entry = its data).
        for pmid, entry in pqal.items()
    ]
    # Sort both corpora by doc_id. The dataset order could change between library versions;
    # a fixed order means the written file is byte-for-byte identical every time, so its
    # SHA-256 fingerprint is reproducible and teammates can check they have the same file.
    strict.sort(key=lambda d: d.doc_id)
    gold.sort(key=lambda d: d.doc_id)
    # Record the final sizes of both corpora.
    stats["strict_size"] = len(strict)
    stats["gold_size"] = len(gold)
    # Sanity checks: count PQA-L articles in each corpus (True counts as 1 in sum()).
    # Expected: 0 in strict and 1,000 in gold.
    stats["pqal_in_strict"] = sum(d.doc_id in pqal_ids for d in strict)
    stats["pqal_in_gold"] = sum(d.doc_id in pqal_ids for d in gold)
    # Give back both corpora and the statistics.
    return strict, gold, stats


# Command-line entry point: downloads the data, builds the corpora and writes them to disk.
# Input: argv = list of command-line arguments (None means "use the real command line").
# Output: the statistics dict (also written to a JSON file and printed).
def main(argv: list[str] | None = None) -> dict:
    # Imported inside the function because the `datasets` library is an optional extra;
    # this way the rest of the module (and its tests) works without it installed.
    from datasets import load_dataset  # needs the `corpus` extra

    # Create the command-line parser; the description appears in --help.
    parser = argparse.ArgumentParser(description="Build the strict and gold retrieval corpora.")
    # --raw-dir: folder containing ori_pqal.json (default data/raw).
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    # --out-dir: folder where strict.jsonl and gold.jsonl are written (default data/corpus).
    parser.add_argument("--out-dir", type=Path, default=CORPUS_DIR)
    # --stats: where to save the statistics JSON file.
    parser.add_argument("--stats", type=Path, default=Path("results/corpus_stats.json"))
    # Read the options the user typed (or the argv list passed in by a test).
    args = parser.parse_args(argv)

    # Download (or load from cache) both Hugging Face subsets at the pinned revision.
    # The result is a dict: {"pqa_a": dataset, "pqa_u": dataset}.
    rows = {
        # Each subset only has a "train" split on the Hub, which holds all its rows.
        source: load_dataset(HF_DATASET, config, split="train", revision=HF_REVISION)
        # Loop over the two (short name, Hugging Face config name) pairs.
        for source, config in HF_CONFIGS.items()
    }
    # Load the 1,000 labelled PQA-L articles that must be removed from strict.
    pqal = load_pqal(args.raw_dir / "ori_pqal.json")
    # Build the two corpora and the statistics.
    strict, gold, stats = build_corpora(rows["pqa_a"], rows["pqa_u"], pqal)

    # Write each corpus to its own file.
    for name, docs in (("strict", strict), ("gold", gold)):
        # e.g. data/corpus/strict.jsonl
        path = args.out_dir / f"{name}.jsonl"
        # Convert each CorpusDoc to a dict and write one JSON line per document.
        write_jsonl(path, (asdict(d) for d in docs))
        # Store the file fingerprint so others can verify they built the exact same corpus.
        stats[f"{name}_sha256"] = sha256_file(path)
    # Also record exactly which dataset and revision were used.
    stats["hf_dataset"] = HF_DATASET
    stats["hf_revision"] = HF_REVISION
    # Make sure the folder for the stats file exists (no error if it already does).
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    # Save the statistics as nicely indented JSON with a final newline.
    args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    # Show the statistics on screen too.
    print(json.dumps(stats, indent=2))
    # Return them so tests or other code can use them.
    return stats


# Runs main() only when this file is executed directly (python -m bmqa.retrieval.corpus),
# not when it is imported by another module or a test.
if __name__ == "__main__":
    main()
