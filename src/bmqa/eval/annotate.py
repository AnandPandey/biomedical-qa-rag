"""Human validation of the NLI claim checker on DEV predictions.

    python -m bmqa.eval.annotate sample --n 160 --seed 0 \
        --strict-corpus data/corpus/strict.jsonl --gold-corpus data/corpus/gold.jsonl
    # two annotators fill the `label` column of copies of sample.csv independently
    python -m bmqa.eval.annotate agreement --a results/annotation/annotator_a.csv \
        --b results/annotation/annotator_b.csv --adjudicated results/annotation/adjudicated.csv

Labels: supported, contradicted, not_supported (see docs/annotation-guideline.md).
"""

# Why this file exists: the NLI model in claims.py is an automatic judge, and automatic judges
# make mistakes. Before we trust its hallucination numbers, we check it against humans:
# 1. "sample": draw a random set of claims from DEV predictions (never test) for people to label.
#    The sample is BLIND: the file shows neither the NLI verdict nor which label is expected,
#    and claims from all systems are shuffled together, so annotators cannot be biased.
# 2. Two annotators label the sample independently; disagreements are settled together
#    ("adjudicated") into one final human label per claim.
# 3. "agreement": measure how well the two humans agree with each other (Cohen's kappa) and
#    how well the NLI model agrees with the final human labels. If humans barely agree, the
#    task is unclear; if NLI disagrees with humans, its numbers should not be trusted.

# Lets modern type hints be used without being evaluated at runtime.
from __future__ import annotations

# argparse: command-line options with two sub-commands ("sample" and "agreement").
import argparse

# csv: read and write spreadsheet-friendly .csv files for the annotators.
import csv

# json: store the list of premises inside one CSV cell and write the result file.
import json

# random: seeded random sampling and shuffling (repeatable).
import random

# Path: file paths.
from pathlib import Path

# load_config: reads the NLI YAML config.
from bmqa.config import load_config

# PROCESSED_DIR: default data folder; load_split: loads the dev examples.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split

# Reuse the claim tools from claims.py so humans and NLI judge exactly the same claims.
from bmqa.eval.claims import CLAIM_LABELS, NLIScorer, decide, is_parse_failure, split_claims

# cohen_kappa: agreement corrected for chance (see stats.py).
from bmqa.eval.stats import cohen_kappa

# load_predictions: reads a predictions .jsonl file.
from bmqa.io import load_predictions

# Default inputs: run 1 DEV predictions of the four LLM systems.
DEFAULT_FILES = [
    # Build the path results/predictions/<system>/dev_run1.jsonl ...
    Path(f"results/predictions/{s}/dev_run1.jsonl")
    # ... for each of these four systems.
    for s in ("closed_book", "oracle", "rag_strict", "rag_gold")
]
# Column names of the annotation CSV; "label" is left empty for the annotators to fill in.
FIELDS = ["item_id", "system", "example_id", "claim", "premise", "premises_json", "label"]


# Turn predictions into annotation items (one per claim).
# Inputs: predictions; premises_for(pred) -> the source texts to judge the claim against;
# min_words -> claim filter. Output: list of row dicts matching FIELDS.
def candidate_items(predictions, premises_for, min_words: int = 4) -> list[dict]:
    # Collected items.
    items = []
    # Look at every prediction.
    for pred in predictions:
        # Skip unparsable replies; they have no rationale.
        if is_parse_failure(pred):
            # Next prediction.
            continue
        # Source texts for this prediction.
        premises = premises_for(pred)
        # One item per claim, numbered from 0.
        for index, claim in enumerate(split_claims(pred.rationale, min_words)):
            # Add one CSV row.
            items.append(
                {
                    # Unique id "system:question:claim number", e.g. "oracle:123:0".
                    "item_id": f"{pred.system}:{pred.id}:{index}",
                    # System that wrote the claim.
                    "system": pred.system,
                    # Question id.
                    "example_id": pred.id,
                    # The sentence to judge.
                    "claim": claim,
                    # Human-readable premises: "[1] text", blank line, "[2] text", ...
                    "premise": "\n\n".join(f"[{i}] {t}" for i, t in enumerate(premises, 1)),
                    # Same premises as a JSON list, so the NLI model can re-read them exactly.
                    "premises_json": json.dumps(premises, ensure_ascii=False),
                    # Empty: to be filled in by the human annotator.
                    "label": "",
                }
            )
    # All items.
    return items


# Stratified sample: the same number of claims from each system, so a chatty system does not
# dominate the sample. Inputs: items grouped by system, total size n, seed. Output: shuffled
# list of items (no labels shown, so the annotation is blind).
def stratified_sample(items_by_system: dict[str, list[dict]], n: int, seed: int) -> list[dict]:
    """Equal numbers of claims per system (as far as available), shuffled, blind to labels."""
    # Seeded generator: the same seed always gives the same sample.
    rng = random.Random(seed)
    # Claims per system, e.g. 160 // 4 = 40 (integer division).
    per_system = n // len(items_by_system)
    # The final sample.
    sample = []
    # Visit systems in alphabetical order so the result does not depend on dict order.
    for system in sorted(items_by_system):
        # Sort items by id first, again so the random draw is repeatable.
        items = sorted(items_by_system[system], key=lambda it: it["item_id"])
        # Draw per_system items without repeats (or all of them if there are fewer).
        sample.extend(rng.sample(items, min(per_system, len(items))))
    # Mix the systems together so annotators cannot guess the system from the row order.
    rng.shuffle(sample)
    # Shuffled sample.
    return sample


# Read one filled-in annotation CSV. Input: file path. Output: dict item_id -> label.
def read_labels(path: Path) -> dict[str, str]:
    # Open the file as UTF-8 text; newline="" is what the csv module expects.
    with open(path, encoding="utf-8", newline="") as f:
        # One entry per row; labels are cleaned (spaces stripped, lower-cased).
        rows = {row["item_id"]: row["label"].strip().lower() for row in csv.DictReader(f)}
    # Any label that is not one of the three allowed ones (including empty cells).
    bad = {v for v in rows.values() if v not in CLAIM_LABELS}
    # Refuse incomplete or misspelled annotation files.
    if bad:
        # Name the file and the bad labels so they can be fixed.
        raise ValueError(f"{path}: unknown or empty labels {sorted(bad)}")
    # Clean labels.
    return rows


# Compare two label lists. Inputs: reference labels (treated as correct) and predicted labels.
# Output: kappa, accuracy, and per-label precision/recall/support.
def label_report(reference: list[str], predicted: list[str]) -> dict:
    # Start with chance-corrected agreement.
    out = {"kappa": cohen_kappa(reference, predicted)}
    # Accuracy: share of items where both lists give the same label.
    out["accuracy"] = sum(r == p for r, p in zip(reference, predicted, strict=True)) / len(
        reference
    )
    # Per-label scores.
    for label in CLAIM_LABELS:
        # True positives: items where both reference and predicted equal this label.
        tp = sum(r == p == label for r, p in zip(reference, predicted, strict=True))
        # How often this label was predicted, and how often it is in the reference.
        n_pred, n_ref = predicted.count(label), reference.count(label)
        # Store the three numbers for this label.
        out[label] = {
            # Precision: of the items predicted as this label, the share that are right.
            "precision": tp / n_pred if n_pred else 0.0,
            # Recall: of the items that truly have this label, the share that were found.
            "recall": tp / n_ref if n_ref else 0.0,
            # Support: how many reference items have this label.
            "support": n_ref,
        }
    # Full report.
    return out


# Sub-command "sample": build the blind annotation CSV. Input: parsed arguments.
# Output: path of the written CSV.
def cmd_sample(args) -> Path:
    # DEV examples by id (dev only: test stays untouched until the final run).
    examples = {ex.id: ex for ex in load_split("dev", processed_dir=args.processed_dir)}
    # Document stores for the two RAG systems, keyed by system name.
    stores = {}
    # Imported here so the module loads even without the retrieval code.
    from bmqa.retrieval.store import DocStore

    # Corpus for the "strict" retrieval system, if given.
    if args.strict_corpus:
        # Load it.
        stores["rag_strict"] = DocStore(args.strict_corpus)
    # Corpus for the "gold" retrieval system, if given.
    if args.gold_corpus:
        # Load it.
        stores["rag_gold"] = DocStore(args.gold_corpus)

    # Items grouped by system.
    items_by_system = {}
    # One prediction file per system.
    for path in args.pred:
        # Load the predictions.
        predictions = load_predictions(path)
        # The system name, taken from the first row.
        system = predictions[0].system

        # Premises for one prediction. "system=system" freezes the current value; without it
        # every function made in this loop would see the LAST system name.
        def premises_for(pred, system=system):
            # Closed-book saw no documents.
            if system == "closed_book":
                return list(examples[pred.id].contexts)  # judged against the source study
            # Oracle saw the gold abstract as one joined text.
            if system == "oracle":
                # One premise: all abstract sections joined with spaces.
                return [" ".join(examples[pred.id].contexts)]
            # RAG systems: use the matching document store.
            store = stores[system]
            # Texts of the documents the model was shown.
            return [store[hit.doc_id] for hit in pred.evidence]

        # Build the items for this system.
        items_by_system[system] = candidate_items(predictions, premises_for)
    # Draw the balanced, shuffled sample.
    sample = stratified_sample(items_by_system, args.n, args.seed)
    # Create the output folder if needed.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Open the CSV for writing.
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        # Writer that turns dicts into rows using the FIELDS column order.
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        # Header row.
        writer.writeheader()
        # One row per sampled claim.
        writer.writerows(sample)
    # Tell the user how many claims were written.
    print(f"{len(sample)} claims written to {args.out}")
    # Return the path.
    return args.out


# Sub-command "agreement": compare annotators with each other and NLI with humans.
# Input: parsed arguments. Output: result dict (also written as JSON).
def cmd_agreement(args) -> dict:
    # Labels of annotator A, annotator B, and the agreed (adjudicated) final labels.
    a, b, adjudicated = read_labels(args.a), read_labels(args.b), read_labels(args.adjudicated)
    # Fixed item order (sorted ids) so all lists line up.
    ids = sorted(adjudicated)
    # All three files must cover exactly the same claims.
    if set(a) != set(ids) or set(b) != set(ids):
        # Otherwise the comparison would be meaningless.
        raise ValueError("the three annotation files must contain the same items")
    # Re-read the original sample to get each claim and its premises.
    with open(args.sample, encoding="utf-8", newline="") as f:
        # item_id -> full CSV row.
        items = {row["item_id"]: row for row in csv.DictReader(f)}
    # NLI settings.
    cfg = load_config(args.config)
    # Load the NLI model.
    scorer = NLIScorer(cfg["model"], cfg["max_length"], cfg["batch_size"])
    # NLI labels in the same order as ids.
    nli = []
    # Judge each sampled claim with NLI, exactly as claims.py would.
    for item_id in ids:
        # Premises were saved as a JSON list in the sample.
        premises = json.loads(items[item_id]["premises_json"])
        # Score (premise, claim) pairs and combine them into one label.
        nli.append(decide(scorer([(p, items[item_id]["claim"]) for p in premises])))
    # Final human labels in the same order.
    human = [adjudicated[i] for i in ids]
    # Collect the results.
    result = {
        # Number of annotated claims.
        "items": len(ids),
        # Human vs human: how clear is the task?
        "inter_annotator": label_report([a[i] for i in ids], [b[i] for i in ids]),
        # NLI vs adjudicated human labels: how trustworthy is the automatic checker?
        "nli_vs_adjudicated": label_report(human, nli),
        # Which NLI model was tested.
        "nli_model": cfg["model"],
    }
    # Create the output folder if needed.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save as pretty-printed JSON.
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Show the result on screen.
    print(json.dumps(result, indent=2))
    # Return it.
    return result


# Command-line entry point with two sub-commands.
def main(argv: list[str] | None = None):
    # Top-level parser.
    parser = argparse.ArgumentParser(description="Human validation of the claim checker (dev).")
    # Sub-commands; the chosen name is stored in args.command and one must be given.
    sub = parser.add_subparsers(dest="command", required=True)

    # Options of "sample".
    s = sub.add_parser("sample", help="draw a blind annotation sample from dev predictions")
    # One or more DEV prediction files (default: the four LLM systems).
    s.add_argument("--pred", type=Path, nargs="+", default=DEFAULT_FILES)
    # Total number of claims to sample.
    s.add_argument("--n", type=int, default=160)
    # Random seed for a repeatable sample.
    s.add_argument("--seed", type=int, default=0)
    # Corpus with texts of the strict-retrieval documents.
    s.add_argument("--strict-corpus", type=Path)
    # Corpus with texts of the gold-retrieval documents.
    s.add_argument("--gold-corpus", type=Path)
    # Folder with processed data.
    s.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # Where to write the sample CSV.
    s.add_argument("--out", type=Path, default=Path("results/annotation/sample.csv"))

    # Options of "agreement".
    g = sub.add_parser("agreement", help="inter-annotator and NLI-vs-human agreement")
    # The sample CSV produced by "sample".
    g.add_argument("--sample", type=Path, default=Path("results/annotation/sample.csv"))
    # Annotator A's labelled copy.
    g.add_argument("--a", type=Path, default=Path("results/annotation/annotator_a.csv"))
    # Annotator B's labelled copy.
    g.add_argument("--b", type=Path, default=Path("results/annotation/annotator_b.csv"))
    # The agreed final labels.
    g.add_argument("--adjudicated", type=Path, default=Path("results/annotation/adjudicated.csv"))
    # NLI model config.
    g.add_argument("--config", type=Path, default=Path("configs/claims_nli.yaml"))
    # Where to write the agreement JSON.
    g.add_argument("--out", type=Path, default=Path("results/annotation/agreement.json"))

    # Parse the command line.
    args = parser.parse_args(argv)
    # Run the chosen sub-command and return its result.
    return cmd_sample(args) if args.command == "sample" else cmd_agreement(args)


# Run main() only when executed as a script.
if __name__ == "__main__":
    # Start the program.
    main()
