"""Score one prediction file.

Usage:  python -m bmqa.eval --pred results/predictions/majority/dev_run1.jsonl --split dev
Writes results/metrics/{system}_{split}_run{run}.json and prints a summary.
"""

# __main__.py is the file Python runs for `python -m bmqa.eval`.

# Makes type hints like `list[str] | None` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --pred and --split.
import argparse

# json: writes the metrics file.
import json

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# PROCESSED_DIR: default data folder; gold_labels: the correct answers of a split (with the
# test guard); warn_test_access: prints a warning when the test split is used.
from bmqa.data.pubmedqa import PROCESSED_DIR, gold_labels, warn_test_access

# compute_metrics: calculates accuracy, macro-F1, per-label scores and the confusion matrix.
from bmqa.eval.metrics import compute_metrics

# load_predictions: reads a prediction .jsonl file into Prediction records.
from bmqa.io import load_predictions

# LABELS = ("yes", "no", "maybe"); SPLITS = ("train", "dev", "test").
from bmqa.schema import LABELS, SPLITS


# main: score one prediction file against the gold labels.
# Input: optional argument list (used by tests). Output: the metrics as a dict.
def main(argv: list[str] | None = None) -> dict:
    # Create the command-line parser.
    parser = argparse.ArgumentParser(description="Score a prediction file.")
    # --pred: the prediction file to score (must be given).
    parser.add_argument("--pred", type=Path, required=True)
    # --split: which split the predictions are for (must be given; train/dev/test only).
    parser.add_argument("--split", choices=SPLITS, required=True)
    # --allow-test: switch needed to score on test; only for the frozen Phase 4 run.
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")
    # --processed-dir: folder with the processed split files (source of gold labels).
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # --out-dir: folder where the metrics JSON file is written.
    parser.add_argument("--out-dir", type=Path, default=Path("results/metrics"))
    # Read the actual arguments.
    args = parser.parse_args(argv)

    # If test is being scored with permission, print the warning first.
    if args.split == "test" and args.allow_test:
        warn_test_access()
    # Read the predictions (this also checks every row has the right fields).
    predictions = load_predictions(args.pred)
    # Collect the distinct system names in the file (a set removes duplicates).
    systems = {p.system for p in predictions}
    # Collect the distinct run numbers in the file.
    runs = {p.run for p in predictions}
    # A file must hold exactly one system and one run; mixing them would blur the scores.
    if len(systems) != 1 or len(runs) != 1:
        raise ValueError(f"one system and run per file expected, got {systems} / {runs}")
    # Take the single system name and run number out of their sets.
    system, run = systems.pop(), runs.pop()

    # Load {id: correct label} for the split (test stays blocked unless --allow-test).
    gold = gold_labels(args.split, args.allow_test, args.processed_dir)
    # Build the metrics dict: identifying information first, then all the scores.
    metrics = {
        # Which system was scored.
        "system": system,
        # Which split.
        "split": args.split,
        # Which run (repeated runs, e.g. different seeds, get different numbers).
        "run": run,
        # The scored file, written with forward slashes on every operating system.
        "prediction_file": args.pred.as_posix(),
        # `**` copies every key of the compute_metrics result into this dict.
        **compute_metrics(gold, predictions),
    }
    # Make sure the output folder exists.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Output file name, e.g. results/metrics/majority_dev_run1.json.
    out = args.out_dir / f"{system}_{args.split}_run{run}.json"
    # Save the metrics as nicely indented JSON (indent=2) with a final newline.
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Print a readable summary; n is the number of scored examples.
    print(f"{system} {args.split} run {run}: n={metrics['n']}")
    # Accuracy and macro-F1 with 4 decimal places.
    print(f"  accuracy {metrics['accuracy']:.4f}   macro-F1 {metrics['macro_f1']:.4f}")
    # One line per label with its F1 score.
    for label in LABELS:
        # Look up the F1 of this label.
        f1 = metrics["per_label"][label]["f1"]
        # `:<5` pads the label to 5 characters so the numbers line up.
        print(f"  F1 {label:<5} {f1:.4f}")
    # How many answers could not be parsed (they were counted as wrong).
    print(f"  parse failures {metrics['parse_failures']}")
    # Where the metrics file was saved.
    print(f"  written to {out}")
    # Return the metrics so tests or other code can use them.
    return metrics


# Run main() only when this module is executed, not when it is imported.
if __name__ == "__main__":
    main()
