"""Majority-class baseline: always predict the most frequent training label.

Usage:  python -m bmqa.baselines --config configs/majority.yaml --split dev
"""

# Why this file exists: a baseline that ignores the question entirely gives the lowest score
# any real system must beat. If a model cannot beat it, the model has learned nothing useful.

# Makes type hints like `list[str] | None` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --split dev.
import argparse

# Counter: a dict that counts how often each item appears.
from collections import Counter

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# config_hash: short fingerprint of the settings; load_config: reads the YAML config.
from bmqa.config import config_hash, load_config

# PROCESSED_DIR: default folder of split files; load_split: loads one split (with the
# test guard); warn_test_access: prints a loud warning when the test split is used.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split, warn_test_access

# write_predictions: saves Prediction records to a JSON Lines file.
from bmqa.io import write_predictions

# Shared definitions: LABELS = ("yes", "no", "maybe"), SPLITS = ("train", "dev", "test"),
# Example = one question record, Prediction = one system answer record.
from bmqa.schema import LABELS, SPLITS, Example, Prediction


# majority_label: find the most common answer in the training data.
# Input: the training examples. Output: one label string ("yes", "no" or "maybe").
# Only train is used, never dev or test, so the baseline does not peek at evaluation data.
def majority_label(train: list[Example]) -> str:
    """Most frequent gold label; ties go to the earlier label in LABELS."""
    # Count how many training examples have each gold answer.
    counts = Counter(ex.final_decision for ex in train)
    # Pick the label with the highest count. max() returns the first maximum it meets,
    # so a tie is broken by LABELS order, which makes the result deterministic.
    return max(LABELS, key=lambda label: counts[label])


# majority_predictions: give the same label to every example.
# Inputs: the examples to answer, the label to use, plus (keyword-only, after `*`) the system
# name and the config hash to record. Output: one Prediction per example.
def majority_predictions(
    examples: list[Example], label: str, *, system: str, cfg_hash: str
) -> list[Prediction]:
    # Build the list with a list comprehension: one Prediction for each example.
    return [
        # Create one Prediction record (the shared format every system writes).
        Prediction(
            # Which question this answer belongs to.
            id=ex.id,
            # Name of the system that made the prediction (from the config).
            system=system,
            # The answer: always the majority label.
            prediction=label,
            # No text was generated, so the raw model output is empty.
            raw_output="",
            # There is no real model; record "majority" instead.
            model="majority",
            # Fingerprint of the config, so results can be traced to their settings.
            config_hash=cfg_hash,
            # The baseline is deterministic, so a single run (run 1) is enough.
            run=1,
        )
        # Repeat for every example in the split.
        for ex in examples
    ]


# main: the command-line entry point.
# Input: optional list of arguments (tests pass them; normally read from the command line).
# Output: the path of the prediction file that was written.
def main(argv: list[str] | None = None) -> Path:
    # Create the command-line parser with a short description shown by --help.
    parser = argparse.ArgumentParser(description="Majority-class baseline.")
    # --config: which YAML config to use (default: configs/majority.yaml).
    parser.add_argument("--config", type=Path, default=Path("configs/majority.yaml"))
    # --split: which split to predict on; only train/dev/test are accepted. Default is dev.
    parser.add_argument("--split", choices=SPLITS, default="dev")
    # --allow-test: a switch (True if given). Needed to touch the test split, which is
    # reserved for the single frozen Phase 4 run so test scores stay honest.
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")
    # --processed-dir: folder holding train/dev/test .jsonl files.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # --out-dir: folder where prediction files are written.
    parser.add_argument("--out-dir", type=Path, default=Path("results/predictions"))
    # Read the actual arguments (from argv, or from the real command line if argv is None).
    args = parser.parse_args(argv)

    # Load the settings from the YAML config.
    cfg = load_config(args.config)
    # If the user really is running on test (and allowed it), print a warning first.
    if args.split == "test" and args.allow_test:
        # Loud reminder that test access is only for the frozen final run.
        warn_test_access()
    # Learn the majority label from the training split only.
    label = majority_label(load_split("train", processed_dir=args.processed_dir))
    # Load the split we want to predict on (load_split refuses "test" without allow_test).
    examples = load_split(args.split, args.allow_test, args.processed_dir)
    # Make one Prediction per example, all with the majority label.
    preds = majority_predictions(examples, label, system=cfg["system"], cfg_hash=config_hash(cfg))
    # Build the output path, e.g. results/predictions/majority/dev_run1.jsonl.
    out = args.out_dir / cfg["system"] / f"{args.split}_run1.jsonl"
    # Save the predictions to that file.
    write_predictions(out, preds)
    # Tell the user what happened; !r prints the label with quotes, e.g. 'yes'.
    print(f"majority label {label!r}: {len(preds)} predictions written to {out}")
    # Return the path so tests (or other code) can open the file.
    return out


# Run main() only when this file is executed directly (python -m bmqa.baselines),
# not when it is imported by another module or a test.
if __name__ == "__main__":
    main()
