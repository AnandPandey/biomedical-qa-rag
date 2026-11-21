"""Create the fixed train/dev/test splits and the processed example files.

The split ID lists in data/splits/ are committed and never regenerated: if they already exist,
this command checks that a fresh computation gives exactly the same lists, then (re)writes
data/processed/ from them.

Usage:  python -m bmqa.data.splits [--raw-dir data/raw]
"""

# Makes type hints like `list[str] | None` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --raw-dir.
import argparse

# json: reads the official test list (test_ground_truth.json).
import json

# random: used to shuffle IDs before cutting off the dev set.
import random

# Counter: counts how many examples of each label a split has (for the summary).
from collections import Counter

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# RAW_DIR: default folder of the downloaded raw files.
from bmqa.data.download import RAW_DIR

# PROCESSED_DIR: default output folder; load_pqal: reads ori_pqal.json;
# write_processed: writes one .jsonl file of Examples per split.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_pqal, write_processed

# LABELS = ("yes", "no", "maybe").
from bmqa.schema import LABELS

# Folder where the committed ID lists (train_ids.txt, dev_ids.txt, test_ids.txt) live.
SPLITS_DIR = Path("data/splits")
# Fixed random seed: the shuffle is the same on every computer, so the split is reproducible.
SEED = 42
# Share of the non-test questions that go to dev (20%); the other 80% are for training.
DEV_FRACTION = 0.2


# make_splits: decide which question IDs go to train, dev and test.
# Inputs: {PMID: label} for all questions, the official test IDs, the seed and dev share.
# Output: {"train": [...], "dev": [...], "test": [...]} with sorted ID lists.
def make_splits(
    labels: dict[str, str],
    test_ids: set[str],
    seed: int = SEED,
    dev_fraction: float = DEV_FRACTION,
) -> dict[str, list[str]]:
    """Official test IDs as given; the rest split per label into train/dev (stratified)."""
    # Find test IDs that do not exist in the labelled data (set difference).
    unknown = test_ids - labels.keys()
    # Any such ID means the two input files do not belong together: stop.
    if unknown:
        raise ValueError(f"{len(unknown)} test IDs are not in the labelled data")
    # A private random generator with a fixed seed, so the shuffle is always identical.
    rng = random.Random(seed)
    # Empty lists to fill with training and development IDs.
    train: list[str] = []
    dev: list[str] = []
    # Split each label separately ("stratified"), so train and dev keep the same
    # yes/no/maybe proportions as the full data. "maybe" is rare, so this matters.
    for label in LABELS:
        # All non-test IDs with this label, sorted first so the input to the shuffle
        # does not depend on the order of the JSON file.
        ids = sorted(pmid for pmid, lab in labels.items() if lab == label and pmid not in test_ids)
        # Shuffle in place with the seeded generator.
        rng.shuffle(ids)
        # How many of this label go to dev (20%, rounded to a whole number).
        n_dev = round(dev_fraction * len(ids))
        # The first n_dev shuffled IDs go to dev ...
        dev.extend(ids[:n_dev])
        # ... and all remaining IDs go to train.
        train.extend(ids[n_dev:])
    # Return each split as a sorted list; test is exactly the official PubMedQA test set.
    return {"train": sorted(train), "dev": sorted(dev), "test": sorted(test_ids)}


# read_ids: read an ID list file (one ID per line).
# Input: path to the .txt file. Output: list of ID strings (blank lines skipped).
def read_ids(path: Path) -> list[str]:
    # Read the whole file, split it into lines, strip spaces, and drop empty lines.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# write_ids: save an ID list, one ID per line.
# Inputs: destination path and the IDs. Output: nothing (writes a file).
def write_ids(path: Path, ids: list[str]) -> None:
    # Create the folder if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open for writing with Unix line endings, so the file is identical on every OS.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        # Join the IDs with newlines and finish with a final newline.
        f.write("\n".join(ids) + "\n")


# main: the command-line entry point.
# Input: optional argument list (used by tests). Output: nothing; writes files and prints counts.
def main(argv: list[str] | None = None) -> None:
    # Create the parser; the description is the first line of this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --raw-dir: folder with the downloaded ori_pqal.json and test_ground_truth.json.
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    # --splits-dir: folder for the ID lists.
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    # --processed-dir: folder for the processed train/dev/test .jsonl files.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # Read the actual arguments.
    args = parser.parse_args(argv)

    # Load all 1,000 labelled questions: {PMID: raw entry}.
    pqal = load_pqal(args.raw_dir / "ori_pqal.json")
    # Open the official test list.
    with open(args.raw_dir / "test_ground_truth.json", encoding="utf-8") as f:
        # test_truth is {PMID: gold label} for the official test questions.
        test_truth = json.load(f)
    # Make {PMID: label} for every labelled question.
    labels = {pmid: entry["final_decision"] for pmid, entry in pqal.items()}
    # Sanity check: list test IDs whose label differs from (or is missing in) ori_pqal.json.
    mismatched = [pmid for pmid, lab in test_truth.items() if labels.get(pmid) != lab]
    # If the two files disagree, something is wrong with the data: stop.
    if mismatched:
        raise ValueError(f"{len(mismatched)} test labels disagree with ori_pqal.json")

    # Compute the splits; set(test_truth) is the set of test PMIDs (the dict keys).
    splits = make_splits(labels, set(test_truth))
    # Check or write the ID list of each split.
    for split, ids in splits.items():
        # e.g. data/splits/dev_ids.txt
        path = args.splits_dir / f"{split}_ids.txt"
        # If the list already exists, it is frozen: never overwrite it.
        if path.exists():
            # A fresh computation must give exactly the same IDs; otherwise results from
            # different team members would not be comparable, so stop loudly.
            if read_ids(path) != ids:
                raise RuntimeError(f"{path} differs from a fresh computation; splits are frozen")
        # First time only: no list yet, so write it.
        else:
            write_ids(path, ids)
    # (Re)write the processed Example files for all three splits.
    write_processed(pqal, splits, args.processed_dir)

    # Print a summary of each split.
    for split, ids in splits.items():
        # Count the labels in this split.
        counts = Counter(labels[pmid] for pmid in ids)
        # Build text of the form "yes <count>, no <count>, maybe <count>".
        detail = ", ".join(f"{lab} {counts[lab]}" for lab in LABELS)
        # Print the split name, its size and the label counts, e.g. "dev: <n> (yes ...)".
        print(f"{split}: {len(ids)} ({detail})")


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
