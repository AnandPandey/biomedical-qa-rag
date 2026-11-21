"""Convert PQA-L to `Example` records and load them by split, with a guard on the test split."""

# Makes type hints like `str | Path` work as plain text on older Python versions.
from __future__ import annotations

# json: reads the raw ori_pqal.json file.
import json

# asdict: converts an Example dataclass into a plain dict so it can be saved as JSON.
from dataclasses import asdict

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# read_jsonl / write_jsonl: our helpers for JSON Lines files (one JSON object per line).
from bmqa.io import read_jsonl, write_jsonl

# SPLITS = ("train", "dev", "test"); Example = the shared record for one question.
from bmqa.schema import SPLITS, Example

# Default folder for the processed split files train.jsonl, dev.jsonl and test.jsonl.
PROCESSED_DIR = Path("data/processed")


# load_pqal: read the raw PubMedQA labelled file.
# Input: path to ori_pqal.json. Output: dict {PMID: raw entry dict}.
def load_pqal(path: str | Path) -> dict[str, dict]:
    """Read ori_pqal.json: a mapping from PMID to its raw PubMedQA entry."""
    # Open the file as UTF-8 text; `with` closes it automatically.
    with open(path, encoding="utf-8") as f:
        # Parse the whole JSON file into a Python dict and return it.
        return json.load(f)


# to_example: convert one raw PubMedQA entry into our shared Example record.
# Inputs: the PubMed ID and the raw entry. Output: an Example.
def to_example(pmid: str, entry: dict) -> Example:
    # Build the Example, renaming the raw upper-case keys to our field names.
    return Example(
        # The PubMed ID, forced to a string so IDs always compare the same way.
        id=str(pmid),
        # The research question, e.g. "Do X cause Y?".
        question=entry["QUESTION"],
        # The abstract text split into sections (without its conclusion); copied to a list.
        contexts=list(entry["CONTEXTS"]),
        # The abstract's conclusion. It states the answer, so it is stored for reference
        # only and must NEVER be given to a model as input (that would be cheating).
        long_answer=entry["LONG_ANSWER"],
        # The gold answer: "yes", "no" or "maybe" (Example checks this when created).
        final_decision=entry["final_decision"],
    )


# write_processed: save one .jsonl file per split.
# Inputs: the raw data, the ID list of each split, and the output folder.
# Output: nothing (writes data/processed/train.jsonl, dev.jsonl, test.jsonl).
def write_processed(
    pqal: dict[str, dict], split_ids: dict[str, list[str]], processed_dir: Path = PROCESSED_DIR
) -> None:
    """Write data/processed/{split}.jsonl in the order of the split ID lists."""
    # Go through each split name and its list of PMIDs.
    for split, ids in split_ids.items():
        # For each ID: look up the raw entry, convert to Example, then to a dict.
        # The round brackets make a generator, so rows are produced one at a time.
        rows = (asdict(to_example(pmid, pqal[pmid])) for pmid in ids)
        # Write those rows to e.g. data/processed/train.jsonl.
        write_jsonl(processed_dir / f"{split}.jsonl", rows)


# load_split: load the examples of one split, with a safety guard on "test".
# Inputs: split name, allow_test flag, data folder. Output: list of Example records.
# Why the guard: the test split must be used exactly once, in the frozen final run. If we
# looked at test scores while developing, we would tune towards them and over-report results.
def load_split(
    split: str, allow_test: bool = False, processed_dir: Path = PROCESSED_DIR
) -> list[Example]:
    """Return the examples of one split.

    Raises PermissionError for "test" unless allow_test=True (see docs/experiment-protocol.md),
    and ValueError for an unknown split name.
    """
    # Reject typos such as "valid" or "Train" early with a clear message.
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    # The test guard: asking for "test" without explicitly allowing it is an error.
    if split == "test" and not allow_test:
        # PermissionError makes the mistake impossible to miss.
        raise PermissionError(
            "The test split is protected. Only the frozen Phase 4 test run may load it "
            "(pass allow_test=True / --allow-test)."
        )
    # Path of the processed file, e.g. data/processed/dev.jsonl.
    path = Path(processed_dir) / f"{split}.jsonl"
    # If the file is missing, the splits have not been created yet.
    if not path.exists():
        # Tell the user exactly which command creates it.
        raise FileNotFoundError(f"{path} not found; run python -m bmqa.data.splits first")
    # Read every line and turn each dict back into an Example (`**row` passes the keys
    # as keyword arguments, e.g. id=..., question=...).
    return [Example(**row) for row in read_jsonl(path)]


# gold_labels: the correct answer for each example of a split.
# Inputs: same as load_split. Output: dict {example id: gold label}.
def gold_labels(
    split: str, allow_test: bool = False, processed_dir: Path = PROCESSED_DIR
) -> dict[str, str]:
    """Map example id to gold label for one split (same guard as load_split)."""
    # Load the split (so the test guard applies here too) and keep only id -> label.
    return {ex.id: ex.final_decision for ex in load_split(split, allow_test, processed_dir)}


# warn_test_access: print a loud warning whenever the protected test split is used,
# so nobody uses it by accident and the logs show when it happened.
def warn_test_access() -> None:
    # First line of the warning.
    print("WARNING: loading the protected TEST split. This must only happen in the frozen")
    # Second line, pointing to the document with the rules.
    print("Phase 4 test run (docs/experiment-protocol.md).")
