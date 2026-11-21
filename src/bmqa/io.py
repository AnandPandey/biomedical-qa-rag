"""JSON Lines helpers shared by every component."""

# Makes type hints like `str | Path` work as plain text on older Python versions.
from __future__ import annotations

# json: turns Python dicts into JSON text and back again.
import json

# Iterable: type hint meaning "anything you can loop over" (list, generator, ...).
from collections.abc import Iterable

# asdict: converts a dataclass object (like Prediction) into a plain dict.
from dataclasses import asdict

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# Prediction: the shared record for one system's answer to one question (see schema.py).
from bmqa.schema import Prediction


# read_jsonl: read a JSON Lines file (one JSON object per line).
# Input: path to the file. Output: a list of dicts, one per non-blank line.
def read_jsonl(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSON Lines file into a list of dicts (blank lines are skipped)."""
    # Open the file as text in UTF-8 so non-English characters are read correctly.
    # `with` closes the file automatically when the block ends.
    with open(path, encoding="utf-8") as f:
        # Parse each line into a dict; `if line.strip()` skips empty or whitespace-only lines.
        return [json.loads(line) for line in f if line.strip()]


# write_jsonl: write many dicts to a JSON Lines file, one dict per line.
# Inputs: destination path and the rows to write. Output: nothing (writes a file).
def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dicts as UTF-8 JSON Lines, creating parent folders as needed."""
    # Accept either a string or a Path, and always work with a Path from here on.
    path = Path(path)
    # Create the folder that will hold the file (and any missing parents);
    # exist_ok=True means "no error if the folder already exists".
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open for writing ("w" replaces old content). newline="\n" forces Unix line endings,
    # so files are byte-identical on Windows, Mac and Linux.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        # Go through the rows one at a time.
        for row in rows:
            # Turn the dict into one line of JSON and end it with a newline.
            # ensure_ascii=False keeps characters like "e-acute" readable instead of \u codes.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# write_predictions: save Prediction objects to a JSON Lines file.
# Inputs: destination path and the predictions. Output: nothing (writes a file).
def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    # Convert each Prediction dataclass to a dict, then reuse write_jsonl to save them.
    write_jsonl(path, (asdict(p) for p in predictions))


# load_predictions: read a prediction file back into Prediction objects.
# Input: path to the file. Output: a list of Prediction records.
def load_predictions(path: str | Path) -> list[Prediction]:
    """Load a prediction file; raises ValueError/TypeError if any row breaks the contract."""
    # Read every row as a dict, then build a Prediction from it. Prediction.from_dict
    # checks the fields, so a badly formed file fails loudly instead of being scored wrongly.
    return [Prediction.from_dict(row) for row in read_jsonl(path)]
