"""In-memory map from doc_id to document text, loaded from a corpus JSONL file."""

# Lets us write type hints such as "str | Path" on every supported Python version.
from __future__ import annotations

# json parses each line of the corpus file.
import json

# Path is an object-oriented way to work with file and folder paths.
from pathlib import Path


# The retriever only returns document IDs; DocStore looks up the full text for an ID so it
# can be shown to the LLM. It behaves like a read-only dict: store[doc_id], "x" in store,
# len(store).
class DocStore:
    # Loads every document of a corpus file into memory.
    # Input: corpus_path = path to strict.jsonl or gold.jsonl.
    def __init__(self, corpus_path: str | Path):
        # The dictionary doc_id -> text (the underscore marks it as internal).
        self._texts: dict[str, str] = {}
        # Open the corpus file as UTF-8 text.
        with open(corpus_path, encoding="utf-8") as f:
            # Read one line (one document) at a time instead of the whole file at once.
            for line in f:  # streamed: the corpus is a few hundred MB
                # Skip empty lines.
                if line.strip():
                    # Parse the JSON line into a dict.
                    row = json.loads(line)
                    # Store the text under its document ID.
                    self._texts[row["doc_id"]] = row["text"]

    # Makes store["123"] work: returns the text for that ID (KeyError if it is unknown).
    def __getitem__(self, doc_id: str) -> str:
        return self._texts[doc_id]

    # Makes "123" in store work: True if the ID is in the corpus.
    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._texts

    # Makes len(store) work: the number of documents loaded.
    def __len__(self) -> int:
        return len(self._texts)
