"""BM25 retriever over a corpus JSONL file, using the `bm25s` library.

Build an index:
    python -m bmqa.retrieval.bm25 --corpus data/corpus/strict.jsonl \
        --index-dir outputs/indexes/bm25_strict
"""

# Lets us write type hints such as "str | Path" on every supported Python version.
from __future__ import annotations

# argparse reads command-line options such as --corpus.
import argparse

# json reads the corpus lines and saves/loads the list of document IDs.
import json

# Path is an object-oriented way to work with file and folder paths.
from pathlib import Path

# Hit is the shared record for one search result: doc_id, score and rank (1 = best).
from bmqa.schema import Hit

# Name of the file (inside the index folder) that stores the doc_id of every indexed document.
# bm25s only knows documents by their position (0, 1, 2, ...), so we need this to map a
# position back to a PubMed ID.
DOC_IDS_FILE = "doc_ids.json"


# Splits texts into word tokens the same way for documents and for queries.
# Input: a list of strings. Output: bm25s token data ready for indexing or searching.
def _tokenize(texts: list[str]):
    # Imported here because bm25s is an optional extra; the module still imports without it.
    import bm25s  # needs the `retrieval` extra

    # PyStemmer: reduces words to their stem, e.g. "fractures" -> "fractur", so that
    # "fracture" in a question matches "fractures" in an abstract.
    import Stemmer

    # stopwords="en" drops very common English words ("the", "of", "and") that carry no
    # meaning for search. The English stemmer is applied to every remaining word.
    return bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"))


# Builds a BM25 index for a corpus file and saves it to disk.
# Inputs: corpus_path = a .jsonl corpus; index_dir = folder to save the index into.
# Output: the number of documents indexed.
def build_index(corpus_path: str | Path, index_dir: str | Path) -> int:
    """Index every document of the corpus; returns the number of documents."""
    # Optional dependency, imported only when needed.
    import bm25s

    # Two parallel lists: doc_ids[i] is the ID of the document whose text is texts[i].
    doc_ids, texts = [], []
    # Open the corpus file for reading as UTF-8 text.
    with open(corpus_path, encoding="utf-8") as f:
        # Read it line by line (one document per line) instead of all at once to save memory.
        for line in f:
            # Skip empty lines (for example a trailing newline at the end of the file).
            if line.strip():
                # Parse the JSON line into a dict.
                row = json.loads(line)
                # Remember the document's ID ...
                doc_ids.append(row["doc_id"])
                # ... and its text, at the same position.
                texts.append(row["text"])
    # Create an empty BM25 model with the library's default parameters.
    retriever = bm25s.BM25()
    # Tokenize all texts and build the index (word statistics used for scoring).
    retriever.index(_tokenize(texts))
    # Make sure we have a Path object (the caller may have passed a string).
    index_dir = Path(index_dir)
    # Create the index folder (and parents) if it does not exist yet.
    index_dir.mkdir(parents=True, exist_ok=True)
    # Save the BM25 index files into that folder (bm25s expects a string path).
    retriever.save(str(index_dir))
    # Save the position -> doc_id list next to the index so we can map results back.
    (index_dir / DOC_IDS_FILE).write_text(json.dumps(doc_ids), encoding="utf-8")
    # Report how many documents were indexed.
    return len(doc_ids)


# A retriever that searches a saved BM25 index. It has a retrieve(query, k) method, so it
# matches the Retriever protocol (interface) defined in bmqa.schema and can be used anywhere
# a Retriever is expected.
class BM25Retriever:
    """Implements the Retriever protocol from bmqa.schema."""

    # Loads a saved index. Input: index_dir = folder written earlier by build_index().
    def __init__(self, index_dir: str | Path):
        # Optional dependency, imported only when needed.
        import bm25s

        # Accept either a string or a Path.
        index_dir = Path(index_dir)
        # Load the BM25 index from disk (the leading underscore means "internal attribute").
        self._bm25 = bm25s.BM25.load(str(index_dir))
        # Load the list that maps index positions back to PubMed IDs.
        self._doc_ids: list[str] = json.loads((index_dir / DOC_IDS_FILE).read_text("utf-8"))

    # Searches for a query. Inputs: query text and k (how many results we want).
    # Output: a list of at most k Hit objects, best first, ranks starting at 1.
    def retrieve(self, query: str, k: int) -> list[Hit]:
        # bm25s fails if asked for more results than there are documents, so cap k at the
        # corpus size (this matters for tiny test corpora; a real corpus is much bigger).
        k = min(k, len(self._doc_ids))
        # Search. We pass a list with one query; bm25s returns a 2-D array of document
        # positions and a 2-D array of scores (one row per query).
        indices, scores = self._bm25.retrieve(_tokenize([query]), k=k)
        # Build one Hit per result.
        return [
            # Map the position to its doc_id; convert numpy numbers to plain Python types.
            Hit(doc_id=self._doc_ids[int(i)], score=float(s), rank=rank)
            # indices[0] and scores[0] are the results for our single query. zip pairs each
            # position with its score (strict=True raises an error if lengths differ), and
            # enumerate(..., start=1) numbers them so the best hit gets rank 1.
            for rank, (i, s) in enumerate(zip(indices[0], scores[0], strict=True), start=1)
        ]


# Command-line entry point: builds an index for one corpus file.
# Input: argv = list of command-line arguments (None means "use the real command line").
def main(argv: list[str] | None = None) -> None:
    # Create the command-line parser; the description appears in --help.
    parser = argparse.ArgumentParser(description="Build a BM25 index for a corpus file.")
    # --corpus: path to strict.jsonl or gold.jsonl (must be given).
    parser.add_argument("--corpus", type=Path, required=True)
    # --index-dir: folder where the index is saved (must be given).
    parser.add_argument("--index-dir", type=Path, required=True)
    # Read the options the user typed (or the argv list passed in by a test).
    args = parser.parse_args(argv)
    # Build and save the index; n is the number of documents indexed.
    n = build_index(args.corpus, args.index_dir)
    # Tell the user what happened.
    print(f"indexed {n} documents into {args.index_dir}")


# Runs main() only when this file is executed directly (python -m bmqa.retrieval.bm25),
# not when it is imported by another module or a test.
if __name__ == "__main__":
    main()
