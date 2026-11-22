# Tests for the BM25 retriever and the DocStore, using a tiny five-document corpus.
# json writes the tiny corpus file.
import json

# pytest is the test framework (fixtures, importorskip, assertions).
import pytest

# importorskip: if the optional library is not installed, SKIP every test in this file
# instead of failing. bm25s and PyStemmer are only installed with the `retrieval` extra.
pytest.importorskip("bm25s")
pytest.importorskip("Stemmer")

# These imports come after importorskip on purpose (they need bm25s). "noqa: E402" tells
# the ruff linter not to complain that the imports are not at the top of the file.
from bmqa.retrieval.bm25 import BM25Retriever, build_index  # noqa: E402
from bmqa.retrieval.store import DocStore  # noqa: E402
from bmqa.schema import Retriever  # noqa: E402

# The tiny test corpus: (doc_id, text) pairs on clearly different topics.
DOCS = [
    ("d1", "Aspirin reduces the risk of myocardial infarction in older adults."),
    ("d2", "Vitamin D supplementation and bone fractures in postmenopausal women."),
    ("d3", "Smoking cessation programmes improve lung function."),
    ("d4", "Statins lower cholesterol and cardiovascular events."),
    ("d5", "Exercise training improves depression scores in adolescents."),
]


# A fixture is setup code pytest runs before each test that asks for it by name ("index").
# tmp_path is a built-in fixture: a fresh temporary folder, deleted later by pytest.
# Output: (path of the corpus file, path of the built index folder).
@pytest.fixture
def index(tmp_path):
    # Where the tiny corpus file is written.
    corpus = tmp_path / "corpus.jsonl"
    # Write one JSON line per document, in the same format as the real corpus.
    corpus.write_text(
        # Join all lines into one string.
        "".join(
            # One line: the document as JSON plus a newline.
            json.dumps({"doc_id": i, "title": "", "text": t, "source": "pqa_a"}) + "\n"
            # i = doc_id, t = text.
            for i, t in DOCS
        ),
        encoding="utf-8",
    )
    # Build the index and check it reports all five documents as indexed.
    assert build_index(corpus, tmp_path / "index") == len(DOCS)
    # Give the test both paths.
    return corpus, tmp_path / "index"


# Checks that BM25 ranks the obviously relevant document first and returns valid hits.
def test_matching_document_ranks_first(index):
    # We only need the index folder; "_" ignores the corpus path.
    _, index_dir = index
    # Load the saved index.
    retriever = BM25Retriever(index_dir)
    # It must satisfy the shared Retriever interface (has a retrieve(query, k) method).
    assert isinstance(retriever, Retriever)
    # Search with a question about vitamin D and fractures, asking for 3 results.
    hits = retriever.retrieve("Does vitamin D prevent fractures?", k=3)
    # d2 is the vitamin D / fractures document, so it must be the top hit.
    # (It is the only document containing the query words "vitamin" and "fractures".)
    assert hits[0].doc_id == "d2"
    # Ranks must be 1, 2, 3, ... in order.
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    # Never more results than asked for.
    assert len(hits) <= 3


# Checks that asking for more documents than exist does not crash: k is capped at 5.
def test_k_larger_than_corpus_is_capped(index):
    # Only the index folder is needed.
    _, index_dir = index
    # Ask for 50 results from a 5-document corpus; we must get exactly 5 back.
    assert len(BM25Retriever(index_dir).retrieve("exercise", k=50)) == len(DOCS)


# Checks that DocStore loads all documents and supports len(), "in" and [] lookup.
def test_doc_store(index):
    # Only the corpus path is needed.
    corpus, _ = index
    # Load the corpus into the store.
    store = DocStore(corpus)
    # All five documents are loaded and "d3" is one of them.
    assert len(store) == len(DOCS) and "d3" in store
    # Looking up d3 returns its text.
    assert store["d3"].startswith("Smoking")
