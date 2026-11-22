# Tests for bmqa.retrieval.corpus. They use small hand-made rows, so no download is needed.
# The function under test: builds the strict and gold corpora from raw rows.
from bmqa.retrieval.corpus import build_corpora


# Helper that makes a fake Hugging Face row with the same shape as the real dataset.
# Inputs: pubid = PubMed ID (an int, like in the real data); text = first abstract section.
# Output: a dict that looks like one PQA-A / PQA-U row.
def _row(pubid, text):
    return {
        # The article's PubMed ID.
        "pubid": pubid,
        # Abstract sections: our text plus a second section "more" (so joining is tested).
        "context": {"contexts": [text, "more"], "labels": [], "meshes": []},
        # Marker strings: the tests check these never end up in a corpus document.
        "question": "QUESTION-TEXT",
        "long_answer": "CONCLUSION-TEXT",
    }


# A fake PQA-L with two labelled articles, "100" and "200" (keys are strings, like the file).
PQAL = {
    "100": {"CONTEXTS": ["labelled abstract"], "LONG_ANSWER": "x", "final_decision": "yes"},
    "200": {"CONTEXTS": ["another"], "LONG_ANSWER": "y", "final_decision": "no"},
}


# Checks the core rules: strict drops PQA-L and duplicates and is sorted; gold adds PQA-L.
def test_strict_excludes_pqal_and_gold_includes_it():
    # PQA-A rows: two normal articles (3, 1) and one PQA-L article (100) that must be removed.
    pqa_a = [_row(3, "a3"), _row(100, "leak"), _row(1, "a1")]
    # PQA-U rows: a new article (2), a duplicate of article 1, and PQA-L article 200.
    pqa_u = [_row(2, "u2"), _row(1, "dup"), _row(200, "leak2")]
    # Build both corpora from the fake data.
    strict, gold, stats = build_corpora(pqa_a, pqa_u, PQAL)

    # Strict holds only 1, 2, 3: sorted by ID, PQA-L removed, duplicate of 1 removed.
    assert [d.doc_id for d in strict] == ["1", "2", "3"]  # sorted, PQA-L removed, deduplicated
    # Article 1 kept its PQA-A text (the PQA-U copy "dup" was dropped); sections are joined.
    assert strict[0].text == "a1 more" and strict[0].source == "pqa_a"  # PQA-A kept first
    # Gold = strict plus both PQA-L articles (a set, so order does not matter here).
    assert {d.doc_id for d in gold} == {"1", "2", "3", "100", "200"}
    # The sanity counters agree: no PQA-L in strict, both PQA-L articles in gold.
    assert stats["pqal_in_strict"] == 0 and stats["pqal_in_gold"] == 2
    # PQA-A: 3 rows read, 1 PQA-L removed (100), no duplicates.
    assert stats["pqa_a"] == {"rows_read": 3, "pqal_removed": 1, "duplicates_removed": 0}
    # PQA-U: 3 rows read, 1 PQA-L removed (200), 1 duplicate removed (1).
    assert stats["pqa_u"] == {"rows_read": 3, "pqal_removed": 1, "duplicates_removed": 1}


# Checks that the question and the conclusion (the answer) are never put into a document,
# otherwise the LLM could read the answer straight from the retrieved text.
def test_question_and_conclusion_never_enter_the_corpus():
    # One PQA-A row, no PQA-U rows; "_" ignores the stats we do not need here.
    strict, gold, _ = build_corpora([_row(1, "a")], [], PQAL)
    # Check every document of both corpora.
    for doc in strict + gold:
        # Neither marker string may appear in the document text.
        assert "QUESTION-TEXT" not in doc.text and "CONCLUSION-TEXT" not in doc.text
        # Titles are always empty (PubMedQA has none).
        assert doc.title == ""
