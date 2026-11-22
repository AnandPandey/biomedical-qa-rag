# Tests for the LLM reply parser (bmqa/llm/parse.py). Run them with `pytest`.
# Import the function under test.
from bmqa.llm.parse import parse_answer


# A clean JSON reply with 2 documents in the prompt is read correctly in every field.
def test_plain_json():
    parsed = parse_answer('{"answer": "no", "rationale": "R.", "citations": [1, 2]}', n_docs=2)
    # The answer label is read as-is.
    assert parsed.label == "no"
    # The rationale text is kept.
    assert parsed.rationale == "R."
    # Both citations point at real documents (1 and 2 of 2), so both are valid.
    assert parsed.citations == [1, 2]
    # ...and nothing is flagged as fabricated.
    assert parsed.invalid_citations == []


# JSON inside a Markdown ```json fence, with extra text before it, is still found, and the
# answer " YES " is trimmed and lower-cased to "yes".
def test_fenced_json_and_case():
    text = 'Here you go:\n```json\n{"answer": " YES ", "rationale": "", "citations": []}\n```'
    # The parser must look inside the fence and normalise the answer.
    assert parse_answer(text, n_docs=0).label == "yes"


# A JSON object surrounded by ordinary words is still extracted.
def test_json_embedded_in_text():
    text = 'Answer follows {"answer": "maybe", "rationale": "x", "citations": []} thanks'
    # The {...} part is found between the first "{" and the last "}".
    assert parse_answer(text, n_docs=0).label == "maybe"


# Bad replies give label None (a parse failure) instead of crashing.
def test_invalid_answer_word_and_broken_json():
    # "probably" is not yes/no/maybe, so it is rejected.
    assert parse_answer('{"answer": "probably", "citations": []}', n_docs=0).label is None
    # Broken JSON (no closing brace) cannot be read at all.
    assert parse_answer('{"answer": "yes", ', n_docs=0).label is None
    # Plain text without JSON is not accepted, even if it says "yes".
    assert parse_answer("yes", n_docs=0).label is None


# With 3 documents, only numbers 1..3 are valid citations; others are "fabricated".
def test_out_of_range_citations_are_fabricated():
    parsed = parse_answer('{"answer": "yes", "rationale": "", "citations": [0, 1, 4, "2"]}', 3)
    # 1 is valid and the string "2" is converted to the number 2, also valid.
    assert parsed.citations == [1, 2]
    # 0 and 4 point at documents that do not exist, so they are fabricated.
    assert parsed.invalid_citations == [0, 4]


# A closed-book prompt has no documents (n_docs=0), so ANY citation is fabricated.
def test_closed_book_citations_are_always_invalid():
    parsed = parse_answer('{"answer": "yes", "rationale": "", "citations": [1]}', n_docs=0)
    # Citation 1 is invalid because there was no document 1.
    assert parsed.invalid_citations == [1]
