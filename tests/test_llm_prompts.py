# Tests for the prompt template (bmqa/llm/prompts.py). Run them with `pytest`.
# pytest is the test framework; here it is needed for pytest.raises (expecting an error).
import pytest

# Import the "no documents" note and the two prompt-building functions under test.
from bmqa.llm.prompts import NO_DOCUMENTS, build_prompt, evidence_block


# A closed-book prompt (empty document list) says there are no documents and shows none.
def test_closed_book_prompt_has_no_documents():
    prompt = build_prompt("Does A cause B?", [])
    # The "no documents" note is included.
    assert NO_DOCUMENTS in prompt
    # No numbered document appears.
    assert "[1]" not in prompt
    # The question is inserted in the expected place.
    assert "Question: Does A cause B?" in prompt


# Documents are numbered [1], [2], ... starting from one.
def test_documents_are_numbered_from_one():
    prompt = build_prompt("Q?", ["first doc", "second doc"])
    # The first document is labelled [1].
    assert "[1] first doc" in prompt
    # The second document is labelled [2].
    assert "[2] second doc" in prompt
    # No [0] (numbering does not start at zero) and no [3] (there are only two documents).
    assert "[0]" not in prompt and "[3]" not in prompt


# Fair comparison check: removing the evidence block from a closed-book prompt and from a
# prompt with documents must leave identical text, so the systems differ only in evidence.
def test_only_the_evidence_block_differs():
    # The same question without and with one document.
    closed = build_prompt("Q?", [])
    with_docs = build_prompt("Q?", ["d"])
    # Cut out each prompt's own evidence block; what remains must be exactly the same.
    assert closed.replace(evidence_block([]), "") == with_docs.replace(evidence_block(["d"]), "")


# Curly braces inside the question must not be treated as template placeholders.
def test_braces_in_question_are_kept_verbatim():
    # "{x}" must appear in the prompt unchanged (and no error must be raised).
    assert "Question: is {x} > 1?" in build_prompt("is {x} > 1?", [])


# Asking for a template version that does not exist is an error, not a silent fallback.
def test_unknown_version_is_rejected():
    # The test passes only if the code inside the `with` block raises ValueError.
    with pytest.raises(ValueError):
        build_prompt("Q?", [], version="v99")
