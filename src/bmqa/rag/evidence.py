"""Evidence function that feeds retrieved abstracts to the Phase 2 runner."""

# Lets us use modern type hints on every supported Python version.
from __future__ import annotations

# EvidenceFn is the type of function the Phase 2 runner expects: it takes an Example and
# returns a list of (Hit, document text) pairs to put into the prompt.
from bmqa.llm.run import EvidenceFn

# Example = one question; Hit = one search result; Retriever = the search interface.
from bmqa.schema import Example, Hit, Retriever


# Creates an evidence function that uses retrieval. This is a "function factory": it
# returns a new function that remembers retriever, store and k.
# Inputs: retriever = anything with retrieve(query, k); store = maps doc_id -> text
# (a DocStore, or a plain dict in tests); k = how many abstracts to give the LLM.
# Output: an EvidenceFn that the unchanged Phase 2 runner can call for every question.
def make_retrieval_evidence(retriever: Retriever, store, k: int) -> EvidenceFn:
    """Query = the question text; returns the top-k (hit, document text) pairs."""

    # The inner function the runner calls once per question.
    # Input: one Example. Output: up to k (Hit, abstract text) pairs, best first.
    def evidence(example: Example) -> list[tuple[Hit, str]]:
        # Search with the question text only (never the long answer, which is the
        # conclusion), then look up each hit's full text in the store.
        return [(hit, store[hit.doc_id]) for hit in retriever.retrieve(example.question, k)]

    # Return the inner function itself (not its result); it is called later by the runner.
    return evidence
