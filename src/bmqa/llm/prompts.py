"""The single, versioned prompt template used by every LLM system.

Only the evidence block differs between systems. Never edit or delete a released version; add a
new one (v2, v3) and report the dev results of every version tried.
"""

# Allows modern type hints such as `list[str]` without evaluating them at runtime.
from __future__ import annotations

# All prompt templates, keyed by version name. Every LLM system (closed-book, oracle, RAG)
# uses the SAME template; only {evidence_block} changes. That way, any difference in accuracy
# between systems is caused by the evidence alone, not by different wording of the prompt.
# Released versions are never edited, so old results can always be reproduced exactly.
TEMPLATES = {
    # Version 1. Adjacent string pieces inside ( ) are joined by Python into one string.
    "v1": (
        # The task: answer with exactly one of the three PubMedQA labels.
        'Answer the biomedical research question below with "yes", "no" or "maybe".\n'
        # Placeholder filled with the documents (or the "no documents" note) by .format().
        "{evidence_block}\n"
        # Empty line to separate the evidence from the question.
        "\n"
        # Placeholder filled with the question text.
        "Question: {question}\n"
        "\n"
        # Output instructions: the three JSON keys the parser expects.
        'Reply in JSON with the keys "answer", "rationale" (at most three sentences explaining '
        'your answer) and "citations" (the numbers of the documents that support your answer).'
    ),
}

# Evidence text for closed-book prompts (no documents): the model must use its own knowledge,
# and it has nothing to cite, so citations should be empty.
NO_DOCUMENTS = (
    'No documents are provided. Answer from your own knowledge and set "citations" to [].'
)
# Heading placed before the numbered documents when evidence IS given.
WITH_DOCUMENTS = "Use only the numbered documents below."


# Build the evidence part of the prompt. Input: a list of document texts (may be empty).
# Output: one string, either the "no documents" note or the heading plus numbered documents.
def evidence_block(docs: list[str]) -> str:
    """Documents are numbered from 1; identifiers such as PMIDs are deliberately not shown."""
    # Empty list (closed-book system): use the "no documents" note.
    if not docs:
        return NO_DOCUMENTS
    # Number each document [1], [2], ... (enumerate counts from 1 here) and separate them by a
    # blank line. Only numbers are shown, never PMIDs, so the model cannot cite a real ID from
    # memory; the parser then checks every citation is a number between 1 and len(docs).
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(docs, start=1))
    # Heading, blank line, then the numbered documents.
    return f"{WITH_DOCUMENTS}\n\n{numbered}"


# Build the full prompt. Inputs: the question, the document texts and the template version
# (default "v1"). Output: the complete prompt string sent to the LLM.
def build_prompt(question: str, docs: list[str], version: str = "v1") -> str:
    """Prompt from the question and document texts only, so long_answer cannot leak in."""
    # Reject unknown versions with a helpful message listing the known ones.
    if version not in TEMPLATES:
        raise ValueError(f"unknown prompt version {version!r}; known: {sorted(TEMPLATES)}")
    # Fill both placeholders. .format() only looks at the template, not at the inserted values,
    # so curly braces inside the question or documents are kept exactly as written.
    return TEMPLATES[version].format(evidence_block=evidence_block(docs), question=question)
