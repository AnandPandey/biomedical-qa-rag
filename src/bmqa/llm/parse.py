"""Parse an LLM reply into answer, rationale and citations."""

# Allows modern type hints such as `str | None` without evaluating them at runtime.
from __future__ import annotations

# json turns the reply text into a Python dict.
import json

# re = regular expressions, used to find JSON wrapped in a Markdown code fence.
import re

# dataclass builds a simple class from field declarations; field() sets per-field options.
from dataclasses import dataclass, field

# LABELS = ("yes", "no", "maybe"): the only valid answers.
from bmqa.schema import LABELS

# Pattern for a Markdown code fence such as ```json { ... } ```. The part in ( ) captures the
# {...} object inside. `.*?` matches as little as possible; re.DOTALL lets `.` match newlines,
# because the JSON object usually spans several lines.
_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# The parser's result. frozen=True makes it read-only after creation (safer to pass around).
@dataclass(frozen=True)
class ParsedAnswer:
    label: str | None  # None if no valid yes/no/maybe answer was found
    rationale: str = ""  # the model's explanation ("" if missing)
    citations: list[int] = field(default_factory=list)  # valid document numbers (1..n_docs)
    invalid_citations: list[int] = field(default_factory=list)  # fabricated citations


# Helper: find a JSON object in the reply text. Input: the raw reply. Output: a dict, or None
# if no JSON object could be read. Models sometimes add text or fences around the JSON, so
# several candidate substrings are tried, from the strictest to the most lenient.
def _load_object(text: str) -> dict | None:
    # Candidate 1: the whole reply (without leading/trailing spaces) is pure JSON.
    candidates = [text.strip()]
    # Candidate 2: JSON inside a Markdown code fence, if there is one.
    fenced = _FENCED.search(text)
    if fenced:
        # group(1) is the {...} part captured by the parentheses in the pattern.
        candidates.append(fenced.group(1))
    # Candidate 3: everything from the first "{" to the last "}" in the text.
    # find/rfind return -1 when the character is missing.
    start, end = text.find("{"), text.rfind("}")
    # Only use it if both braces exist and the "{" comes before the "}".
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    # Try each candidate in order and return the first one that is a valid JSON object.
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        # Not valid JSON: move on to the next candidate.
        except json.JSONDecodeError:
            continue
        # Valid JSON, but we need an object (dict), not e.g. a list or a number.
        if isinstance(data, dict):
            return data
    # Nothing worked: the reply is unparseable.
    return None


# Main parser. Inputs: the reply text and n_docs (how many documents were in the prompt; 0 for
# closed-book). Output: a ParsedAnswer. It never raises on bad replies: problems are reported
# as label=None, which the runner records as a parse failure.
def parse_answer(text: str, n_docs: int) -> ParsedAnswer:
    """Accept plain JSON, JSON in a Markdown fence, or the first {...} object in the text."""
    # Extract the JSON object from the reply, if any.
    data = _load_object(text)
    # No JSON at all: return a result with no label (a parse failure).
    if data is None:
        return ParsedAnswer(label=None)
    # Read "answer" (default ""), make it text, trim spaces and lower-case it ("YES " -> "yes").
    answer = str(data.get("answer", "")).strip().lower()
    # Keep it only if it is one of yes/no/maybe; anything else (e.g. "probably") -> None.
    label = answer if answer in LABELS else None
    # Read "rationale"; `or ""` also turns a missing or null value into an empty string.
    rationale = str(data.get("rationale") or "").strip()

    # Read "citations"; a missing or null value becomes an empty list.
    raw = data.get("citations") or []
    # If the model gave a single value instead of a list (e.g. 2), wrap it in a list.
    if not isinstance(raw, list):
        raw = [raw]
    # Two output lists: citations of real documents, and invented (fabricated) ones.
    valid, invalid = [], []
    for item in raw:
        # Skip true/false: in Python bool is a kind of int (True == 1), which would be wrong.
        if isinstance(item, bool):
            continue
        # Convert to a whole number; strings like "2" also work.
        try:
            number = int(item)
        # Values that are not numbers (e.g. "doc A" or None) are ignored.
        except (TypeError, ValueError):
            continue
        # A number between 1 and n_docs points at a real document -> valid. Anything else
        # (0, too large, or any citation in a closed-book prompt) is fabricated -> invalid.
        (valid if 1 <= number <= n_docs else invalid).append(number)
    # Return everything as positional arguments: label, rationale, citations, invalid ones.
    return ParsedAnswer(label, rationale, valid, invalid)
