"""Shared records and interfaces used by every component.

This module is the code form of docs/interfaces.md. Change both together, in a dedicated PR.
Only the standard library is used so that every component can import it cheaply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Label = Literal["yes", "no", "maybe"]
LABELS: tuple[Label, ...] = ("yes", "no", "maybe")

Split = Literal["train", "dev", "test"]
SPLITS: tuple[Split, ...] = ("train", "dev", "test")


def _check_label(value: str, field_name: str) -> None:
    if value not in LABELS:
        raise ValueError(f"{field_name} must be one of {LABELS}, got {value!r}")


@dataclass(frozen=True)
class Example:
    """One PubMedQA question. `long_answer` must never be part of a model input."""

    id: str
    question: str
    contexts: list[str]
    long_answer: str
    final_decision: Label

    def __post_init__(self) -> None:
        _check_label(self.final_decision, "final_decision")


@dataclass(frozen=True)
class CorpusDoc:
    """One abstract in the retrieval corpus (conclusion section removed)."""

    doc_id: str
    title: str
    text: str
    source: Literal["pqa_a", "pqa_u", "pqa_l"]


@dataclass(frozen=True)
class Hit:
    """One retrieved document. `rank` starts at 1 (best)."""

    doc_id: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool


@dataclass(frozen=True)
class Prediction:
    """One system's answer to one example. Written by every system, classifier included."""

    id: str
    system: str
    prediction: Label
    raw_output: str
    model: str
    config_hash: str
    run: int = 1
    probs: dict[str, float] | None = None
    rationale: str | None = None
    evidence: list[Hit] = field(default_factory=list)
    prompt_version: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        _check_label(self.prediction, "prediction")
        if self.run < 1:
            raise ValueError(f"run must be >= 1, got {self.run}")
        if self.probs is not None:
            if set(self.probs) != set(LABELS):
                raise ValueError(f"probs must have exactly the keys {LABELS}")
            if not math.isclose(sum(self.probs.values()), 1.0, abs_tol=1e-6):
                raise ValueError("probs must sum to 1")
        ranks = [hit.rank for hit in self.evidence]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("evidence must be ordered by rank 1, 2, 3, ...")

    @classmethod
    def from_dict(cls, data: dict) -> Prediction:
        """Build from one parsed JSONL row; raises ValueError or TypeError if it is invalid."""
        data = dict(data)
        data["evidence"] = [Hit(**hit) for hit in data.get("evidence", [])]
        return cls(**data)


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Hit]:
        """Return at most `k` hits, best first."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """The model is fixed when the client is created, so one run cannot mix models."""

    def generate(self, prompt: str, *, max_tokens: int) -> LLMResponse: ...
