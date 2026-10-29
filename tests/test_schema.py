import json

import pytest

from bmqa.schema import (
    Example,
    Hit,
    LLMClient,
    LLMResponse,
    Prediction,
    Retriever,
)

# A hand-written row in the documented prediction format (not real data).
VALID_ROW = {
    "id": "00000001",
    "system": "rag_strict",
    "prediction": "yes",
    "probs": None,
    "rationale": "Doc 1 reports the effect.",
    "evidence": [
        {"doc_id": "00000010", "score": 12.5, "rank": 1},
        {"doc_id": "00000011", "score": 9.0, "rank": 2},
    ],
    "raw_output": '{"answer": "yes"}',
    "model": "test-model",
    "prompt_version": "v1",
    "config_hash": "abc123",
    "seed": None,
    "run": 1,
}


def test_valid_prediction_row_parses():
    pred = Prediction.from_dict(json.loads(json.dumps(VALID_ROW)))
    assert pred.prediction == "yes"
    assert [hit.rank for hit in pred.evidence] == [1, 2]


def test_classifier_style_prediction_needs_no_llm_fields():
    pred = Prediction(
        id="00000001",
        system="classifier",
        prediction="maybe",
        raw_output="",
        model="test-checkpoint",
        config_hash="abc123",
        probs={"yes": 0.2, "no": 0.3, "maybe": 0.5},
        seed=13,
    )
    assert pred.evidence == []
    assert pred.rationale is None


@pytest.mark.parametrize(
    "change",
    [
        {"prediction": "Yes"},
        {"run": 0},
        {"probs": {"yes": 0.5, "no": 0.4}},
        {"probs": {"yes": 0.5, "no": 0.4, "maybe": 0.2}},
        {"evidence": [{"doc_id": "a", "score": 1.0, "rank": 2}]},
        {"unexpected_field": 1},
    ],
)
def test_invalid_prediction_rows_are_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        Prediction.from_dict({**VALID_ROW, **change})


def test_example_rejects_unknown_label():
    with pytest.raises(ValueError):
        Example(id="1", question="q", contexts=["c"], long_answer="a", final_decision="unsure")


def test_fakes_satisfy_the_protocols():
    class FakeRetriever:
        def retrieve(self, query, k):
            return [Hit(doc_id="d1", score=1.0, rank=1)][:k]

    class FakeLLM:
        def generate(self, prompt, *, max_tokens):
            return LLMResponse(
                text="yes", model="fake", input_tokens=1, output_tokens=1, cached=False
            )

    assert isinstance(FakeRetriever(), Retriever)
    assert isinstance(FakeLLM(), LLMClient)
