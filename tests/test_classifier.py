"""Classifier helpers that work without PyTorch; training itself is checked by a manual run."""

# pytest: the test framework; pytest.approx compares floating-point numbers with a tolerance.
import pytest

# ID2LABEL / LABEL2ID convert between label names and the numbers the model uses;
# make_predictions turns model probabilities into Prediction rows; text_pair builds the input.
from bmqa.classifier.data import ID2LABEL, LABEL2ID, make_predictions, text_pair

# to_example converts a raw PubMedQA entry into an Example record.
from bmqa.data.pubmedqa import to_example


# The two label mappings must agree, otherwise model outputs would be decoded wrongly.
def test_label_mapping_round_trips():
    # Labels are numbered in LABELS order: yes = 0, no = 1, maybe = 2.
    assert LABEL2ID == {"yes": 0, "no": 1, "maybe": 2}
    # Going label -> number -> label must give back the same label for every label.
    assert all(ID2LABEL[i] == label for label, i in LABEL2ID.items())


# The long answer states the conclusion, so feeding it to the model would be cheating.
# pqal_sample is a fixture from conftest.py with the 6 synthetic entries.
def test_input_never_contains_long_answer(pqal_sample):
    # Check every synthetic entry (pmid is the id, entry is the raw dict).
    for pmid, entry in pqal_sample.items():
        # Build the model input: the question and the joined abstract contexts.
        question, context = text_pair(to_example(pmid, entry))
        # The tag that marks every long answer in the fixtures must not appear anywhere.
        assert "SYNTHETIC-CONCLUSION" not in question + context


# make_predictions must rescale probabilities to sum to 1 and pick the highest one (argmax).
def test_make_predictions_normalises_and_takes_argmax(pqal_sample):
    # Use the first two synthetic entries (PMIDs 90000001 and 90000002) as examples.
    examples = [to_example(p, e) for p, e in list(pqal_sample.items())[:2]]
    # Fake model probabilities in LABELS order (yes, no, maybe), one row per example:
    #   row 1: highest is 0.7 at position 1 -> "no"
    #   row 2: highest is 0.5 at position 0 -> "yes"
    # The other keyword arguments are metadata that should be copied onto each Prediction.
    preds = make_predictions(
        examples,
        [[0.1, 0.7, 0.2], [0.5, 0.25, 0.25]],
        model_name="m",
        cfg_hash="c",
        run=2,
        seed=1,
    )
    # The argmax labels worked out above.
    assert [p.prediction for p in preds] == ["no", "yes"]
    # Row 1 already sums to 1.0, so the "maybe" probability stays 0.2 (approx: float maths).
    assert preds[0].probs["maybe"] == pytest.approx(0.2)
    # Metadata is copied through, and system defaults to "classifier".
    assert all(p.seed == 1 and p.run == 2 and p.system == "classifier" for p in preds)
