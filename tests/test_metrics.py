# Tests for bmqa.eval.metrics: accuracy, per-label precision/recall/F1, macro-F1, the
# confusion matrix, and compute_metrics, which scores a list of Prediction objects.

# pytest: the test framework; pytest.approx and pytest.raises are used below.
import pytest

# The metric functions under test.
from bmqa.eval.metrics import accuracy, compute_metrics, confusion_matrix, macro_f1, per_label

# Prediction is the record every system writes; compute_metrics takes a list of them.
from bmqa.schema import Prediction

# Hand-computed example (yes, no, maybe):
#   gold: yes yes no maybe no      pred: yes no no yes no
#   yes:   P = 1/2, R = 1/2, F1 = 0.5
#   no:    P = 2/3, R = 2/2, F1 = 0.8
#   maybe: never predicted -> P = 0, R = 0, F1 = 0
#   accuracy = 3/5 = 0.6, macro-F1 = (0.5 + 0.8 + 0) / 3
# (P = precision = correct / times predicted; R = recall = correct / times it is the gold
#  answer; F1 = 2 * P * R / (P + R). For "no": 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 0.8.)
# The correct answers ("gold" labels) for 5 imaginary questions.
GOLD = ["yes", "yes", "no", "maybe", "no"]
# What an imaginary system predicted for the same 5 questions (positions 1, 3, 5 are right).
PRED = ["yes", "no", "no", "yes", "no"]


# Checks every metric against the numbers worked out by hand in the comment block above.
def test_hand_computed_metrics():
    # pytest.approx(x) compares floating-point numbers with a tiny tolerance, because values
    # like 2/3 cannot be stored exactly and "==" could fail on rounding noise.
    # 3 of 5 predictions match the gold label -> 3/5 = 0.6.
    assert accuracy(GOLD, PRED) == pytest.approx(0.6)
    # per_label returns {label: {"precision", "recall", "f1", "support"}}.
    scores = per_label(GOLD, PRED)
    # "yes": predicted twice, 1 correct (P = 1/2); gold twice, 1 found (R = 1/2) -> F1 = 0.5.
    assert scores["yes"]["f1"] == pytest.approx(0.5)
    # "no": predicted 3 times, 2 correct -> precision 2/3.
    assert scores["no"]["precision"] == pytest.approx(2 / 3)
    # "no": recall 2/2 = 1, so F1 = 2 * (2/3) * 1 / (2/3 + 1) = 0.8.
    assert scores["no"]["f1"] == pytest.approx(0.8)
    # "maybe" is never predicted, so P = R = 0 and F1 is defined as exactly 0.0.
    assert scores["maybe"]["f1"] == 0.0
    # Macro-F1 = plain average of the three F1s = (0.5 + 0.8 + 0) / 3 = 1.3 / 3.
    assert macro_f1(GOLD, PRED) == pytest.approx(1.3 / 3)
    # Rows = gold label, columns = predicted label, both in order yes, no, maybe:
    #   gold yes:   1 predicted yes, 1 predicted no          -> [1, 1, 0]
    #   gold no:    both predicted no                        -> [0, 2, 0]
    #   gold maybe: predicted yes                            -> [1, 0, 0]
    assert confusion_matrix(GOLD, PRED) == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


# Reproduces the majority-class baseline score on the real dev label counts, so the number
# reported for that baseline can be trusted.
def test_majority_on_dev_label_distribution():
    # Dev has 55 yes, 34 no, 11 maybe; always predicting "yes":
    # Build 100 gold labels with exactly those counts (list * n repeats the list n times).
    gold = ["yes"] * 55 + ["no"] * 34 + ["maybe"] * 11
    # The baseline answers "yes" for all 100 questions.
    pred = ["yes"] * 100
    # Only the 55 "yes" questions are right -> 55 / 100 = 0.55.
    assert accuracy(gold, pred) == pytest.approx(0.55)
    # "yes": P = 55/100 = 0.55, R = 55/55 = 1, F1 = 2 * 0.55 * 1 / (0.55 + 1) = 1.1 / 1.55.
    # "no" and "maybe" are never predicted -> F1 = 0. Macro-F1 = (1.1 / 1.55 + 0 + 0) / 3.
    assert macro_f1(gold, pred) == pytest.approx((1.1 / 1.55) / 3)
    # The same value rounded to 4 decimals, as it would appear in a results table.
    assert round(macro_f1(gold, pred), 4) == 0.2366


# Helper (not a test) that builds a minimal valid Prediction with a given id and label.
# raw is the raw model output; it defaults to an empty string.
def _pred(pid, label, raw=""):
    # Fill the other required fields with short dummy values.
    return Prediction(
        id=pid, system="s", prediction=label, raw_output=raw, model="m", config_hash="c"
    )


# If a model's output could not be parsed, its row is marked with "PARSE_FAILURE:" and must be
# scored as wrong, even if the placeholder label happens to equal the gold label.
def test_parse_failures_count_as_wrong():
    # Gold answers for two questions, keyed by id.
    gold = {"a": "yes", "b": "no"}
    # "a" is correct; "b" has the right label "no" but its raw output is a parse failure.
    preds = [_pred("a", "yes"), _pred("b", "no", raw="PARSE_FAILURE: not json")]
    # Score the predictions; returns a dict of metrics.
    metrics = compute_metrics(gold, preds)
    # Only "a" counts as correct -> 1 / 2 = 0.5 (not 1.0).
    assert metrics["accuracy"] == pytest.approx(0.5)
    # Exactly one parse failure is reported.
    assert metrics["parse_failures"] == 1


# Predictions must cover exactly the ids of the split: none missing, none extra, none twice.
# Otherwise a score could look better or worse than it really is.
def test_id_sets_must_match():
    # Gold labels for two ids.
    gold = {"a": "yes", "b": "no"}
    # Missing id "b" -> ValueError.
    with pytest.raises(ValueError):
        compute_metrics(gold, [_pred("a", "yes")])
    # Extra id "c" that has no gold label -> ValueError.
    with pytest.raises(ValueError):
        compute_metrics(gold, [_pred("a", "yes"), _pred("b", "no"), _pred("c", "no")])
    # Duplicate id "a" (and "b" missing) -> ValueError.
    with pytest.raises(ValueError):
        compute_metrics(gold, [_pred("a", "yes"), _pred("a", "yes")])
