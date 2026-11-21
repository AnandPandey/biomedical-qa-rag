"""Classification metrics for PubMedQA predictions (standard library only).

Macro-F1 is the unweighted mean of the per-label F1 scores over yes/no/maybe, with precision
0 for a label that is never predicted, matching scikit-learn's f1_score(average="macro") used
by the official PubMedQA evaluation script.
"""

# Makes type hints like `str | None` work as plain text on older Python versions.
from __future__ import annotations

# Counter: a dict that counts how often each item appears; used for all the tallies below.
from collections import Counter

# LABELS = ("yes", "no", "maybe"); Prediction = the shared record of one system answer.
from bmqa.schema import LABELS, Prediction

# A system writes raw_output starting with this text when it could not turn its output into
# yes/no/maybe. Such rows are scored as wrong (never silently dropped), so a system cannot
# raise its score by refusing hard questions.
PARSE_FAILURE_PREFIX = "PARSE_FAILURE:"


# _check_lengths: make sure there is exactly one prediction per gold label.
# The leading underscore means "internal helper, not meant for other modules".
def _check_lengths(gold: list[str], pred: list) -> None:
    # Different lengths would pair the wrong answers together, so stop.
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")


# accuracy: share of predictions that equal the gold label.
# Inputs: gold labels and predicted labels (same order). Output: a number from 0.0 to 1.0.
def accuracy(gold: list[str], pred: list[str | None]) -> float:
    """`None` in `pred` (an unparseable answer) always counts as wrong."""
    # Make sure the two lists line up.
    _check_lengths(gold, pred)
    # zip pairs each gold with its prediction (strict=True errors if lengths differ);
    # g == p is True (1) or False (0), so the sum is the number correct. Divide by the total.
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


# per_label: precision, recall, F1 and support for each of yes/no/maybe.
# Inputs: gold and predicted labels. Output: {label: {"precision", "recall", "f1", "support"}}.
def per_label(gold: list[str], pred: list[str | None]) -> dict[str, dict[str, float]]:
    # Make sure the two lists line up.
    _check_lengths(gold, pred)
    # Count each (gold, predicted) pair, e.g. ("yes", "no") -> how often yes was called no.
    pairs = Counter(zip(gold, pred, strict=True))
    # How many times each label is the correct answer.
    gold_counts = Counter(gold)
    # How many times each label was predicted (None counts are kept but never used below).
    pred_counts = Counter(pred)
    # Results will be collected here.
    out = {}
    # Compute the scores for each label in turn.
    for label in LABELS:
        # True positives: gold is this label AND the prediction is this label.
        tp = pairs[(label, label)]
        # Precision: of all times we predicted this label, the share that was right.
        # If the label was never predicted, use 0.0 instead of dividing by zero.
        precision = tp / pred_counts[label] if pred_counts[label] else 0.0
        # Recall: of all examples truly having this label, the share we found.
        recall = tp / gold_counts[label] if gold_counts[label] else 0.0
        # F1: harmonic mean of precision and recall; 0.0 when both are 0 (avoids 0/0).
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Store the four numbers for this label.
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            # Support: how many gold examples of this label exist.
            "support": gold_counts[label],
        }
    # Return the scores of all three labels.
    return out


# macro_f1: the main score of the project.
# Inputs: gold and predicted labels. Output: the plain average of the three per-label F1s.
# Each label counts equally, so a system that ignores the rare "maybe" class is penalised,
# unlike plain accuracy.
def macro_f1(gold: list[str], pred: list[str | None]) -> float:
    # Get the per-label scores.
    scores = per_label(gold, pred)
    # Add up the three F1 scores and divide by 3.
    return sum(scores[label]["f1"] for label in LABELS) / len(LABELS)


# confusion_matrix: a 3x3 table of counts showing which labels get confused with which.
# Inputs: gold and predicted labels. Output: list of 3 rows, each a list of 3 counts.
def confusion_matrix(gold: list[str], pred: list[str | None]) -> list[list[int]]:
    """Rows = gold label, columns = predicted label, in LABELS order (None is not counted)."""
    # Make sure the two lists line up.
    _check_lengths(gold, pred)
    # Count each (gold, predicted) pair.
    pairs = Counter(zip(gold, pred, strict=True))
    # Outer loop builds one row per gold label g; inner loop one cell per predicted label p.
    # Pairs whose prediction is None never match a column, so parse failures are left out.
    return [[pairs[(g, p)] for p in LABELS] for g in LABELS]


# effective_label: the label that is actually scored for one prediction row.
# Input: a Prediction. Output: its label, or None if the row is a parse failure.
def effective_label(prediction: Prediction) -> str | None:
    """The label to score: None when the row is marked as a parse failure."""
    # A raw output starting with "PARSE_FAILURE:" means the answer was unusable.
    if prediction.raw_output.startswith(PARSE_FAILURE_PREFIX):
        # None never equals a gold label, so this row will count as wrong.
        return None
    # Otherwise score the predicted label as written.
    return prediction.prediction


# compute_metrics: all scores for one prediction file.
# Inputs: {id: gold label} for the split, and the predictions. Output: a dict of metrics.
def compute_metrics(gold: dict[str, str], predictions: list[Prediction]) -> dict:
    """Score predictions against gold labels; the ID sets must match exactly."""
    # The example IDs in prediction order.
    ids = [p.id for p in predictions]
    # A set drops duplicates; if it is shorter, some question was answered twice.
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in predictions")
    # missing = gold IDs with no prediction; extra = predicted IDs not in this split.
    missing, extra = gold.keys() - set(ids), set(ids) - gold.keys()
    # Every question must be answered exactly once; otherwise scores are not comparable
    # between systems (skipping hard questions would unfairly raise a score).
    if missing or extra:
        raise ValueError(
            f"predictions must cover exactly the split: {len(missing)} missing, {len(extra)} extra"
        )
    # Gold labels in the same order as the predictions.
    gold_list = [gold[p.id] for p in predictions]
    # Labels to score in the same order (None for parse failures).
    pred_list = [effective_label(p) for p in predictions]
    # Build and return the full metrics dict.
    return {
        # Number of scored examples.
        "n": len(predictions),
        # Share of exactly correct answers.
        "accuracy": accuracy(gold_list, pred_list),
        # Main score: average F1 over yes/no/maybe.
        "macro_f1": macro_f1(gold_list, pred_list),
        # Precision/recall/F1/support of each label.
        "per_label": per_label(gold_list, pred_list),
        # The confusion matrix plus its label order, so the table can be read on its own.
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows_gold_cols_pred": confusion_matrix(gold_list, pred_list),
        },
        # How many rows were parse failures (True counts as 1 in the sum).
        "parse_failures": sum(label is None for label in pred_list),
    }
