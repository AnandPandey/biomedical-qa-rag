# Tests for bmqa.eval.compare: loading metric files, averaging over runs, Markdown tables.

# json: write fake metric files.
import json

# pytest: test framework; pytest.approx compares floats with a small tolerance.
import pytest

# Functions under test.
from bmqa.eval.compare import aggregate, load_metrics, markdown_table


# Helper: build a fake metrics dict like the real metric files.
# Inputs: system name, run number, accuracy, macro-F1 (also used as every label's F1).
def _metrics(system, run, acc, f1):
    # Per-label scores for yes/no/maybe; only "f1" matters here, the rest are zeros.
    per_label = {
        label: {"f1": f1, "precision": 0, "recall": 0, "support": 0}
        for label in ("yes", "no", "maybe")
    }
    # The full metrics record.
    return {
        "system": system,  # system name
        "split": "test",  # data split
        "run": run,  # run number
        "accuracy": acc,  # overall accuracy
        "macro_f1": f1,  # macro-F1
        "per_label": per_label,  # per-label scores
    }


# aggregate must average over runs, keep only main systems of the right split, in order.
def test_aggregate_mean_and_std(tmp_path):
    # Three closed_book runs with accuracy 0.6, 0.7, 0.8 (tmp_path = pytest's temp folder).
    for run, (acc, f1) in enumerate([(0.6, 0.4), (0.7, 0.5), (0.8, 0.6)], start=1):
        # File name pattern the loader expects, e.g. closed_book_test_run1.json.
        path = tmp_path / f"closed_book_test_run{run}.json"
        # Write the fake metrics as JSON.
        path.write_text(json.dumps(_metrics("closed_book", run, acc, f1)), encoding="utf-8")
    # One majority-baseline run.
    (tmp_path / "majority_test_run1.json").write_text(
        json.dumps(_metrics("majority", 1, 0.552, 0.237)), encoding="utf-8"
    )
    (tmp_path / "rag_strict_k3_test_run1.json").write_text(  # ablations are not in the table
        json.dumps(_metrics("rag_strict_k3", 1, 0.9, 0.9)), encoding="utf-8"
    )
    (tmp_path / "closed_book_dev_run1.json").write_text(  # other split ignored
        json.dumps({**_metrics("closed_book", 1, 0.0, 0.0), "split": "dev"}), encoding="utf-8"
    )
    # Load test-split metrics and aggregate them.
    rows = aggregate(load_metrics(tmp_path, "test"))
    # Only main systems, in MAIN_SYSTEMS order (majority before closed_book).
    assert [r["system"] for r in rows] == ["majority", "closed_book"]
    # The closed_book row.
    closed = rows[1]
    # Three test runs (the dev file was ignored).
    assert closed["runs"] == 3
    # Mean = (0.6 + 0.7 + 0.8) / 3 = 0.7.
    assert closed["accuracy_mean"] == pytest.approx(0.7)
    # Sample std = sqrt(((-0.1)**2 + 0**2 + 0.1**2) / (3 - 1)) = sqrt(0.01) = 0.1.
    assert closed["accuracy_std"] == pytest.approx(0.1)
    # A single run has no spread, so std is defined as 0.0.
    assert rows[0]["macro_f1_std"] == 0.0


# markdown_table must produce header, separator and row lines exactly.
def test_markdown_table():
    # Two columns, one row.
    text = markdown_table(["A", "B"], [[1, "x"]])
    # Header "| A | B |", separator "|---|---|" (one "---|" per column), row "| 1 | x |".
    assert text == "| A | B |\n|---|---|\n| 1 | x |\n"
