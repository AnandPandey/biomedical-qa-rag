# Tests for the majority-class baseline (bmqa.baselines), which always predicts the label
# that is most common in the training data.

# main runs the baseline command; majority_label picks the most common label;
# majority_predictions builds one Prediction per example using that label.
from bmqa.baselines import main, majority_label, majority_predictions

# load_split reads a split's examples from a processed-data folder.
from bmqa.data.pubmedqa import load_split

# load_predictions reads a prediction .jsonl file back into Prediction objects.
from bmqa.io import load_predictions

# Example is the record for one PubMedQA question.
from bmqa.schema import Example


# Helper (not a test) that makes a tiny Example with id i and gold label `label`;
# the text fields are dummy one-letter strings because the baseline never reads them.
def _ex(i, label):
    return Example(id=str(i), question="q", contexts=["c"], long_answer="a", final_decision=label)


# majority_label must return the most frequent label, and break ties in a fixed way.
def test_majority_label_and_ties():
    # Two "no" vs one "yes" -> "no" is the majority.
    assert majority_label([_ex(1, "no"), _ex(2, "no"), _ex(3, "yes")]) == "no"
    # One "maybe" vs one "yes" is a tie; the earlier label in LABELS (yes, no, maybe) wins.
    assert majority_label([_ex(1, "maybe"), _ex(2, "yes")]) == "yes"  # tie: LABELS order


# Every prediction must carry the chosen label and the fixed baseline metadata.
def test_majority_predictions_are_valid():
    # Two examples with different gold labels; the baseline must ignore gold and say "yes".
    preds = majority_predictions(
        [_ex(1, "no"), _ex(2, "yes")], "yes", system="majority", cfg_hash="abc"
    )
    # One prediction per example, both "yes".
    assert [p.prediction for p in preds] == ["yes", "yes"]
    # The baseline is deterministic, so it is always run 1, and model is set to "majority".
    assert all(p.run == 1 and p.model == "majority" for p in preds)


# Runs the full command on a temporary dataset and checks the written prediction file.
# tmp_path: built-in pytest fixture giving an empty temporary folder.
# processed_dir: fixture from conftest.py with train = 2,4,6 and dev = 5 (synthetic data).
def test_command_writes_dev_predictions(tmp_path, processed_dir):
    # Path for a minimal YAML config file.
    cfg = tmp_path / "majority.yaml"
    # The config only needs the system name, which also names the output folder.
    cfg.write_text("system: majority\n", encoding="utf-8")
    # Call the command with command-line style arguments; it returns the output file path.
    # Everything points into temporary folders so real data and results are not touched.
    out = main(
        [
            "--config",
            str(cfg),
            "--split",
            "dev",
            "--processed-dir",
            str(processed_dir),
            "--out-dir",
            str(tmp_path / "pred"),
        ]
    )
    # Read the written predictions back (this also validates every row).
    preds = load_predictions(out)
    # There must be one prediction per dev example, in the same order as the dev split.
    assert [p.id for p in preds] == [ex.id for ex in load_split("dev", processed_dir=processed_dir)]
