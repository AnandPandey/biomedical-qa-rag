# Tests for the data layer: turning raw PubMedQA entries into Example records, loading splits
# (with the protected test split), and creating the fixed train/dev/test split lists.

# shutil: standard-library file utilities; shutil.copy copies a file to a new location.
import shutil

# Counter counts how many times each value appears, e.g. Counter(["a", "a", "b"]) -> a:2, b:1.
from collections import Counter

# pytest: the test framework; pytest.raises checks that an expected error is raised.
import pytest

# to_example converts one raw entry to an Example; load_split reads a split from disk;
# gold_labels maps example id -> correct answer for a split.
from bmqa.data.pubmedqa import gold_labels, load_split, to_example

# The command-line entry point of the splits module, renamed so the name "main" is clearer.
from bmqa.data.splits import main as splits_main

# make_splits computes the split lists; read_ids reads one ID-per-line text file.
from bmqa.data.splits import make_splits, read_ids

# Path to the tests/fixtures folder, defined in conftest.py (pytest puts tests/ on the
# import path, so conftest can be imported like a normal module).
from conftest import FIXTURES


# to_example must copy each raw field into the right Example attribute.
# pqal_sample is a fixture from conftest.py: pytest passes in the 6 synthetic entries.
def test_to_example_maps_fields(pqal_sample):
    # Convert the first synthetic entry (PMID 90000001, label "yes").
    ex = to_example("90000001", pqal_sample["90000001"])
    # The id is the PMID string we passed in.
    assert ex.id == "90000001"
    # QUESTION -> question (the fixture text starts with "Synthetic question 1").
    assert ex.question.startswith("Synthetic question 1")
    # CONTEXTS -> contexts: the two abstract sentences, in order.
    assert ex.contexts == ["Synthetic background 1.", "Synthetic results 1: symptom B fell."]
    # final_decision -> final_decision: the gold label is "yes" in the fixture.
    assert ex.final_decision == "yes"
    # LONG_ANSWER -> long_answer: the fixture marks it with a recognisable tag.
    assert "SYNTHETIC-CONCLUSION-1" in ex.long_answer
    # The conclusion (which gives away the answer) must not leak into the contexts.
    assert all("SYNTHETIC-CONCLUSION" not in c for c in ex.contexts)


# load_split and gold_labels must read back what the processed_dir fixture wrote.
# processed_dir (conftest.py) is a temp folder with train = 2,4,6; dev = 5; test = 1,3.
def test_load_split_returns_examples(processed_dir):
    # Load the dev split from the temporary folder instead of the real data/processed.
    dev = load_split("dev", processed_dir=processed_dir)
    # Dev should contain exactly one example, PMID 90000005.
    assert [ex.id for ex in dev] == ["90000005"]
    # Train gold labels come straight from the fixture: 2 is yes, 4 is no, 6 is maybe.
    assert gold_labels("train", processed_dir=processed_dir) == {
        "90000002": "yes",
        "90000004": "no",
        "90000006": "maybe",
    }


# The test split is protected so it cannot be used by accident while developing.
def test_test_split_is_guarded(processed_dir):
    # Without allow_test=True, loading "test" must raise PermissionError.
    with pytest.raises(PermissionError):
        load_split("test", processed_dir=processed_dir)
    # gold_labels uses the same guard, so the test answers cannot be peeked at either.
    with pytest.raises(PermissionError):
        gold_labels("test", processed_dir=processed_dir)
    # With the explicit opt-in it works and returns the 2 test examples (PMIDs 1 and 3).
    assert len(load_split("test", allow_test=True, processed_dir=processed_dir)) == 2


# A misspelt or unknown split name must fail loudly instead of returning nothing.
def test_unknown_split_is_rejected(processed_dir):
    # "validation" is not one of train/dev/test, so ValueError is expected.
    with pytest.raises(ValueError):
        load_split("validation", processed_dir=processed_dir)


# Helper (not a test: the name does not start with "test_") that builds 100 fake labels:
# 60 yes, 30 no and 10 maybe, with ids like "yes000", "no012", "maybe009".
def _synthetic_labels():
    # Start with an empty {id: label} dict.
    labels = {}
    # Loop over each label together with how many ids it should get.
    for label, n in (("yes", 60), ("no", 30), ("maybe", 10)):
        # Create n ids for this label.
        for i in range(n):
            # {i:03d} pads the number to 3 digits (0 -> "000"), so ids sort nicely.
            labels[f"{label}{i:03d}"] = label
    # Return the finished dict of 100 ids.
    return labels


# make_splits must be repeatable (same seed -> same result), stratified (each label is split
# in the same proportion) and disjoint (no id in two splits).
def test_splits_are_deterministic_stratified_and_disjoint():
    # 100 fake labelled ids from the helper above.
    labels = _synthetic_labels()
    # Pretend one id of each label is in the official test set.
    test_ids = {"yes000", "no000", "maybe000"}
    # Compute the splits once with the default seed.
    first = make_splits(labels, test_ids)
    # Deterministic: running again with the same inputs gives identical lists.
    assert first == make_splits(labels, test_ids)
    # The seed matters: a different seed shuffles differently, so the lists change.
    assert first != make_splits(labels, test_ids, seed=7)

    # The test split is exactly the given test ids (sorted), never reshuffled.
    assert first["test"] == sorted(test_ids)
    # Join all three lists together to check for overlap and coverage.
    all_ids = first["train"] + first["dev"] + first["test"]
    # No duplicates (list length == set length) and every one of the 100 ids is used.
    assert len(all_ids) == len(set(all_ids)) == len(labels)

    # Count how many dev ids have each label.
    dev_counts = Counter(labels[i] for i in first["dev"])
    # 20 % of the non-test ids of each label: 59 -> 12, 29 -> 6, 9 -> 2
    # (0.2 * 59 = 11.8 rounds to 12; 0.2 * 29 = 5.8 -> 6; 0.2 * 9 = 1.8 -> 2.)
    assert dev_counts == Counter({"yes": 12, "no": 6, "maybe": 2})


# A test id that does not exist in the labelled data points to a data error, so it must fail.
def test_unknown_test_ids_are_rejected():
    # "missing" is not a key of the labels dict -> ValueError.
    with pytest.raises(ValueError):
        make_splits({"a": "yes"}, {"missing"})


# Runs the whole split command on the fixture files, then checks that saved split lists are
# frozen: a second run must agree with them, and an edited list must be refused.
def test_split_command_end_to_end_and_frozen(tmp_path):
    # Make a fake "data/raw" folder inside the temporary folder.
    raw = tmp_path / "raw"
    # Actually create that folder on disk.
    raw.mkdir()
    # Copy the synthetic files in under the real file names the command expects.
    shutil.copy(FIXTURES / "ori_pqal_sample.json", raw / "ori_pqal.json")
    shutil.copy(FIXTURES / "test_ground_truth_sample.json", raw / "test_ground_truth.json")
    # Command-line arguments, as a list of strings, pointing every folder into tmp_path
    # so the real data/ folder is never touched.
    args = [
        "--raw-dir",
        str(raw),
        "--splits-dir",
        str(tmp_path / "splits"),
        "--processed-dir",
        str(tmp_path / "processed"),
    ]
    # First run: computes the splits and writes the *_ids.txt files and processed files.
    splits_main(args)
    # The test ids come from test_ground_truth_sample.json, which lists PMIDs 1 and 3.
    assert read_ids(tmp_path / "splits" / "test_ids.txt") == ["90000001", "90000003"]
    splits_main(args)  # second run: same lists, accepted

    # Simulate someone editing the committed dev list by hand.
    (tmp_path / "splits" / "dev_ids.txt").write_text("90000002\n", encoding="utf-8")
    with pytest.raises(RuntimeError):  # committed splits must never change
        # The fresh computation no longer matches the saved file, so the command refuses.
        splits_main(args)
