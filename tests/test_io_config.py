# Tests for bmqa.io (reading/writing JSON Lines and prediction files) and bmqa.config
# (loading YAML experiment configs and hashing them). pytest finds this file because its name
# starts with "test_", and runs every function whose name starts with "test_".

# json: standard-library module to turn Python dicts into JSON text (used to write a bad row).
import json

# pytest: the test framework; here we use pytest.raises to check that errors are raised.
import pytest

# config_hash makes a short fingerprint of a config dict; load_config reads a YAML file.
from bmqa.config import config_hash, load_config

# The JSON Lines helpers under test: plain dict rows and typed Prediction rows.
from bmqa.io import load_predictions, read_jsonl, write_jsonl, write_predictions

# Hit (one retrieved document) and Prediction (one system answer) are the shared record types.
from bmqa.schema import Hit, Prediction


# Writing rows and reading them back must give exactly the same data ("round trip").
# tmp_path is a built-in pytest fixture: a fresh, empty temporary folder for this test only.
def test_jsonl_round_trip(tmp_path):
    # Two simple rows; the first contains a non-ASCII character to prove UTF-8 is handled.
    rows = [{"a": 1, "text": "é unicode"}, {"a": 2, "text": "second"}]
    # The "nested" sub-folder does not exist yet, so this also checks that write_jsonl
    # creates missing parent folders.
    path = tmp_path / "nested" / "rows.jsonl"
    # Write the rows, one JSON object per line.
    write_jsonl(path, rows)
    # Reading the file back must return a list equal to the original rows.
    assert read_jsonl(path) == rows


# Same round-trip idea for Prediction objects, including the nested evidence list.
def test_predictions_round_trip(tmp_path):
    # Build one valid Prediction; every required field is filled with a small dummy value.
    # evidence holds one Hit with rank=1, which satisfies the "ranks start at 1" rule.
    pred = Prediction(
        id="1",
        system="rag_strict",
        prediction="no",
        raw_output="{}",
        model="m",
        config_hash="abc",
        evidence=[Hit(doc_id="d", score=1.5, rank=1)],
    )
    # Where the prediction file will be written inside the temporary folder.
    path = tmp_path / "p.jsonl"
    # Save the list of predictions (here just one) as JSON Lines.
    write_predictions(path, [pred])
    # Loading must rebuild an equal Prediction, including the Hit inside evidence.
    assert load_predictions(path) == [pred]


# load_predictions must refuse a row that breaks the contract, instead of loading bad data.
def test_load_predictions_rejects_invalid_row(tmp_path):
    # File that will hold the invalid row.
    path = tmp_path / "bad.jsonl"
    # A row with all required fields, but "unsure" is not an allowed label (yes/no/maybe).
    row = {
        "id": "1",
        "system": "s",
        "prediction": "unsure",
        "raw_output": "",
        "model": "m",
        "config_hash": "c",
    }
    # Write the row by hand (bypassing Prediction, which would reject it immediately).
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    # pytest.raises(ValueError) is a context manager: the test passes only if the code inside
    # the "with" block raises ValueError; if nothing is raised, the test fails.
    with pytest.raises(ValueError):
        # Prediction's validation raises ValueError for the unknown label.
        load_predictions(path)


# The config hash identifies an experiment, so it must depend on content, not on key order.
def test_config_hash_ignores_key_order_but_not_values(tmp_path):
    # Same keys and values written in a different order -> same hash (keys are sorted first).
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    # A different value -> a different hash, so changed settings are noticed.
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    # The hash is the first 12 hex characters of a SHA-256 digest, even for an empty config.
    assert len(config_hash({})) == 12


# load_config reads a YAML mapping, and rejects YAML whose top level is not a mapping.
def test_load_config(tmp_path):
    # Temporary YAML file for this test.
    path = tmp_path / "c.yaml"
    # Write a small valid config: a string value and a list value.
    path.write_text("system: majority\nseeds: [0, 1]\n", encoding="utf-8")
    # The YAML must come back as the equivalent Python dict.
    assert load_config(path) == {"system": "majority", "seeds": [0, 1]}
    # Overwrite the file with a YAML list ("- item"), which is not a key/value mapping.
    path.write_text("- not a mapping\n", encoding="utf-8")
    # load_config must raise ValueError because a config has to be a mapping (dict).
    with pytest.raises(ValueError):
        load_config(path)
