"""Shared test fixtures. All data here is synthetic; tests never need the real dataset."""

# conftest.py is a special file name for pytest: pytest loads it automatically before running
# the tests in this folder, so every fixture defined here can be used by every test file
# without importing it.

# Path is the standard-library class for building and joining file-system paths safely.
from pathlib import Path

# pytest is the testing framework; we need it here for the @pytest.fixture decorator.
import pytest

# load_pqal reads a PubMedQA JSON file into a dict; write_processed writes the per-split
# .jsonl files that the rest of the code loads. Both are the real project functions.
from bmqa.data.pubmedqa import load_pqal, write_processed

# Folder holding the small, fake (synthetic) PubMedQA files used by the tests.
# __file__ is this file's path, .parent is the tests/ folder, and / joins "fixtures" onto it.
FIXTURES = Path(__file__).parent / "fixtures"


# A "fixture" is a helper that prepares data for tests. A test asks for it simply by naming
# a parameter with the same name (e.g. def test_x(pqal_sample): ...). pytest then calls this
# function and passes the return value in. It runs fresh for every test that uses it.
@pytest.fixture
def pqal_sample() -> dict[str, dict]:
    # Load the 6 fake entries (PMIDs 90000001-90000006) as {pmid: raw entry dict}.
    # Labels in the file: 1 yes, 2 yes, 3 no, 4 no, 5 maybe, 6 maybe.
    return load_pqal(FIXTURES / "ori_pqal_sample.json")


# This fixture builds a temporary "data/processed" folder so tests can load splits from disk.
# It asks for two other fixtures: tmp_path (built into pytest: a brand-new empty temporary
# folder per test, deleted later, so tests never touch the real data/ folder) and pqal_sample
# (defined above). Fixtures can depend on other fixtures in this way.
@pytest.fixture
def processed_dir(tmp_path, pqal_sample) -> Path:
    """data/processed-style folder: test = 90000001, 90000003; dev = 90000005; rest train."""
    # Hand-picked split assignment so tests know exactly which IDs land in which split:
    # train gets one of each label (2 = yes, 4 = no, 6 = maybe), dev gets 5 (maybe),
    # and test gets 1 (yes) and 3 (no), matching test_ground_truth_sample.json.
    splits = {
        "train": ["90000002", "90000004", "90000006"],
        "dev": ["90000005"],
        "test": ["90000001", "90000003"],
    }
    # Write train.jsonl, dev.jsonl and test.jsonl into the temporary folder.
    write_processed(pqal_sample, splits, tmp_path)
    # Give the folder path to the test, which passes it as processed_dir=... to the loaders.
    return tmp_path
