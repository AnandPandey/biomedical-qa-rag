"""Download PQA-L and the official test list from the PubMedQA repository at a pinned commit.

Usage:  python -m bmqa.data.download [--raw-dir data/raw] [--force]
"""

# Makes type hints like `str | Path` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --force.
import argparse

# hashlib: provides SHA-256, used to check that downloaded files are exactly right.
import hashlib

# json: used to open the downloaded JSON files and count their entries.
import json

# urllib.request: Python's built-in tool for downloading a file from a URL.
import urllib.request

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# The exact Git commit of the PubMedQA GitHub repository we download from. "Pinning" a commit
# means everyone gets the same files forever, even if the authors later change the repository.
COMMIT = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
# Base web address of the raw data files at that commit (f-string inserts COMMIT).
BASE_URL = f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{COMMIT}/data"
# Default local folder where the downloaded (unchanged, "raw") files are stored.
RAW_DIR = Path("data/raw")

# SHA-256 of the files at COMMIT (recorded 2026-11-21). A mismatch means the data changed.
# FILES maps each file name to its expected SHA-256 checksum (a 64-character fingerprint).
# ori_pqal.json: the 1,000 expert-labelled questions (PQA-L).
# test_ground_truth.json: the official list of test question IDs with their answers.
FILES = {
    "ori_pqal.json": "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d",
    "test_ground_truth.json": "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
}


# sha256_file: compute the SHA-256 checksum of a file.
# Input: path to the file. Output: the checksum as a 64-character hex string.
# Two files have the same checksum only if their bytes are identical.
def sha256_file(path: str | Path) -> str:
    # Start an empty SHA-256 calculation.
    digest = hashlib.sha256()
    # Open the file in binary mode ("rb"): we hash the exact bytes, not decoded text.
    with open(path, "rb") as f:
        # Read 1 MiB at a time (1 << 20 = 1,048,576 bytes) so big files do not fill memory.
        # iter(func, b"") keeps calling func until it returns b"" (end of file).
        for chunk in iter(lambda: f.read(1 << 20), b""):
            # Feed this chunk into the running checksum.
            digest.update(chunk)
    # Return the final checksum written as hexadecimal text.
    return digest.hexdigest()


# download: fetch the data files and check them.
# Inputs: target folder, and force=True to download again even if files exist.
# Output: dict mapping each file name to its local path.
def download(raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Fetch every file in FILES into raw_dir and verify its checksum."""
    # Create the target folder (and parents) if it does not exist yet.
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Collect name -> local path here.
    paths = {}
    # Handle each expected file and its expected checksum.
    for name, expected in FILES.items():
        # Local path where this file should live.
        path = raw_dir / name
        # Download only if asked to (force) or if the file is not there yet.
        if force or not path.exists():
            # Download first to a temporary ".part" file, so a broken download never
            # leaves a half-written file under the real name.
            tmp = path.with_suffix(".part")
            # Fetch the file from GitHub and save it to the temporary path.
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", tmp)
            # Download finished: rename the temporary file to the real name in one step.
            tmp.replace(path)
        # Compute the checksum of the local file (new or already present).
        actual = sha256_file(path)
        # Verify the checksum: if it differs, the file is corrupted or not the pinned
        # version, and every result built on it would be untrustworthy.
        if actual != expected:
            # Stop immediately with both checksums shown.
            raise RuntimeError(f"{path}: SHA-256 {actual} does not match the pinned {expected}")
        # The file is verified; remember where it is.
        paths[name] = path
    # Return all verified file paths.
    return paths


# main: the command-line entry point.
# Input: optional argument list (used by tests). Output: nothing; prints a summary.
def main(argv: list[str] | None = None) -> None:
    # Create the parser; the description is the first line of this module's docstring.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --raw-dir: where to store the downloaded files (default data/raw).
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    # --force: a switch; when given, download again even if the files already exist.
    parser.add_argument("--force", action="store_true", help="download even if files exist")
    # Read the actual arguments.
    args = parser.parse_args(argv)

    # Download (if needed) and verify every file.
    paths = download(args.raw_dir, args.force)
    # Print a short report for each file.
    for path in paths.values():
        # Open the JSON file as UTF-8 text.
        with open(path, encoding="utf-8") as f:
            # json.load gives a dict keyed by PMID; len() is the number of questions.
            print(f"{path}: {len(json.load(f))} entries, checksum OK")


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
