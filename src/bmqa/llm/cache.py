"""On-disk cache of LLM responses: one JSON file per request.

The run number is part of the key, so repeated runs (run 1, 2, 3) make independent calls while
re-running the same run is free and returns identical text.
"""

# Lets us write type hints like `dict | None` and `str | Path` as plain text annotations,
# which are not evaluated at runtime (keeps hints simple and works on older Pythons too).
from __future__ import annotations

# hashlib gives us SHA-256, used to turn a request into a short, fixed-length file name.
import hashlib

# json converts Python dicts to text (to store on disk) and back again.
import json

# Path is an object-oriented way to build and use file paths that works on Windows and Linux.
from pathlib import Path

# Default folder where cached replies are stored. It is relative to the folder you run from
# (the project root), and lives under outputs/ so it is never mixed with source code.
DEFAULT_CACHE_DIR = Path("outputs/llm_cache")


# Build the cache key: a unique "fingerprint" string for one LLM request.
# Inputs (keyword-only, because of the `*`): model name, effort level, max_tokens, the full
# prompt text and the run number. Output: a 64-character hexadecimal SHA-256 string.
# Every setting that can change the reply is included, so a changed prompt or model never
# returns an old, wrong reply. `run` is included on purpose: run 1, 2 and 3 must be
# independent samples of the model, so each run gets its own key (and its own API call),
# while repeating the SAME run hits the cache and gives identical text for free.
def cache_key(
    *,
    model: str,
    effort: str,
    max_tokens: int,
    prompt: str,
    run: int,
    provider: str | None = None,
    extra: dict | None = None,
) -> str:
    # Collect all the request settings in one dictionary.
    payload = {
        "model": model,  # which model answers
        "effort": effort,  # how much reasoning effort the model spends (e.g. "low", "none")
        "max_tokens": max_tokens,  # upper limit on reply length
        "prompt": prompt,  # the exact text sent to the model
        "run": run,  # repetition number (1, 2, 3...)
    }
    # Optional provider and extra configuration / quantization identity.
    if provider is not None:
        payload["provider"] = provider
    if extra:
        payload.update(extra)
    # Turn the dict into one canonical text form: sorted keys and no extra spaces mean the
    # same settings always give exactly the same text (and therefore the same hash).
    # ensure_ascii=False keeps non-English characters as they are instead of \u escapes.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # Encode the text to bytes, hash it with SHA-256 and return the hash as hex text.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# A tiny key-value store on disk: one JSON file per cached LLM reply.
# Using files (not memory) means the cache survives between program runs, so re-running an
# experiment does not pay for the same API calls twice.
class ResponseCache:
    # Constructor. Input: the folder to store files in (a string or a Path).
    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR):
        # Always store a Path object, even if the caller passed a plain string.
        self.directory = Path(directory)

    # Helper (leading underscore = "internal use"): the file path for one key.
    # Files go into a subfolder named after the first 2 characters of the key, e.g.
    # outputs/llm_cache/ab/abcd....json. This spreads thousands of files over up to 256
    # folders, because one folder with very many files becomes slow to browse.
    def _path(self, key: str) -> Path:
        # `/` joins path parts; key[:2] is the first two characters of the key.
        return self.directory / key[:2] / f"{key}.json"

    # Look up a cached reply. Input: the key. Output: the stored dict, or None if not cached.
    def get(self, key: str) -> dict | None:
        # Work out where this key's file would be.
        path = self._path(key)
        # No file means this request was never made before: report "not cached".
        if not path.exists():
            return None
        # Read the file as UTF-8 text and convert the JSON back into a Python dict.
        return json.loads(path.read_text(encoding="utf-8"))

    # Save a reply. Inputs: the key and the record (a dict) to store. Returns nothing.
    def put(self, key: str, record: dict) -> None:
        """Write atomically, so an interrupted run never leaves a half-written entry."""
        # Work out the final file path for this key.
        path = self._path(key)
        # Create the 2-character subfolder (and parents) if needed; no error if it exists.
        path.parent.mkdir(parents=True, exist_ok=True)
        # A temporary file next to the final one: same name but ending in .tmp.
        tmp = path.with_suffix(".tmp")
        # Write the full record to the temporary file first (indent=1 keeps it readable).
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        # Rename the temp file onto the real name in one step. If the program crashes during
        # the write above, only the .tmp file is broken; the real cache file is never half-full.
        tmp.replace(path)
