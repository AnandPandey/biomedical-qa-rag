"""Experiment configs: YAML files under configs/, identified by a short content hash."""

# Makes type hints like `str | Path` work as plain text on older Python versions.
from __future__ import annotations

# hashlib: provides SHA-256, used to build a fingerprint of a config.
import hashlib

# json: used to turn the config into a fixed, canonical text form before hashing.
import json

# Path: an object-oriented way to work with file paths.
from pathlib import Path

# yaml (the PyYAML package): reads YAML config files into Python dicts.
import yaml


# load_config: read one YAML config file.
# Input: path to the .yaml file. Output: the settings as a dict.
def load_config(path: str | Path) -> dict:
    # Open the YAML file as UTF-8 text; `with` closes it automatically.
    with open(path, encoding="utf-8") as f:
        # safe_load only builds plain data (dicts, lists, strings, numbers), never
        # arbitrary Python objects, so a config file cannot run code.
        cfg = yaml.safe_load(f)
    # A config must be a key: value mapping; an empty file or a plain list is a mistake.
    if not isinstance(cfg, dict):
        # Stop with a clear message naming the bad file.
        raise ValueError(f"{path} must contain a YAML mapping")
    # Give the settings back to the caller.
    return cfg


# config_hash: make a short ID that changes whenever any config value changes.
# Input: the config dict. Output: a 12-character hexadecimal string.
# Every prediction stores this hash, so we can always tell which settings produced it.
def config_hash(cfg: dict) -> str:
    """First 12 hex characters of the SHA-256 of the canonical JSON form of `cfg`."""
    # Turn the dict into JSON with sorted keys and no extra spaces. This "canonical" form
    # means the same settings always give the same text, whatever order the YAML used.
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    # Hash the UTF-8 bytes with SHA-256, write it as hex, and keep the first 12 characters
    # (short enough to read, long enough that two configs practically never collide).
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
