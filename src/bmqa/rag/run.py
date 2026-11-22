"""Run a RAG system: retrieve abstracts, then the same prompt/LLM as closed_book.

Usage:
    python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --run 1
    python -m bmqa.rag.run --config configs/rag_strict_bm25.yaml --split dev --k 3   # ablation
With --k the system is named {system}_k{K}, so ablation runs never overwrite each other.
"""

# Lets us write type hints such as "int | None" on every supported Python version.
from __future__ import annotations

# argparse reads command-line options such as --config and --k.
import argparse

# Path is an object-oriented way to work with file and folder paths.
from pathlib import Path

# load_config reads a YAML config file into a Python dict.
from bmqa.config import load_config

# From the Phase 2 runner (reused unchanged so RAG and closed_book differ only in evidence):
# add_common_arguments adds --config/--split/--run/--limit/--dry-run/--allow-test;
# execute runs the LLM over a split and writes predictions; load_env reads a local .env file.
from bmqa.llm.run import add_common_arguments, execute, load_env

# Builds the function that retrieves abstracts for each question.
from bmqa.rag.evidence import make_retrieval_evidence


# Applies an optional --k override to a loaded config.
# Inputs: cfg = config dict; k_override = the --k value, or None if not given.
# Output: the config to use (the original if no override, otherwise a modified copy).
def resolve_config(cfg: dict, k_override: int | None) -> dict:
    """Apply a --k override; the system name and config hash then reflect the new k."""
    # No --k given: use the config exactly as written in the YAML file.
    if k_override is None:
        return cfg
    # Return a NEW dict ({**cfg} copies all keys) with k replaced and "_k<K>" added to the
    # system name, e.g. rag_strict -> rag_strict_k3. Results are saved in a folder named after
    # the system, so each k in the ablation gets its own folder and never overwrites another
    # run (or the main rag_strict results). The original cfg is left untouched.
    return {**cfg, "k": k_override, "system": f"{cfg['system']}_k{k_override}"}


# Command-line entry point: runs one RAG system on one split.
# Input: argv = list of command-line arguments (None means "use the real command line").
# Output: (path of the predictions file, path of the usage log) as returned by execute().
def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    # Imported here because they need the optional bm25s library; resolve_config and the
    # tests that use it work without it.
    from bmqa.retrieval.bm25 import BM25Retriever
    from bmqa.retrieval.store import DocStore

    # Create the command-line parser; the description appears in --help.
    parser = argparse.ArgumentParser(description="Run a RAG system (rag_strict or rag_gold).")
    # Add the same options as the Phase 2 runner (--config, --split, --run, ...).
    add_common_arguments(parser)
    # --k: optional; overrides how many documents are retrieved (used for the k ablation).
    parser.add_argument("--k", type=int, help="override the number of retrieved documents")
    # Read the options the user typed (or the argv list passed in by a test).
    args = parser.parse_args(argv)
    # Load the API key etc. from a .env file if one exists.
    load_env()

    # Load the YAML config and apply the --k override (if any).
    cfg = resolve_config(load_config(args.config), args.k)
    # The "retriever" section of the config (type, corpus, index_dir).
    retriever_cfg = cfg["retriever"]
    # Only BM25 is implemented; stop with a clear error for anything else.
    if retriever_cfg["type"] != "bm25":
        # !r shows the value with quotes, so typos and empty strings are easy to spot.
        raise ValueError(f"unsupported retriever type {retriever_cfg['type']!r}")
    # Build the evidence function: load the BM25 index, load the corpus texts, use k docs.
    evidence_fn = make_retrieval_evidence(
        BM25Retriever(retriever_cfg["index_dir"]), DocStore(retriever_cfg["corpus"]), cfg["k"]
    )
    # Hand everything to the unchanged Phase 2 runner and return its two output paths.
    return execute(
        # The (possibly renamed) config: system name, llm settings, prompt version, prices.
        cfg,
        # Where the documents for each question come from (here: retrieval).
        evidence_fn,
        # Which split to run on (dev by default; test is blocked unless --allow-test).
        split=args.split,
        # Run number, so repeated runs of the same system are stored separately.
        run=args.run,
        # Optional: only the first N questions (quick smoke test).
        limit=args.limit,
        # If True, a fake LLM is used: no API key needed and no cost.
        dry_run=args.dry_run,
        # Must be set explicitly to touch the test split (only for the final frozen run).
        allow_test=args.allow_test,
    )


# Runs main() only when this file is executed directly (python -m bmqa.rag.run),
# not when it is imported by another module or a test.
if __name__ == "__main__":
    main()
