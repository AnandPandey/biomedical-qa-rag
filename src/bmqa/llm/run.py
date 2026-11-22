"""Generic LLM runner: examples + evidence function -> Prediction rows and a usage log.

Usage:
    python -m bmqa.llm.run --config configs/closed_book.yaml --split dev --run 1
    python -m bmqa.llm.run --config configs/oracle.yaml --split dev --run 1 --dry-run
Phase 3 reuses `execute` / `run_system` with its own retrieval evidence function.
"""

# Allows modern type hints such as `int | None` without evaluating them at runtime.
from __future__ import annotations

# argparse reads command-line options such as --config and --split.
import argparse

# json writes the usage log file.
import json

# Callable is used to describe the type "a function that takes X and returns Y".
from collections.abc import Callable

# datetime/UTC record when a run started and finished, independent of time zone.
from datetime import UTC, datetime

# Path represents file and folder paths.
from pathlib import Path

# load_config reads a YAML config into a dict; config_hash gives a short ID of its content.
from bmqa.config import config_hash, load_config

# PROCESSED_DIR = data/processed; load_split loads the examples of one split;
# warn_test_access prints a warning whenever the protected test split is used.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split, warn_test_access

# "PARSE_FAILURE:" - the marker the scorer looks for to count an answer as wrong.
from bmqa.eval.metrics import PARSE_FAILURE_PREFIX

# Writes a list of Prediction objects to a JSON Lines file.
from bmqa.io import write_predictions

# Builds the fake or real LLM client from the config.
from bmqa.llm.client import make_client

# Turns the model's reply text into label, rationale and citations.
from bmqa.llm.parse import parse_answer

# Builds the prompt from the question and the evidence documents.
from bmqa.llm.prompts import build_prompt

# Shared data types: valid split names, one question (Example), one evidence document (Hit)
# and one system answer (Prediction).
from bmqa.schema import SPLITS, Example, Hit, Prediction

# Type name for an "evidence function": given an Example, it returns a list of
# (Hit, document text) pairs. Each system differs only in which evidence function it uses.
EvidenceFn = Callable[[Example], list[tuple[Hit, str]]]  # (hit, document text) pairs

# Stop reasons whose text cannot be trusted as a complete answer.
# "max_tokens" = the reply was cut off; "refusal" = the model declined to answer.
FAILED_STOP_REASONS = {"max_tokens", "refusal"}
PLACEHOLDER_LABEL = "maybe"  # stored for parse failures; the scorer counts them as wrong


# Evidence function for the closed-book system: no documents at all.
# Input: one Example (unused). Output: an empty list.
def no_evidence(example: Example) -> list[tuple[Hit, str]]:
    return []


# Evidence function for the oracle system: give the model the question's own abstract.
# Input: one Example. Output: a list with one (Hit, text) pair.
def gold_evidence(example: Example) -> list[tuple[Hit, str]]:
    """The example's own abstract contexts as one document (oracle upper bound)."""
    # The Hit points at the example itself (rank 1, score 1.0); the text joins all context
    # paragraphs with spaces. long_answer (the conclusion) is NOT used, so the answer never
    # leaks into the prompt.
    return [(Hit(doc_id=example.id, score=1.0, rank=1), " ".join(example.contexts))]


# Maps the `evidence:` value in a YAML config to the function to use.
EVIDENCE_FUNCTIONS: dict[str, EvidenceFn] = {"none": no_evidence, "gold": gold_evidence}


# Core loop: ask the LLM about every example and collect Prediction rows plus usage counts.
# Inputs: the examples, a client (fake or real), an evidence function, and (keyword-only,
# after `*`) the system name, prompt version, config hash, run number and max_tokens.
# Output: (list of Predictions in the same order as the examples, usage summary dict).
def run_system(
    examples: list[Example],
    client,
    evidence_fn: EvidenceFn,
    *,
    system: str,
    prompt_version: str,
    config_hash: str,
    run: int,
    max_tokens: int,
) -> tuple[list[Prediction], dict]:
    """Return predictions (in input order) and a usage summary."""
    # Collected predictions, one per example.
    predictions = []
    # Running totals for the usage log.
    usage = {
        "n": 0,  # number of examples answered
        "input_tokens": 0,  # all prompt tokens (cached and new)
        "output_tokens": 0,  # all reply tokens (cached and new)
        "cached_responses": 0,  # replies taken from the cache (free)
        "uncached_input_tokens": 0,  # prompt tokens actually paid for in this run
        "uncached_output_tokens": 0,  # reply tokens actually paid for in this run
        "parse_failures": 0,  # replies we could not turn into a valid answer
        "stop_reasons": {},  # how often each stop reason occurred
    }
    # Process the examples one by one, in order.
    for example in examples:
        # Get this example's evidence documents (none, gold abstract, or retrieved ones).
        docs = evidence_fn(example)
        # Build the prompt from the question and the document texts only (the Hits are not
        # shown to the model; `_` ignores the Hit part of each pair).
        prompt = build_prompt(example.question, [text for _, text in docs], prompt_version)
        # Ask the LLM (or read the cached reply).
        response = client.generate(prompt, max_tokens=max_tokens)
        # Read the client's last stop reason; getattr(..., None) is safe if it has none.
        stop = getattr(client, "last_stop_reason", None)
        # Parse the reply; n_docs lets the parser flag citations to non-existent documents.
        parsed = parse_answer(response.text, n_docs=len(docs))
        # Failure if no valid label was found, or if the reply was cut off / refused.
        failed = parsed.label is None or stop in FAILED_STOP_REASONS

        # Update the usage totals.
        usage["n"] += 1
        usage["input_tokens"] += response.input_tokens
        usage["output_tokens"] += response.output_tokens
        # Cached replies cost nothing, so they are counted separately from paid tokens.
        if response.cached:
            usage["cached_responses"] += 1
        else:
            # Only new API calls are added to the tokens used for the cost estimate.
            usage["uncached_input_tokens"] += response.input_tokens
            usage["uncached_output_tokens"] += response.output_tokens
        # Count this stop reason (str() so None becomes the key "None"; .get(..., 0) starts
        # a new reason at zero).
        usage["stop_reasons"][str(stop)] = usage["stop_reasons"].get(str(stop), 0) + 1
        # Adding a bool adds 1 for True and 0 for False.
        usage["parse_failures"] += failed

        # Store one Prediction per example. Failures are KEPT, not dropped: dropping them
        # would hide the model's mistakes and make accuracy look better than it is. They are
        # marked so the scorer counts them as wrong.
        predictions.append(
            Prediction(
                id=example.id,  # which question this answers
                system=system,  # e.g. "closed_book" or "oracle"
                # A Prediction must hold a valid label, so failures get the placeholder.
                prediction=PLACEHOLDER_LABEL if failed else parsed.label,
                # Failures get the "PARSE_FAILURE:" prefix, which the scorer treats as wrong;
                # the original reply is kept after it for debugging.
                raw_output=f"{PARSE_FAILURE_PREFIX} {response.text}" if failed else response.text,
                model=client.model,  # model name for the record
                config_hash=config_hash,  # ties the row to the exact config used
                run=run,  # repetition number
                rationale=None if failed else parsed.rationale,  # no rationale on failures
                evidence=[hit for hit, _ in docs],  # the documents shown, in rank order
                prompt_version=prompt_version,  # which template version was used
            )
        )
    # Hand back all predictions and the usage summary.
    return predictions, usage


# Full pipeline for one system on one split: load data, run, save predictions and usage log.
# Inputs: the config dict, an evidence function and keyword-only options (split, run number,
# optional example limit, dry run, test-split permission, data/results folders and an
# optional ready-made client, which tests use). Output: (predictions file, usage log file).
def execute(
    cfg: dict,
    evidence_fn: EvidenceFn,
    *,
    split: str,
    run: int,
    limit: int | None = None,
    dry_run: bool = False,
    allow_test: bool = False,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path | None = None,
    client=None,
) -> tuple[Path, Path]:
    """Load the split, run the system described by cfg, write predictions and the usage log."""
    # Using the protected test split (allowed only in the final run): print a warning.
    if split == "test" and allow_test:
        warn_test_access()
    # Load the examples; [:limit] keeps the first `limit` ones (all of them if limit is None).
    examples = load_split(split, allow_test, processed_dir)[:limit]
    # The `llm:` block of the config (provider, model, effort, max_tokens).
    llm = cfg["llm"]
    # Use the client passed in (tests), otherwise build one from the config.
    client = client or make_client(llm, run, dry_run=dry_run)
    # Choose the output folder: dry runs go to outputs/dry_run so fake answers never mix
    # with real results; real runs go to results/.
    results_dir = results_dir or (Path("outputs/dry_run") if dry_run else Path("results"))
    # Start time as ISO text in UTC, for the log.
    started = datetime.now(UTC).isoformat(timespec="seconds")

    # Run the system over all examples.
    predictions, usage = run_system(
        examples,
        client,
        evidence_fn,
        system=cfg["system"],
        prompt_version=cfg["prompt_version"],
        config_hash=config_hash(cfg),  # short hash identifying this exact config
        run=run,
        max_tokens=llm["max_tokens"],
    )
    # A dry run makes no API calls, so it costs nothing.
    # Otherwise read prices per million tokens from the config ({} if missing).
    prices = {} if dry_run else cfg.get("price_per_million_tokens") or {}
    # Estimated cost in US dollars: paid tokens times price per million, rounded to 4 decimals.
    # Cached tokens are left out because they were not paid for in this run.
    usage["estimated_cost_usd"] = round(
        usage["uncached_input_tokens"] * prices.get("input", 0) / 1e6
        + usage["uncached_output_tokens"] * prices.get("output", 0) / 1e6,
        4,
    )
    # Everything worth knowing about this run, saved as the usage log.
    log = {
        "system": cfg["system"],
        "split": split,
        "run": run,
        "model": client.model,
        "effort": llm.get("effort"),  # .get() so a missing effort gives None, not an error
        "prompt_version": cfg["prompt_version"],
        "config_hash": config_hash(cfg),
        "dry_run": dry_run,
        "limit": limit,
        "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        **usage,  # copy in all usage counters (tokens, failures, cost...)
    }
    # e.g. results/predictions/closed_book/dev_run1.jsonl
    pred_path = results_dir / "predictions" / cfg["system"] / f"{split}_run{run}.jsonl"
    # e.g. results/runs/closed_book_dev_run1.json
    log_path = results_dir / "runs" / f"{cfg['system']}_{split}_run{run}.json"
    # Write the predictions (this helper creates the folders itself).
    write_predictions(pred_path, predictions)
    # Make sure the log folder exists, then write the log as indented JSON.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    # Short summary on screen: answers, cache hits, failures and estimated cost.
    print(
        f"{cfg['system']} {split} run {run}: {usage['n']} answers, "
        f"{usage['cached_responses']} from cache, {usage['parse_failures']} parse failures, "
        f"~${usage['estimated_cost_usd']}"
    )
    # Show where the two output files were written.
    print(f"  predictions: {pred_path}\n  usage log:   {log_path}")
    # Return both paths (tests and later phases use them).
    return pred_path, log_path


# Add the command-line options shared by all LLM runners (Phase 3 reuses this too).
# Input: an argparse parser, which is modified in place. Returns nothing.
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    # --config: path to the YAML experiment config (required).
    parser.add_argument("--config", type=Path, required=True)
    # --split: which data split to answer; must be train/dev/test, default dev.
    parser.add_argument("--split", choices=SPLITS, default="dev")
    # --run: repetition number (1, 2, 3); also part of the cache key and output file names.
    parser.add_argument("--run", type=int, default=1)
    # --limit: answer only the first N examples, for a quick, cheap check.
    parser.add_argument("--limit", type=int, help="only the first N examples (smoke test)")
    # --dry-run: a flag (no value); uses the fake client, so no key is needed and no money spent.
    parser.add_argument("--dry-run", action="store_true", help="fake client, no API key, no cost")
    # --allow-test: a flag that unlocks the protected test split (final Phase 4 run only).
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")


# Load settings (such as ANTHROPIC_API_KEY) from a local .env file into the environment.
def load_env() -> None:
    """Read a local .env file if python-dotenv is installed (the `llm` extra)."""
    # python-dotenv is optional, so try to import it.
    try:
        from dotenv import load_dotenv
    # Not installed: silently do nothing (environment variables may be set another way).
    except ImportError:
        return
    # Copy the variables from .env into the environment. Never print them: .env holds the
    # secret API key.
    load_dotenv()


# Command-line entry point. Input: an optional list of arguments (tests can pass their own;
# None means "read from the real command line"). Output: the two written file paths.
def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    # Create the parser with a short description shown by --help.
    parser = argparse.ArgumentParser(description="Run an LLM system (closed_book or oracle).")
    # Register the shared options.
    add_common_arguments(parser)
    # Read and check the command-line arguments.
    args = parser.parse_args(argv)
    # Load the API key etc. from .env, if present.
    load_env()
    # Read the YAML config file into a dict.
    cfg = load_config(args.config)
    # Pick the evidence function named in the config ("none" or "gold").
    evidence_fn = EVIDENCE_FUNCTIONS[cfg["evidence"]]
    # Run everything with the chosen options and return the output paths.
    return execute(
        cfg,
        evidence_fn,
        split=args.split,
        run=args.run,
        limit=args.limit,
        dry_run=args.dry_run,
        allow_test=args.allow_test,
    )


# Runs main() only when this file is started as a program (python -m bmqa.llm.run),
# not when it is imported by another module.
if __name__ == "__main__":
    main()
