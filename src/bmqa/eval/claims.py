"""Unsupported-claim and fabricated-citation analysis of LLM rationales.

Each rationale sentence (a "claim") is checked with a natural-language-inference (NLI) model
against two premise sets:
  evidence-grounded: the documents the model was shown (none for closed-book);
  gold-grounded:     the example's own abstract sections, for every system.

Usage (GPU recommended, needs the `classifier` extra):
    python -m bmqa.eval.claims --config configs/claims_nli.yaml \
        --pred results/predictions/rag_strict/test_run1.jsonl --split test --allow-test \
        --corpus data/corpus/strict.jsonl
"""

# Big picture: an LLM can give the right yes/no answer but explain it with made-up facts
# ("hallucinations"). This file measures that. Each sentence of the model's explanation
# (rationale) is a "claim". An NLI model reads a premise (a source text) and a hypothesis (the
# claim) and says whether the premise entails (supports), contradicts, or is neutral to it.
# We also count "fabricated citations": the model cites document [3] when it only saw 2.

# Lets type hints such as "str | None" work without being evaluated at runtime.
from __future__ import annotations

# argparse: reads command-line options such as --pred and --split.
import argparse

# json: reads and writes JSON text (the summary file).
import json

# re: regular expressions, used to split text into sentences.
import re

# Callable: type hint for "a function"; used for the premise-provider and scorer arguments.
from collections.abc import Callable

# Path: convenient, OS-independent file paths.
from pathlib import Path

# load_config: reads a YAML config file into a Python dict.
from bmqa.config import load_config

# PROCESSED_DIR: default data folder; load_split: loads examples of one split;
# warn_test_access: prints a loud warning whenever the protected test split is read.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split, warn_test_access

# PARSE_FAILURE_PREFIX = "PARSE_FAILURE:"; marks rows where the LLM reply could not be parsed.
from bmqa.eval.metrics import PARSE_FAILURE_PREFIX

# load_predictions: reads a predictions .jsonl file; write_jsonl: writes rows to .jsonl.
from bmqa.io import load_predictions, write_jsonl

# parse_answer: parses the LLM's JSON reply and reports citations that point to missing docs.
from bmqa.llm.parse import parse_answer

# SPLITS = ("train", "dev", "test"); Prediction = one system answer for one question.
from bmqa.schema import SPLITS, Prediction

# The three labels a claim can get, in a fixed order (used for tables and validation).
CLAIM_LABELS = ("supported", "contradicted", "not_supported")
# Sentence splitter: split on whitespace that comes right after ".", "!" or "?".
# "(?<=[.!?])" is a look-behind: it checks the previous character without removing it,
# so the punctuation stays attached to its sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# Split a rationale into claims (sentences).
# Inputs: the rationale text (may be None) and the minimum word count for a claim.
# Output: list of sentences with at least min_words words; very short ones such as "Yes."
# carry no checkable fact, so they are dropped.
def split_claims(rationale: str | None, min_words: int = 4) -> list[str]:
    """Sentences of the rationale with at least `min_words` words."""
    # No rationale (None or empty string) -> no claims.
    if not rationale:
        # Return an empty list so callers can still loop over the result.
        return []
    # Split into sentences and trim spaces around each one (a lazy generator, not a list).
    sentences = (s.strip() for s in _SENTENCE_END.split(rationale.strip()))
    # Keep only sentences whose word count (split on spaces) reaches min_words.
    return [s for s in sentences if len(s.split()) >= min_words]


# Combine NLI results for several premises into one claim label.
# Input: one probability dict per premise, e.g. {"entailment": 0.8, "neutral": 0.1, ...}.
# Output: "supported", "contradicted" or "not_supported".
# Rule: a claim is supported if at least one premise entails it (one source is enough);
# otherwise contradicted if some premise contradicts it; otherwise not supported.
def decide(premise_probs: list[dict[str, float]]) -> str:
    """Claim label from NLI probabilities for each premise.

    supported if any premise's most likely label is entailment; otherwise contradicted if any
    premise's most likely label is contradiction; otherwise not_supported.
    """
    # For each premise, the label with the highest probability (the NLI model's verdict).
    top = [max(p, key=p.__getitem__) for p in premise_probs]
    # Any premise entails the claim -> supported (this rule wins over contradiction).
    if "entailment" in top:
        # The claim is backed by at least one source text.
        return "supported"
    # No support, but at least one premise contradicts the claim.
    if "contradiction" in top:
        # The claim conflicts with a source text.
        return "contradicted"
    # Every premise was neutral (or there were none): the claim is unbacked.
    return "not_supported"


# True when the prediction row is a parse failure (LLM reply was not valid JSON).
# Such rows have no usable rationale or citations, so they are skipped in the analysis.
def is_parse_failure(pred: Prediction) -> bool:
    # Parse failures are stored with raw_output starting with "PARSE_FAILURE:".
    return pred.raw_output.startswith(PARSE_FAILURE_PREFIX)


# Find citations to documents the model never received ("fabricated" citations).
# Input: one prediction. Output: list of invalid document numbers (empty list = all fine).
def fabricated_citations(pred: Prediction) -> list[int]:
    """Cited document numbers that the model was never given (any citation for closed-book)."""
    # A failed parse has no trustworthy citations, so report none.
    if is_parse_failure(pred):
        # Nothing to check.
        return []
    # Re-parse the reply; n_docs = how many documents the model saw. Numbers outside 1..n_docs
    # are invalid. Closed-book saw 0 documents, so ANY citation it gives is fabricated.
    return parse_answer(pred.raw_output, n_docs=len(pred.evidence)).invalid_citations


# The real NLI checker: a pretrained "cross-encoder" model from Hugging Face that reads a
# premise and a hypothesis together and outputs probabilities for each NLI label.
class NLIScorer:
    """Wraps a Hugging Face NLI cross-encoder; label names are read from the model config."""

    # Load the model once. Inputs: model name, max tokens per pair, pairs per batch.
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16):
        # Imported here (not at the top) so the rest of the file works without torch installed.
        import torch

        # Hugging Face helpers: load the model weights and the matching tokenizer.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Keep a reference to torch for use in __call__.
        self._torch = torch
        # Use the GPU when one is available (much faster), otherwise the CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # The tokenizer turns text into the number IDs the model understands.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the classification model and move it to the chosen device.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        # Evaluation mode: switches off training-only behaviour such as dropout.
        self.model.eval()
        # Label names in output order, read from the model's own config (never hard-coded),
        # lower-cased, e.g. ["contradiction", "entailment", "neutral"].
        self.labels = [
            self.model.config.id2label[i].lower() for i in range(len(self.model.config.id2label))
        ]
        # Longer inputs are cut to this many tokens.
        self.max_length = max_length
        # How many pairs are sent to the model at once.
        self.batch_size = batch_size

    # Score pairs. Input: list of (premise, hypothesis) pairs.
    # Output: one dict per pair mapping label name -> probability (the three sum to 1).
    def __call__(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """(premise, hypothesis) pairs -> probabilities keyed by label name."""
        # Results for all pairs, in the same order as the input.
        out = []
        # no_grad: we only predict, so skip gradient bookkeeping (saves memory and time).
        with self._torch.no_grad():
            # Walk through the pairs in steps of batch_size.
            for start in range(0, len(pairs), self.batch_size):
                # The current slice of pairs.
                batch = pairs[start : start + self.batch_size]
                # Tokenize premises and hypotheses together as pairs.
                enc = self.tokenizer(
                    # First texts: the premises.
                    [p for p, _ in batch],
                    # Second texts: the hypotheses (claims).
                    [h for _, h in batch],
                    # If too long, shorten only the premise so the claim is never cut.
                    truncation="only_first",
                    # Upper limit on tokens per pair.
                    max_length=self.max_length,
                    # Pad shorter pairs so the batch forms a rectangular tensor.
                    padding=True,
                    # Return PyTorch tensors ("pt") and move them to the model's device.
                    return_tensors="pt",
                ).to(self.device)
                # Run the model, turn raw scores (logits) into probabilities with softmax,
                # then move them to the CPU as a plain Python list of lists.
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
                # Pair each probability with its label name and add one dict per pair.
                out.extend(dict(zip(self.labels, row, strict=True)) for row in probs)
        # All probability dicts, one per input pair.
        return out


# Type alias: any function with the same input/output as NLIScorer.__call__. Tests pass a
# tiny fake scorer with this shape so they run fast without downloading the real model.
Scorer = Callable[[list[tuple[str, str]]], list[dict[str, float]]]


# Check every claim of every prediction against both premise sets.
# Inputs: predictions; evidence_texts(pred) -> texts the model was shown; gold_texts(pred) ->
# the example's own abstract sections; scorer -> NLI function; min_words -> claim filter.
# Output: one dict (row) per claim with its two labels.
def check_claims(
    predictions: list[Prediction],  # the system answers to analyse
    evidence_texts: Callable[[Prediction], list[str]],  # premises: documents shown to model
    gold_texts: Callable[[Prediction], list[str]],  # premises: the gold abstract sections
    scorer: Scorer,  # NLI function (real model or test stub)
    min_words: int = 4,  # shorter sentences are not counted as claims
) -> list[dict]:
    """One row per claim with its evidence-grounded and gold-grounded label."""
    # Collect output rows here.
    rows = []
    # Go through the predictions one by one.
    for pred in predictions:
        # Parse failures have no usable rationale: skip them.
        if is_parse_failure(pred):
            # Move on to the next prediction.
            continue
        # Get both premise lists for this prediction.
        evidence, gold = evidence_texts(pred), gold_texts(pred)
        # Number each claim (0, 1, 2, ...) in order of appearance.
        for index, claim in enumerate(split_claims(pred.rationale, min_words)):
            # Evidence-grounded label: pair the claim with every shown document and decide.
            # Closed-book saw no documents, so the label is None (not applicable).
            ev_label = decide(scorer([(p, claim) for p in evidence])) if evidence else None
            # Store one row describing this claim.
            rows.append(
                {
                    # Question id.
                    "id": pred.id,
                    # Which system produced the answer.
                    "system": pred.system,
                    # Which repeated run (1, 2, 3...).
                    "run": pred.run,
                    # Position of the claim inside the rationale.
                    "claim_index": index,
                    # The claim sentence itself.
                    "claim": claim,
                    # Verdict against the documents the model was shown (or None).
                    "evidence_label": ev_label,
                    # Verdict against the gold abstract (available for every system).
                    "gold_label": decide(scorer([(p, claim) for p in gold])),
                }
            )
    # All claim rows.
    return rows


# Summarise claim rows and predictions into a few headline numbers for one system run.
# Inputs: rows from check_claims and the full prediction list. Output: a summary dict.
def summarize(rows: list[dict], predictions: list[Prediction]) -> dict:
    # Answers that were parsed successfully (only these can contain claims).
    answers = [p for p in predictions if not is_parse_failure(p)]
    # Total number of claims checked.
    n_claims = len(rows)

    # Inner helper: share of each claim label in one column ("evidence_label" or "gold_label").
    # Output: e.g. {"supported": 0.5, "contradicted": 0.1, "not_supported": 0.4} or None.
    def shares(key: str) -> dict | None:
        # Keep only rows where the label exists (None = not applicable, e.g. closed-book).
        labelled = [r[key] for r in rows if r[key] is not None]
        # No labelled claims at all -> no shares to report.
        if not labelled:
            # None means "not applicable" in the summary.
            return None
        # Count of each label divided by the number of labelled claims.
        return {label: labelled.count(label) / len(labelled) for label in CLAIM_LABELS}

    # Ids of answers with at least one claim NOT supported by the gold abstract.
    unsupported_ids = {r["id"] for r in rows if r["gold_label"] != "supported"}
    # Answers that cite at least one document they never received.
    fabricated = [p for p in answers if fabricated_citations(p)]
    # The summary dictionary (all shares are between 0 and 1).
    return {
        # Number of prediction rows, including parse failures.
        "answers": len(predictions),
        # How many replies could not be parsed.
        "parse_failures": len(predictions) - len(answers),
        # Total claims checked.
        "claims": n_claims,
        # Average claims per parsed answer (0.0 if there are no parsed answers).
        "claims_per_answer": n_claims / len(answers) if answers else 0.0,
        # Label shares against the documents shown (None for closed-book).
        "evidence_grounded": shares("evidence_label"),
        # Label shares against the gold abstract.
        "gold_grounded": shares("gold_label"),
        # Share of parsed answers with at least one claim not supported by gold.
        "answers_with_claim_not_supported_by_gold": len(unsupported_ids) / len(answers)
        if answers  # avoid dividing by zero when every reply failed to parse
        else 0.0,
        # Share of parsed answers containing a fabricated citation.
        "answers_with_fabricated_citation": len(fabricated) / len(answers) if answers else 0.0,
    }


# Command-line entry point: analyse one prediction file and save the results.
# Input: optional argument list (tests can pass one). Output: the summary dict.
def main(argv: list[str] | None = None) -> dict:
    # Parser for the command-line options.
    parser = argparse.ArgumentParser(description="Unsupported-claim analysis of a prediction file.")
    # YAML config with the NLI model name and settings.
    parser.add_argument("--config", type=Path, default=Path("configs/claims_nli.yaml"))
    # The predictions .jsonl file to analyse (required).
    parser.add_argument("--pred", type=Path, required=True)
    # Which data split the predictions belong to (train, dev or test).
    parser.add_argument("--split", choices=SPLITS, required=True)
    # Explicit opt-in flag to read the protected test split (only for the final frozen run).
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")
    # Corpus file holding the texts of retrieved documents (needed for RAG systems).
    parser.add_argument("--corpus", type=Path, help="corpus JSONL with the retrieved documents")
    # Folder with the processed dataset files.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # Where output files are written.
    parser.add_argument("--out-dir", type=Path, default=Path("results/claims"))
    # Read the options (from argv if given, otherwise from the real command line).
    args = parser.parse_args(argv)

    # Load the NLI settings from YAML.
    cfg = load_config(args.config)
    # Reading the test split is allowed but must be loudly announced.
    if args.split == "test" and args.allow_test:
        # Print the warning.
        warn_test_access()
    # Load the system's predictions.
    predictions = load_predictions(args.pred)
    # Dict from question id to example (gives access to the gold abstract "contexts").
    examples = {ex.id: ex for ex in load_split(args.split, args.allow_test, args.processed_dir)}
    # Document store; stays None when no --corpus was given.
    store = None
    # Only load the corpus if one was given.
    if args.corpus:
        # Imported here so this module does not require the retrieval code otherwise.
        from bmqa.retrieval.store import DocStore

        # Load the corpus so document ids can be turned into texts.
        store = DocStore(args.corpus)

    # Premise provider 1: the texts of the documents this prediction's model was shown.
    def evidence_texts(pred: Prediction) -> list[str]:
        # Collected document texts.
        texts = []
        # Each hit is one retrieved document (id, score, rank).
        for hit in pred.evidence:
            # Normal case: look the document up in the corpus.
            if store is not None and hit.doc_id in store:
                # Add the document text.
                texts.append(store[hit.doc_id])
            elif hit.doc_id == pred.id:  # oracle: the example's own abstract
                # Join the abstract sections into one text, as the oracle model saw it.
                texts.append(" ".join(examples[pred.id].contexts))
            else:  # text not available: stop and tell the user what to pass
                # Raise an error naming the missing document.
                raise KeyError(f"text of document {hit.doc_id} not found; pass --corpus")
        # All document texts for this prediction.
        return texts

    # Premise provider 2: the gold abstract sections of the question (same for every system).
    def gold_texts(pred: Prediction) -> list[str]:
        # A copy of the list of abstract sections.
        return list(examples[pred.id].contexts)

    # Load the real NLI model with the configured settings.
    scorer = NLIScorer(cfg["model"], cfg["max_length"], cfg["batch_size"])
    # Label every claim.
    rows = check_claims(predictions, evidence_texts, gold_texts, scorer, cfg["min_words"])
    # Compute headline numbers.
    summary = summarize(rows, predictions)

    # System name and run number, taken from the first prediction (one file = one run).
    system, run = predictions[0].system, predictions[0].run
    # Save every claim row, e.g. results/claims/rag_strict_test_run1.jsonl.
    write_jsonl(args.out_dir / f"{system}_{args.split}_run{run}.jsonl", rows)
    # One shared summary file per split, collecting all systems and runs.
    summary_path = args.out_dir / f"summary_{args.split}.json"
    # Load existing summaries so earlier runs are kept; start empty if the file is new.
    all_summaries = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    )
    # Add or replace this run's entry, e.g. key "rag_strict_run1".
    all_summaries[f"{system}_run{run}"] = {
        # System name.
        "system": system,
        # Run number.
        "run": run,
        # Which NLI model produced the labels (for reproducibility).
        "nli_model": cfg["model"],
        # Copy in all summary numbers ("**" unpacks the dict).
        **summary,
    }
    # Write the updated summaries back as pretty-printed JSON.
    summary_path.write_text(json.dumps(all_summaries, indent=2) + "\n", encoding="utf-8")
    # Also show this run's summary on screen.
    print(json.dumps(summary, indent=2))
    # Return it (handy for tests).
    return summary


# Run main() only when executed as a script, not when imported.
if __name__ == "__main__":
    # Start the command-line program.
    main()
