"""Inputs and outputs of the classifier. No PyTorch import here, so tests run without it."""

# Makes type hints like `tuple[str, str]` work as plain text on older Python versions.
from __future__ import annotations

# LABELS = ("yes", "no", "maybe"); Example = one question; Prediction = one answer record.
from bmqa.schema import LABELS, Example, Prediction

# A neural network works with numbers, not words, so each label gets a class number.
# LABEL2ID = {"yes": 0, "no": 1, "maybe": 2} (enumerate pairs each label with its position).
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
# The reverse mapping: {0: "yes", 1: "no", 2: "maybe"}, to turn numbers back into labels.
ID2LABEL = dict(enumerate(LABELS))


# text_pair: the two pieces of text the model reads for one example.
# Input: an Example. Output: (question, all context sections joined into one string).
def text_pair(example: Example) -> tuple[str, str]:
    """(question, abstract contexts). `long_answer` states the answer and is never used."""
    # Return the question and the contexts joined with spaces. long_answer is deliberately
    # left out: it is the abstract's conclusion and gives the answer away, so using it
    # would make the task trivial and the scores meaningless.
    return example.question, " ".join(example.contexts)


# encode: turn examples into the number tensors (token IDs etc.) that the model expects.
# Inputs: examples, a Hugging Face tokenizer, and the maximum length in tokens.
# Output: a dict-like batch of PyTorch tensors (input_ids, attention_mask, ...).
def encode(examples: list[Example], tokenizer, max_length: int):
    """Tokenise (question, contexts) pairs; only the contexts are truncated."""
    # Build a text pair for every example, then zip(*...) "unzips" the pairs into one tuple
    # of all questions and one tuple of all contexts.
    questions, contexts = zip(*(text_pair(ex) for ex in examples), strict=True)
    # Run the tokenizer on all pairs at once and return the result.
    return tokenizer(
        # First text of each pair: the questions.
        list(questions),
        # Second text of each pair: the contexts.
        list(contexts),
        # If a pair is too long, cut only the second text (contexts), so the question,
        # which is short and essential, is always kept whole.
        truncation="only_second",
        # Upper limit on tokens per pair (BERT models accept at most 512).
        max_length=max_length,
        # Pad every pair to exactly max_length so they can be stacked into one tensor.
        padding="max_length",
        # "pt" = return PyTorch tensors.
        return_tensors="pt",
    )


# make_predictions: turn model probabilities into the shared Prediction records.
# Inputs: the examples, one list of 3 probabilities per example (yes, no, maybe order),
# and (keyword-only, after `*`) details to record about the run.
# Output: one Prediction per example.
def make_predictions(
    examples: list[Example],
    probs: list[list[float]],
    *,
    model_name: str,
    cfg_hash: str,
    run: int,
    seed: int,
    system: str = "classifier",
) -> list[Prediction]:
    """Turn per-example class probabilities (LABELS order) into Prediction rows."""
    # Predictions will be collected here.
    preds = []
    # Walk through the examples and their probability rows together.
    for ex, row in zip(examples, probs, strict=True):
        # Sum of the three probabilities (should be about 1.0).
        total = sum(row)
        # Make {label: probability}, dividing by the total so the values sum to exactly 1.
        p = {label: float(row[i]) / total for i, label in enumerate(LABELS)}
        # Create the Prediction record and add it to the list.
        preds.append(
            Prediction(
                # Which question this answer is for.
                id=ex.id,
                # System name, e.g. "classifier".
                system=system,
                # The label with the highest probability (p.__getitem__ looks up p[label]).
                prediction=max(LABELS, key=p.__getitem__),
                # The classifier produces no text, so the raw output is empty.
                raw_output="",
                # The pretrained model the classifier was fine-tuned from.
                model=model_name,
                # Fingerprint of the config used.
                config_hash=cfg_hash,
                # Run number (one run per training seed).
                run=run,
                # Keep all three probabilities for later analysis (e.g. calibration).
                probs=p,
                # Random seed of this run, so it can be reproduced.
                seed=seed,
            )
        )
    # Return all predictions.
    return preds
