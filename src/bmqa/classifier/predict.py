"""Write predictions from a saved classifier checkpoint.

Usage:
    python -m bmqa.classifier.predict --checkpoint outputs/classifier/seed0 \
        --config configs/classifier_pubmedbert.yaml --split dev --run 1 --seed 0
Phase 4 adds `--split test --allow-test` for the frozen test run.
"""

# A "checkpoint" is a folder with a trained model saved by train.py. This script loads it
# and answers every question of a split without training again.

# Makes type hints like `list[str] | None` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --checkpoint and --split.
import argparse

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# encode: turns examples into model inputs; make_predictions: turns probabilities into
# Prediction records.
from bmqa.classifier.data import encode, make_predictions

# predict_probs: runs the model and returns class probabilities (shared with training).
from bmqa.classifier.train import predict_probs

# config_hash: short fingerprint of the settings; load_config: reads the YAML config.
from bmqa.config import config_hash, load_config

# PROCESSED_DIR: default data folder; load_split: loads a split (with the test guard);
# warn_test_access: prints a warning when the test split is used.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split, warn_test_access

# write_predictions: saves Prediction records to a JSON Lines file.
from bmqa.io import write_predictions

# SPLITS = ("train", "dev", "test").
from bmqa.schema import SPLITS


# main: the command-line entry point.
# Input: optional argument list (used by tests). Output: path of the written prediction file.
def main(argv: list[str] | None = None) -> Path:
    # PyTorch and transformers are big libraries, so they are imported inside the function:
    # other modules (and tests) can import this file without having them installed.
    # torch: PyTorch, the deep learning library that runs the model.
    import torch

    # AutoModelForSequenceClassification: loads a transformer with a classification head;
    # AutoTokenizer: loads the matching tokenizer (text -> token IDs).
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Create the command-line parser.
    parser = argparse.ArgumentParser(description="Predict with a saved classifier checkpoint.")
    # --checkpoint: folder of the saved model (must be given).
    parser.add_argument("--checkpoint", type=Path, required=True)
    # --config: YAML config with model_name, max_length and system name.
    parser.add_argument("--config", type=Path, default=Path("configs/classifier_pubmedbert.yaml"))
    # --split: which split to predict on (train/dev/test only; default dev).
    parser.add_argument("--split", choices=SPLITS, default="dev")
    # --run: run number written into the predictions and the output file name (required).
    parser.add_argument("--run", type=int, required=True)
    # --seed: training seed of this checkpoint, recorded for reproducibility (required).
    parser.add_argument("--seed", type=int, required=True)
    # --allow-test: switch needed to use the test split; only for the frozen Phase 4 run.
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")
    # --batch-size: how many examples the model processes at once (lower it if memory runs out).
    parser.add_argument("--batch-size", type=int, default=16)
    # --processed-dir: folder holding the processed split files.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # --out-dir: folder where prediction files are written.
    parser.add_argument("--out-dir", type=Path, default=Path("results/predictions"))
    # Read the actual arguments.
    args = parser.parse_args(argv)

    # Load the settings from the YAML config.
    cfg = load_config(args.config)
    # If test is being used with permission, print the warning first.
    if args.split == "test" and args.allow_test:
        warn_test_access()
    # Load the examples (load_split refuses "test" unless --allow-test was given).
    examples = load_split(args.split, args.allow_test, args.processed_dir)
    # Use the GPU ("cuda") if one is available, because it is much faster; otherwise the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer saved with the checkpoint, so text is split exactly as in training.
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # Load the trained model weights and move the model to the chosen device.
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    # Encode the examples and get yes/no/maybe probabilities for each of them.
    probs = predict_probs(
        model, encode(examples, tokenizer, cfg["max_length"]), args.batch_size, device
    )
    # Turn the probabilities into Prediction records with all run details.
    preds = make_predictions(
        # The questions that were answered.
        examples,
        # Their probabilities, in yes/no/maybe order.
        probs,
        # Name of the pretrained model, from the config.
        model_name=cfg["model_name"],
        # Fingerprint of the config.
        cfg_hash=config_hash(cfg),
        # Run number and seed from the command line.
        run=args.run,
        seed=args.seed,
        # System name from the config (e.g. "classifier").
        system=cfg["system"],
    )
    # Output path, e.g. results/predictions/classifier/dev_run1.jsonl.
    out = args.out_dir / cfg["system"] / f"{args.split}_run{args.run}.jsonl"
    # Save the predictions.
    write_predictions(out, preds)
    # Tell the user where they went.
    print(f"{len(preds)} predictions written to {out}")
    # Return the path for tests or other code.
    return out


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
