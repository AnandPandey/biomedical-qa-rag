"""Fine-tune the classifier for each seed, keep the epoch with the best dev macro-F1.

Usage (GPU recommended, see docs/colab.md):
    python -m bmqa.classifier.train --config configs/classifier_pubmedbert.yaml
Smoke test on CPU:
    python -m bmqa.classifier.train --config configs/classifier_pubmedbert.yaml \
        --seeds 0 --epochs 1 --limit 16 --no-save
"""

# "Fine-tuning" means starting from a model already pretrained on biomedical text and
# training it a little more on our train split so it learns to answer yes/no/maybe.
# We train several times with different random seeds because results vary from seed to seed;
# reporting all seeds shows how stable the classifier really is.

# Makes type hints like `Path | None` work as plain text on older Python versions.
from __future__ import annotations

# argparse: reads command-line options such as --seeds and --epochs.
import argparse

# json: writes the checkpoint record file.
import json

# random: Python's own random number generator, seeded in set_seed().
import random

# Path: an object-oriented way to work with file and folder paths.
from pathlib import Path

# ID2LABEL / LABEL2ID: map between class numbers and labels; encode: examples -> tensors;
# make_predictions: probabilities -> Prediction records.
from bmqa.classifier.data import ID2LABEL, LABEL2ID, encode, make_predictions

# config_hash: short fingerprint of the settings; load_config: reads the YAML config.
from bmqa.config import config_hash, load_config

# sha256_file: checksum of a file; used to record a fingerprint of each saved model.
from bmqa.data.download import sha256_file

# PROCESSED_DIR: default data folder; load_split: loads train or dev examples.
from bmqa.data.pubmedqa import PROCESSED_DIR, load_split

# macro_f1: the score used to pick the best epoch.
from bmqa.eval.metrics import macro_f1

# write_predictions: saves Prediction records to a JSON Lines file.
from bmqa.io import write_predictions

# Example: the shared record for one question (used in type hints).
from bmqa.schema import Example


# set_seed: fix every random number generator used during training.
# Input: the seed number. Output: nothing.
# Why: weight initialisation, shuffling and dropout are random. Fixing the seeds makes a run
# repeatable, so the same seed gives (nearly) the same model again.
def set_seed(seed: int) -> None:
    # Imported here so this module can be imported without numpy/torch installed.
    # numpy: numerical library (some libraries use its random generator).
    import numpy as np

    # torch: PyTorch, the deep learning library.
    import torch

    # Seed Python's built-in random module.
    random.seed(seed)
    # Seed numpy's random generator.
    np.random.seed(seed)
    # Seed PyTorch on the CPU.
    torch.manual_seed(seed)
    # Seed PyTorch on every GPU (does nothing if there is no GPU).
    torch.cuda.manual_seed_all(seed)


# predict_probs: run the model on encoded inputs and return class probabilities.
# Inputs: the model, the encoded inputs (dict of tensors), batch size and device.
# Output: one list of 3 probabilities per example, in the model's label order (yes, no, maybe).
def predict_probs(model, enc, batch_size: int, device) -> list[list[float]]:
    """Class probabilities for already-encoded inputs, in the model's label order."""
    # Imported here so the module can be imported without PyTorch.
    import torch

    # Switch to evaluation mode: turns off dropout so predictions are deterministic.
    model.eval()
    # Names of the input tensors, e.g. ["input_ids", "token_type_ids", "attention_mask"].
    keys = list(enc.keys())
    # Probabilities of all examples will be collected here.
    out: list[list[float]] = []
    # no_grad: we are not training, so skip gradient bookkeeping (faster, less memory).
    with torch.no_grad():
        # Step through the examples in chunks of batch_size: start = 0, batch_size, ...
        for start in range(0, len(enc[keys[0]]), batch_size):
            # Slice the same rows out of every input tensor and move them to the device.
            batch = {k: enc[k][start : start + batch_size].to(device) for k in keys}
            # Run the model (`**batch` passes the tensors as keyword arguments). "Logits" are
            # the raw, unnormalised scores for each class.
            logits = model(**batch).logits
            # softmax turns scores into probabilities that sum to 1 (dim=-1: across classes);
            # .cpu().tolist() moves them back from the GPU and makes plain Python lists.
            out.extend(torch.softmax(logits, dim=-1).cpu().tolist())
    # Return the probabilities of all examples.
    return out


# train_seed: fine-tune one model with one seed and keep its best epoch.
# Inputs: the config, the seed, train and dev examples, and (keyword-only) the number of
# epochs, the device, and the folder to save the best model in (None = do not save).
# Output: (best epoch number, best dev macro-F1, dev probabilities at that epoch).
def train_seed(
    cfg: dict,
    seed: int,
    train: list[Example],
    dev: list[Example],
    *,
    epochs: int,
    device,
    save_dir: Path | None,
) -> tuple[int, float, list[list[float]]]:
    """Train one seed; return (best epoch, best dev macro-F1, dev probabilities at that epoch)."""
    # Heavy libraries imported only when training actually runs.
    # torch: PyTorch.
    import torch

    # DataLoader: serves the training data in (shuffled) mini-batches;
    # TensorDataset: wraps several tensors so row i of each belongs together.
    from torch.utils.data import DataLoader, TensorDataset

    # From Hugging Face transformers:
    from transformers import (
        # Loads a pretrained transformer with a new classification layer on top.
        AutoModelForSequenceClassification,
        # Loads the tokenizer that matches the model.
        AutoTokenizer,
        # Learning-rate schedule: warm up, then decrease linearly to 0.
        get_linear_schedule_with_warmup,
    )

    # Fix all random generators before creating the model, so this seed is reproducible.
    set_seed(seed)
    # Download (or load from cache) the tokenizer of the pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Load the pretrained model with a fresh 3-class output layer and move it to the device.
    # num_labels=3; id2label/label2id store the label names inside the saved model.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    # Turn the training examples into tensors (question + contexts, never long_answer).
    train_enc = encode(train, tokenizer, cfg["max_length"])
    # Same for the dev examples, used to choose the best epoch.
    dev_enc = encode(dev, tokenizer, cfg["max_length"])
    # Names of the input tensors, kept in a fixed order.
    keys = list(train_enc.keys())
    # Gold labels of the training examples as class numbers (yes=0, no=1, maybe=2).
    labels = torch.tensor([LABEL2ID[ex.final_decision] for ex in train])
    # Build the loader that gives training mini-batches.
    loader = DataLoader(
        # Each item = (input tensors..., label); the label is last.
        TensorDataset(*(train_enc[k] for k in keys), labels),
        # Number of examples per training step.
        batch_size=cfg["batch_size"],
        # Shuffle the order every epoch so the model does not learn the order.
        shuffle=True,
        # A generator seeded with this seed, so the shuffle order is reproducible.
        generator=torch.Generator().manual_seed(seed),
    )
    # AdamW optimiser: updates the model weights from the gradients.
    # lr = step size; weight_decay = small penalty on large weights to reduce overfitting.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    # Total number of training steps = epochs x batches per epoch.
    total_steps = epochs * len(loader)
    # Learning rate rises from 0 during the first warmup_ratio of steps (gentle start that
    # avoids damaging the pretrained weights), then falls linearly to 0 at the end.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps
    )
    # Gold labels of dev, used to score each epoch.
    dev_gold = [ex.final_decision for ex in dev]

    # Best result so far. best_f1 starts at -1.0 so the first epoch always counts as better.
    best_epoch, best_f1, best_probs = 0, -1.0, []
    # One epoch = one full pass over the training data. Epochs are numbered from 1.
    for epoch in range(1, epochs + 1):
        # Training mode: turns dropout on.
        model.train()
        # One training step per mini-batch.
        for batch in loader:
            # All items except the last are inputs: pair them with their names and move them
            # to the device.
            inputs = {k: t.to(device) for k, t in zip(keys, batch[:-1], strict=True)}
            # Forward pass: giving labels makes the model also compute the loss
            # (how wrong it is; lower is better).
            loss = model(**inputs, labels=batch[-1].to(device)).loss
            # Backward pass: compute the gradient of the loss for every weight.
            loss.backward()
            # Clip gradients to a total size of at most 1.0 to avoid unstable huge updates.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the weights using the gradients.
            optimizer.step()
            # Move the learning rate one step along its schedule.
            scheduler.step()
            # Reset the gradients; otherwise PyTorch would add them to the next step's.
            optimizer.zero_grad()
        # After each epoch, get dev probabilities (bigger batches are fine without training).
        probs = predict_probs(model, dev_enc, cfg["batch_size"] * 2, device)
        # For each dev example pick the class number with the highest probability and
        # convert it to its label.
        dev_pred = [ID2LABEL[max(range(len(p)), key=p.__getitem__)] for p in probs]
        # Score this epoch on dev with macro-F1 (dev only; test is never used for choices).
        f1 = macro_f1(dev_gold, dev_pred)
        # Progress line: the last batch's loss and the dev score.
        print(f"seed {seed} epoch {epoch}: loss {loss.item():.4f}  dev macro-F1 {f1:.4f}")
        # Keep this epoch only if it is strictly better. With `>` (not `>=`) a tie keeps the
        # earlier epoch: the simpler, less-trained model, and a fixed rule for everyone.
        if f1 > best_f1:  # strictly better, so ties keep the earlier epoch
            # Remember this epoch as the best one.
            best_epoch, best_f1, best_probs = epoch, f1, probs
            # Save the model only when a save folder was given (not in --no-save smoke tests).
            if save_dir is not None:
                # Write the model weights and config (overwrites the previous best).
                model.save_pretrained(save_dir)
                # Save the tokenizer next to it so predict.py can load both together.
                tokenizer.save_pretrained(save_dir)
    # Return the best epoch, its dev score and its dev probabilities.
    return best_epoch, best_f1, best_probs


# main: the command-line entry point; trains one model per seed.
# Input: optional argument list (used by tests). Output: nothing; writes files and prints.
def main(argv: list[str] | None = None) -> None:
    # PyTorch is imported here so the module can be imported without it.
    import torch

    # Create the command-line parser.
    parser = argparse.ArgumentParser(description="Fine-tune on train, select epochs on dev.")
    # --config: the YAML config with all training settings.
    parser.add_argument("--config", type=Path, default=Path("configs/classifier_pubmedbert.yaml"))
    # --seeds: optional list of seeds (nargs="*" accepts zero or more numbers).
    parser.add_argument("--seeds", type=int, nargs="*", help="override the seeds in the config")
    # --epochs: optional number of epochs instead of the config value.
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    # --limit: use only the first N examples, for a quick test run.
    parser.add_argument("--limit", type=int, help="use only the first N train/dev examples")
    # --no-save: switch for smoke tests; nothing is written to disk.
    parser.add_argument("--no-save", action="store_true", help="smoke test: write nothing")
    # --processed-dir: folder holding the processed split files.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # --checkpoint-dir: folder where each seed's best model is saved.
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/classifier"))
    # --results-dir: folder for predictions and the checkpoint record file.
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    # Read the actual arguments.
    args = parser.parse_args(argv)

    # Load the training settings.
    cfg = load_config(args.config)
    # Use the seeds from the command line if any were given, otherwise those in the config.
    seeds = args.seeds if args.seeds else cfg["seeds"]
    # Use --epochs if given (not None/0), otherwise the config value.
    epochs = args.epochs or cfg["epochs"]
    # Use the GPU if available (much faster), otherwise the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Show which device is used.
    print(f"device: {device}")
    # Load train; [: args.limit] keeps the first N rows (all rows when limit is None).
    train = load_split("train", processed_dir=args.processed_dir)[: args.limit]
    # Load dev the same way. Test is never loaded here: it plays no part in training.
    dev = load_split("dev", processed_dir=args.processed_dir)[: args.limit]

    # One record per trained seed will be collected here.
    records = []
    # Train one model per seed.
    for seed in seeds:
        # Run number: the seed's position in the config list plus 1 (seed 0 -> run 1);
        # a seed not in the config gets the next free number instead.
        run = cfg["seeds"].index(seed) + 1 if seed in cfg["seeds"] else len(records) + 1
        # Checkpoint folder such as outputs/classifier/seed0, or None when not saving.
        save_dir = None if args.no_save else args.checkpoint_dir / f"seed{seed}"
        # Train this seed and get its best epoch, score and dev probabilities.
        best_epoch, best_f1, probs = train_seed(
            cfg, seed, train, dev, epochs=epochs, device=device, save_dir=save_dir
        )
        # Report the result of this seed.
        print(f"seed {seed}: best epoch {best_epoch}, dev macro-F1 {best_f1:.4f}")
        # In a smoke test, write nothing and go on to the next seed.
        if args.no_save:
            continue
        # Turn the best epoch's dev probabilities into Prediction records.
        preds = make_predictions(
            dev, probs, model_name=cfg["model_name"], cfg_hash=config_hash(cfg), run=run, seed=seed
        )
        # Save them, e.g. results/predictions/classifier/dev_run1.jsonl.
        write_predictions(
            args.results_dir / "predictions" / cfg["system"] / f"dev_run{run}.jsonl", preds
        )
        # Record the details of this seed's checkpoint.
        records.append(
            {
                # Seed and run number.
                "seed": seed,
                "run": run,
                # Which epoch was kept and its dev score.
                "best_epoch": best_epoch,
                "dev_macro_f1": best_f1,
                # Checkpoint folder name (relative to the checkpoint dir).
                "checkpoint": f"seed{seed}",
                # Checksum of the saved weights, so anyone can later verify they use the
                # exact same model file.
                "model_safetensors_sha256": sha256_file(save_dir / "model.safetensors"),
            }
        )
    # Write the record file only if at least one seed was saved.
    if records:
        # Path results/classifier_checkpoints.json.
        out = args.results_dir / "classifier_checkpoints.json"
        # Make sure the folder exists.
        out.parent.mkdir(parents=True, exist_ok=True)
        # Save the records as indented JSON with a final newline.
        out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        # Tell the user where the records went.
        print(f"checkpoint records written to {out}")


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
