"""Final comparison tables and figures from committed predictions and metrics.

Usage:  python -m bmqa.eval.compare --split test --allow-test
Writes results/tables/{main_results,significance,hallucination}.md (+ main_results.csv) and,
if matplotlib is installed (`report` extra), results/figures/{macro_f1,claims}.png.
"""

# Big picture: this is the last step of the project. It reads the saved metrics and
# predictions of every system, then builds the tables and figures for the report:
# 1. Main results: mean +/- standard deviation over repeated runs, plus a bootstrap 95% CI.
# 2. Significance: four comparisons chosen BEFORE seeing test results (so we cannot
#    cherry-pick), each tested with McNemar + paired bootstrap, then Holm-corrected.
# 3. Hallucination: the claim-checking summary produced by bmqa.eval.claims.

# Lets modern type hints be used without being evaluated at runtime.
from __future__ import annotations

# argparse: command-line options.
import argparse

# csv: write the main table as a CSV file too.
import csv

# json: read metric and summary files.
import json

# statistics: mean (fmean) and sample standard deviation (stdev).
import statistics

# Path: file paths.
from pathlib import Path

# PROCESSED_DIR: data folder; gold_labels: id -> correct answer; warn_test_access: warning.
from bmqa.data.pubmedqa import PROCESSED_DIR, gold_labels, warn_test_access

# Metrics: accuracy, macro-F1, and effective_label (None for parse failures).
from bmqa.eval.metrics import accuracy, effective_label, macro_f1

# Statistics helpers from stats.py.
from bmqa.eval.stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap

# load_predictions: read a predictions .jsonl file.
from bmqa.io import load_predictions

# LABELS = ("yes", "no", "maybe"); SPLITS = ("train", "dev", "test").
from bmqa.schema import LABELS, SPLITS

# The systems shown in the main table, in display order (ablations are left out).
MAIN_SYSTEMS = ["majority", "classifier", "closed_book", "oracle", "rag_strict", "rag_gold"]
# Pre-declared primary comparisons (A vs B), using run 1 of each system (classifier: seed 0).
# Fixing these in advance keeps the analysis honest: only these 4 tests are run and corrected.
PRIMARY = [
    # Does retrieval from the strict corpus beat no retrieval?
    ("rag_strict", "closed_book"),
    # Does retrieval from the gold corpus beat no retrieval?
    ("rag_gold", "closed_book"),
    # How much is lost by retrieving instead of being handed the right abstract?
    ("oracle", "rag_strict"),
    # Does a small fine-tuned classifier match the LLM with retrieval?
    ("classifier", "rag_strict"),
]


# Load all metric files of one split. Inputs: metrics folder, split name.
# Output: dict system -> list of metric dicts (one per run).
def load_metrics(metrics_dir: Path, split: str) -> dict[str, list[dict]]:
    # Result being built.
    by_system: dict[str, list[dict]] = {}
    # Files named like "closed_book_test_run1.json", in sorted (stable) order.
    for path in sorted(metrics_dir.glob(f"*_{split}_run*.json")):
        # Read the JSON metrics of one run.
        m = json.loads(path.read_text(encoding="utf-8"))
        # Keep only main systems (e.g. ablation "rag_strict_k3" is skipped).
        if m["system"] in MAIN_SYSTEMS:
            # Create the list on first use, then append this run.
            by_system.setdefault(m["system"], []).append(m)
    # All runs grouped by system.
    return by_system


# Mean and sample standard deviation of a list; std is 0.0 if there is only one value
# (the spread of a single number is undefined, and stdev would raise an error).
def _mean_std(values: list[float]) -> tuple[float, float]:
    # Returns the tuple (mean, std).
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


# Build one summary row per system: mean/std of accuracy and macro-F1, and mean per-label F1.
# Input: output of load_metrics. Output: list of row dicts in MAIN_SYSTEMS order.
def aggregate(metrics_by_system: dict[str, list[dict]]) -> list[dict]:
    # Rows of the main table.
    rows = []
    # Fixed display order.
    for system in MAIN_SYSTEMS:
        # All runs of this system (None if missing).
        runs = metrics_by_system.get(system)
        # Skip systems with no metric files.
        if not runs:
            # Next system.
            continue
        # (mean, std) of accuracy over runs.
        acc = _mean_std([m["accuracy"] for m in runs])
        # (mean, std) of macro-F1 over runs.
        f1 = _mean_std([m["macro_f1"] for m in runs])
        # Start the row.
        row = {
            # System name.
            "system": system,
            # Number of runs (seeds) averaged.
            "runs": len(runs),
            # Mean accuracy.
            "accuracy_mean": acc[0],
            # Run-to-run spread of accuracy.
            "accuracy_std": acc[1],
            # Mean macro-F1 (average of the yes/no/maybe F1 scores).
            "macro_f1_mean": f1[0],
            # Run-to-run spread of macro-F1.
            "macro_f1_std": f1[1],
        }
        # Add the mean F1 of each label (columns f1_yes, f1_no, f1_maybe).
        for label in LABELS:
            # Average this label's F1 over runs.
            row[f"f1_{label}"] = statistics.fmean(m["per_label"][label]["f1"] for m in runs)
        # Store the row.
        rows.append(row)
    # All rows.
    return rows


# Load gold and predicted labels in the same (sorted id) order for one prediction file.
# Inputs: predictions file, dict id -> gold label. Output: (gold list, predicted list).
def _labels(pred_path: Path, gold: dict[str, str]) -> tuple[list[str], list]:
    # Predictions indexed by question id.
    preds = {p.id: p for p in load_predictions(pred_path)}
    # Fixed order of question ids, so every system lines up question by question (pairing).
    ids = sorted(gold)
    # Gold labels and predicted labels (None for parse failures) in that order.
    return [gold[i] for i in ids], [effective_label(preds[i]) for i in ids]


# Build a Markdown table. Inputs: column names and rows. Output: the table as text.
def markdown_table(header: list[str], rows: list[list]) -> str:
    # Header line "| A | B |" and separator line "|---|---|".
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # One line per row; every cell converted to text.
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    # Join with newlines and end with a newline.
    return "\n".join(lines) + "\n"


# Command-line entry point: build all tables and (optionally) figures.
def main(argv: list[str] | None = None) -> None:
    # Parser for the options.
    parser = argparse.ArgumentParser(description="Comparison tables and figures.")
    # Split to evaluate (default: test).
    parser.add_argument("--split", choices=SPLITS, default="test")
    # Explicit opt-in to read the protected test split.
    parser.add_argument("--allow-test", action="store_true", help="Phase 4 frozen run only")
    # Root results folder.
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    # Processed data folder.
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    # Number of bootstrap resamples.
    parser.add_argument("--n-boot", type=int, default=10_000)
    # Flag to skip drawing figures.
    parser.add_argument("--no-figures", action="store_true")
    # Read the options.
    args = parser.parse_args(argv)

    # Announce reading of the test split.
    if args.split == "test" and args.allow_test:
        # Print the warning.
        warn_test_access()
    # Short name for the results folder.
    res = args.results_dir
    # Correct answers: question id -> label.
    gold = gold_labels(args.split, args.allow_test, args.processed_dir)
    # Aggregated rows of the main table.
    rows = aggregate(load_metrics(res / "metrics", args.split))
    # Path of the run-1 prediction file of each system.
    run1 = {s: res / "predictions" / s / f"{args.split}_run1.jsonl" for s in MAIN_SYSTEMS}

    # Main table with 95 % bootstrap CIs of run 1.
    # Text rows for the Markdown table.
    table = []
    # One table row per system.
    for row in rows:
        # Gold and predicted labels of run 1.
        g, p = _labels(run1[row["system"]], gold)
        # 95% CI of macro-F1 (how uncertain the score is due to the limited test set).
        row["macro_f1_ci_run1"] = bootstrap_ci(g, p, macro_f1, n_boot=args.n_boot)
        # 95% CI of accuracy.
        row["accuracy_ci_run1"] = bootstrap_ci(g, p, accuracy, n_boot=args.n_boot)
        # Add the formatted row.
        table.append(
            [
                # System name.
                row["system"],
                # Number of runs.
                row["runs"],
                # Accuracy as "mean +/- std" with 3 decimals.
                f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
                # Macro-F1 as "mean +/- std".
                f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}",
                # The CI as "[low, high]" ("*" unpacks the (low, high) tuple).
                "[{:.3f}, {:.3f}]".format(*row["macro_f1_ci_run1"]),
                # Per-label F1 columns (yes, no, maybe).
                *(f"{row[f'f1_{label}']:.3f}" for label in LABELS),
            ]
        )
    # Folder for the tables.
    tables = res / "tables"
    # Create it if needed.
    tables.mkdir(parents=True, exist_ok=True)
    # Column titles of the main table.
    header = [
        "System",
        "Runs",
        "Accuracy (mean ± std)",
        "Macro-F1 (mean ± std)",
        "Macro-F1 95% CI (run 1)",
        "F1 yes",
        "F1 no",
        "F1 maybe",
    ]
    # Write the Markdown version of the main table.
    (tables / "main_results.md").write_text(markdown_table(header, table), encoding="utf-8")
    # Also write the raw numbers as CSV (handy for spreadsheets).
    with open(tables / "main_results.csv", "w", encoding="utf-8", newline="") as f:
        # Column names are the keys of the first row dict.
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        # Header line.
        writer.writeheader()
        # One line per system.
        writer.writerows(rows)

    # Primary comparisons: McNemar + paired bootstrap, Holm-adjusted.
    # Raw results of each comparison.
    tests = []
    # Each pre-declared pair (A, B).
    for a, b in PRIMARY:
        # Skip a comparison if either prediction file is missing.
        if not (run1[a].exists() and run1[b].exists()):
            # Next pair.
            continue
        # Gold and A's labels.
        g, pa = _labels(run1[a], gold)
        # B's labels (gold is the same, so it is ignored with "_").
        _, pb = _labels(run1[b], gold)
        # McNemar on per-question correctness: True where the prediction equals the gold.
        only_a, only_b, p = mcnemar_exact(
            # A correct?
            [x == y for x, y in zip(g, pa, strict=True)],
            # B correct?
            [x == y for x, y in zip(g, pb, strict=True)],
        )
        # Macro-F1 difference A - B with its 95% paired bootstrap CI.
        diff = paired_bootstrap(g, pa, pb, macro_f1, n_boot=args.n_boot)
        # Keep everything for the table; index 4 is the raw p-value.
        tests.append([a, b, only_a, only_b, p, diff])
    # Holm-correct the four raw p-values together (empty list if no tests ran).
    adjusted = holm([t[4] for t in tests]) if tests else []
    # Formatted rows of the significance table.
    sig_rows = [
        [
            # Name of the comparison.
            f"{a} vs {b}",
            # Observed difference with a sign, e.g. "+0.052".
            f"{d['difference']:+.3f}",
            # CI of the difference; if it excludes 0, the gap is reliable.
            f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
            # Questions only A got right.
            oa,
            # Questions only B got right.
            ob,
            # Raw McNemar p-value.
            f"{p:.4f}",
            # Holm-adjusted p-value (compare this one with 0.05).
            f"{adj:.4f}",
        ]
        # Unpack each test and pair it with its adjusted p-value.
        for (a, b, oa, ob, p, d), adj in zip(tests, adjusted, strict=True)
    ]
    # Column titles of the significance table.
    sig_header = [
        "Comparison (run 1)",
        "Δ macro-F1",
        "95% CI",
        "Only A correct",
        "Only B correct",
        "McNemar p",
        "Holm-adjusted p",
    ]
    # Write the significance table.
    (tables / "significance.md").write_text(markdown_table(sig_header, sig_rows), encoding="utf-8")

    # Hallucination summary from bmqa.eval.claims.
    # Summary file written by claims.py.
    summary_path = res / "claims" / f"summary_{args.split}.json"
    # Load it, or use an empty dict if claims.py has not been run.
    claims = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    # Formatted rows.
    hall_rows = []
    # key = "system_runN", s = that run's summary; sorted for a stable order.
    for key, s in sorted(claims.items()):
        # Evidence-grounded shares (None for closed-book -> empty dict).
        ev = s["evidence_grounded"] or {}
        # Gold-grounded shares.
        gd = s["gold_grounded"] or {}
        # Add one row; missing values are shown as "nan" (not a number).
        hall_rows.append(
            [
                # System and run.
                key,
                # Number of claims.
                s["claims"],
                # Average claims per answer.
                f"{s['claims_per_answer']:.2f}",
                # Share of claims supported by the documents shown.
                f"{ev.get('supported', float('nan')):.3f}",
                # Share supported by the gold abstract.
                f"{gd.get('supported', float('nan')):.3f}",
                # Share contradicted by the gold abstract.
                f"{gd.get('contradicted', float('nan')):.3f}",
                # Share of answers with a fabricated citation.
                f"{s['answers_with_fabricated_citation']:.3f}",
            ]
        )
    # Column titles of the hallucination table.
    hall_header = [
        "System/run",
        "Claims",
        "Claims per answer",
        "Supported by evidence shown",
        "Supported by gold abstract",
        "Contradicted by gold abstract",
        "Answers with fabricated citation",
    ]
    # Write the hallucination table.
    (tables / "hallucination.md").write_text(
        markdown_table(hall_header, hall_rows), encoding="utf-8"
    )
    # Report where tables went.
    print(f"tables written to {tables}")

    # Draw figures unless --no-figures was given.
    if not args.no_figures:
        # Save PNG figures into results/figures.
        plot_figures(rows, claims, res / "figures")


# Draw two figures: macro-F1 bar chart with CI error bars, and stacked claim-label shares.
# Inputs: main table rows, claim summaries, output folder. Output: none (files are saved).
def plot_figures(rows: list[dict], claims: dict, out_dir: Path) -> None:
    # matplotlib is optional; skip figures gracefully if it is not installed.
    try:
        # Plotting library.
        import matplotlib

        # "Agg" backend draws straight to files, no window needed (works on servers).
        matplotlib.use("Agg")
        # The plotting interface.
        import matplotlib.pyplot as plt
    except ImportError:  # matplotlib missing
        # Explain why no figures are produced.
        print("matplotlib not installed (`report` extra): figures skipped")
        # Stop here.
        return
    # Create the figures folder if needed.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar labels (system names).
    systems = [r["system"] for r in rows]
    # Bar heights (mean macro-F1).
    means = [r["macro_f1_mean"] for r in rows]
    # Error bar length below the bar: mean minus CI low end.
    lows = [r["macro_f1_mean"] - r["macro_f1_ci_run1"][0] for r in rows]
    # Error bar length above the bar: CI high end minus mean.
    highs = [r["macro_f1_ci_run1"][1] - r["macro_f1_mean"] for r in rows]
    # New figure of 7 x 3.5 inches with one set of axes.
    fig, ax = plt.subplots(figsize=(7, 3.5))
    # Draw the bars.
    ax.bar(
        # x positions/labels.
        systems,
        # Heights.
        means,
        # Error bars; negative lengths (CI of run 1 may not contain the mean) clipped to 0.
        yerr=[[max(v, 0) for v in lows], [max(v, 0) for v in highs]],
        # Small caps at the ends of the error bars.
        capsize=4,
        # Bar colour (a blue hex code).
        color="#4C72B0",
    )
    # y-axis title.
    ax.set_ylabel("Macro-F1")
    # Scores range from 0 to 1.
    ax.set_ylim(0, 1)
    # Figure title.
    ax.set_title("Test macro-F1 (mean over runs; error bars: 95% CI of run 1)")
    # Fit everything nicely inside the figure.
    fig.tight_layout()
    # Save as a PNG at 200 dots per inch.
    fig.savefig(out_dir / "macro_f1.png", dpi=200)
    # Free the figure's memory.
    plt.close(fig)

    # Only draw the claims figure if claim summaries exist.
    if claims:
        # One bar per system/run, in sorted order.
        keys = sorted(claims)
        # Running bottom of each stacked bar (starts at 0).
        bottom = [0.0] * len(keys)
        # New figure.
        fig, ax = plt.subplots(figsize=(7, 3.5))
        # One stacked segment per label, each with its colour (green, red, grey).
        for label, color in (
            ("supported", "#55A868"),
            ("contradicted", "#C44E52"),
            ("not_supported", "#BBBBBB"),
        ):
            # Gold-grounded share of this label for each bar (0 if unavailable).
            values = [(claims[k]["gold_grounded"] or {}).get(label, 0.0) for k in keys]
            # Draw this segment on top of the previous ones.
            ax.bar(keys, values, bottom=bottom, label=label.replace("_", " "), color=color)
            # Raise the bottom for the next segment.
            bottom = [b + v for b, v in zip(bottom, values, strict=True)]
        # y-axis title.
        ax.set_ylabel("Share of rationale sentences")
        # Figure title.
        ax.set_title("Rationale sentences checked against the gold abstract (NLI)")
        # Legend explaining the colours.
        ax.legend(loc="lower right", fontsize=8)
        # Tidy layout.
        fig.tight_layout()
        # Save the PNG.
        fig.savefig(out_dir / "claims.png", dpi=200)
        # Free memory.
        plt.close(fig)
    # Report where figures went.
    print(f"figures written to {out_dir}")


# Run main() only when executed as a script.
if __name__ == "__main__":
    # Start the program.
    main()
