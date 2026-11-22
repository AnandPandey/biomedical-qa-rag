"""Uncertainty and significance for paired classification results (standard library only).

`metric` arguments are functions such as bmqa.eval.metrics.accuracy or macro_f1, called as
metric(gold_labels, predicted_labels).
"""

# Big picture for beginners: one accuracy number on the test set is not the full story.
# If we had drawn a slightly different set of test questions, the number would wiggle a bit.
# This file measures that wiggle (confidence intervals) and checks whether the gap between
# two systems is bigger than what chance alone could produce (significance tests).

# Allows modern type hints (like list[str]) to be written without being evaluated at runtime.
from __future__ import annotations

# random: Python's built-in random number generator; we seed it so results are repeatable.
import random

# Counter: a dictionary that counts how many times each item appears (used for kappa).
from collections import Counter

# Callable = "something you can call like a function"; Sequence = list-like (list, tuple...).
from collections.abc import Callable, Sequence

# comb(n, k) = "n choose k" = number of ways to pick k items out of n (used by McNemar).
from math import comb

# Type alias: a metric is a function that takes (gold labels, predicted labels) and returns a
# float score, e.g. accuracy([...], [...]) -> 0.73. The alias just makes signatures shorter.
Metric = Callable[[list[str], list], float]


# Helper: build the list of bootstrap resamples.
# Inputs: n = number of questions, n_boot = how many resamples, seed = random seed.
# Output: n_boot lists, each holding n question positions drawn WITH replacement
# (the same question may appear several times, another may be missing - that is the point).
def _resample_indices(n: int, n_boot: int, seed: int) -> list[list[int]]:
    # A private random generator with a fixed seed: same seed -> same resamples every run.
    rng = random.Random(seed)
    # Inner list: pick n random positions in 0..n-1; outer list: repeat that n_boot times.
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]


# Helper: turn many bootstrap scores into a (low, high) interval by cutting off the extremes.
# Inputs: values = the bootstrap scores, alpha = total share to cut (0.05 -> a 95% interval).
# Output: the value at the 2.5th percentile and the value at the 97.5th percentile.
def _percentiles(values: list[float], alpha: float) -> tuple[float, float]:
    # Sort the scores from smallest to largest so positions correspond to percentiles.
    values = sorted(values)
    # Lower bound: skip the lowest alpha/2 share (e.g. with 10,000 values -> index 250).
    low = values[int(alpha / 2 * len(values))]
    # Upper bound: the last value before the highest alpha/2 share (e.g. index 9,749).
    high = values[int((1 - alpha / 2) * len(values)) - 1]
    # Return both ends of the interval as a tuple.
    return low, high


# Percentile bootstrap confidence interval (CI) for one system's score.
# Idea: pretend the test set is the whole world, draw many fake test sets from it (with
# replacement), score the system on each one, and look at how much the score spreads.
# The middle 95% of those scores is the 95% CI: a range of plausible "true" scores.
# Inputs: gold labels, predicted labels (same order), the metric function, number of
# resamples, random seed and alpha. Output: (low, high) ends of the interval.
def bootstrap_ci(
    gold: Sequence[str],  # correct answers, one per question
    pred: Sequence,  # the system's answers, same order as gold (None = parse failure)
    metric: Metric,  # scoring function, e.g. accuracy or macro_f1
    n_boot: int = 10_000,  # how many fake test sets to draw (more = smoother interval)
    seed: int = 0,  # fixed seed so the interval is identical every time we rerun
    alpha: float = 0.05,  # 0.05 -> a 95% interval (100% - 5%)
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval: resample questions with replacement."""
    # Collect the metric value of every resampled test set here.
    values = []
    # Loop over each resample (a list of question positions).
    for idx in _resample_indices(len(gold), n_boot, seed):
        # Build the resampled gold and predicted lists and score them with the metric.
        values.append(metric([gold[i] for i in idx], [pred[i] for i in idx]))
    # Cut off the extreme alpha/2 on each side to get the confidence interval.
    return _percentiles(values, alpha)


# Paired bootstrap of a DIFFERENCE between two systems A and B on the same questions.
# "Paired" means each fake test set is used for both systems, so both see exactly the same
# questions; this removes noise caused by easy vs hard questions and compares them fairly.
# If the whole CI of (A - B) is above 0, A is reliably better; if it contains 0, we cannot tell.
# Output: a dict with the observed difference and its CI ends.
def paired_bootstrap(
    gold: Sequence[str],  # correct answers
    pred_a: Sequence,  # answers of system A
    pred_b: Sequence,  # answers of system B (same question order)
    metric: Metric,  # scoring function used for both systems
    n_boot: int = 10_000,  # number of resamples
    seed: int = 0,  # fixed seed for reproducibility
    alpha: float = 0.05,  # 0.05 -> 95% interval
) -> dict:
    """Observed metric(A) - metric(B) and its CI, using the same resampled questions for both."""
    # The real difference on the actual test set (e.g. 0.80 - 0.60 = +0.20).
    observed = metric(list(gold), list(pred_a)) - metric(list(gold), list(pred_b))
    # Store the difference measured on every resampled test set.
    diffs = []
    # Each idx is one fake test set (list of question positions).
    for idx in _resample_indices(len(gold), n_boot, seed):
        # Gold labels of this fake test set, shared by both systems.
        g = [gold[i] for i in idx]
        # Score A and B on the SAME questions and keep A minus B.
        diffs.append(metric(g, [pred_a[i] for i in idx]) - metric(g, [pred_b[i] for i in idx]))
    # Turn the differences into a (low, high) interval.
    low, high = _percentiles(diffs, alpha)
    # Return the observed gap plus its interval as a small dictionary.
    return {"difference": observed, "ci_low": low, "ci_high": high}


# Exact McNemar test: are two systems' error rates really different on the same questions?
# Questions both get right or both get wrong tell us nothing about which is better, so the
# test only looks at disagreements: b = only A right, c = only B right.
# If A and B were equally good, each disagreement would be a fair coin flip (50/50), so
# b should be about half of b + c. The p-value is the chance of a split at least this lopsided
# under that fair-coin assumption. Small p (e.g. < 0.05) -> the difference is unlikely chance.
# Inputs: two lists of True/False (was the answer correct?). Output: (b, c, p_value).
def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired correctness.

    b = questions only A got right, c = questions only B got right.
    """
    # Paired data must line up one-to-one; refuse mismatched inputs.
    if len(correct_a) != len(correct_b):
        # Stop with a clear error message instead of computing nonsense.
        raise ValueError("paired inputs must have the same length")
    # b: count questions where A is correct and B ("o" for other) is not (True counts as 1).
    b = sum(a and not o for a, o in zip(correct_a, correct_b, strict=True))
    # c: count questions where B is correct and A is not.
    c = sum(o and not a for a, o in zip(correct_a, correct_b, strict=True))
    # n: total number of disagreements (the "coin flips").
    n = b + c
    # No disagreements at all: no evidence of any difference.
    if n == 0:
        # p = 1.0 means "completely consistent with no difference".
        return b, c, 1.0
    # One tail: probability of getting min(b, c) or fewer "heads" in n fair coin flips,
    # i.e. sum of C(n, i) for i = 0..min(b, c), divided by all 2**n possible outcomes.
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    # Two-sided: double the tail (a lopsided result in either direction counts); cap at 1.
    return b, c, min(1.0, 2 * tail)


# Holm-Bonferroni correction for running several tests at once.
# Why: with a 5% false-alarm rate per test, running 4 tests gives a much higher chance that at
# least one looks "significant" by pure luck. Holm raises (adjusts) the p-values to keep the
# overall false-alarm rate at 5%, while being less strict than plain Bonferroni (p * m).
# Recipe: sort p-values from smallest; multiply the smallest by m, the next by m - 1, ...,
# the largest by 1; never let an adjusted value be smaller than the one before; cap at 1.
# Input: raw p-values. Output: adjusted p-values in the SAME order as the input.
def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    # m: how many tests (comparisons) we are correcting for.
    m = len(p_values)
    # Positions of the p-values sorted from smallest p to largest p.
    order = sorted(range(m), key=lambda i: p_values[i])
    # Result list, filled in at the original positions.
    adjusted = [0.0] * m
    # Largest adjusted value so far; keeps the adjusted values from going down.
    running = 0.0
    # rank = 0 for the smallest p, 1 for the next...; i = its position in the input.
    for rank, i in enumerate(order):
        # Multiply by (m - rank): m for the smallest, m - 1 for the next, and so on.
        running = max(running, (m - rank) * p_values[i])
        # Store at the original position; a probability can never exceed 1.
        adjusted[i] = min(1.0, running)
    # Adjusted p-values, same order as the input list.
    return adjusted


# Cohen's kappa: how much two label lists agree BEYOND what random guessing would give.
# Plain "% agreement" is misleading: if both raters say "supported" 90% of the time, they
# agree a lot just by chance. Kappa = (observed - expected) / (1 - expected).
# 1 = perfect agreement, 0 = no better than chance, below 0 = worse than chance.
# Inputs: two equal-length label lists (e.g. annotator A and annotator B). Output: kappa.
def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Agreement between two label sequences, corrected for chance."""
    # Kappa needs at least one item and the same number of items in both lists.
    if len(a) != len(b) or not a:
        # Explain what went wrong.
        raise ValueError("kappa needs two non-empty sequences of the same length")
    # n: number of items labelled by both.
    n = len(a)
    # Observed agreement: share of items where both lists have the same label.
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    # How often each list uses each label (e.g. {"supported": 5, "contradicted": 2}).
    count_a, count_b = Counter(a), Counter(b)
    # Chance agreement: for each label, P(A says it) * P(B says it), summed over labels.
    # (count_a/n) * (count_b/n) summed = sum(count_a * count_b) / n**2.
    expected = sum(count_a[label] * count_b[label] for label in count_a) / n**2
    # Special case: both always use one same label, so 1 - expected would be 0 (divide by 0).
    if expected == 1.0:
        # Treat full agreement as kappa 1, anything else as 0, to avoid dividing by zero.
        return 1.0 if observed == 1.0 else 0.0
    # The kappa formula: improvement over chance, divided by the maximum possible improvement.
    return (observed - expected) / (1 - expected)
