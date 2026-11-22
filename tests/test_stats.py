# Tests for bmqa.eval.stats. Run with: pytest tests/test_stats.py
# The expected numbers below were worked out by hand so we know the code does the right maths.

# pytest: the test framework; pytest.approx compares floats allowing tiny rounding errors.
import pytest

# accuracy(gold, pred): share of predictions that equal the gold label.
from bmqa.eval.metrics import accuracy

# The five statistics functions under test.
from bmqa.eval.stats import bootstrap_ci, cohen_kappa, holm, mcnemar_exact, paired_bootstrap


# Checks the McNemar test against a p-value computed by hand.
def test_mcnemar_hand_computed():
    # b = 2 (only A right), c = 8 (only B right): p = 2 * (C(10,0)+C(10,1)+C(10,2)) / 2**10
    # Worked out: n = 2 + 8 = 10 disagreements, min(b, c) = 2.
    # C(10,0) + C(10,1) + C(10,2) = 1 + 10 + 45 = 56; 56 / 1024 = 0.0546875 (one tail).
    # Two-sided: 2 * 0.0546875 = 0.109375.
    # System A: 2 right, then 8 wrong, then 5 right (15 questions in total).
    a = [True] * 2 + [False] * 8 + [True] * 5
    # System B: 2 wrong, then 8 right, then 5 right; the last 5 are "both right" (ignored).
    b = [False] * 2 + [True] * 8 + [True] * 5
    # Expect b = 2, c = 8 and the hand-computed p-value 0.109375.
    assert mcnemar_exact(a, b) == (2, 8, pytest.approx(0.109375))
    # Identical results -> no disagreements -> b = c = 0 and p = 1.0 by definition.
    assert mcnemar_exact([True, False], [True, False]) == (0, 0, 1.0)


# Checks the Holm correction against values computed by hand.
def test_holm():
    # m = 4. Sorted p: 0.01, 0.03, 0.04, 0.2 -> multiply by 4, 3, 2, 1:
    # 0.01*4 = 0.04; 0.03*3 = 0.09; 0.04*2 = 0.08 -> raised to 0.09 (never decrease);
    # 0.2*1 = 0.2. Put back in input order: [0.04, 0.09, 0.09, 0.2].
    assert holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    # m = 2. Sorted: 0.5*2 = 1.0; 0.6*1 = 0.6 -> raised to 1.0. Both capped at 1.0.
    assert holm([0.6, 0.5]) == pytest.approx([1.0, 1.0])


# Checks Cohen's kappa on perfect agreement, chance-level agreement and bad input.
def test_cohen_kappa():
    # Identical lists: observed = 1, expected = (2*2 + 1*1) / 3**2 = 5/9,
    # kappa = (1 - 5/9) / (1 - 5/9) = 1.0 (perfect agreement).
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # observed 0.5, expected 0.5 -> kappa 0
    # Observed: items 1 and 4 match -> 2/4 = 0.5. Each list has 2 "a" and 2 "b", so
    # expected = (2*2 + 2*2) / 4**2 = 8/16 = 0.5. kappa = (0.5 - 0.5) / (1 - 0.5) = 0.
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    # Empty lists are invalid input and must raise ValueError.
    with pytest.raises(ValueError):
        # This call is expected to fail inside the "with" block.
        cohen_kappa([], [])


# Shared toy data: 50 questions, 30 with gold "yes" and 20 with gold "no".
GOLD = ["yes"] * 30 + ["no"] * 20
# System A: 25 of the 30 "yes" right, 5 wrong; 15 of the 20 "no" right, 5 wrong.
# Accuracy A = (25 + 15) / 50 = 40/50 = 0.8.
PRED_A = ["yes"] * 25 + ["no"] * 5 + ["no"] * 15 + ["yes"] * 5
# System B always says "yes": right on the 30 "yes" only -> accuracy 30/50 = 0.6.
PRED_B = ["yes"] * 50


# The bootstrap must be repeatable with a fixed seed and its CI must include the real score.
def test_bootstrap_is_deterministic_and_contains_estimate():
    # 95% CI of A's accuracy using 500 resamples (small for a fast test) and seed 1.
    ci = bootstrap_ci(GOLD, PRED_A, accuracy, n_boot=500, seed=1)
    # Same inputs and seed -> exactly the same interval.
    assert ci == bootstrap_ci(GOLD, PRED_A, accuracy, n_boot=500, seed=1)
    # The observed accuracy (0.8) should lie inside the interval [low, high].
    assert ci[0] <= accuracy(GOLD, PRED_A) <= ci[1]


# The paired bootstrap must report the right observed difference and a CI around it.
def test_paired_bootstrap_difference():
    # Compare A against B on accuracy, 500 resamples, seed 1.
    result = paired_bootstrap(GOLD, PRED_A, PRED_B, accuracy, n_boot=500, seed=1)
    # Observed difference = accuracy A - accuracy B = 0.8 - 0.6 = 0.2.
    assert result["difference"] == pytest.approx(0.8 - 0.6)
    # The observed difference should lie inside its own confidence interval.
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]
