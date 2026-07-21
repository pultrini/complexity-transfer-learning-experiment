"""
statistical_tests.py
=====================
Paired non-parametric comparisons between experiment configurations across
statistical iterations (e.g. comparing target-task accuracy when the
max-complexity checkpoint is selected from different training windows).

Uses the Wilcoxon signed-rank test: appropriate here because each pair of
observations being compared comes from the *same* iteration (same random
seed / data ordering), so the comparison is paired, and the small sample
sizes typical of these experiments (a handful of iterations) make a
normality assumption (as required by a paired t-test) unsafe to rely on.

IMPORTANT CAVEAT ON STATISTICAL POWER:
With N paired observations, the Wilcoxon signed-rank test's smallest
achievable two-sided p-value is 1 / 2^(N-1) (e.g. ~0.031 for N=5, ~0.002 for
N=10). With N=5 iterations, only a *perfect* ranking across all pairs can
ever reach the conventional alpha=0.05 threshold -- in practice this test is
underpowered at N=5. Prefer N>=10 iterations for any comparison you intend
to report as statistically significant in a paper.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class WilcoxonResult:
    """Result of a paired Wilcoxon signed-rank comparison, with effect size."""

    statistic: float
    p_value: float
    effect_size_r: float
    n_pairs: int
    n_ties_dropped: int
    median_diff: float
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size_r": self.effect_size_r,
            "n_pairs": self.n_pairs,
            "n_ties_dropped": self.n_ties_dropped,
            "median_diff": self.median_diff,
            "interpretation": self.interpretation,
        }


def _interpret_effect_size(r: float) -> str:
    """Rough qualitative bands for |r|, following common conventions for
    rank-biserial correlation (analogous to Cohen's guidelines for r)."""
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    if abs_r < 0.3:
        return "small"
    if abs_r < 0.5:
        return "medium"
    return "large"


def wilcoxon_signed_rank(
    x: list[float], y: list[float], alternative: str = "two-sided"
) -> WilcoxonResult:
    """Compare two paired samples (e.g. metric values from the same
    iterations under two different configurations) via the Wilcoxon
    signed-rank test.

    Args:
        x: Values for configuration A, one per iteration.
        y: Values for configuration B, one per iteration, in the same
            iteration order as ``x``.
        alternative: 'two-sided', 'greater', or 'less' (x vs y).

    Returns:
        A WilcoxonResult with the test statistic, p-value, effect size, and
        a plain-language interpretation of the effect size magnitude.

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths, or if fewer
            than 2 non-tied pairs remain (the test is undefined below that).
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} vs {len(y)}.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    diffs = x_arr - y_arr

    n_total = len(diffs)
    n_ties = int(np.sum(diffs == 0))

    if (n_total - n_ties) < 2:
        raise ValueError(
            f"Not enough non-tied pairs to run Wilcoxon signed-rank "
            f"({n_total - n_ties} remain after dropping {n_ties} ties out of {n_total})."
        )

    result = stats.wilcoxon(x_arr, y_arr, alternative=alternative, zero_method="wilcox")

    n_effective = n_total - n_ties
    if alternative == "two-sided":
        z_approx = stats.norm.ppf(1 - result.pvalue / 2)
    else:
        z_approx = stats.norm.ppf(1 - result.pvalue)
    effect_size_r = z_approx / np.sqrt(n_effective)
    if np.median(diffs) < 0:
        effect_size_r = -effect_size_r

    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect_size_r=float(effect_size_r),
        n_pairs=n_total,
        n_ties_dropped=n_ties,
        median_diff=float(np.median(diffs)),
        interpretation=_interpret_effect_size(effect_size_r),
    )


def compare_configurations(
    results: dict[str, list[float]],
    baseline: str,
    alternative: str = "two-sided",
) -> dict[str, WilcoxonResult]:
    """Run pairwise Wilcoxon comparisons of every configuration against a baseline.

    Args:
        results: Mapping of configuration name (e.g. window fraction label
            like '20%', '40%', ... or optimizer name) to a list of metric
            values, one per iteration, in matching iteration order across
            all configurations.
        baseline: Which key in ``results`` to compare every other key against
            (e.g. '100%' for the full-training-window complexity checkpoint,
            as the current default).
        alternative: Passed through to ``wilcoxon_signed_rank``.

    Returns:
        Mapping of configuration name (excluding the baseline) to its
        WilcoxonResult versus the baseline.
    """
    if baseline not in results:
        raise ValueError(f"Baseline {baseline!r} not found in results keys: {list(results.keys())}")

    baseline_values = results[baseline]
    comparisons = {}
    for name, values in results.items():
        if name == baseline:
            continue
        comparisons[name] = wilcoxon_signed_rank(values, baseline_values, alternative=alternative)

    return comparisons
