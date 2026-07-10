import json
import math
from pathlib import Path

VISION_METRICS: dict[str, str] = {
    "accuracy": "Max_accuracy",
    "loss": "Min_loss",
    "complexity": "Max_complexity",
}


LM_METRICS: dict[str, str] = {
    "loss": "Min_loss",
    "complexity": "Max_complexity",
}


class StatisticsCalculator:
    """Computes mean and standard deviation from aggregated round metrics."""

    def __init__(
        self,
        model_names: list[str],
        ddof: int = 0,
        metrics: dict[str, str] | None = None,
    ):
        """
        Args:
            model_names: Names of the models/steps to compute statistics for.
            ddof: Delta degrees of freedom for the standard deviation, matching
                NumPy's convention. Use 0 for population std (divide by N) or
                1 for sample std (divide by N-1).
            metrics: Mapping of internal metric key to the field name found in
                the raw per-round data (e.g. {"loss": "Min_loss"}). Defaults
                to ``VISION_METRICS`` for backward compatibility; pass
                ``LM_METRICS`` for language model experiments, which have no
                accuracy field.
        """
        self.model_names = model_names
        self.ddof = ddof
        self.metrics = metrics if metrics is not None else VISION_METRICS

    def calculate(self, data: dict) -> dict[str, dict[str, str]]:
        """Compute mean and standard deviation for each model and metric."""
        num_rounds = len(data)
        if num_rounds == 0:
            return {}

        means = self._compute_means(data, num_rounds)
        stds = self._compute_stds(data, means, num_rounds)

        return {
            model: {
                field: f"{means[metric][model]:.5f} +- {stds[metric][model]:.5f}"
                for metric, field in self.metrics.items()
            }
            for model in self.model_names
        }

    def _compute_means(self, data: dict, num_rounds: int) -> dict[str, dict[str, float]]:
        sums = {metric: dict.fromkeys(self.model_names, 0.0) for metric in self.metrics}

        for round_data in data.values():
            for model in self.model_names:
                for metric, field in self.metrics.items():
                    sums[metric][model] += round_data[model].get(field, 0.0)

        return {
            metric: {model: total / num_rounds for model, total in totals.items()}
            for metric, totals in sums.items()
        }

    def _compute_stds(
        self, data: dict, means: dict[str, dict[str, float]], num_rounds: int
    ) -> dict[str, dict[str, float]]:
        sum_sq_diffs = {metric: dict.fromkeys(self.model_names, 0.0) for metric in self.metrics}

        for round_data in data.values():
            for model in self.model_names:
                for metric, field in self.metrics.items():
                    value = round_data[model].get(field, 0.0)
                    sum_sq_diffs[metric][model] += (value - means[metric][model]) ** 2

        denom = max(num_rounds - self.ddof, 1)

        return {
            metric: {model: math.sqrt(total / denom) for model, total in totals.items()}
            for metric, totals in sum_sq_diffs.items()
        }

    def save(self, results: dict, output_path: str) -> None:
        """Save the results to a JSON file."""
        Path(output_path).write_text(json.dumps(results, indent=4), encoding="utf-8")
