import numpy as np


class LMCComplexity:
    """Computes the LMC (López-Ruiz, Mancini, Calbet) complexity of numerical data."""

    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to the [0, 1] range."""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def calculate_probabilities(self, data: np.ndarray) -> np.ndarray:
        """Compute the probability distribution of the data over ``num_bins`` bins.

        Only bins with nonzero counts are returned.
        """
        counts, _ = np.histogram(data, bins=self.num_bins, density=False)
        total = np.sum(counts)
        if total == 0:
            return np.zeros(self.num_bins)
        probabilities = counts / total
        return probabilities[probabilities > 0]

    def shannon_entropy(self, probabilities: np.ndarray) -> float:
        """Compute the Shannon entropy of a probability distribution."""
        nonzero = probabilities[probabilities > 0]
        return -np.sum(nonzero * np.log(nonzero))

    def disequilibrium(self, probabilities: np.ndarray) -> float:
        """Compute the disequilibrium (distance from the uniform distribution)."""
        equilibrium_prob = 1.0 / self.num_bins
        return np.sqrt(np.sum((probabilities - equilibrium_prob) ** 2))

    def compute(self, data: np.ndarray) -> tuple[float, float, float]:
        """Compute the full LMC complexity measure.

        Args:
            data: Raw numerical data to analyze.

        Returns:
            A tuple ``(entropy, disequilibrium, complexity)``, where
            ``complexity = entropy * disequilibrium``.
        """
        normalized_data = self.normalize(data)
        probabilities = self.calculate_probabilities(normalized_data)
        entropy = self.shannon_entropy(probabilities)
        disequilibrium_value = self.disequilibrium(probabilities)
        complexity = entropy * disequilibrium_value
        return entropy, disequilibrium_value, complexity
