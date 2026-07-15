import math

import numpy as np
import torch


class LMCComplexity:
    """Computes the LMC (Lopez-Ruiz, Mancini, Calbet) complexity of a model's
    flattened weight vector, using adaptive histogram binning.

    The number of bins is chosen per the Freedman-Diaconis rule, based on the
    interquartile range (IQR) and sample size of the weight distribution,
    rather than a fixed bin count. This adapts naturally to weight vectors of
    very different scale (e.g. a ~25M-parameter CNN vs. a much larger
    Transformer language model), which a fixed bin count would not.

    Operates natively on torch tensors (GPU-resident when the input tensor
    is) to avoid an unnecessary CPU round-trip for large parameter vectors.
    """

    def __init__(self, max_bin_sample_size: int = 10_000):
        """
        Args:
            max_bin_sample_size: Maximum number of weights sampled to estimate
                the IQR used for bin-width selection. Sampling (rather than
                using the full weight vector) keeps this step fast for very
                large models, at negligible cost to bin-width accuracy.
        """
        self.max_bin_sample_size = max_bin_sample_size

    def compute(self, weights: torch.Tensor | np.ndarray) -> tuple[float, float, float]:
        """Compute the full LMC complexity measure for a weight vector.

        Args:
            weights: A 1-D tensor of model weights (e.g. from
                ``torch.nn.utils.parameters_to_vector``). Any device/dtype is
                accepted; computation stays on the input tensor's device.

        Returns:
            A tuple ``(entropy, disequilibrium, complexity)``, where
            ``complexity = entropy * disequilibrium``.
        """
        weights = torch.as_tensor(weights)

        probabilities = self._compute_probabilities(weights)
        entropy = self._shannon_entropy(probabilities)
        disequilibrium = self._disequilibrium(probabilities)
        complexity = entropy * disequilibrium
        return entropy, disequilibrium, complexity

    def _compute_probabilities(self, weights: torch.Tensor) -> torch.Tensor:
        """Build a probability distribution over adaptively-sized bins.

        Each weight contributes to its two nearest bin centers with linear
        interpolation weights, which reduces quantization noise relative to
        hard bin assignment -- useful given that the bin count itself is
        derived from the data and can be small for very large models.
        """
        weights_min = weights.min()
        weights_max = weights.max()
        data_range = weights_max - weights_min

        if data_range == 0:
            # Degenerate case: all weights identical (e.g. right after
            # zero-initialization). Treat as a single-bin distribution.
            return torch.ones(1, device=weights.device, dtype=weights.dtype)

        normalized = (weights - weights_min) / (data_range + 1e-10)
        num_bins = self._select_num_bins(normalized)

        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float32, device=weights.device)
        bin_width = bin_edges[1] - bin_edges[0]

        bin_pos = normalized / bin_width
        bin_floor = torch.floor(bin_pos)
        bin_left = bin_floor.long().clamp(0, num_bins - 1)
        bin_right = (bin_floor + 1).long().clamp(0, num_bins - 1)

        frac = bin_pos - bin_floor
        weight_left = 1.0 - frac
        weight_right = frac

        hist = torch.zeros(num_bins, device=weights.device, dtype=torch.float32)
        hist.scatter_add_(0, bin_left, weight_left)
        hist.scatter_add_(0, bin_right, weight_right)

        probabilities = hist / hist.sum()

        eps = 1e-10
        probabilities = torch.clamp(probabilities, eps, 1.0)
        return probabilities / probabilities.sum()

    def _select_num_bins(self, normalized_weights: torch.Tensor) -> int:
        """Select the bin count via the Freedman-Diaconis rule.

        Falls back to Sturges' rule (sqrt of sample size) when the IQR is
        zero, which the Freedman-Diaconis rule cannot handle directly (it
        would imply a zero bin width).
        """
        n = normalized_weights.numel()
        sample_size = min(self.max_bin_sample_size, n)

        if n > sample_size:
            indices = torch.randperm(n, device=normalized_weights.device)[:sample_size]
            sample = normalized_weights[indices]
        else:
            sample = normalized_weights

        q1 = torch.quantile(sample.float(), 0.25)
        q3 = torch.quantile(sample.float(), 0.75)
        iqr = (q3 - q1).item()

        if iqr == 0:
            return max(1, int(np.ceil(math.sqrt(n))))

        bin_width = 2 * iqr * (n ** (-1 / 3))
        return max(1, int(np.ceil(1.0 / bin_width)))

    def _shannon_entropy(self, probabilities: torch.Tensor) -> float:
        """Compute the Shannon entropy of a probability distribution."""
        return -(probabilities * torch.log(probabilities)).sum().item()

    def _disequilibrium(self, probabilities: torch.Tensor) -> float:
        """Compute the disequilibrium (squared distance from uniform)."""
        num_bins = probabilities.numel()
        uniform_prob = 1.0 / num_bins
        return ((probabilities - uniform_prob) ** 2).sum().item()
