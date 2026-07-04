# Complexity Transfer Learning Experiments

[![ORCID](https://img.shields.io/badge/ORCID-XXXX--XXXX--XXXX--XXXX-brightgreen)](https://orcid.org/0000-0003-2018-8007)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> **Companion repository for the paper: "Internal Pattern Complexity as an Optimal Moment Marker for Transfer Learning"**

## Abstract

Transfer learning is typically applied by selecting a pretrained checkpoint based on validation loss, with little consideration for what internal state of the network is actually being transferred. This repository provides the experimental pipeline used to test an alternative selection criterion grounded in statistical complexity theory: whether a source-model checkpoint selected by **maximum LMC (López-Ruiz–Mancini–Calbet) complexity** of its weight distribution transfers as effectively as—or more effectively than—one selected by **minimum validation loss**. Experiments are conducted across two source→target dataset pairs (TissueMNIST→BloodMNIST and MNIST→FashionMNIST) and two convolutional architectures (ResNet-50 and EfficientNetV2-S), with all runs tracked via MLflow for full reproducibility.

## Research Question

> Do neural network checkpoints selected by maximum statistical complexity (LMC) of their weight distribution yield better transfer learning outcomes than checkpoints selected by minimum validation loss?

## Background: The LMC Complexity Measure

The LMC measure (López-Ruiz, Mancini & Calbet, 1995) quantifies the statistical complexity of a probability distribution by combining two complementary notions:

- **Shannon entropy (H)** — quantifies the degree of disorder or information content of the distribution. Entropy alone is maximized both by a perfectly uniform distribution (high disorder, low structure) and can be low for a highly ordered but trivial one; it does not by itself distinguish "structured" from "trivial" regimes.
- **Disequilibrium (D)** — quantifies the distance of the distribution from a uniform reference distribution.

The LMC complexity is defined as the product of the two:

**C = H × D**

The underlying intuition is that genuinely complex systems — including, we hypothesize, the weight distributions of neural networks at meaningful stages of training — occupy an intermediate regime: neither maximally disordered (high entropy, low disequilibrium) nor maximally ordered/uniform (low entropy). In this codebase, the measure is applied to the flattened vector of a model's trained parameters (`utils/complexity.py`), treating the weight distribution as the signal of interest.

## System Architecture

```
CLI (src/main.py)
    │
    ▼
Orchestrator (src/orchestrator.py)
    │
    ├── run()          → standard loop over MODEL_CONFIGS
    │
    └── run_workflow() → transfer learning pipeline
         │
         ▼
       Experiment (src/experiment.py)
         │
         ├── DatasetManager (src/dataset.py) → DataLoader
         ├── ModelFactory (src/model.py) → nn.Module
         ├── Trainer (src/trainer.py) → train_epoch / validate_epoch
         │      └── LMCComplexity (utils/complexity.py)
         ├── MLflow logging
         └── JSON results
              │
              ▼
       StatisticsCalculator (utils/statistics.py) → mean ± std
```

## Methodology

### Transfer Learning Workflows

Each workflow executes four stages designed to isolate the effect of checkpoint selection criterion from the effect of transfer learning itself:

| Stage | Description | Purpose |
|-------|-------------|---------|
| 1. Source Pretraining | Train a model on the source dataset from random initialization | Produces the candidate checkpoints |
| 2. Target Control (No Transfer) | Train on the target dataset from random initialization | Baseline: quantifies target-task difficulty without any transfer |
| 3. Target + Min Loss | Initialize target training from the source checkpoint with lowest validation loss | Standard/conventional selection criterion |
| 4. Target + Max Complexity | Initialize target training from the source checkpoint with highest LMC complexity | Criterion under investigation |

All target-stage checkpoints are loaded with `strict=False`, since the final classification layer differs in shape between source and target tasks; only compatible layers are transferred, and the classification head is reinitialized.

### Available Workflows

| Workflow | Source Dataset | Target Dataset | Source Epochs | Target Epochs |
|----------|-----------------|-----------------|:---:|:---:|
| `medmnist` | TissueMNIST (8 classes) | BloodMNIST (8 classes) | 50 | 10 |
| `mnist` | MNIST (10 classes) | FashionMNIST (10 classes) | 50 | 10 |

### Checkpoint Selection

During source-dataset training, two checkpoints are tracked and persisted independently, each updated whenever a new best value is observed for its respective criterion:

- `{dataset}_min_loss.pth` — model weights from the epoch with the lowest validation loss
- `{dataset}_max_complexity.pth` — model weights from the epoch with the highest LMC complexity

### Statistical Reporting

To account for stochasticity in training (weight initialization, data shuffling, and GPU non-determinism), each configuration is repeated across `N` independent iterations. Final results are reported as **mean ± sample standard deviation (ddof = 1)** across iterations, computed by `utils/statistics.py`.

## Model Architectures

| Architecture | Input Adaptation | Notes |
|--------------|-------------------|-------|
| **ResNet-50** | First convolution modified to accept 1-channel (grayscale) input | Final fully connected layer resized to the target number of classes |
| **EfficientNetV2-S** | First convolution modified to accept 1-channel (grayscale) input | Final classifier layer resized to the target number of classes |

Both architectures are instantiated from `torchvision.models` **without** ImageNet-pretrained weights, so that any transfer-learning effect observed in the experiments is attributable solely to the source-task pretraining performed within this pipeline, not to external pretraining.

## Requirements

- Python >= 3.13
- PyTorch with CUDA 12.6 support (for GPU-accelerated training)
- An NVIDIA GPU with CUDA support (recommended; CPU execution is supported for testing but not practical for full-scale runs)

## Installation

### With uv (recommended)

```bash
git clone git@github.com:pultrini/complexity-transfer-learning-experiment.git
cd complexity-transfer-learning-experiments

uv sync
```

### With pip

```bash
git clone git@github.com:pultrini/complexity-transfer-learning-experiment.git
cd complexity-transfer-learning-experiments

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### Core Dependencies

- `medmnist` >= 3.0.2 — standardized biomedical image datasets
- `mlflow` >= 3.14.0 — experiment tracking and reproducibility
- `numpy` >= 2.5.0 — numerical computation
- `torch` >= 2.11.0 — deep learning framework
- `torchvision` >= 0.26.0 — vision model architectures and datasets

## Usage

### Local Execution

```bash
# Run the MedMNIST workflow with 10 statistical iterations
python -m src.main --workflow medmnist --iterations 10

# Run the MNIST workflow with 5 iterations
python -m src.main --workflow mnist --iterations 5

# Run with a specific architecture
python -m src.main --workflow medmnist --architecture efficientnet_v2_s --iterations 10

# Run on CPU (no GPU available)
python -m src.main --workflow medmnist --device cpu --iterations 3
```

### CLI Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--workflow` | Workflow to execute | `medmnist`, `mnist` | None |
| `--iterations` | Number of statistical iterations | Any positive integer | 1 |
| `--architecture` | Model architecture | `resnet50`, `efficientnet_v2_s` | `resnet50` |
| `--device` | Training device | `cpu`, `cuda`, `cuda:N` | Auto-detected |
| `--models-dir` | Checkpoint output directory | Any path | `models/` |
| `--metrics-dir` | Per-run metrics output directory | Any path | `results/metrics/` |
| `--results-dir` | Final aggregated results directory | Any path | `results/` |

### Containerized Execution (Docker)

Docker Compose provisions a complete, isolated experimental environment — MLflow tracking server, MinIO (S3-compatible artifact storage), and PostgreSQL (MLflow backend store) — enabling fully reproducible runs independent of the host environment:

```bash
docker compose up --build
```

MLflow UI: `http://localhost:5000`

#### Services

| Service | Description | Port |
|---------|-------------|------|
| `mlflow` | Experiment tracking server | 5000 |
| `postgres` | MLflow metadata backend store | 5432 |
| `minio` | S3-compatible artifact storage | 9000 / 9001 |
| `minio-init` | One-shot creation of the `mlflow` bucket | — |
| `app` | GPU-enabled training container | — |

## Repository Structure

```
complexity_transfer_learning_experiments/
├── config/
│   ├── __init__.py
│   ├── dataset_registry.py    # Registry of supported datasets and their metadata
│   ├── experiment_config.py   # Per-run experiment configuration (dataclass)
│   └── workflows.py           # Transfer learning workflow definitions
├── src/
│   ├── __init__.py
│   ├── dataset.py             # DatasetManager: loads MedMNIST and torchvision datasets
│   ├── experiment.py          # Experiment: orchestrates a single training run with MLflow
│   ├── main.py                # CLI entry point
│   ├── model.py                # ModelFactory: builds ResNet-50 / EfficientNetV2-S
│   ├── orchestrator.py        # Orchestrator: runs multiple iterations and aggregates results
│   └── trainer.py             # Trainer: training/validation loop with complexity tracking
├── utils/
│   ├── complexity.py          # LMCComplexity: computes the LMC complexity measure
│   └── statistics.py          # StatisticsCalculator: mean/std aggregation across iterations
├── data/                      # Datasets (gitignored)
├── models/                    # Checkpoints (gitignored)
├── results/                   # Metrics and aggregated results (gitignored)
├── docker-compose.yml         # Full stack: MLflow + MinIO + Postgres + GPU-enabled app
├── Dockerfile                 # Multi-stage build: MLflow server + training app
├── pyproject.toml             # Project metadata and dependency specification
└── README.md                  # This file
```

## Metrics and Results

### Per-Epoch Metrics

Logged to MLflow at every training epoch:

- `val_accuracy` — validation set accuracy
- `val_loss` — validation set loss
- `complexity` — LMC complexity of the model's weight distribution (H × D)
- `disequilibrium` — the D component of LMC complexity
- `entropy` — the H (Shannon entropy) component of LMC complexity

### Final Aggregation

After all iterations of a given configuration complete, the following are computed across iterations:

- **Mean** of each metric
- **Sample standard deviation** (std, ddof = 1) of each metric

### Output Artifacts

```
results/
├── metrics/
│   ├── {workflow}_{dataset}_{step}.json    # Per-iteration metrics
│   └── {workflow}_all_metrics.json         # All metrics aggregated across iterations
├── {workflow}_results.json                 # Final mean ± std summary per workflow
└── results.json                            # Final results for the standard (non-workflow) mode
```

### Experiment Tracking

All runs are logged to MLflow, providing:

- Side-by-side comparison across runs, architectures, and checkpoint-selection strategies
- Convergence plots for accuracy, loss, and complexity
- Versioned artifacts (checkpoints, result files) attached to each run

## Reproducibility Statement

All experiments in this repository are fully parameterized and seed-controlled at the configuration level, with every run's hyperparameters, metrics, and artifacts persisted to MLflow. The Docker Compose environment is provided specifically to allow independent verification of results without dependency on the original host system's software environment.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pultrini2026complexity,
  title={[PAPER TITLE]},
  author={Pultrini, Davi and Scatena, Rafael Oddone and Murta Jr., Luiz Otavio},
  journal={[JOURNAL NAME]},
  year={2026},
  note={Repository: https://github.com/pultrini/complexity-transfer-learning-experiment}
}
```

## Authors

- **Davi Pultrini** — University of São Paulo (USP)
  ORCID: [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
- **Rafael Oddone Scatena**
  ORCID: [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
- **Luiz Otavio Murta Jr.**
  ORCID: [XXXX-XXXX-XXXX-XXXX](https://orcid.org/XXXX-XXXX-XXXX-XXXX)

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.