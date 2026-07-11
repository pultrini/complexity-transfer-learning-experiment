from dataclasses import dataclass, field
from typing import Literal

Strategy = Literal["normal", "complexity", "min_loss"]


@dataclass
class LMWorkflowStep:
    """A single step within a language-model transfer learning workflow."""

    name: str
    dataset_name: str
    num_epochs: int
    strategy: Strategy
    checkpoint_path: str | None = None
    output_file: str | None = None
    mlflow_run_name: str | None = None


@dataclass
class LMWorkflow:
    """A complete language-model transfer learning workflow."""

    name: str
    mlflow_experiment_name: str
    source_dataset: str
    target_dataset: str
    steps: list[LMWorkflowStep] = field(default_factory=list)

    DEFAULT_TARGET_EPOCHS = 10

    @classmethod
    def create_wikitext_shakespeare_workflow(
        cls,
        models_dir: str = "models",
        metrics_dir: str = "results/metrics",
        source_epochs: int = 10,
        target_epochs: int = 10,
    ) -> "LMWorkflow":
        """Create the WikiText-2 -> Tiny-Shakespeare transfer learning workflow.

        Requires 'tiny_shakespeare' to have train/val files configured in
        LM_DATASET_REGISTRY (see config/lm_registry.py) -- the original
        Tiny-Shakespeare source only ships a test split.
        """
        source = "wikitext2"
        target = "tiny_shakespeare"

        return cls(
            name="wikitext_shakespeare",
            mlflow_experiment_name="wikitext_shakespeare_transfer",
            source_dataset=source,
            target_dataset=target,
            steps=[
                LMWorkflowStep(
                    name="source",
                    dataset_name=source,
                    num_epochs=source_epochs,
                    strategy="normal",
                    checkpoint_path=None,
                    output_file=f"{metrics_dir}/wikitext_shakespeare_{source}_source.json",
                    mlflow_run_name=f"{source}_source_run",
                ),
                LMWorkflowStep(
                    name="target_no_transfer",
                    dataset_name=target,
                    num_epochs=target_epochs,
                    strategy="normal",
                    checkpoint_path=None,
                    output_file=f"{metrics_dir}/wikitext_shakespeare_{target}_no_transfer.json",
                    mlflow_run_name=f"{target}_no_transfer_run",
                ),
                LMWorkflowStep(
                    name="target_min_loss",
                    dataset_name=target,
                    num_epochs=target_epochs,
                    strategy="normal",
                    checkpoint_path=f"{models_dir}/{source}_min_loss.pth",
                    output_file=f"{metrics_dir}/wikitext_shakespeare_{target}_min_loss.json",
                    mlflow_run_name=f"{target}_min_loss_run",
                ),
                LMWorkflowStep(
                    name="target_max_complexity",
                    dataset_name=target,
                    num_epochs=target_epochs,
                    strategy="normal",
                    checkpoint_path=f"{models_dir}/{source}_max_complexity.pth",
                    output_file=f"{metrics_dir}/wikitext_shakespeare_{target}_max_complexity.json",
                    mlflow_run_name=f"{target}_max_complexity_run",
                ),
            ],
        )


LM_WORKFLOWS = {
    "wikitext_shakespeare": LMWorkflow.create_wikitext_shakespeare_workflow,
}
