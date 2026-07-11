import argparse

from config.workflows import WORKFLOWS


def build_vision_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "vision", help="Run image classification transfer learning experiments."
    )
    parser.add_argument("--workflow", choices=list(WORKFLOWS.keys()), default=None)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--architecture", choices=["resnet50", "efficientnet_v2_s"], default="resnet50"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics")
    parser.add_argument("--results-dir", type=str, default="results")


def build_language_parser(subparsers) -> None:
    from config.lm_workflows import LM_WORKFLOWS

    parser = subparsers.add_parser(
        "language", help="Run language model transfer learning experiments."
    )
    parser.add_argument("--workflow", choices=list(LM_WORKFLOWS.keys()), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--tokenizer", type=str, default="roberta-base")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap the number of training windows, for quick smoke tests.",
    )


def run_vision(args: argparse.Namespace) -> None:
    from src.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        max_iterations=args.iterations,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        results_dir=args.results_dir,
        device=args.device,
        model_architecture=args.architecture,
    )

    if args.workflow:
        workflow_factory = WORKFLOWS[args.workflow]
        workflow = workflow_factory(models_dir=args.models_dir, metrics_dir=args.metrics_dir)
        orchestrator.run_workflow(workflow)
    else:
        orchestrator.run()


def run_language(args: argparse.Namespace) -> None:
    from config.lm_workflows import LM_WORKFLOWS
    from src.lm_orchestrator import LMOrchestrator

    orchestrator = LMOrchestrator(
        max_iterations=args.iterations,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        results_dir=args.results_dir,
        device=args.device,
        data_root=args.data_root,
        tokenizer_name=args.tokenizer,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
    )

    workflow_factory = LM_WORKFLOWS[args.workflow]
    workflow = workflow_factory(models_dir=args.models_dir, metrics_dir=args.metrics_dir)
    orchestrator.run_workflow(workflow)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run transfer learning experiments (vision or language models)."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    build_vision_parser(subparsers)
    build_language_parser(subparsers)

    args = parser.parse_args()

    if args.task == "vision":
        run_vision(args)
    elif args.task == "language":
        run_language(args)


if __name__ == "__main__":
    main()