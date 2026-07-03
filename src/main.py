import argparse

from config.workflows import WORKFLOWS
from src.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa experimentos de treinamento e transfer learning."
    )
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOWS.keys()),
        default=None,
        help=(
            "Nome do workflow de transfer learning a rodar (ex: 'medmnist', 'mnist'). "
            "Se omitido, roda o loop padrão com todos os MODEL_CONFIGS."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Número de iterações estatísticas a rodar (default: 5).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Diretório onde os checkpoints são salvos/lidos (default: 'models').",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="results/metrics",
        help="Diretório onde as métricas por rodada são salvas (default: 'results/metrics').",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Diretório onde os resultados finais agregados são salvos (default: 'results').",
    )
    parser.add_argument(
        "--architecture",
        choices=["resnet50", "efficientnet_v2_s"],
        default="resnet50",
        help="Arquitetura do modelo a treinar (default: 'resnet50').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device a usar, ex: 'cuda:0', 'cuda:1', 'cpu'. Se omitido, escolhe cuda:0 automaticamente.",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
