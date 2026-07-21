#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ITERATIONS=5
DEVICE="cuda:1"

# Ajuste aqui o otimizador e o learning rate para esta rodada.
OPTIMIZER="adamw"        # "adam" | "adamw" | "sgd"
LEARNING_RATE="0.0005"

FAILURES_LOG="failures.log"
PROGRESS_LOG="progress.log"
: > "${FAILURES_LOG}"
: > "${PROGRESS_LOG}"

log_progress() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "${PROGRESS_LOG}"
}

log_failure() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - FALHOU: $1" >> "${FAILURES_LOG}"
}

VISION_WORKFLOWS=("medmnist" "mnist" "tinyimagenet_catsdogs")
ARCHITECTURES=("resnet50" "efficientnet_v2_s")

for WORKFLOW in "${VISION_WORKFLOWS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do

        # Grava em pastas separadas por otimizador+LR, para nao sobrescrever
        # resultados de outras rodadas com hiperparametros diferentes.
        MODELS_DIR="models/${OPTIMIZER}_${LEARNING_RATE}/${WORKFLOW}/${ARCH}"
        RESULTS_DIR="results/${OPTIMIZER}_${LEARNING_RATE}/${WORKFLOW}/${ARCH}"
        METRICS_DIR="${RESULTS_DIR}/metrics"

        echo "========================================================="
        echo "Iniciando experimento (vision):"
        echo "   Workflow:    ${WORKFLOW}"
        echo "   Arquitetura: ${ARCH}"
        echo "   Otimizador:  ${OPTIMIZER}"
        echo "   Learning Rate: ${LEARNING_RATE}"
        echo "========================================================="
        log_progress "INICIADO: vision ${WORKFLOW}+${ARCH} (opt=${OPTIMIZER}, lr=${LEARNING_RATE})"

        if uv run python -m src.main vision \
            --workflow "${WORKFLOW}" \
            --iterations "${ITERATIONS}" \
            --architecture "${ARCH}" \
            --device "${DEVICE}" \
            --optmizer "${OPTIMIZER}" \
            --learning-rate "${LEARNING_RATE}" \
            --models-dir "${MODELS_DIR}" \
            --metrics-dir "${METRICS_DIR}" \
            --results-dir "${RESULTS_DIR}"; then
            echo "Experimento ${WORKFLOW} com ${ARCH} concluido!"
            log_progress "CONCLUIDO: vision ${WORKFLOW}+${ARCH}"
        else
            echo "ERRO ao rodar ${WORKFLOW} + ${ARCH} -- continuando para o proximo"
            log_failure "vision ${WORKFLOW}+${ARCH}"
        fi
        echo ""
    done
done

LM_WORKFLOWS=("wikitext_shakespeare")

for LM_WORKFLOW in "${LM_WORKFLOWS[@]}"; do

    # Agora o subcomando "language" ja aceita --optmizer tambem.
    MODELS_DIR="models/${OPTIMIZER}_${LEARNING_RATE}/${LM_WORKFLOW}"
    RESULTS_DIR="results/${OPTIMIZER}_${LEARNING_RATE}/${LM_WORKFLOW}"
    METRICS_DIR="${RESULTS_DIR}/metrics"

    echo "========================================================="
    echo "Iniciando experimento (language):"
    echo "   Workflow: ${LM_WORKFLOW}"
    echo "   Otimizador: ${OPTIMIZER}"
    echo "   Learning Rate: ${LEARNING_RATE}"
    echo "========================================================="
    log_progress "INICIADO: language ${LM_WORKFLOW} (opt=${OPTIMIZER}, lr=${LEARNING_RATE})"

    if uv run python -m src.main language \
        --workflow "${LM_WORKFLOW}" \
        --iterations "${ITERATIONS}" \
        --device "${DEVICE}" \
        --optmizer "${OPTIMIZER}" \
        --learning-rate "${LEARNING_RATE}" \
        --models-dir "${MODELS_DIR}" \
        --metrics-dir "${METRICS_DIR}" \
        --results-dir "${RESULTS_DIR}"; then
        echo "Experimento ${LM_WORKFLOW} concluido!"
        log_progress "CONCLUIDO: language ${LM_WORKFLOW}"
    else
        echo "ERRO ao rodar ${LM_WORKFLOW} -- continuando para o proximo"
        log_failure "language ${LM_WORKFLOW}"
    fi
    echo ""
done

echo "========================================================="
if [ -s "${FAILURES_LOG}" ]; then
    echo "Experimentos finalizados COM FALHAS. Veja ${FAILURES_LOG}:"
    cat "${FAILURES_LOG}"
else
    echo "Todos os experimentos foram finalizados com sucesso!"
fi
echo "Log completo de progresso em ${PROGRESS_LOG}"
echo "========================================================="
