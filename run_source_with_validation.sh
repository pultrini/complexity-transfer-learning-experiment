#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ajuste estas configuracoes para a rodada desejada.
ITERATIONS=5
DEVICE="cuda:0"
OPTIMIZER="adam"       # "adam" | "adamw" | "sgd"
LEARNING_RATE="0.001"
USE_WINDOWS=false

VISION_WORKFLOWS=("medmnist" "mnist" "tinyimagenet_catsdogs")
ARCHITECTURES=("resnet50" "efficientnet_v2_s")

FAILURES_LOG="failures_source_with_validation.log"
PROGRESS_LOG="progress_source_with_validation.log"
: > "${FAILURES_LOG}"
: > "${PROGRESS_LOG}"

log_progress() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "${PROGRESS_LOG}"
}

log_failure() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - FALHOU: $1" >> "${FAILURES_LOG}"
}

USE_WINDOWS_FLAG=()
if [ "${USE_WINDOWS}" = true ]; then
    USE_WINDOWS_FLAG=(--use-windows)
fi

for WORKFLOW in "${VISION_WORKFLOWS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        SUFFIX="${OPTIMIZER}_${LEARNING_RATE}_source_with_validation"
        if [ "${USE_WINDOWS}" = true ]; then
            SUFFIX="${SUFFIX}_windows"
        fi

        MODELS_DIR="models/${SUFFIX}/${WORKFLOW}/${ARCH}"
        RESULTS_DIR="results/${SUFFIX}/${WORKFLOW}/${ARCH}"
        METRICS_DIR="${RESULTS_DIR}/metrics"

        echo "========================================================="
        echo "Iniciando experimento com validacao incorporada ao source:"
        echo "   Workflow:      ${WORKFLOW}"
        echo "   Arquitetura:   ${ARCH}"
        echo "   Otimizador:    ${OPTIMIZER}"
        echo "   Learning rate: ${LEARNING_RATE}"
        echo "   Use windows:   ${USE_WINDOWS}"
        echo "========================================================="

        RUN_DESCRIPTION="vision ${WORKFLOW}+${ARCH} (opt=${OPTIMIZER}, lr=${LEARNING_RATE}, windows=${USE_WINDOWS}, source=train+validation)"
        log_progress "INICIADO: ${RUN_DESCRIPTION}"

        if uv run python -m src.main vision \
            --workflow "${WORKFLOW}" \
            --iterations "${ITERATIONS}" \
            --architecture "${ARCH}" \
            --device "${DEVICE}" \
            --optmizer "${OPTIMIZER}" \
            --learning-rate "${LEARNING_RATE}" \
            --models-dir "${MODELS_DIR}" \
            --metrics-dir "${METRICS_DIR}" \
            --results-dir "${RESULTS_DIR}" \
            --source-train-with-validation \
            "${USE_WINDOWS_FLAG[@]}"; then
            echo "Experimento ${WORKFLOW} com ${ARCH} concluido!"
            log_progress "CONCLUIDO: ${RUN_DESCRIPTION}"
        else
            echo "ERRO ao rodar ${WORKFLOW} + ${ARCH} -- continuando para o proximo"
            log_failure "${RUN_DESCRIPTION}"
        fi
        echo ""
    done
done

echo "========================================================="
if [ -s "${FAILURES_LOG}" ]; then
    echo "Experimentos finalizados COM FALHAS. Veja ${FAILURES_LOG}:"
    cat "${FAILURES_LOG}"
else
    echo "Todos os experimentos foram finalizados com sucesso!"
fi
echo "Log de progresso: ${PROGRESS_LOG}"
echo "========================================================="
