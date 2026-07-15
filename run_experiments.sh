#!/usr/bin/env bash

set -uo pipefail

#cd /code

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ITERATIONS=5
DEVICE="cuda:1"

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

VISION_WORKFLOWS=("medmnist" "mnist")
ARCHITECTURES=("resnet50" "efficientnet_v2_s")

for WORKFLOW in "${VISION_WORKFLOWS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do

        MODELS_DIR="models/${WORKFLOW}/${ARCH}"
        RESULTS_DIR="results/${WORKFLOW}/${ARCH}"
        METRICS_DIR="${RESULTS_DIR}/metrics"

        echo "========================================================="
        echo "Iniciando experimento (vision):"
        echo "   Workflow:    ${WORKFLOW}"
        echo "   Arquitetura: ${ARCH}"
        echo "========================================================="
        log_progress "INICIADO: vision ${WORKFLOW} + ${ARCH}"

        if uv run python -m src.main vision \
            --workflow "${WORKFLOW}" \
            --iterations "${ITERATIONS}" \
            --architecture "${ARCH}" \
            --device "${DEVICE}" \
            --models-dir "${MODELS_DIR}" \
            --metrics-dir "${METRICS_DIR}" \
            --results-dir "${RESULTS_DIR}"; then
            echo "Experimento ${WORKFLOW} com ${ARCH} concluido!"
            log_progress "CONCLUIDO: vision ${WORKFLOW} + ${ARCH}"
        else
            echo "ERRO ao rodar ${WORKFLOW} + ${ARCH} -- continuando para o proximo"
            log_failure "vision ${WORKFLOW} + ${ARCH}"
        fi
        echo ""
    done
done


LM_WORKFLOWS=("wikitext_shakespeare")

for LM_WORKFLOW in "${LM_WORKFLOWS[@]}"; do

    MODELS_DIR="models/${LM_WORKFLOW}"
    RESULTS_DIR="results/${LM_WORKFLOW}"
    METRICS_DIR="${RESULTS_DIR}/metrics"

    echo "========================================================="
    echo "Iniciando experimento (language):"
    echo "   Workflow: ${LM_WORKFLOW}"
    echo "========================================================="
    log_progress "INICIADO: language ${LM_WORKFLOW}"

    if uv run python -m src.main language \
        --workflow "${LM_WORKFLOW}" \
        --iterations "${ITERATIONS}" \
        --device "${DEVICE}" \
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
