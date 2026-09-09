#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ITERATIONS=5
DEVICE="cuda:1"

LM_WORKFLOWS=("wikitext_shakespeare")

# Configurações que serão executadas:
# SGD  -> 0.0001
# Adam -> 0.0001
# Adam -> 0.0005
OPTIMIZERS=("sgd" "adam" "adam")
LEARNING_RATES=("0.0001" "0.0001" "0.0005")

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

for LM_WORKFLOW in "${LM_WORKFLOWS[@]}"; do

    for i in "${!OPTIMIZERS[@]}"; do

        OPTIMIZER="${OPTIMIZERS[$i]}"
        LEARNING_RATE="${LEARNING_RATES[$i]}"

        # Pasta separada para cada configuração
        SUFFIX="${OPTIMIZER}_${LEARNING_RATE}"

        MODELS_DIR="models/${SUFFIX}/${LM_WORKFLOW}"
        RESULTS_DIR="results/${SUFFIX}/${LM_WORKFLOW}"
        METRICS_DIR="${RESULTS_DIR}/metrics"

        echo "========================================================="
        echo "Iniciando experimento (language):"
        echo "   Workflow:     ${LM_WORKFLOW}"
        echo "   Otimizador:   ${OPTIMIZER}"
        echo "   Learning Rate: ${LEARNING_RATE}"
        echo "   Iterations:   ${ITERATIONS}"
        echo "   Device:       ${DEVICE}"
        echo "========================================================="

        log_progress \
            "INICIADO: language ${LM_WORKFLOW} (opt=${OPTIMIZER}, lr=${LEARNING_RATE})"

        if uv run python -m src.main language \
            --workflow "${LM_WORKFLOW}" \
            --iterations "${ITERATIONS}" \
            --device "${DEVICE}" \
            --optmizer "${OPTIMIZER}" \
            --learning-rate "${LEARNING_RATE}" \
            --models-dir "${MODELS_DIR}" \
            --metrics-dir "${METRICS_DIR}" \
            --results-dir "${RESULTS_DIR}"; then

            echo "Experimento ${LM_WORKFLOW} (${OPTIMIZER}, lr=${LEARNING_RATE}) concluído!"

            log_progress \
                "CONCLUIDO: language ${LM_WORKFLOW} (opt=${OPTIMIZER}, lr=${LEARNING_RATE})"

        else

            echo "ERRO ao rodar ${LM_WORKFLOW} (${OPTIMIZER}, lr=${LEARNING_RATE})"
            echo "Continuando para a próxima configuração..."

            log_failure \
                "language ${LM_WORKFLOW} (opt=${OPTIMIZER}, lr=${LEARNING_RATE})"

        fi

        echo ""

    done

done

echo "========================================================="

if [ -s "${FAILURES_LOG}" ]; then

    echo "Experimentos finalizados COM FALHAS."
    echo ""
    echo "Falhas registradas em ${FAILURES_LOG}:"
    cat "${FAILURES_LOG}"

else

    echo "Todos os experimentos foram finalizados com sucesso!"

fi

echo ""
echo "Log completo de progresso em ${PROGRESS_LOG}"
echo "========================================================="
