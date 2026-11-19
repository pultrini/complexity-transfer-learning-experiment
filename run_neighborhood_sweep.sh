#!/usr/bin/env bash
# Sweep de checkpoints na vizinhanca do ponto de transferencia.
# Para cada iteracao: pre-treina o source (como no workflow principal), guarda os
# checkpoints t*-R..t*+R ao redor do Max-C e do Min-loss e faz fine-tuning no
# target a partir de cada um. Depois roda a analise (regret, posto, Kendall tau).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Raise the open-file limit for this shell (up to the hard limit) to avoid
# "Too many open files" with 8 persistent DataLoader workers per loader.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65535 2>/dev/null || true
echo "open-file limit: $(ulimit -n)"

ITERATIONS="${ITERATIONS:-5}"
DEVICE="${DEVICE:-cuda:0}"
OPTIMIZER="${OPTIMIZER:-adam}"          # "adam" | "adamw" | "sgd"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"

# Raio da vizinhanca (epocas) e espacamento entre os offsets.
# R=3, STEP=1 -> offsets -3,-2,-1,+1,+2,+3 por ancora (12 fine-tunings extras
# por iteracao com as duas ancoras).
SWEEP_RADIUS="${SWEEP_RADIUS:-3}"
SWEEP_STEP="${SWEEP_STEP:-1}"
SWEEP_ANCHORS=(${SWEEP_ANCHORS:-max_complexity min_loss})

VISION_WORKFLOWS=(${WORKFLOWS:-medmnist mnist tinyimagenet_catsdogs})
ARCHITECTURES=(${ARCHITECTURES:-resnet50 efficientnet_v2_s})

FAILURES_LOG="sweep_failures.log"
PROGRESS_LOG="sweep_progress.log"
: > "${FAILURES_LOG}"
: > "${PROGRESS_LOG}"
log_progress() { echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "${PROGRESS_LOG}"; }
log_failure()  { echo "$(date '+%Y-%m-%d %H:%M:%S') - FALHOU: $1" >> "${FAILURES_LOG}"; }

for WORKFLOW in "${VISION_WORKFLOWS[@]}"; do
    for ARCH in "${ARCHITECTURES[@]}"; do
        SUFFIX="${OPTIMIZER}_${LEARNING_RATE}_sweep_r${SWEEP_RADIUS}_s${SWEEP_STEP}"
        MODELS_DIR="models/${SUFFIX}/${WORKFLOW}/${ARCH}"
        RESULTS_DIR="results/${SUFFIX}/${WORKFLOW}/${ARCH}"
        METRICS_DIR="${RESULTS_DIR}/metrics"

        echo "=== sweep: ${WORKFLOW} / ${ARCH} (opt=${OPTIMIZER}, lr=${LEARNING_RATE}, R=${SWEEP_RADIUS}) ==="
        log_progress "INICIADO: ${WORKFLOW}+${ARCH} ${SUFFIX}"

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
            --sweep-radius "${SWEEP_RADIUS}" \
            --sweep-step "${SWEEP_STEP}" \
            --sweep-anchors "${SWEEP_ANCHORS[@]}"; then
            log_progress "CONCLUIDO: ${WORKFLOW}+${ARCH}"
            uv run python scripts/analyze_neighborhood_sweep.py \
                "${METRICS_DIR}/${WORKFLOW}_all_metrics.json" \
                --out-dir "${RESULTS_DIR}/sweep_analysis" \
                --label "${WORKFLOW}/${ARCH}/${OPTIMIZER}_${LEARNING_RATE}" \
                || log_failure "analise ${WORKFLOW}+${ARCH}"
        else
            log_failure "${WORKFLOW}+${ARCH} ${SUFFIX}"
        fi
    done
done
