#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# S1/S2/S3 correspond to the three vision workflows in config/workflows.py.
SCENARIOS=("mnist")
ARCHITECTURES=("resnet50" "efficientnet_v2_s")
VARIANTS=(
    "adaptive_linear"
    "fixed_64"
    "fixed_256"
    "fixed_1024"
    "normalized"
    "jensen_shannon"
    "hard_binning"
    "no_bn_bias"
    "quantile_normalization"
    "mean_per_layer"
)

ITERATIONS="${ITERATIONS:-3}"
DEVICE="${DEVICE:-cuda:0}"
ROOT_DIR="${ROBUSTNESS_ROOT:-results/complexity_robustness}"
FAILURES_LOG="${ROOT_DIR}/failures.log"
MANIFEST="${ROOT_DIR}/manifest.tsv"

mkdir -p "${ROOT_DIR}"
: > "${FAILURES_LOG}"
printf "scenario\tarchitecture\tvariant\tmetrics_file\n" > "${MANIFEST}"

for scenario in "${SCENARIOS[@]}"; do
    for architecture in "${ARCHITECTURES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            run_dir="${ROOT_DIR}/${scenario}/${architecture}/${variant}"
            models_dir="${run_dir}/models"
            metrics_dir="${run_dir}/metrics"
            results_dir="${run_dir}/results"
            source_metrics="${metrics_dir}/${scenario}_all_metrics.json"

            echo "Running ${scenario} / ${architecture} / ${variant}"
            if uv run python -m src.main vision \
                --workflow "${scenario}" \
                --iterations "${ITERATIONS}" \
                --device "${DEVICE}" \
                --architecture "${architecture}" \
                --complexity-variant "${variant}" \
                --models-dir "${models_dir}" \
                --metrics-dir "${metrics_dir}" \
                --results-dir "${results_dir}"; then
                printf "%s\t%s\t%s\t%s\n" \
                    "${scenario}" "${architecture}" "${variant}" "${source_metrics}" >> "${MANIFEST}"
            else
                printf "%s\t%s\t%s\n" \
                    "$(date '+%Y-%m-%d %H:%M:%S')" \
                    "${scenario}/${architecture}/${variant}" \
                    "failed" >> "${FAILURES_LOG}"
            fi
        done
    done
done

if [[ -s "${FAILURES_LOG}" ]]; then
    echo "Some robustness runs failed; see ${FAILURES_LOG}."
    exit 1
fi

echo "Robustness matrix completed. Manifest: ${MANIFEST}"
