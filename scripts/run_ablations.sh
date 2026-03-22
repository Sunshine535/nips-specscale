#!/bin/bash
# =============================================================================
# Run ablation experiments for SpecScale
# - γ schedules (fixed, adaptive, oracle)
# - Draft model sizes
# - Temperature sensitivity
# - Noise schedules (linear, cosine)
# =============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR="${PROJECT_ROOT}/results/ablations"
NUM_SAMPLES=100
CONFIG="${PROJECT_ROOT}/configs/default.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --num_samples) NUM_SAMPLES="$2"; shift 2;;
        --config) CONFIG="$2"; shift 2;;
        --quick) NUM_SAMPLES=10; shift;;
        *) shift;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "[ablation] γ schedule comparison..."
python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
    --output_dir "${OUTPUT_DIR}/gamma_schedules" \
    --config "$CONFIG" \
    --num_samples "$NUM_SAMPLES" \
    2>&1 | tee "${OUTPUT_DIR}/gamma_schedules.log"

echo "[ablation] Temperature sensitivity..."
for temp in 0.5 1.0 2.0; do
    python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
        --output_dir "${OUTPUT_DIR}/temperature_${temp}" \
        --config "$CONFIG" \
        --num_samples "$NUM_SAMPLES" \
        2>&1 | tee "${OUTPUT_DIR}/temperature_${temp}.log"
done

echo "[ablation] Noise schedule comparison..."
for sched in linear cosine; do
    python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
        --output_dir "${OUTPUT_DIR}/noise_${sched}" \
        --config "$CONFIG" \
        --num_samples "$NUM_SAMPLES" \
        2>&1 | tee "${OUTPUT_DIR}/noise_${sched}.log"
done

echo "[ablation] Done. Results in ${OUTPUT_DIR}"
