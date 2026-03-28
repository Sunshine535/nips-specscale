#!/usr/bin/env bash
# =============================================================================
# SpecScale — Master experiment pipeline
#
# Merges SpecDraft (AR LLM speculative decoding scaling laws) and SpecDiff
# (DiT speculative denoising scaling laws) under one unified S(d,T,γ) narrative,
# with cross-modality comparison and ImageNet FID reporting.
#
# Phases:
#   0  Download all models (LLM draft/target pairs + DiT checkpoints)
#   1  LLM measurement sweep — 11 pairs × 7 γ × 4 datasets × 3 seeds
#   2  LLM scaling-law fit S(d,T,γ) — run_scaling_law_llm.py
#   3  DiT acceptance sweep — 3 pairs × 5 γ × 3 guidance × 3 seeds
#   4  DiT scaling-law fit with timestep modulation h(t) — run_scaling_law_dit.py
#   5  Cross-modality unified analysis — run_unified_comparison.py
#   6  DiT ImageNet 256×256 evaluation (FID-50K / IS) — run_imagenet_eval.py
#   7  Ablations — γ schedules, temperatures, noise schedules
#   8  Final paper figures + tables — generate_paper_figures_tables.py
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   QUICK=--quick bash scripts/run_all_experiments.sh  # (dev only)
#   bash scripts/run_all_experiments.sh --from-phase 3
#   bash scripts/run_all_experiments.sh --only-phase 5
#   bash scripts/run_all_experiments.sh --dry-run
#
# Requires: NVIDIA GPUs (see README). Sources standard gpu_utils.sh from either
#   ../_shared/gpu_utils.sh (monorepo layout) or scripts/gpu_utils.sh (standalone).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# --- Standard GPU utilities (monorepo _shared OR local copy) ---
if [[ -f "$PROJECT_ROOT/../_shared/gpu_utils.sh" ]]; then
  # github_repos/nips-specscale → github_repos/_shared/
  source "$PROJECT_ROOT/../_shared/gpu_utils.sh"
elif [[ -f "$SCRIPT_DIR/gpu_utils.sh" ]]; then
  source "$SCRIPT_DIR/gpu_utils.sh"
else
  echo "[FATAL] gpu_utils.sh not found. Copy github_repos/_shared/gpu_utils.sh to scripts/ or use monorepo layout." >&2
  exit 1
fi

auto_setup

# --- Activate project virtualenv (expected: setup.sh → .venv) ---
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
export PATH="${HOME}/.local/bin:${PATH}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# MODEL_BASE_DIR: when set, all HF model names are resolved as local paths
# e.g. MODEL_BASE_DIR=/data/models → "Qwen/Qwen3.5-9B" becomes "/data/models/Qwen/Qwen3.5-9B"
_resolve_model() {
  local name="$1"
  if [[ -n "${MODEL_BASE_DIR:-}" ]] && [[ -d "${MODEL_BASE_DIR}/${name}" ]]; then
    echo "${MODEL_BASE_DIR}/${name}"
  else
    echo "${name}"
  fi
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FROM_PHASE=0
ONLY_PHASE=-1
QUICK=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all_experiments.sh [options]

Options:
  --quick          Smoke mode: tiny sample counts, subset of pairs/gammas/seeds
  --from-phase N   Start pipeline at phase N (0–8)
  --only-phase N   Run exactly phase N
  --dry-run        Print intended commands only (no subprocesses)
  -h, --help       Show this help

Phases:
  0  Model download (LLM + DiT)
  1  LLM benchmark sweep (benchmark_speculative.py)
  2  LLM scaling law (run_scaling_law_llm.py)
  3  DiT acceptance sweep (run_acceptance_sweep_dit.py)
  4  DiT scaling law + h(t) (run_scaling_law_dit.py)
  5  Unified cross-modality analysis (run_unified_comparison.py)
  6  ImageNet FID-50K (run_imagenet_eval.py)
  7  Ablations (run_ablations.sh)
  8  Final figures & tables (generate_paper_figures_tables.py)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)       QUICK="--quick"; shift ;;
    --from-phase)  FROM_PHASE="$2"; shift 2 ;;
    --only-phase)  ONLY_PHASE="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

QUICK_ARGS=()
if [[ -n "${QUICK}" ]]; then
  QUICK_ARGS+=("${QUICK}")
fi

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').log"
touch "$PIPELINE_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H%M%S')] $*" | tee -a "$PIPELINE_LOG"
}

should_run_phase() {
  local p="$1"
  if [[ "${ONLY_PHASE}" -ge 0 ]]; then
    [[ "${p}" -eq "${ONLY_PHASE}" ]]
  else
    [[ "${p}" -ge "${FROM_PHASE}" ]]
  fi
}

# ---------------------------------------------------------------------------
# Experiment grids (full vs --quick)
# ---------------------------------------------------------------------------
if [[ -n "${QUICK}" ]]; then
  log ">>> QUICK MODE: reduced grids and sample counts"
  LLM_NUM_SAMPLES=8
  LLM_MAX_NEW_TOKENS=64
  DIT_SWEEP_SAMPLES=24
  DIT_EVAL_IMAGES=128
  DIT_EVAL_BATCH=4
  ABLATION_SAMPLES=32

  LLM_PAIRS=(
    "Qwen/Qwen3.5-0.8B:Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-4B:Qwen/Qwen3.5-27B"
  )
  LLM_GAMMAS=(4 8)
  LLM_DATASETS=(gsm8k math)
  LLM_SEEDS=(42)

  DIT_SWEEP_LABEL="quick_subset"
else
  LLM_NUM_SAMPLES=50
  LLM_MAX_NEW_TOKENS=128
  DIT_SWEEP_SAMPLES=100
  DIT_EVAL_IMAGES=50000
  DIT_EVAL_BATCH=16
  ABLATION_SAMPLES=10000

  # Nine draft→target pairs using actual Qwen3.5 dense models (0.8B/2B/4B/9B/27B)
  # Covers size ratios from 1:3.5 to 1:33.75 for S(d,T,γ) paper grid
  LLM_PAIRS=(
    "Qwen/Qwen3.5-0.8B:Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-0.8B:Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-0.8B:Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-2B:Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-2B:Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-2B:Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-4B:Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-4B:Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-9B:Qwen/Qwen3.5-27B"
  )
  LLM_GAMMAS=(2 3 4 5 6 7 8)
  LLM_DATASETS=(gsm8k math)
  LLM_SEEDS=(0)

  DIT_SWEEP_LABEL="full_paper_grid"
fi

# ---------------------------------------------------------------------------
log "=============================================================="
log "SpecScale — full pipeline"
log "  Project root : ${PROJECT_ROOT}"
log "  GPUs         : ${NUM_GPUS}"
log "  GPU class    : ${GPU_CLASS}"
log "  Quick        : ${QUICK:-no}"
log "  From phase   : ${FROM_PHASE}"
log "  Only phase   : $([[ ${ONLY_PHASE} -ge 0 ]] && echo "${ONLY_PHASE}" || echo all)"
log "  Dry-run      : ${DRY_RUN}"
log "  Model base   : ${MODEL_BASE_DIR:-<hub>}"
log "  Log file     : ${PIPELINE_LOG}"
log "=============================================================="

# ===========================================================================
# Phase 0 — Download LLM + DiT weights (skip if MODEL_BASE_DIR is set)
# ===========================================================================
if should_run_phase 0; then
  log "######## Phase 0: download models (LLM pairs + DiT) ########"
  P0_LOG="${LOG_DIR}/phase0_download.log"

  if [[ -n "${MODEL_BASE_DIR:-}" ]]; then
    log "MODEL_BASE_DIR is set (${MODEL_BASE_DIR}), skipping download — using local models."
  elif [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: Phase 0 — would download Qwen3.5 family + facebook/DiT-*-2-256 checkpoints"
  elif [[ -f "${SCRIPT_DIR}/download_models.py" ]]; then
    python "${SCRIPT_DIR}/download_models.py" \
      --llm_models Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B \
      --dit_models facebook/DiT-XL-2-256 facebook/DiT-L-2-256 facebook/DiT-B-2-256 facebook/DiT-S-2-256 \
      "${QUICK_ARGS[@]}" \
      2>&1 | tee -a "${P0_LOG}"
  else
    log "Note: scripts/download_models.py not found — using inline Hugging Face prefetch."
    python - <<'PY' 2>&1 | tee -a "${P0_LOG}"
import gc
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("phase0")

llms = [
    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
]

log.info("Prefetching LLM tokenizers + weights (CPU)...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    for name in llms:
        log.info("LLM: %s", name)
        AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        n = sum(p.numel() for p in m.parameters()) / 1e9
        log.info("  OK (%.2fB params)", n)
        del m
        gc.collect()
except Exception as e:
    log.error("LLM download failed: %s", e)
    sys.exit(1)

log.info("Prefetching DiT checkpoints (diffusers)...")
try:
    from diffusers import DiTPipeline
    import torch
    for repo in [
        "facebook/DiT-XL-2-256",
        "facebook/DiT-L-2-256",
        "facebook/DiT-B-2-256",
        "facebook/DiT-S-2-256",
    ]:
        log.info("DiT: %s", repo)
        pipe = DiTPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
        n = sum(p.numel() for p in pipe.transformer.parameters()) / 1e6
        log.info("  OK (%.1fM transformer params)", n)
        del pipe
        gc.collect()
except Exception as e:
    log.warning("DiT diffusers prefetch skipped or failed: %s", e)

log.info("Phase 0 complete.")
PY
  fi
fi

# ===========================================================================
# Phase 1 — LLM benchmark sweep (GPU-slot scheduler: fill all GPUs, never idle)
# ===========================================================================
if should_run_phase 1; then
  log "######## Phase 1: LLM sweep (benchmark_speculative.py, ${NUM_GPUS}-GPU parallel) ########"
  OUT_LLMSW="${PROJECT_ROOT}/results/llm_sweep"
  mkdir -p "${OUT_LLMSW}"
  P1_LOG="${LOG_DIR}/phase1_llm_sweep.log"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    log "Phase 1: per-job logs → ${LOG_DIR}/phase1_<tag>.log"
    : > "${P1_LOG}"
  fi

  GPU_SLOT_DIR=$(mktemp -d)
  trap "rm -rf ${GPU_SLOT_DIR}" EXIT

  P1_PIDS=()
  P1_FAIL=0

  for SEED in "${LLM_SEEDS[@]}"; do
    SEED_DIR="${OUT_LLMSW}/seed_${SEED}"
    mkdir -p "${SEED_DIR}"
    for PAIR in "${LLM_PAIRS[@]}"; do
      DRAFT="$(_resolve_model "${PAIR%%:*}")"
      TARGET="$(_resolve_model "${PAIR##*:}")"
      TAG="seed${SEED}_$(echo "${PAIR}" | tr '/:' '_')"

      if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "DRY-RUN: python benchmark_speculative.py --draft_model ${DRAFT} --target_model ${TARGET} --gamma ${LLM_GAMMAS[*]} --datasets ${LLM_DATASETS[*]} --num_samples ${LLM_NUM_SAMPLES} --max_new_tokens ${LLM_MAX_NEW_TOKENS} --seed ${SEED} --output_dir ${SEED_DIR} ${QUICK_ARGS[*]:-}"
        continue
      fi

      JOB_LOG="${LOG_DIR}/phase1_${TAG}.log"
      (
        # Acquire a free GPU slot (spin until one is available)
        GPU=-1
        while true; do
          for (( g=0; g<NUM_GPUS; g++ )); do
            if mkdir "${GPU_SLOT_DIR}/gpu_${g}" 2>/dev/null; then
              GPU=$g; break 2
            fi
          done
          sleep 1
        done

        echo "=== Phase 1 job ${TAG} (GPU ${GPU}) started $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
        CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/benchmark_speculative.py" \
          --draft_model "${DRAFT}" \
          --target_model "${TARGET}" \
          --gamma "${LLM_GAMMAS[@]}" \
          --datasets "${LLM_DATASETS[@]}" \
          --num_samples "${LLM_NUM_SAMPLES}" \
          --max_new_tokens "${LLM_MAX_NEW_TOKENS}" \
          --seed "${SEED}" \
          --output_dir "${SEED_DIR}" \
          "${QUICK_ARGS[@]}"
        RC=$?
        echo "=== Phase 1 job ${TAG} (GPU ${GPU}) finished rc=${RC} $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="

        rmdir "${GPU_SLOT_DIR}/gpu_${GPU}"
        exit ${RC}
      ) > "${JOB_LOG}" 2>&1 &

      CHILD_PID=$!
      P1_PIDS+=("${CHILD_PID}")
      echo "=== Queued ${TAG} pid=${CHILD_PID} ===" >> "${P1_LOG}"
      log "  Queued ${TAG} (pid ${CHILD_PID})"

      sleep 0.5
    done
  done

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    log "Phase 1: ${#P1_PIDS[@]} jobs queued across ${NUM_GPUS} GPUs, waiting..."
    for pid in "${P1_PIDS[@]}"; do
      if ! wait "${pid}"; then
        log "WARNING: Phase 1 job pid=${pid} failed"
        P1_FAIL=$((P1_FAIL + 1))
      fi
    done
    rm -rf "${GPU_SLOT_DIR}"
    trap - EXIT
    if [[ ${P1_FAIL} -gt 0 ]]; then
      log "ERROR: ${P1_FAIL}/${#P1_PIDS[@]} Phase 1 jobs failed"
      exit 1
    fi
    log "Phase 1 complete: all ${#P1_PIDS[@]} jobs succeeded"
  fi
fi

# ===========================================================================
# Phase 2 — LLM scaling law
# ===========================================================================
if should_run_phase 2; then
  log "######## Phase 2: LLM scaling law (run_scaling_law_llm.py) ########"
  mkdir -p "${PROJECT_ROOT}/results/llm_scaling"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python run_scaling_law_llm.py --results_dir results/llm_sweep --output_dir results/llm_scaling"
  else
    python "${SCRIPT_DIR}/run_scaling_law_llm.py" \
      --results_dir "${PROJECT_ROOT}/results/llm_sweep" \
      --output_dir "${PROJECT_ROOT}/results/llm_scaling" \
      2>&1 | tee -a "${LOG_DIR}/phase2_llm_scaling.log"
  fi
fi

# ===========================================================================
# Phase 3 — DiT acceptance sweep (3×5×3×3 full grid, multi-GPU parallel)
# ===========================================================================
if should_run_phase 3; then
  log "######## Phase 3: DiT acceptance (${NUM_GPUS}-GPU parallel) ########"
  mkdir -p "${PROJECT_ROOT}/results/dit_acceptance"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python run_acceptance_sweep_dit.py --full_sweep ×${NUM_GPUS} workers"
  else
    P3_PIDS=()
    P3_FAIL=0
    for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
      (
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
          --full_sweep \
          --num_samples "${DIT_SWEEP_SAMPLES}" \
          --batch_size 4 \
          --output_dir "${PROJECT_ROOT}/results/dit_acceptance" \
          --config "${PROJECT_ROOT}/configs/default.yaml" \
          --draft_device cuda:0 \
          --target_device cuda:0 \
          --worker_id "${gpu}" \
          --num_workers "${NUM_GPUS}" \
          --skip_existing
      ) > "${LOG_DIR}/phase3_dit_gpu${gpu}.log" 2>&1 &
      P3_PIDS+=($!)
      log "  Phase 3 worker ${gpu} started (pid ${P3_PIDS[-1]})"
    done

    log "Phase 3: ${#P3_PIDS[@]} workers running across ${NUM_GPUS} GPUs, waiting..."
    for pid in "${P3_PIDS[@]}"; do
      if ! wait "${pid}"; then
        log "WARNING: Phase 3 worker pid=${pid} failed"
        P3_FAIL=$((P3_FAIL + 1))
      fi
    done
    if [[ ${P3_FAIL} -gt 0 ]]; then
      log "ERROR: ${P3_FAIL}/${#P3_PIDS[@]} Phase 3 workers failed (continuing)"
    else
      log "Phase 3 complete: all ${#P3_PIDS[@]} workers succeeded"
    fi
  fi
fi

# ===========================================================================
# Phase 4 — DiT scaling law + h(t)
# ===========================================================================
if should_run_phase 4; then
  log "######## Phase 4: DiT scaling law (run_scaling_law_dit.py) ########"
  mkdir -p "${PROJECT_ROOT}/results/dit_scaling"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python run_scaling_law_dit.py --input_dir results/dit_acceptance --fit_ht_modulation"
  else
    python "${SCRIPT_DIR}/run_scaling_law_dit.py" \
      --input_dir "${PROJECT_ROOT}/results/dit_acceptance" \
      --output_dir "${PROJECT_ROOT}/results/dit_scaling" \
      --figure_format pdf \
      2>&1 | tee -a "${LOG_DIR}/phase4_dit_scaling.log"
  fi
fi

# ===========================================================================
# Phase 5 — Unified cross-modality analysis
# ===========================================================================
if should_run_phase 5; then
  log "######## Phase 5: unified comparison (run_unified_comparison.py) ########"
  mkdir -p "${PROJECT_ROOT}/results/unified"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python run_unified_comparison.py (LLM + DiT scaling dirs)"
  else
    python "${SCRIPT_DIR}/run_unified_comparison.py" \
      --llm_results_dir "${PROJECT_ROOT}/results/llm_scaling" \
      --dit_results_dir "${PROJECT_ROOT}/results/dit_scaling" \
      --output_dir "${PROJECT_ROOT}/results/unified" \
      2>&1 | tee -a "${LOG_DIR}/phase5_unified.log"
  fi
fi

# ===========================================================================
# Phase 6 — ImageNet FID-50K (parallel: baselines + specdenoise on separate GPUs)
# ===========================================================================
if should_run_phase 6; then
  log "######## Phase 6: ImageNet eval (multi-GPU parallel) ########"
  mkdir -p "${PROJECT_ROOT}/results/imagenet_eval"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python run_imagenet_eval.py --num_images ${DIT_EVAL_IMAGES} ×5 configs"
  else
    P6_PIDS=()
    P6_FAIL=0

    P6_COMMON=(
      --num_images "${DIT_EVAL_IMAGES}"
      --batch_size "${DIT_EVAL_BATCH}"
      --image_size 256
      --target_model "DiT-XL/2"
      --guidance_scale 4.0
      --seed 42
      --output_dir "${PROJECT_ROOT}/results/imagenet_eval"
      --config "${PROJECT_ROOT}/configs/default.yaml"
    )

    # 3 baselines on GPUs 0-2
    gpu=0
    for steps in 50 25 20; do
      solver="ddim"
      [[ ${steps} -eq 20 ]] && solver="dpmsolver++"
      (
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_imagenet_eval.py" \
          --method baseline --solver "${solver}" --num_steps "${steps}" \
          --draft_device cuda:0 --target_device cuda:0 \
          "${P6_COMMON[@]}"
      ) > "${LOG_DIR}/phase6_baseline_${solver}_${steps}.log" 2>&1 &
      P6_PIDS+=($!)
      log "  Phase 6 baseline ${solver}-${steps} on GPU ${gpu} (pid ${P6_PIDS[-1]})"
      gpu=$((gpu + 1))
    done

    # 2 SpecDenoise configs on GPUs 3-4
    for gamma_spec in "5:false" "10:true"; do
      g="${gamma_spec%%:*}"
      adaptive="${gamma_spec##*:}"
      adaptive_flag=""
      [[ "${adaptive}" == "true" ]] && adaptive_flag="--adaptive_gamma"
      (
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_imagenet_eval.py" \
          --method specdenoise --draft_model "DiT-S/2" \
          --gamma "${g}" ${adaptive_flag} \
          --draft_device cuda:0 --target_device cuda:0 \
          "${P6_COMMON[@]}"
      ) > "${LOG_DIR}/phase6_specdenoise_g${g}.log" 2>&1 &
      P6_PIDS+=($!)
      log "  Phase 6 specdenoise γ=${g} on GPU ${gpu} (pid ${P6_PIDS[-1]})"
      gpu=$((gpu + 1))
    done

    log "Phase 6: ${#P6_PIDS[@]} eval configs across ${gpu} GPUs, waiting..."
    for pid in "${P6_PIDS[@]}"; do
      if ! wait "${pid}"; then
        log "WARNING: Phase 6 pid=${pid} failed"
        P6_FAIL=$((P6_FAIL + 1))
      fi
    done
    if [[ ${P6_FAIL} -gt 0 ]]; then
      log "ERROR: ${P6_FAIL}/${#P6_PIDS[@]} Phase 6 jobs failed (continuing)"
    else
      log "Phase 6 complete: all ${#P6_PIDS[@]} eval configs succeeded"
    fi
  fi
fi

# ===========================================================================
# Phase 7 — Ablations (6 configs across 6 GPUs in parallel)
# ===========================================================================
if should_run_phase 7; then
  log "######## Phase 7: ablations (multi-GPU parallel) ########"
  ABL_DIR="${PROJECT_ROOT}/results/ablations"
  ABL_CONFIG="${PROJECT_ROOT}/configs/default.yaml"
  mkdir -p "${ABL_DIR}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: 6 ablation configs in parallel across GPUs"
  else
    P7_PIDS=()
    P7_FAIL=0
    gpu=0

    # Ablation 1: γ schedule comparison
    (
      CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
        --output_dir "${ABL_DIR}/gamma_schedules" \
        --config "${ABL_CONFIG}" \
        --num_samples "${ABLATION_SAMPLES}" \
        --draft_device cuda:0 --target_device cuda:0
    ) > "${LOG_DIR}/phase7_gamma_sched.log" 2>&1 &
    P7_PIDS+=($!); log "  Ablation: γ schedules on GPU ${gpu}"; gpu=$((gpu+1))

    # Ablation 2-4: Temperature sensitivity (0.5, 1.0, 2.0)
    for temp in 0.5 1.0 2.0; do
      (
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
          --output_dir "${ABL_DIR}/temperature_${temp}" \
          --config "${ABL_CONFIG}" \
          --num_samples "${ABLATION_SAMPLES}" \
          --temperature "${temp}" \
          --draft_device cuda:0 --target_device cuda:0
      ) > "${LOG_DIR}/phase7_temp_${temp}.log" 2>&1 &
      P7_PIDS+=($!); log "  Ablation: temp=${temp} on GPU ${gpu}"; gpu=$((gpu+1))
    done

    # Ablation 5-6: Noise schedules (linear, cosine)
    for sched in linear cosine; do
      (
        CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/run_acceptance_sweep_dit.py" \
          --output_dir "${ABL_DIR}/noise_${sched}" \
          --config "${ABL_CONFIG}" \
          --num_samples "${ABLATION_SAMPLES}" \
          --noise_schedule "${sched}" \
          --draft_device cuda:0 --target_device cuda:0
      ) > "${LOG_DIR}/phase7_noise_${sched}.log" 2>&1 &
      P7_PIDS+=($!); log "  Ablation: noise=${sched} on GPU ${gpu}"; gpu=$((gpu+1))
    done

    log "Phase 7: ${#P7_PIDS[@]} ablations across ${gpu} GPUs, waiting..."
    for pid in "${P7_PIDS[@]}"; do
      if ! wait "${pid}"; then
        log "WARNING: Phase 7 pid=${pid} failed"
        P7_FAIL=$((P7_FAIL + 1))
      fi
    done
    if [[ ${P7_FAIL} -gt 0 ]]; then
      log "ERROR: ${P7_FAIL}/${#P7_PIDS[@]} Phase 7 ablations failed (continuing)"
    else
      log "Phase 7 complete: all ${#P7_PIDS[@]} ablations succeeded"
    fi
  fi
fi

# ===========================================================================
# Phase 8 — Final figures + tables
# ===========================================================================
if should_run_phase 8; then
  log "######## Phase 8: paper figures + tables ########"
  mkdir -p "${PROJECT_ROOT}/results/final/figures"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "DRY-RUN: python generate_paper_figures_tables.py → results/final"
  else
    python "${SCRIPT_DIR}/generate_paper_figures_tables.py" \
      --llm_scaling "${PROJECT_ROOT}/results/llm_scaling" \
      --dit_scaling "${PROJECT_ROOT}/results/dit_scaling" \
      --unified "${PROJECT_ROOT}/results/unified" \
      --imagenet_eval "${PROJECT_ROOT}/results/imagenet_eval" \
      --ablations "${PROJECT_ROOT}/results/ablations" \
      --output_dir "${PROJECT_ROOT}/results/final" \
      "${QUICK_ARGS[@]}" \
      2>&1 | tee -a "${LOG_DIR}/phase8_final_figures.log"
  fi
fi

log "=============================================================="
log "SpecScale pipeline finished."
log "  Master log: ${PIPELINE_LOG}"
log "  Key dirs:"
log "    results/llm_sweep          Phase 1 raw LLM JSONs"
log "    results/llm_scaling        Phase 2 S(d,T,γ) fits (LLM)"
log "    results/dit_acceptance     Phase 3 DiT acceptance JSONs"
log "    results/dit_scaling        Phase 4 DiT + h(t) fits"
log "    results/unified            Phase 5 cross-modality"
log "    results/imagenet_eval      Phase 6 FID-50K / IS"
log "    results/ablations          Phase 7"
log "    results/final              Phase 8 paper assets"
log "=============================================================="

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
