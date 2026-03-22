#!/bin/bash
# ============================================================================
# Shared GPU utilities for all NeurIPS 2026 projects
# Source this file in any experiment script:
#   source "$(dirname "$0")/../_shared/gpu_utils.sh" 2>/dev/null || source "$(dirname "$0")/gpu_utils.sh"
# ============================================================================

detect_gpus() {
    # Detect available NVIDIA GPUs and set environment variables
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[ERROR] nvidia-smi not found. CUDA driver not installed?"
        exit 1
    fi

    export NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "[ERROR] No GPUs detected."
        exit 1
    fi

    # Build CUDA_VISIBLE_DEVICES string: 0,1,...,N-1
    local gpu_ids=""
    for ((i=0; i<NUM_GPUS; i++)); do
        [ -n "$gpu_ids" ] && gpu_ids="${gpu_ids},"
        gpu_ids="${gpu_ids}${i}"
    done
    export CUDA_VISIBLE_DEVICES="$gpu_ids"

    # GPU memory in MiB (of first GPU)
    export GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')

    # Determine if we can run large models
    if [ "${GPU_MEM_MIB:-0}" -ge 70000 ]; then
        export GPU_CLASS="a100_80g"
    elif [ "${GPU_MEM_MIB:-0}" -ge 35000 ]; then
        export GPU_CLASS="a100_40g"
    elif [ "${GPU_MEM_MIB:-0}" -ge 20000 ]; then
        export GPU_CLASS="a10_24g"
    else
        export GPU_CLASS="consumer"
    fi

    echo "============================================"
    echo " GPU Configuration"
    echo "============================================"
    echo "  GPUs detected     : $NUM_GPUS"
    echo "  CUDA_VISIBLE      : $CUDA_VISIBLE_DEVICES"
    echo "  GPU memory (each) : ${GPU_MEM_MIB} MiB"
    echo "  GPU class          : $GPU_CLASS"
    echo "============================================"
}

# Select torchrun command based on GPU count
get_torchrun_cmd() {
    local nproc="${1:-$NUM_GPUS}"
    echo "torchrun --nproc_per_node=$nproc --master_port=$(( RANDOM % 10000 + 20000 ))"
}

# Select accelerate launch command
get_accelerate_cmd() {
    local nproc="${1:-$NUM_GPUS}"
    echo "accelerate launch --num_processes=$nproc --mixed_precision=bf16"
}

# Select per_device_batch_size based on GPU memory and model size
auto_batch_size() {
    local model_params_b="${1:-9}"  # model size in billions
    local base_bs="${2:-4}"        # base batch size for 80GB GPU with 9B model
    local mem=${GPU_MEM_MIB:-80000}

    local scale=$(echo "$mem / 80000 * 9 / $model_params_b" | bc -l 2>/dev/null || echo "1.0")
    local bs=$(echo "$base_bs * $scale" | bc 2>/dev/null | cut -d. -f1)
    bs=${bs:-$base_bs}
    [ "$bs" -lt 1 ] && bs=1
    [ "$bs" -gt 32 ] && bs=32
    echo "$bs"
}

# Generate accelerate config YAML
generate_accelerate_config() {
    local output_path="${1:-accelerate_config.yaml}"
    local nproc="${2:-$NUM_GPUS}"

    cat > "$output_path" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
main_training_tp_size: 1
mixed_precision: bf16
num_machines: 1
num_processes: ${nproc}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
    echo "  Generated accelerate config: $output_path (${nproc} GPUs)"
}

# Generate FSDP accelerate config for large models
generate_fsdp_config() {
    local output_path="${1:-fsdp_config.yaml}"
    local nproc="${2:-$NUM_GPUS}"

    cat > "$output_path" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_tp_size: 1
mixed_precision: bf16
num_machines: 1
num_processes: ${nproc}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
    echo "  Generated FSDP config: $output_path (${nproc} GPUs)"
}

# Common environment setup
setup_env() {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    export TRANSFORMERS_CACHE="${HF_HOME}/hub"
    export TOKENIZERS_PARALLELISM=false
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export OMP_NUM_THREADS=8
}

# Auto-detect and setup everything
auto_setup() {
    setup_env
    detect_gpus
}

# Run if sourced directly for testing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    auto_setup
    echo ""
    echo "Torchrun cmd : $(get_torchrun_cmd)"
    echo "Accelerate   : $(get_accelerate_cmd)"
    echo "Batch size 9B: $(auto_batch_size 9 4)"
    echo "Batch size 27B: $(auto_batch_size 27 2)"
fi
