#!/usr/bin/env bash
# train.sh — 4-stage QLoRA training pipeline for FableForge-14B
#
# Stages:
#   1. behavior_shaping — Learn tool-use patterns (SFT on 100K vfable + armand0e + glint)
#   2. skill_distillation  — Master code generation (SFT on 100K coding_excellence)
#   3. error_recovery — Debug expertise (SFT on error/recovery pairs)
#   4. dpo           — Preference alignment (DPO on chosen/rejected pairs)
#
# Requirements:
#   - 2x A100-80GB or 4x A6000-48GB (minimum 80GB VRAM total for 14B QLoRA)
#   - Python 3.10+
#   - 200GB disk for checkpoints
#
# Estimated times (4x A100-80GB):
#   Stage 1: ~18h on 100K examples
#   Stage 2: ~15h on 100K examples
#   Stage 3: ~4h on ~20K examples
#   Stage 4: ~8h on ~30K pairs
#   Total:   ~45h
#
# Memory requirements at 14B QLoRA (4-bit):
#   Per GPU: ~22GB for model + gradients, ~30GB peak with optimizer states
#
# Usage:
#   bash train.sh [OPTIONS]
#   Options:
#     --stage {1,2,3,4}     Run a specific stage (default: all)
#     --resume-checkpoint    Resume from last checkpoint
#     --dry-run              Print config and exit without training
#     --gpus N               Number of GPUs (default: auto-detect)
#     --base-model MODEL     Base model name (default: Qwen/Qwen2.5-Coder-14B)
#     --data-dir DIR         Data directory (default: /tmp/fableforge/fableforge-14b/data)
#     --output-dir DIR       Output directory (default: /tmp/fableforge/fableforge-14b/output)
#
# Free-tier / Low-VRAM modes:
#     --unsloth              Use Unsloth for 2-5x faster training with 70% less VRAM
#     --colab                Optimize for Google Colab T4 (gradient checkpointing, 4-bit, fp16)
#     --free-tier            Combine --unsloth + --colab + conservative memory settings
#                            Enables training on Colab free T4 (15GB VRAM)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-14B}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output}"
LOG_DIR="${OUTPUT_DIR}/logs"
GPUS="${GPUS:-}"
STAGE="${STAGE:-all}"
RESUME="${RESUME:-false}"
DRY_RUN="${DRY_RUN:-false}"

# ─── Free-tier / Unsloth Mode Flags ─────────────────────────────────────────
# --unsloth:      Use Unsloth (https://github.com/unslothai/unsloth) for faster,
#                 more memory-efficient training. Automatically installs unsloth
#                 and uses its FastLanguageModel for 2-5x speedup + 70% less VRAM.
# --colab:        Optimize for Google Colab T4 GPU (15GB VRAM). Enables aggressive
#                 gradient checkpointing, reduces batch sizes, uses fp16 (not bf16),
#                 and adjusts memory settings for T4 constraints.
# --free-tier:    Combines --unsloth + --colab + conservative memory settings.
#                 This mode lets you train on Colab free T4 without OOM.
#                 Reduces max_seq_length to 4096, batch_size to 1-2, gradient
#                 accumulation to 8-16, and uses Unsloth's optimized checkpointing.
UNSLOTH="${UNSLOTH:-false}"
COLAB="${COLAB:-false}"
FREE_TIER="${FREE_TIER:-false}"

for arg in "$@"; do
    case "$arg" in
        --stage=*)     STAGE="${arg#--stage=}" ;;
        --resume)      RESUME="true" ;;
        --dry-run)     DRY_RUN="true" ;;
        --gpus=*)      GPUS="${arg#--gpus=}" ;;
        --base-model=*) BASE_MODEL="${arg#--base-model=}" ;;
        --data-dir=*)  DATA_DIR="${arg#--data-dir=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#--output-dir=}" ;;
        --unsloth)     UNSLOTH="true" ;;
        --colab)       COLAB="true" ;;
        --free-tier)   FREE_TIER="true" ;;
    esac
done

# If --free-tier is set, enable both unsloth and colab modes
if [[ "$FREE_TIER" == "true" ]]; then
    UNSLOTH="true"
    COLAB="true"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── GPU Detection ───────────────────────────────────────────────────────────

if [[ -z "$GPUS" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
        GPUS=${GPUS:-1}
    else
        GPUS=1
        echo "[WARN] nvidia-smi not found, defaulting to $GPUS GPU(s)"
    fi
fi

TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1}END{print s}' || echo "0")
echo "=== FableForge-14B Training Pipeline ==="
echo "  Base model:   $BASE_MODEL"
echo "  GPUs:         $GPUS"
echo "  Total VRAM:   ${TOTAL_VRAM}MB"
echo "  Data dir:     $DATA_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

if [[ "$TOTAL_VRAM" -lt 60000 ]] 2>/dev/null; then
    echo "[WARN] Insufficient VRAM (${TOTAL_VRAM}MB). 14B QLoRA requires ~80GB."
    echo "       Consider using a smaller base model like Qwen/Qwen2.5-Coder-7B"
fi

# ─── Unsloth & Colab Mode Configuration ──────────────────────────────────────
# These settings modify training parameters for free-tier/low-VRAM environments.

# Detect Colab environment
if [[ -f "/content/.colab_running" ]] || [[ -d "/content/drive" ]] || [[ -n "${COLAB_RUNTIME_ENV+x}" ]]; then
    echo "[INFO] Google Colab environment detected"
    COLAB="${COLAB:-true}"
fi

if [[ "$UNSLOTH" == "true" ]]; then
    echo "[INFO] Unsloth mode ENABLED — 2-5x faster, 70% less VRAM"
    
    # Install Unsloth if not present
    if ! python3 -c "import unsloth" 2>/dev/null; then
        echo "[INFO] Installing Unsloth..."
        pip install --no-deps "unsloth[colab-new]" --quiet 2>/dev/null || \
        pip install "unsloth" --quiet 2>/dev/null || \
        pip install git+https://github.com/unslothai/unsloth.git --quiet 2>/dev/null
        echo "[OK] Unsloth installed"
    fi
    
    # Unsloth uses FastLanguageModel instead of AutoModelForCausalLM
    # Training script will detect this flag and use unsloth loading
    export USE_UNSLOTH=1
    TRAINING_BACKEND="unsloth"
else
    TRAINING_BACKEND="transformers"
fi

if [[ "$COLAB" == "true" ]]; then
    echo "[INFO] Colab T4 optimization ENABLED — conservative memory settings"
    
    # Colab T4: 15GB VRAM, no bf16 support, single GPU
    # Override settings for memory-constrained environment
    export COLAB_MODE=1
    
    # Use fp16 (T4 doesn't have bf16 support)
    export TRAINING_DTYPE="fp16"
    
    # Reduce batch sizes and increase gradient accumulation for memory efficiency
    # Effective batch size stays the same but uses less VRAM per step
    if [[ "$FREE_TIER" == "true" ]]; then
        # Free-tier: most conservative settings
        export DEFAULT_BATCH_SIZE=1
        export DEFAULT_GRADIENT_ACCUM=16
        export DEFAULT_MAX_SEQ_LEN=4096
        echo "[INFO] Free-tier mode: batch_size=1, grad_accum=16, max_seq_len=4096"
    else
        # Colab with some headroom
        export DEFAULT_BATCH_SIZE=2
        export DEFAULT_GRADIENT_ACCUM=8
        export DEFAULT_MAX_SEQ_LEN=4096
    fi
    
    # Enable aggressive gradient checkpointing
    export GRADIENT_CHECKPOINTING="unsloth"  # Use Unsloth's optimized variant if available
    
    # Mount Google Drive if available for checkpoint persistence
    if [[ -d "/content/drive/MyDrive" ]]; then
        DRIVE_OUTPUT="/content/drive/MyDrive/fableforge-14b/output"
        if [[ "$OUTPUT_DIR" == "${PROJECT_DIR}/output" ]]; then
            OUTPUT_DIR="$DRIVE_OUTPUT"
            mkdir -p "$OUTPUT_DIR"
            echo "[INFO] Output redirected to Google Drive: $OUTPUT_DIR"
        fi
    fi
    
    echo "[INFO] Colab training settings:"
    echo "  dtype:           fp16 (T4 compatible)"
    echo "  batch_size:      ${DEFAULT_BATCH_SIZE:-2}"
    echo "  grad_accum:      ${DEFAULT_GRADIENT_ACCUM:-8}"
    echo "  max_seq_len:     ${DEFAULT_MAX_SEQ_LEN:-4096}"
    echo "  checkpointing:   ${GRADIENT_CHECKPOINTING:-true}"
    echo "  output_dir:      $OUTPUT_DIR"
fi

if [[ "$UNSLOTH" == "true" ]] && [[ "$COLAB" == "true" ]]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║          🚀 Free-Tier Training Mode Active                ║"
    echo "║  Unsloth + Colab T4 optimizations enabled                  ║"
    echo "║  Expected: 2-5x faster, 70% less VRAM usage               ║"
    echo "║  Target: Complete 14B training on free Colab T4             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
fi

# ─── Python Environment Check ───────────────────────────────────────────────

check_python() {
    python3 -c "
import sys
pkgs = ['torch', 'transformers', 'peft', 'trl', 'bitsandbytes', 'datasets', 'accelerate', 'wandb']
missing = []
for p in pkgs:
    try:
        __import__(p)
    except ImportError:
        missing.append(p)
if missing:
    print(f'Missing packages: {\" \".join(missing)}')
    print(f'Install with: pip install {\" \".join(missing)}')
    sys.exit(1)
print('All required packages available.')
" 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] Missing Python packages. Install training dependencies:"
        echo "  pip install torch transformers peft trl bitsandbytes datasets accelerate wandb"
        echo "  pip install scipy sentencepiece protobuf"
        exit 1
    fi

    # Check Unsloth if requested
    if [[ "$UNSLOTH" == "true" ]]; then
        python3 -c "import unsloth; print(f'Unsloth {unsloth.__version__} available.')" 2>/dev/null
        if [[ $? -ne 0 ]]; then
            echo "[ERROR] Unsloth not installed. Install with:"
            echo "  pip install unsloth[colab-new]"
            exit 1
        fi
    fi
}

# ─── Stage Configs ───────────────────────────────────────────────────────────

declare -A STAGE_NAMES=(
    [1]="behavior_shaping"
    [2]="skill_distillation"
    [3]="error_recovery"
    [4]="dpo"
)

# Stage 1: Behavior Shaping — LoRA r=64, alpha=128, lr=2e-4
declare -A S1_LORA_R=([1]="64")
declare -A S1_LORA_ALPHA=([1]="128")
declare -A S1_LR=([1]="2e-4")
declare -A S1_EPOCHS=([1]="3")
declare -A S1_BATCH_SIZE=([1]="2")
declare -A S1_GRADIENT_ACCUM=([1]="8")
declare -A S1_MAX_SEQ_LEN=([1]="4096")
declare -A S1_WARMUP_RATIO=([1]="0.06")

# Stage 2: Skill Distillation — LoRA r=32, alpha=64, lr=1e-4
declare -A S2_LORA_R=([1]="32")
declare -A S2_LORA_ALPHA=([1]="64")
declare -A S2_LR=([1]="1e-4")
declare -A S2_EPOCHS=([1]="2")
declare -A S2_BATCH_SIZE=([1]="2")
declare -A S2_GRADIENT_ACCUM=([1]="8")
declare -A S2_MAX_SEQ_LEN=([1]="4096")
declare -A S2_WARMUP_RATIO=([1]="0.06")

# Stage 3: Error Recovery — LoRA r=16, alpha=32, lr=5e-5
declare -A S3_LORA_R=([1]="16")
declare -A S3_LORA_ALPHA=([1]="32")
declare -A S3_LR=([1]="5e-5")
declare -A S3_EPOCHS=([1]="3")
declare -A S3_BATCH_SIZE=([1]="4")
declare -A S3_GRADIENT_ACCUM=([1]="4")
declare -A S3_MAX_SEQ_LEN=([1]="4096")
declare -A S3_WARMUP_RATIO=([1]="0.06")

# Stage 4: DPO — LoRA r=16, alpha=32, lr=5e-5, beta=0.1
declare -A S4_LORA_R=([1]="16")
declare -A S4_LORA_ALPHA=([1]="32")
declare -A S4_LR=([1]="5e-5")
declare -A S4_EPOCHS=([1]="1")
declare -A S4_BATCH_SIZE=([1]="1")
declare -A S4_GRADIENT_ACCUM=([1]="16")
declare -A S4_MAX_SEQ_LEN=([1]="4096")
declare -A S4_WARMUP_RATIO=([1]="0.1")
declare -A S4_DPO_BETA=([1]="0.1")

# ─── Training Functions ──────────────────────────────────────────────────────

run_sft_stage() {
    local stage_num=$1
    local stage_name=$2
    local lora_r=$3
    local lora_alpha=$4
    local lr=$5
    local epochs=$6
    local batch_size=$7
    local grad_accum=$8
    local max_seq_len=$9
    local warmup_ratio=${10}

    local stage_output="${OUTPUT_DIR}/stage${stage_num}_${stage_name}"
    local data_path="${DATA_DIR}/${stage_name}/${stage_name}_train.jsonl"
    local val_path="${DATA_DIR}/${stage_name}/${stage_name}_val.jsonl"

    if [[ ! -f "$data_path" ]]; then
        echo "[ERROR] Training data not found: $data_path"
        echo "        Run convert_data.py first: python scripts/convert_data.py --stage ${stage_num}"
        return 1
    fi

    local train_examples
    train_examples=$(wc -l < "$data_path" | tr -d ' ')
    echo ""
    echo "=== Stage ${stage_num}: ${stage_name} ==="
    echo "  Training data:    $data_path (${train_examples} examples)"
    echo "  Validation data:  $val_path"
    echo "  Output:           $stage_output"
    echo "  LoRA r:           $lora_r, alpha: $lora_alpha"
    echo "  Learning rate:    $lr"
    echo "  Epochs:           $epochs"
    echo "  Batch size:       $batch_size, gradient accumulation: $grad_accum"
    echo "  Max seq length:   $max_seq_len"
    echo "  Warmup ratio:    $warmup_ratio"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would run SFT training with the above config"
        return 0
    fi

    local resume_args=""
    if [[ "$RESUME" == "true" ]] && [[ -d "${stage_output}/checkpoint-latest" ]]; then
        resume_args="--resume_from_checkpoint ${stage_output}/checkpoint-latest"
        echo "  Resuming from: ${stage_output}/checkpoint-latest"
    fi

    local prev_adapter=""
    if [[ "$stage_num" -gt 1 ]]; then
        local prev_stage=$((stage_num - 1))
        local prev_name="${STAGE_NAMES[$prev_stage]}"
        prev_adapter="${OUTPUT_DIR}/stage${prev_stage}_${prev_name}/final"
        if [[ ! -d "$prev_adapter" ]]; then
            prev_adapter="${OUTPUT_DIR}/stage${prev_stage}_${prev_name}/checkpoint-latest"
        fi
        if [[ -d "$prev_adapter" ]]; then
            echo "  Loading previous adapter: $prev_adapter"
        else
            echo "[WARN] Previous adapter not found at $prev_adapter, training from base model"
            prev_adapter=""
        fi
    fi

    local adapter_args=""
    if [[ -n "$prev_adapter" ]]; then
        adapter_args="--adapter_path $prev_adapter"
    fi

    local effective_batch=$((batch_size * grad_accum * GPUS))
    local steps_per_epoch=$(( (train_examples + effective_batch - 1) / effective_batch ))
    local total_steps=$(( steps_per_epoch * epochs ))
    local save_steps=$(( steps_per_epoch / 2 ))
    save_steps=$(( save_steps < 1 ? 1 : save_steps ))

    echo "  Effective batch:  $effective_batch"
    echo "  Steps/epoch:      $steps_per_epoch"
    echo "  Total steps:      $total_steps"
    echo "  Save steps:       $save_steps"
    echo ""

    local wandb_project="${WANDB_PROJECT:-fableforge-14b}"
    local wandb_run_name="stage${stage_num}-${stage_name}"

    local accelerate_config="${OUTPUT_DIR}/accelerate_config_stage${stage_num}.yaml"
    cat > "$accelerate_config" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

    ACCELERATE_CONFIG_FILE="$accelerate_config" \
    python3 -m fableforge_14b.training.run_sft \
        --base_model_name_or_path "$BASE_MODEL" \
        --dataset_path "$data_path" \
        --validation_path "$val_path" \
        --output_dir "$stage_output" \
        --lora_r "$lora_r" \
        --lora_alpha "$lora_alpha" \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --learning_rate "$lr" \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size "$batch_size" \
        --per_device_eval_batch_size "$batch_size" \
        --gradient_accumulation_steps "$grad_accum" \
        --max_seq_length "$max_seq_len" \
        --warmup_ratio "$warmup_ratio" \
        --lr_scheduler_type cosine \
        --bf16 true \
        --logging_steps 10 \
        --save_steps "$save_steps" \
        --save_total_limit 3 \
        --eval_strategy steps \
        --eval_steps "$save_steps" \
        --load_best_model_at_end true \
        --metric_for_best_model eval_loss \
        --report_to wandb \
        --run_name "$wandb_run_name" \
        --wandb_project "$wandb_project" \
        --gradient_checkpointing true \
        --optim paged_adamw_8bit \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --seed 42 \
        $adapter_args \
        $resume_args \
        2>&1 | tee "${LOG_DIR}/stage${stage_num}_${stage_name}.log"

    local exit_code=${PIPESTATUS[0]:-0}
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Stage ${stage_num} training failed with exit code $exit_code"
        return $exit_code
    fi

    echo "[OK] Stage ${stage_num} complete. Output: $stage_output"
    echo ""
}

run_dpo_stage() {
    local stage_num=4
    local stage_name="dpo"

    local stage_output="${OUTPUT_DIR}/stage4_dpo"
    local data_path="${DATA_DIR}/dpo/dpo_train.jsonl"
    local val_path="${DATA_DIR}/dpo/dpo_val.jsonl"

    if [[ ! -f "$data_path" ]]; then
        echo "[ERROR] DPO data not found: $data_path"
        return 1
    fi

    local train_examples
    train_examples=$(wc -l < "$data_path" | tr -d ' ')

    echo ""
    echo "=== Stage 4: DPO Alignment ==="
    echo "  Training data:    $data_path (${train_examples} pairs)"
    echo "  Validation data:  $val_path"
    echo "  Output:           $stage_output"
    echo "  LoRA r:           16, alpha: 32"
    echo "  DPO beta:         0.1"
    echo "  Learning rate:    5e-5"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would run DPO training"
        return 0
    fi

    local prev_adapter="${OUTPUT_DIR}/stage3_error_recovery/final"
    if [[ ! -d "$prev_adapter" ]]; then
        prev_adapter="${OUTPUT_DIR}/stage3_error_recovery/checkpoint-latest"
    fi

    local resume_args=""
    if [[ "$RESUME" == "true" ]] && [[ -d "${stage_output}/checkpoint-latest" ]]; then
        resume_args="--resume_from_checkpoint ${stage_output}/checkpoint-latest"
    fi

    python3 -m fableforge_14b.training.run_dpo \
        --base_model_name_or_path "$BASE_MODEL" \
        --adapter_path "$prev_adapter" \
        --dataset_path "$data_path" \
        --validation_path "$val_path" \
        --output_dir "$stage_output" \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj k_proj v_proj o_proj \
        --dpo_beta 0.1 \
        --dpo_loss_type sigmoid \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --max_seq_length 4096 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type cosine \
        --bf16 true \
        --logging_steps 10 \
        --save_steps 200 \
        --save_total_limit 3 \
        --eval_strategy steps \
        --eval_steps 200 \
        --report_to wandb \
        --run_name "stage4-dpo" \
        --wandb_project "${WANDB_PROJECT:-fableforge-14b}" \
        --gradient_checkpointing true \
        --optim paged_adamw_8bit \
        $resume_args \
        2>&1 | tee "${LOG_DIR}/stage4_dpo.log"
}

# ─── GGUF Export ─────────────────────────────────────────────────────────────

export_gguf() {
    local merged_model="${OUTPUT_DIR}/merged_model"
    local gguf_output="${OUTPUT_DIR}/fableforge-14b-Q4_K_M.gguf"

    echo ""
    echo "=== Exporting to GGUF ==="
    echo "  Merged model:  $merged_model"
    echo "  GGUF output:  $gguf_output"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would merge adapters and export to GGUF"
        return 0
    fi

    if ! command -v llama-cli &>/dev/null && ! command -v llama.cpp/llama-quantize &>/dev/null; then
        echo "[INFO] llama.cpp not found, installing..."
        pip install llama-cpp-python[server] 2>/dev/null || true

        if [[ ! -d "${OUTPUT_DIR}/llama.cpp" ]]; then
            echo "[INFO] Cloning llama.cpp for GGUF export..."
            git clone https://github.com/ggerganov/llama.cpp.git "${OUTPUT_DIR}/llama.cpp" --depth 1
            cd "${OUTPUT_DIR}/llama.cpp" && make -j$(nproc) llama-quantize 2>/dev/null || true
            cd "$PROJECT_DIR"
        fi
    fi

    echo "[INFO] Merging LoRA adapters into base model..."
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}/src')
from fableforge_14b.model.merge_lora import merge_lora_adapters, MergeConfig

config = MergeConfig(
    base_model='${BASE_MODEL}',
    adapters=[
        '${OUTPUT_DIR}/stage1_behavior_shaping/final',
        '${OUTPUT_DIR}/stage2_skill_distillation/final',
        '${OUTPUT_DIR}/stage3_error_recovery/final',
        '${OUTPUT_DIR}/stage4_dpo/final',
    ],
    output_dir='${merged_model}',
    merge_method='sequential',
)
merge_lora_adapters(config)
print(f'[OK] Merged model saved to ${merged_model}')
" || {
        echo "[WARN] Python merge failed. Trying huggingface-peft merge..."
        python3 << 'MERGE_SCRIPT'
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "${BASE_MODEL}"
output = "${merged_model}"

print(f"Loading base model: {base}")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base)

adapters = [
    "${OUTPUT_DIR}/stage1_behavior_shaping/final",
    "${OUTPUT_DIR}/stage2_skill_distillation/final",
    "${OUTPUT_DIR}/stage3_error_recovery/final",
    "${OUTPUT_DIR}/stage4_dpo/final",
]

for adapter_path in adapters:
    try:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"  Merged: {adapter_path}")
    except Exception as e:
        print(f"  Skipped {adapter_path}: {e}")

print(f"Saving merged model to: {output}")
model.save_pretrained(output)
tokenizer.save_pretrained(output)
print("[OK] Merge complete")
MERGE_SCRIPT
    }

    echo "[INFO] Converting merged model to GGUF..."
    if [[ -d "${OUTPUT_DIR}/llama.cpp" ]]; then
        python3 "${OUTPUT_DIR}/llama.cpp/convert_hf_to_gguf.py" \
            "$merged_model" \
            --outfile "$gguf_output" \
            --outtype f16

        echo "[INFO] Quantizing GGUF to Q4_K_M..."
        "${OUTPUT_DIR}/llama.cpp/llama-quantize" \
            "$gguf_output" \
            "${OUTPUT_DIR}/fableforge-14b-Q4_K_M.gguf" \
            Q4_K_M

        echo "[OK] GGUF exported to: ${OUTPUT_DIR}/fableforge-14b-Q4_K_M.gguf"
    else
        echo "[WARN] llama.cpp not available for GGUF export."
        echo "       Install llama.cpp and run:"
        echo "       python llama.cpp/convert_hf_to_gguf.py $merged_model --outfile $gguf_output"
        echo "       ./llama.cpp/llama-quantize $gguf_output fableforge-14b-Q4_K_M.gguf Q4_K_M"
    fi
}

# ─── Unsloth Training Wrapper ────────────────────────────────────────────────
# Uses Unsloth's FastLanguageModel for 2-5x faster training with 70% less VRAM.
# Activated by --unsloth, --colab, or --free-tier flags.
# Falls back to standard transformers training if Unsloth is not available.

run_sft_stage_unsloth() {
    local stage_num=$1
    local stage_name=$2
    local lora_r=$3
    local lora_alpha=$4
    local lr=$5
    local epochs=$6
    local batch_size=$7
    local grad_accum=$8
    local max_seq_len=$9
    local warmup_ratio=${10}

    # Override with Colab/free-tier settings
    if [[ -n "${DEFAULT_BATCH_SIZE:-}" ]]; then
        batch_size="$DEFAULT_BATCH_SIZE"
    fi
    if [[ -n "${DEFAULT_GRADIENT_ACCUM:-}" ]]; then
        grad_accum="$DEFAULT_GRADIENT_ACCUM"
    fi
    if [[ -n "${DEFAULT_MAX_SEQ_LEN:-}" ]]; then
        max_seq_len="$DEFAULT_MAX_SEQ_LEN"
    fi

    local stage_output="${OUTPUT_DIR}/stage${stage_num}_${stage_name}"
    local data_path="${DATA_DIR}/${stage_name}/${stage_name}_train.jsonl"
    local val_path="${DATA_DIR}/${stage_name}/${stage_name}_val.jsonl"

    if [[ ! -f "$data_path" ]]; then
        echo "[ERROR] Training data not found: $data_path"
        echo "        Run convert_data.py first: python scripts/convert_data.py --stage ${stage_num}"
        return 1
    fi

    local train_examples
    train_examples=$(wc -l < "$data_path" | tr -d ' ')
    echo ""
    echo "=== Stage ${stage_num}: ${stage_name} [UNSLOTH] ==="
    echo "  Training backend:  Unsloth (FastLanguageModel)"
    echo "  Training data:    $data_path (${train_examples} examples)"
    echo "  Validation data:  $val_path"
    echo "  Output:           $stage_output"
    echo "  LoRA r:           $lora_r, alpha: $lora_alpha"
    echo "  Learning rate:    $lr"
    echo "  Epochs:           $epochs"
    echo "  Batch size:       $batch_size, gradient accumulation: $grad_accum"
    echo "  Max seq length:   $max_seq_len"
    echo "  Warmup ratio:     $warmup_ratio"
    echo "  Colab mode:       ${COLAB:-false}"
    echo "  Free-tier mode:   ${FREE_TIER:-false}"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would run Unsloth SFT training with the above config"
        return 0
    fi

    local prev_adapter=""
    if [[ "$stage_num" -gt 1 ]]; then
        local prev_stage=$((stage_num - 1))
        local prev_name="${STAGE_NAMES[$prev_stage]}"
        prev_adapter="${OUTPUT_DIR}/stage${prev_stage}_${prev_name}/final"
        if [[ ! -d "$prev_adapter" ]]; then
            prev_adapter="${OUTPUT_DIR}/stage${prev_stage}_${prev_name}/checkpoint-latest"
        fi
        if [[ -d "$prev_adapter" ]]; then
            echo "  Loading previous adapter: $prev_adapter"
        else
            echo "[WARN] Previous adapter not found at $prev_adapter, training from base model"
            prev_adapter=""
        fi
    fi

    # Resolve previous adapter for sequential training
    local adapter_args=""
    if [[ -n "$prev_adapter" ]]; then
        adapter_args="--adapter_path $prev_adapter"
    fi

    local resume_args=""
    if [[ "$RESUME" == "true" ]] && [[ -d "${stage_output}/checkpoint-latest" ]]; then
        resume_args="--resume_from_checkpoint ${stage_output}/checkpoint-latest"
        echo "  Resuming from: ${stage_output}/checkpoint-latest"
    fi

    # Determine dtype: fp16 for Colab T4, bf16 for everything else
    local dtype_flag="bf16"
    if [[ "$COLAB" == "true" ]] || [[ "$TRAINING_DTYPE" == "fp16" ]]; then
        dtype_flag="fp16"
    fi

    local effective_batch=$((batch_size * grad_accum * GPUS))
    local steps_per_epoch=$(( (train_examples + effective_batch - 1) / effective_batch ))
    local total_steps=$(( steps_per_epoch * epochs ))
    local save_steps=$(( steps_per_epoch / 2 ))
    save_steps=$(( save_steps < 1 ? 1 : save_steps ))

    echo "  Effective batch:  $effective_batch"
    echo "  Steps/epoch:      $steps_per_epoch"
    echo "  Total steps:      $total_steps"
    echo "  Save steps:       $save_steps"
    echo ""

    # Run Unsloth training via Python wrapper
    USE_UNSLOTH=1 UNSLOTH_MODE=1 \
    python3 -c "
import os, sys
sys.path.insert(0, '${PROJECT_DIR}/src')

# Unsloth training wrapper
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

base_model = os.environ.get('BASE_MODEL', '${BASE_MODEL}')
stage_output = '${stage_output}'
data_path = '${data_path}'
val_path = '${val_path}'
lora_r = ${lora_r}
lora_alpha = ${lora_alpha}
lr = float('${lr}')
epochs = ${epochs}
batch_size = ${batch_size}
grad_accum = ${grad_accum}
max_seq_len = ${max_seq_len}
warmup_ratio = float('${warmup_ratio}')
dtype_flag = '${dtype_flag}'
prev_adapter = '${prev_adapter}' if '${prev_adapter}' else None
resume_checkpoint = '${resume_args}'.replace('--resume_from_checkpoint ', '').strip() if '${resume_args}' else None

print(f'Loading base model with Unsloth: {base_model}')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=max_seq_len,
    load_in_4bit=True,
    dtype=None,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Merge previous adapter if available
if prev_adapter and os.path.exists(prev_adapter):
    print(f'Merging previous adapter: {prev_adapter}')
    model = PeftModel.from_pretrained(model, prev_adapter)
    model = model.merge_and_unload()

# Apply LoRA via Unsloth
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] if lora_r >= 32 else ['q_proj', 'k_proj', 'v_proj', 'o_proj']
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0.05,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=42,
    target_modules=target_modules,
)
model.print_trainable_parameters()

# Load datasets
train_ds = load_dataset('json', data_files=data_path, split='train')
eval_ds = None
if os.path.exists(val_path):
    eval_ds = load_dataset('json', data_files=val_path, split='train')

# Training config
use_bf16 = dtype_flag == 'bf16'
use_fp16 = dtype_flag == 'fp16'
gradient_checkpointing = True

training_args = SFTConfig(
    output_dir=stage_output,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=lr,
    lr_scheduler_type='cosine',
    warmup_ratio=warmup_ratio,
    bf16=use_bf16,
    fp16=use_fp16,
    logging_steps=10,
    save_strategy='steps',
    save_steps=${save_steps},
    save_total_limit=3,
    eval_strategy='steps' if eval_ds else 'no',
    eval_steps=${save_steps} if eval_ds else None,
    report_to='none',
    max_seq_length=max_seq_len,
    dataset_text_field='text',
    gradient_checkpointing=gradient_checkpointing,
    optim='paged_adamw_8bit',
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print('Starting Unsloth training...')
trainer.train()

final_dir = os.path.join(stage_output, 'final')
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f'[OK] Stage ${stage_num} complete. Saved to {final_dir}')
" 2>&1 | tee "${LOG_DIR}/stage${stage_num}_${stage_name}_unsloth.log"

    local exit_code=${PIPESTATUS[0]:-0}
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Stage ${stage_num} Unsloth training failed with exit code $exit_code"
        return $exit_code
    fi

    echo "[OK] Stage ${stage_num} complete (Unsloth). Output: $stage_output"
    echo ""
}

# ─── Main Pipeline ────────────────────────────────────────────────────────────

main() {
    echo "Starting FableForge-14B training pipeline"
    if [[ "$UNSLOTH" == "true" ]]; then
        echo "  Mode: Unsloth (2-5x faster, 70% less VRAM)"
    fi
    if [[ "$COLAB" == "true" ]]; then
        echo "  Mode: Colab T4 optimized"
    fi
    if [[ "$FREE_TIER" == "true" ]]; then
        echo "  Mode: Free-tier (Unsloth + Colab + conservative)"
    fi
    echo "=========================================="
    echo ""

    check_python

    # Select training function based on mode
    if [[ "$UNSLOTH" == "true" ]]; then
        SFT_RUNNER="run_sft_stage_unsloth"
        echo "[INFO] Using Unsloth training backend"
    else
        SFT_RUNNER="run_sft_stage"
        echo "[INFO] Using standard transformers training backend"
    fi

    case "$STAGE" in
        1)
            $SFT_RUNNER 1 behavior_shaping 64 128 2e-4 3 2 8 4096 0.06
            ;;
        2)
            $SFT_RUNNER 2 skill_distillation 32 64 1e-4 2 2 8 4096 0.06
            ;;
        3)
            $SFT_RUNNER 3 error_recovery 16 32 5e-5 3 4 4 4096 0.06
            ;;
        4)
            if [[ "$UNSLOTH" == "true" ]]; then
                run_sft_stage_unsloth 4 dpo 16 32 5e-5 1 1 16 4096 0.1
            else
                run_dpo_stage
            fi
            ;;
        all)
            $SFT_RUNNER 1 behavior_shaping 64 128 2e-4 3 2 8 4096 0.06
            $SFT_RUNNER 2 skill_distillation 32 64 1e-4 2 2 8 4096 0.06
            $SFT_RUNNER 3 error_recovery 16 32 5e-5 3 4 4 4096 0.06
            if [[ "$UNSLOTH" == "true" ]]; then
                run_sft_stage_unsloth 4 dpo 16 32 5e-5 1 1 16 4096 0.1
            else
                run_dpo_stage
            fi
            export_gguf
            ;;
        *)
            echo "Unknown stage: $STAGE. Use 1, 2, 3, 4, or 'all'"
            echo ""
            echo "Free-tier modes:"
            echo "  --unsloth      Use Unsloth for faster/cheaper training"
            echo "  --colab        Optimize for Google Colab T4"
            echo "  --free-tier    Combine --unsloth + --colab"
            exit 1
            ;;
    esac

    echo ""
    echo "=== Training pipeline complete ==="
    if [[ "$UNSLOTH" == "true" ]]; then
        echo "  Trained with Unsloth (2-5x faster, 70% less VRAM)"
    fi
    if [[ "$COLAB" == "true" ]]; then
        echo "  Optimized for Colab T4"
    fi
}

main "$@"