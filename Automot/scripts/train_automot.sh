#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOMOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
QWEN3VL_PATH="${QWEN3VL_PATH:-}"
AUTOMOT_MODEL_PATH="${AUTOMOT_MODEL_PATH:-}"
AUTOMOT_RESUME_FROM="${AUTOMOT_RESUME_FROM:-}"
PDM_DATA_DIR="${PDM_DATA_DIR:-}"
PDM_JSONL_DIR="${PDM_JSONL_DIR:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${AUTOMOT_ROOT}/checkpoints/traj_meta}"
RESULTS_DIR="${RESULTS_DIR:-${AUTOMOT_ROOT}/results}"

TRAIN_CONFIG_FILE="${TRAIN_CONFIG_FILE:-${AUTOMOT_ROOT}/mot/data/automot/configs/automot_traj_train.yaml}"
EVAL_CONFIG_FILE="${EVAL_CONFIG_FILE:-${AUTOMOT_ROOT}/mot/data/automot/configs/automot_traj_eval.yaml}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29550}"

TOTAL_STEPS="${TOTAL_STEPS:-30000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-1500}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DO_EVAL="${DO_EVAL:-True}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-167}"

MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-11520}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-11520}"
EXPECTED_NUM_TOKENS="${EXPECTED_NUM_TOKENS:-11520}"
REASONING_QUERY_MAX_NUM_TOKENS="${REASONING_QUERY_MAX_NUM_TOKENS:-8}"

SHARDING_STRATEGY="${SHARDING_STRATEGY:-HYBRID_SHARD}"
CPU_OFFLOAD="${CPU_OFFLOAD:-False}"
BACKWARD_PREFETCH="${BACKWARD_PREFETCH:-BACKWARD_PRE}"

WANDB_PROJECT="${WANDB_PROJECT:-AutoMoT}"
WANDB_NAME="${WANDB_NAME:-automot-traj-$(date +%m%d_%H%M)}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WANDB_DIR="${WANDB_DIR:-${RESULTS_DIR}/wandb}"

require_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    echo "Set ${name} before running ${BASH_SOURCE[0]}" >&2
    exit 1
  fi
}

require_var AUTOMOT_MODEL_PATH
require_var PDM_DATA_DIR
require_var PDM_JSONL_DIR

if [[ -z "${QWEN3VL_PATH}" ]]; then
  QWEN3VL_PATH="${AUTOMOT_MODEL_PATH}"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${AUTOMOT_ROOT}:${AUTOMOT_ROOT}/mot:${PYTHONPATH}"
else
  export PYTHONPATH="${AUTOMOT_ROOT}:${AUTOMOT_ROOT}/mot"
fi

export CUDA_VISIBLE_DEVICES
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export AUTOMOT_COMPILE_BLOCK_MASK="${AUTOMOT_COMPILE_BLOCK_MASK:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3VL_PATH AUTOMOT_MODEL_PATH PDM_DATA_DIR PDM_JSONL_DIR
export WANDB_PROJECT WANDB_NAME WANDB_DIR

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}" "${WANDB_DIR}"

RESUME_ARGS=()
if [[ -n "${AUTOMOT_RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume_from "${AUTOMOT_RESUME_FROM}")
fi

"${PYTHON_EXECUTABLE}" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  "${AUTOMOT_ROOT}/train/train_automot.py" \
  --train_config_file "${TRAIN_CONFIG_FILE}" \
  --eval_config_file "${EVAL_CONFIG_FILE}" \
  --layer_module Qwen3VLMoTDecoderLayer \
  --use_flex False \
  --model_path "${AUTOMOT_MODEL_PATH}" \
  --qwen3vl_path "${QWEN3VL_PATH}" \
  --results_dir "${RESULTS_DIR}" \
  --finetune_from_hf True \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --tie_word_embeddings True \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "${WANDB_NAME}" \
  --num_shard "${NUM_GPUS}" \
  --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --expected_num_tokens "${EXPECTED_NUM_TOKENS}" \
  --reasoning_query_max_num_tokens "${REASONING_QUERY_MAX_NUM_TOKENS}" \
  --total_steps "${TOTAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --do_eval "${DO_EVAL}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_max_steps "${EVAL_MAX_STEPS}" \
  --sharding_strategy "${SHARDING_STRATEGY}" \
  --cpu_offload "${CPU_OFFLOAD}" \
  --backward_prefetch "${BACKWARD_PREFETCH}" \
  --wandb_offline "${WANDB_OFFLINE}" \
  "${RESUME_ARGS[@]}"
