#!/usr/bin/env bash
# 阶段 2：在阶段 1 checkpoint 上仅训练 Gate（backbone / heads / selector 冻结）
#
# 用法示例：
#   RESUME_FROM=/path/to/experiments/latent_mrm_llama3.1_baseline/best \
#   OUTPUT_DIR=/path/to/experiments/latent_mrm_llama3.1_baseline_gate \
#   bash run_train_gate.sh

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

LATENT_DIR="${REWARD_MODEL_ROOT:-/mnt/afs/250010036/reward_model}/latent_reward_model"
ACCEL_CONFIG="${LATENT_DIR}/accel_ds2.yaml"

BACKBONE_TYPE="${BACKBONE_TYPE:-llama3.1_baseline}"
STAGE1_NAME="${STAGE1_NAME:-latent_mrm_${BACKBONE_TYPE}}"
RESUME_FROM="${RESUME_FROM:-${LATENT_DIR}/experiments/${STAGE1_NAME}/best}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${STAGE1_NAME}_gate}"
OUTPUT_DIR="${OUTPUT_DIR:-${LATENT_DIR}/experiments/${EXPERIMENT_NAME}}"

TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

SWANLAB_PROJECT="${SWANLAB_PROJECT:-latentMRM}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ ! -d "${RESUME_FROM}" ]]; then
  echo "[错误] 阶段1 checkpoint 不存在: ${RESUME_FROM}"
  echo "  请先完成阶段1训练，或设置 RESUME_FROM=.../best"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "[阶段2 Gate] resume=${RESUME_FROM}"
echo "[阶段2 Gate] output=${OUTPUT_DIR}"
echo "[阶段2 Gate] 训练结束后自动导出: ${OUTPUT_DIR}/eval_export/best/"
echo "[阶段2 Gate] 若需补导出: PYTHONPATH=. python scripts/export_for_eval.py ${OUTPUT_DIR}/best"

cd "${LATENT_DIR}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  scripts/train_rm.py \
  --backbone_type "${BACKBONE_TYPE}" \
  --train_data_path "${TRAIN_JSONL}" \
  --eval_data_path "${VAL_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from "${RESUME_FROM}" \
  --k_dimensions 10 \
  --num_pos_heads 10 \
  --use_gate \
  --train_stage gate \
  --lambda_gate 1.0 \
  --gate_hidden_size 1024 \
  --gate_num_layers 3 \
  --gate_temperature 10.0 \
  --gate_lr 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --batch_size 4 \
  --grad_accum 32 \
  --num_epochs 2 \
  --max_grad_norm 1.0 \
  --eval_steps 10 \
  --logging_steps 1 \
  --save_steps 500 \
  --max_length 4096 \
  --max_eval_samples 2000 \
  --max_train_samples 0 \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}" \
  --swanlab_mode "${SWANLAB_MODE}"
