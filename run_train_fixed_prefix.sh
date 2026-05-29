#!/usr/bin/env bash
# 无 selector：维度 0 .. (K+-1) 为正向维，其余为负向维
#   单阶段：TRAIN_STAGE=fixed_joint（heads+gate）
#   两阶段：bash run_train_two_stage_fixed_prefix.sh（fixed_latent → fixed_gate）

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
K_DIM="${K_DIM:-10}"
K_PLUS="${K_PLUS:-10}"
TRAIN_STAGE="${TRAIN_STAGE:-fixed_latent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-latent_mrm_${BACKBONE_TYPE}_k${K_DIM}_prefix${K_PLUS}_${TRAIN_STAGE}}"
OUTPUT_DIR="${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"

TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${LATENT_DIR}"

EXTRA=()
if [[ "${TRAIN_STAGE}" == "fixed_joint" ]]; then
  EXTRA+=(--use_gate)
fi

echo "[启动] ${TRAIN_STAGE}  K+=dim[0,${K_PLUS})  K=${K_DIM}  output=${OUTPUT_DIR}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  scripts/train_rm.py \
  --backbone_type "${BACKBONE_TYPE}" \
  --train_data_path "${TRAIN_JSONL}" \
  --eval_data_path "${VAL_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_stage "${TRAIN_STAGE}" \
  --k_dimensions "${K_DIM}" \
  --num_pos_heads "${K_PLUS}" \
  --lambda_neg 1.0 \
  "${EXTRA[@]}" \
  --head_lr 1e-5 \
  --backbone_lr 1e-7 \
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
  --run_name "${EXPERIMENT_NAME}" \
  --swanlab_experiment_name "${EXPERIMENT_NAME}" \
  "$@"
