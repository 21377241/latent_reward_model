#!/usr/bin/env bash
# Latent Reward Model — Accelerate + DeepSpeed ZeRO-2
# 复用 solution1/accel_ds2.yaml 与 ds_zero2.json（与 baseline 一致）

set -euo pipefail

REPO_ROOT="${REWARD_MODEL_ROOT:-/mnt/afs/250010036/reward_model}"
LATENT_DIR="${REPO_ROOT}/latent_reward_model"
ACCEL_CONFIG="${REPO_ROOT}/solution1/accel_ds2.yaml"

BACKBONE_TYPE="${BACKBONE_TYPE:-llama3_baseline}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-latent_mrm_${BACKBONE_TYPE}}"
OUTPUT_DIR="${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

# 数据：默认使用 latent_reward_model/data 下 jsonl（需先运行 prepare_full_data.py）
TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "[错误] Accelerate 配置不存在: ${ACCEL_CONFIG}"
  exit 1
fi

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [[ "${NGPU}" -lt 4 ]]; then
  echo "[警告] 检测到 GPU 数量 < 4，accel_ds2.yaml 默认配置为 4 进程"
fi

echo "[启动] backbone=${BACKBONE_TYPE}  output=${OUTPUT_DIR}"
echo "[Accelerate] config=${ACCEL_CONFIG}"

cd "${REPO_ROOT}/solution1"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  "${LATENT_DIR}/scripts/train_rm.py" \
  --backbone_type "${BACKBONE_TYPE}" \
  --train_data_path "${TRAIN_JSONL}" \
  --eval_data_path "${VAL_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --k_dimensions 8 \
  --lambda_neg 1.0 \
  --beta_dir 0.2 \
  --target_tau 0.8 \
  --num_pos_heads 5 \
  --head_lr 1e-5 \
  --backbone_lr 1e-7 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --batch_size 4 \
  --grad_accum 32 \
  --num_epochs 2 \
  --max_grad_norm 1.0 \
  --eval_steps 50 \
  --logging_steps 10 \
  --save_steps 500 \
  --max_length 4096 \
  --max_eval_samples 2000 \
  --run_name "${EXPERIMENT_NAME}"
