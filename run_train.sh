#!/usr/bin/env bash
# Latent Reward Model — Accelerate + DeepSpeed ZeRO-2
# 配置见本目录 accel_ds2.yaml、ds_zero2.json

set -euo pipefail

# ─── 环境（与 baseline 脚本一致）────────────────────────────────────────────
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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-latent_mrm_${BACKBONE_TYPE}}"
OUTPUT_DIR="${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

# 数据：默认使用 latent_reward_model/data 下 jsonl（需先运行 prepare_full_data.py）
TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

# SwanLab（API Key 由环境变量注入，勿写入脚本）
SWANLAB_PROJECT="${SWANLAB_PROJECT:-latentMRM}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-${BACKBONE_TYPE}}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
elif [[ "${SWANLAB_MODE}" == "cloud" ]]; then
  echo "[警告] SWANLAB_MODE=cloud 但未设置 SWANLAB_API_KEY；请先 export SWANLAB_API_KEY，或 SWANLAB_MODE=disabled"
fi

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
echo "[SwanLab] project=${SWANLAB_PROJECT}  experiment=${SWANLAB_EXPERIMENT_NAME}  mode=${SWANLAB_MODE}"

export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "${LATENT_DIR}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  scripts/train_rm.py \
  --backbone_type "${BACKBONE_TYPE}" \
  --train_data_path "${TRAIN_JSONL}" \
  --eval_data_path "${VAL_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --k_dimensions 10 \
  --lambda_neg 1.0 \
  --beta_dir 0.2 \
  --target_tau 0.8 \
  --num_pos_heads 10 \
  --train_stage latent \
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
  --max_train_samples 0 \
  --run_name "${EXPERIMENT_NAME}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}_5000train_samples" \
  --swanlab_mode "${SWANLAB_MODE}"
