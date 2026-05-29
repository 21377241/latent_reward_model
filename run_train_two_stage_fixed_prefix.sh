#!/usr/bin/env bash
# 无 selector 两阶段训练（固定前缀 K+/K-）
#   阶段 1：fixed_latent（backbone + reward_heads）
#   阶段 2：fixed_gate（仅 gating_network，resume 阶段1 best）
#
# 用法:
#   cd /mnt/afs/250010036/reward_model/latent_reward_model
#   bash run_train_two_stage_fixed_prefix.sh
#
#   K_PLUS=6 NUM_POS_HEADS=6 bash run_train_two_stage_fixed_prefix.sh
#   SKIP_STAGE1=1 bash run_train_two_stage_fixed_prefix.sh   # 只训阶段2
#
# 单阶段 fixed_joint 见 run_train_fixed_prefix.sh

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

LATENT_DIR="${REWARD_MODEL_ROOT:-/mnt/afs/250010036/reward_model}/latent_reward_model"

_detect_num_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local _s="${CUDA_VISIBLE_DEVICES// /}"
    if [[ "${_s}" == *","* ]]; then
      echo "${_s}" | awk -F, '{print NF}'
      return
    fi
    echo 1
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l || echo 1
}

_NUM_GPUS="$(_detect_num_gpus)"
if [[ -z "${ACCEL_CONFIG:-}" ]]; then
  if [[ "${_NUM_GPUS}" -ge 4 ]]; then
    ACCEL_CONFIG="${LATENT_DIR}/accel_ds2.yaml"
  else
    ACCEL_CONFIG="${LATENT_DIR}/accel_1gpu.yaml"
  fi
fi

BACKBONE_TYPE="${BACKBONE_TYPE:-llama3.1_baseline}"
K_DIMENSIONS="${K_DIMENSIONS:-${K_DIM:-10}}"
NUM_POS_HEADS="${NUM_POS_HEADS:-${K_PLUS:-10}}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-_k${K_DIMENSIONS}_prefix${NUM_POS_HEADS}_fixed_2stage}"

# 阶段1 / 阶段2 独立实验名（目录 + SwanLab）
STAGE1_NAME="${STAGE1_NAME:-latent_mrm_${BACKBONE_TYPE}${EXPERIMENT_SUFFIX}_latent}"
STAGE2_NAME="${STAGE2_NAME:-${STAGE1_NAME}_gate}"
STAGE1_OUT="${STAGE1_OUT:-${LATENT_DIR}/experiments/${STAGE1_NAME}}"
STAGE2_OUT="${STAGE2_OUT:-${LATENT_DIR}/experiments/${STAGE2_NAME}}"
RESUME_TAG="${RESUME_TAG:-best}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
HEAD_LR="${HEAD_LR:-1e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-7}"
GATE_LR="${GATE_LR:-1e-4}"
LAMBDA_NEG="${LAMBDA_NEG:-1.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"

SWANLAB_PROJECT="${SWANLAB_PROJECT:-latentMRM}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

if [[ "${NUM_POS_HEADS}" -gt "${K_DIMENSIONS}" ]]; then
  echo "[错误] NUM_POS_HEADS=${NUM_POS_HEADS} 不能大于 K_DIMENSIONS=${K_DIMENSIONS}"
  exit 1
fi

if [[ ! -f "${ACCEL_CONFIG}" ]]; then
  echo "[错误] Accelerate 配置不存在: ${ACCEL_CONFIG}"
  exit 1
fi
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[错误] 训练数据不存在: ${TRAIN_JSONL}"
  exit 1
fi

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
elif [[ "${SWANLAB_MODE}" == "cloud" ]]; then
  echo "[警告] SWANLAB_MODE=cloud 但未设置 SWANLAB_API_KEY；请 export 或 SWANLAB_MODE=disabled"
fi

export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"
RESUME_FROM="${STAGE1_OUT}/${RESUME_TAG}"

_common_args() {
  echo \
    --backbone_type "${BACKBONE_TYPE}" \
    --train_data_path "${TRAIN_JSONL}" \
    --eval_data_path "${VAL_JSONL}" \
    --k_dimensions "${K_DIMENSIONS}" \
    --num_pos_heads "${NUM_POS_HEADS}" \
    --lambda_neg "${LAMBDA_NEG}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --max_grad_norm 1.0 \
    --eval_steps 1 \
    --logging_steps 1 \
    --save_steps 500 \
    --max_length 4096 \
    --max_eval_samples 2000 \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_mode "${SWANLAB_MODE}"
}

echo "════════════════════════════════════════════════════════════"
echo " 无 selector 两阶段（fixed_prefix）"
echo "  backbone=${BACKBONE_TYPE}  K=${K_DIMENSIONS}  K+=dim[0,${NUM_POS_HEADS})"
echo "  阶段1 fixed_latent → ${STAGE1_OUT}"
echo "       SwanLab: ${STAGE1_NAME}"
echo "  阶段2 fixed_gate   → ${STAGE2_OUT}"
echo "       SwanLab: ${STAGE2_NAME}"
echo "       resume: ${RESUME_FROM}"
echo "  ACCEL_CONFIG=${ACCEL_CONFIG}  gpus=${_NUM_GPUS}"
echo "════════════════════════════════════════════════════════════"

cd "${LATENT_DIR}"

# ─── 阶段 1：backbone + heads ─────────────────────────────────────────────
if [[ "${SKIP_STAGE1}" == "1" ]]; then
  if [[ -f "${RESUME_FROM}/latent_heads.pt" ]]; then
    echo ""
    echo ">>> [阶段 1/2] 跳过 SKIP_STAGE1=1，使用 ${RESUME_FROM}"
    echo ""
  else
    echo "[错误] SKIP_STAGE1=1 但 checkpoint 不存在: ${RESUME_FROM}"
    exit 1
  fi
else
  echo ""
  echo ">>> [阶段 1/2] train_stage=fixed_latent  epochs=${STAGE1_EPOCHS}"
  echo ""

  # shellcheck disable=SC2046
  accelerate launch --config_file "${ACCEL_CONFIG}" \
    scripts/train_rm.py \
    $(_common_args) \
    --output_dir "${STAGE1_OUT}" \
    --train_stage fixed_latent \
    --head_lr "${HEAD_LR}" \
    --backbone_lr "${BACKBONE_LR}" \
    --num_epochs "${STAGE1_EPOCHS}" \
    --run_name "${STAGE1_NAME}" \
    --swanlab_experiment_name "${STAGE1_NAME}"
fi

if [[ ! -f "${RESUME_FROM}/latent_heads.pt" ]]; then
  echo "[错误] 阶段1 未产生 checkpoint: ${RESUME_FROM}"
  exit 1
fi

# ─── 阶段 2：仅 gate ───────────────────────────────────────────────────────
echo ""
echo ">>> [阶段 2/2] train_stage=fixed_gate  epochs=${STAGE2_EPOCHS}"
echo ""

# shellcheck disable=SC2046
accelerate launch --config_file "${ACCEL_CONFIG}" \
  scripts/train_rm.py \
  $(_common_args) \
  --output_dir "${STAGE2_OUT}" \
  --resume_from "${RESUME_FROM}" \
  --use_gate \
  --train_stage fixed_gate \
  --lambda_gate 1.0 \
  --gate_hidden_size 1024 \
  --gate_num_layers 3 \
  --gate_temperature 10.0 \
  --gate_lr "${GATE_LR}" \
  --num_epochs "${STAGE2_EPOCHS}" \
  --run_name "${STAGE2_NAME}" \
  --swanlab_experiment_name "${STAGE2_NAME}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " 完成"
echo "  阶段1 : ${RESUME_FROM}"
echo "  阶段2 : ${STAGE2_OUT}/best"
echo "  评测  : ${STAGE2_OUT}/eval_export/best/"
echo "════════════════════════════════════════════════════════════"
