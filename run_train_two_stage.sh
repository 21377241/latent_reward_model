#!/usr/bin/env bash
# Latent Reward Model — 两阶段串行训练
#   阶段 1：latent（backbone + reward_heads + selector）
#   阶段 2：gate（冻结阶段1，仅训 gating_network）
#
# 用法:
#   cd /mnt/afs/250010036/reward_model/latent_reward_model
#   bash run_train_two_stage.sh
#
#   NUM_POS_HEADS=7 SWANLAB_MODE=disabled bash run_train_two_stage.sh
#
#   阶段1 已有、只训 gate：
#   SKIP_STAGE1=1 NUM_POS_HEADS=6 bash run_train_two_stage.sh
#
#   K+=5,6,8,9 批量两阶段+评测：bash run_kplus_sweep_train_eval.sh
#
# 默认不会写入 experiments/latent_mrm_llama3.1_baseline（旧实验 K+=10 全正向），
# 而是独立目录，例如 experiments/latent_mrm_llama3.1_baseline_k10_kplus7_2stage/

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

LATENT_DIR="${REWARD_MODEL_ROOT:-/mnt/afs/250010036/reward_model}/latent_reward_model"

# 按可见 GPU 数选择 Accelerate 配置（避免「只暴露 1 卡却 launch 4 进程」）
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

# ─── 可配置超参（环境变量覆盖）────────────────────────────────────────────
BACKBONE_TYPE="${BACKBONE_TYPE:-llama3.1_baseline}"
K_DIMENSIONS="${K_DIMENSIONS:-10}"
NUM_POS_HEADS="${NUM_POS_HEADS:-7}"
# 与 run_train.sh 默认目录 latent_mrm_<backbone>（K+=10）区分开
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-_k${K_DIMENSIONS}_kplus${NUM_POS_HEADS}_2stage}"

STAGE1_NAME="${STAGE1_NAME:-latent_mrm_${BACKBONE_TYPE}${EXPERIMENT_SUFFIX}}"
STAGE1_OUT="${STAGE1_OUT:-${LATENT_DIR}/experiments/${STAGE1_NAME}}"
STAGE2_NAME="${STAGE2_NAME:-${STAGE1_NAME}_gate}"
STAGE2_OUT="${STAGE2_OUT:-${LATENT_DIR}/experiments/${STAGE2_NAME}}"
RESUME_TAG="${RESUME_TAG:-best}"
SKIP_STAGE1="${SKIP_STAGE1:-0}"

# 旧单阶段实验（run_train.sh 默认 num_pos_heads=10，目录名无 kplus 后缀）
LEGACY_STAGE1_DIRS=(
  "${LATENT_DIR}/experiments/latent_mrm_llama3.1_baseline"
  "${LATENT_DIR}/experiments/latent_mrm_llama3_baseline"
)

TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
HEAD_LR="${HEAD_LR:-1e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-7}"
GATE_LR="${GATE_LR:-1e-4}"
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
  echo "[错误] 训练数据不存在: ${TRAIN_JSONL}（请先运行 scripts/prepare_full_data.py）"
  exit 1
fi

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
elif [[ "${SWANLAB_MODE}" == "cloud" ]]; then
  echo "[警告] SWANLAB_MODE=cloud 但未设置 SWANLAB_API_KEY；请 export SWANLAB_API_KEY 或 SWANLAB_MODE=disabled"
fi

export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

_realpath_or_echo() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "${p}" 2>/dev/null || echo "${p}"
  else
    echo "${p}"
  fi
}

STAGE1_OUT_ABS="$(_realpath_or_echo "${STAGE1_OUT}")"
STAGE2_OUT_ABS="$(_realpath_or_echo "${STAGE2_OUT}")"
for legacy in "${LEGACY_STAGE1_DIRS[@]}"; do
  legacy_abs="$(_realpath_or_echo "${legacy}")"
  if [[ "${STAGE1_OUT_ABS}" == "${legacy_abs}" ]] || [[ "${STAGE2_OUT_ABS}" == "${legacy_abs}" ]]; then
    echo "[错误] 输出目录指向旧实验 ${legacy}"
    echo "  该目录为 run_train.sh 默认结果（通常 num_pos_heads=10），与当前 NUM_POS_HEADS=${NUM_POS_HEADS} 不兼容。"
    echo "  请勿设置 STAGE1_OUT/STAGE2_OUT 到该路径；使用本脚本默认目录或自定义 STAGE1_NAME。"
    exit 1
  fi
  if [[ "${STAGE1_OUT_ABS}" == "${legacy_abs}"* ]] && [[ "${STAGE1_OUT_ABS}" != "${legacy_abs}" ]]; then
    echo "[警告] STAGE1_OUT 位于旧实验目录树下: ${STAGE1_OUT}"
  fi
done

mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}"

RESUME_FROM="${STAGE1_OUT}/${RESUME_TAG}"
# 阶段2 只从本次阶段1 输出恢复，绝不默认读 latent_mrm_*_baseline/best

_common_args() {
  echo \
    --backbone_type "${BACKBONE_TYPE}" \
    --train_data_path "${TRAIN_JSONL}" \
    --eval_data_path "${VAL_JSONL}" \
    --k_dimensions "${K_DIMENSIONS}" \
    --num_pos_heads "${NUM_POS_HEADS}" \
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
echo " 两阶段 Latent MRM（独立实验目录，不覆盖 run_train.sh 默认 baseline）"
echo "  backbone=${BACKBONE_TYPE}  K=${K_DIMENSIONS}  K+=${NUM_POS_HEADS}"
echo "  阶段1 → ${STAGE1_OUT}"
echo "  阶段2 → ${STAGE2_OUT}  (resume 本次阶段1: ${RESUME_FROM})"
echo "  ACCEL_CONFIG=${ACCEL_CONFIG}  visible_gpus=${_NUM_GPUS}"
echo "  global_batch=$(( _NUM_GPUS * BATCH_SIZE * GRAD_ACCUM ))"
echo "════════════════════════════════════════════════════════════"

cd "${LATENT_DIR}"

# ─── 阶段 1 ───────────────────────────────────────────────────────────────
if [[ "${SKIP_STAGE1}" == "1" ]]; then
  if [[ -f "${RESUME_FROM}/latent_heads.pt" ]]; then
    echo ""
    echo ">>> [阶段 1/2] 跳过（SKIP_STAGE1=1，使用已有 ${RESUME_FROM}）"
    echo ""
  else
    echo "[错误] SKIP_STAGE1=1 但阶段1 checkpoint 不存在: ${RESUME_FROM}"
    exit 1
  fi
else
  echo ""
  echo ">>> [阶段 1/2] train_stage=latent  epochs=${STAGE1_EPOCHS}"
  echo ""

  # shellcheck disable=SC2046
  accelerate launch --config_file "${ACCEL_CONFIG}" \
    scripts/train_rm.py \
    $(_common_args) \
    --output_dir "${STAGE1_OUT}" \
    --train_stage latent \
    --head_lr "${HEAD_LR}" \
    --backbone_lr "${BACKBONE_LR}" \
    --num_epochs "${STAGE1_EPOCHS}" \
    --swanlab_experiment_name "${STAGE1_NAME}"
fi

if [[ ! -d "${RESUME_FROM}" ]] || [[ ! -f "${RESUME_FROM}/latent_heads.pt" ]]; then
  echo "[错误] 阶段1 未产生有效 checkpoint: ${RESUME_FROM}"
  echo "  仅训阶段2: SKIP_STAGE1=1 bash run_train_two_stage.sh"
  echo "  或尝试 RESUME_TAG=final"
  exit 1
fi

# ─── 阶段 2 ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [阶段 2/2] train_stage=gate  epochs=${STAGE2_EPOCHS}"
echo ""

# shellcheck disable=SC2046
accelerate launch --config_file "${ACCEL_CONFIG}" \
  scripts/train_rm.py \
  $(_common_args) \
  --output_dir "${STAGE2_OUT}" \
  --resume_from "${RESUME_FROM}" \
  --use_gate \
  --train_stage gate \
  --lambda_gate 1.0 \
  --gate_hidden_size 1024 \
  --gate_num_layers 3 \
  --gate_temperature 10.0 \
  --gate_lr "${GATE_LR}" \
  --num_epochs "${STAGE2_EPOCHS}" \
  --swanlab_experiment_name "${STAGE2_NAME}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " 完成"
echo "  阶段1 checkpoint : ${RESUME_FROM}"
echo "  阶段2 checkpoint : ${STAGE2_OUT}/best"
echo "  评测导出         : ${STAGE2_OUT}/eval_export/best/"
echo "════════════════════════════════════════════════════════════"
