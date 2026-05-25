#!/usr/bin/env bash
# =============================================================================
# Latent Reward Model — 串行训练多个 backbone
#
# 用法（在 latent_reward_model/ 或任意目录）：
#   export SWANLAB_API_KEY="..."   # cloud 模式需要，由你自行提供
#   bash run_train_serial.sh
#
#   # 指定子集（空格分隔）
#   bash run_train_serial.sh llama3_baseline armorm_baseline
#
# 说明：
#   - 默认顺序：llama3_baseline → llama3.1_baseline → armorm_baseline
#   - 单个失败不中断，最后汇总；全部成功 exit 0，否则 exit 1
#   - 每个 backbone 调用 run_train.sh，输出与 SwanLab 实验名按 backbone 区分
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TRAIN="${SCRIPT_DIR}/run_train.sh"

VALID_TYPES=(llama3_baseline llama3.1_baseline armorm_baseline)

if [[ $# -gt 0 ]]; then
  MODEL_TYPES=("$@")
else
  MODEL_TYPES=(llama3_baseline llama3.1_baseline armorm_baseline)
fi

for t in "${MODEL_TYPES[@]}"; do
  ok=false
  for v in "${VALID_TYPES[@]}"; do
    [[ "${t}" == "${v}" ]] && ok=true && break
  done
  if [[ "${ok}" != "true" ]]; then
    echo "[错误] 未知 backbone_type: ${t}，可选: ${VALID_TYPES[*]}"
    exit 1
  fi
done

LATENT_DIR="${REWARD_MODEL_ROOT:-/mnt/afs/250010036/reward_model}/latent_reward_model"
TRAIN_JSONL="${LATENT_DIR}/data/ultrafeedback_train.jsonl"
VAL_JSONL="${LATENT_DIR}/data/ultrafeedback_val.jsonl"

echo "========================================================"
echo " Latent Reward Model 串行训练"
echo " 待训练: ${MODEL_TYPES[*]}"
echo " SwanLab project: ${SWANLAB_PROJECT:-latentMRM}"
echo " SwanLab mode:    ${SWANLAB_MODE:-cloud}"
echo "========================================================"
echo ""

for f in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[错误] 数据文件不存在: ${f}"
    echo "请先运行: cd ${LATENT_DIR} && PYTHONPATH=. python scripts/prepare_full_data.py"
    exit 1
  fi
done
echo "[检查] jsonl 数据 OK"
echo ""

TOTAL=${#MODEL_TYPES[@]}
declare -A RESULTS

for (( IDX=0; IDX<TOTAL; IDX++ )); do
  BACKBONE_TYPE="${MODEL_TYPES[$IDX]}"
  EXPERIMENT_NAME="latent_mrm_${BACKBONE_TYPE}"
  SWANLAB_EXPERIMENT_NAME="LatentRewardModel_${BACKBONE_TYPE}"

  echo "========================================================"
  echo " [$(( IDX+1 ))/${TOTAL}] backbone=${BACKBONE_TYPE}"
  echo "  output_dir       : ${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"
  echo "  swanlab_experiment: ${SWANLAB_EXPERIMENT_NAME}"
  echo "========================================================"
  echo ""

  EXIT_CODE=0
  BACKBONE_TYPE="${BACKBONE_TYPE}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME}" \
  bash "${RUN_TRAIN}" || EXIT_CODE=$?

  echo ""
  echo "[结束] $(date '+%Y-%m-%d %H:%M:%S')  exit_code=${EXIT_CODE}"
  if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[完成] ${BACKBONE_TYPE} → ${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"
    RESULTS["${BACKBONE_TYPE}"]="${LATENT_DIR}/experiments/${EXPERIMENT_NAME}"
  else
    echo "[失败] ${BACKBONE_TYPE}（exit_code=${EXIT_CODE}）"
    RESULTS["${BACKBONE_TYPE}"]="FAILED(exit_code=${EXIT_CODE})"
  fi
  echo ""
done

echo "========================================================"
echo " 全部训练结束  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
OVERALL_OK=true
for BACKBONE_TYPE in "${MODEL_TYPES[@]}"; do
  RESULT="${RESULTS[${BACKBONE_TYPE}]:-未运行}"
  if [[ "${RESULT}" == FAILED* ]]; then
    OVERALL_OK=false
    echo "  [FAIL] ${BACKBONE_TYPE}"
    echo "         ${RESULT}"
  else
    echo "  [ OK ] ${BACKBONE_TYPE}"
    echo "         ${RESULT}"
  fi
done
echo "========================================================"

if [[ "${OVERALL_OK}" == "true" ]]; then
  exit 0
else
  echo "[警告] 部分 backbone 训练失败，请检查上方日志。"
  exit 1
fi
