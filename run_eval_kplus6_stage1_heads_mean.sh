#!/usr/bin/env bash
# K+=6 阶段1 checkpoint — 对 K=10 个 head 等权平均（score_mode=heads_mean）后跑三 benchmark
#
# 用法:
#   cd /mnt/afs/250010036/reward_model/latent_reward_model
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_eval_kplus6_stage1_heads_mean.sh
#
# 说明:
#   - 聚合: reward = (1/K) * sum_{i=1..K} head_i，K=10
#   - 与 heads_sum 仅差常数倍，pair 排序一致；显式等权便于与 gate 模型对比
#   - 阶段1 无 gate，use_gate=false

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LATENT_DIR="${SCRIPT_DIR}"

CKPT_TAG="${CKPT_TAG:-best}"
STAGE1_NAME="latent_mrm_llama3.1_baseline_k10_kplus6_2stage"
CKPT_DIR="${CKPT_DIR:-${LATENT_DIR}/experiments/${STAGE1_NAME}/${CKPT_TAG}}"
EXPORT_DIR="${EXPORT_DIR:-${LATENT_DIR}/experiments/${STAGE1_NAME}/eval_export_heads_mean/${CKPT_TAG}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${STAGE1_NAME}_heads_mean}"
GLOBAL_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/evals/${EXPERIMENT_NAME}_${GLOBAL_TS}}"

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"

if [[ ! -f "${CKPT_DIR}/latent_heads.pt" ]]; then
  echo "[错误] 阶段1 checkpoint 不存在: ${CKPT_DIR}" >&2
  exit 1
fi

echo "[1/2] 导出评测目录 score_mode=heads_mean"
export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${LATENT_DIR}"
"${PYTHON}" "${LATENT_DIR}/scripts/export_for_eval.py" "${CKPT_DIR}" \
  --out_dir "${EXPORT_DIR}" \
  --score_mode heads_mean

echo "[2/2] 三 benchmark 评测 → ${OUTPUT_DIR}"
cd "${ROOT_DIR}"
export EXPERIMENT_NAME EXPORT_DIR CKPT_DIR CKPT_TAG
bash latent_reward_model/run_eval_benchmarks.sh

echo "完成: ${OUTPUT_DIR}/summary_all.json"
