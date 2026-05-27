#!/usr/bin/env bash
# 导出验证集多维度 head 输出，检查高相关 / 坍缩
set -euo pipefail

LATENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

CKPT="${CKPT:-${LATENT_DIR}/experiments/latent_mrm_llama3.1_baseline_gate/eval_export/best}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
TOP_K="${TOP_K:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${LATENT_DIR}"

python scripts/dump_multidim_eval_samples.py \
  --ckpt "${CKPT}" \
  --num-samples "${NUM_SAMPLES}" \
  --top-k "${TOP_K}" \
  --device cuda
