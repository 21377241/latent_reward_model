#!/usr/bin/env bash
# 检查核心源码是否齐全，防止误删后仍继续训练/同步。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REQUIRED=(
  modeling_latent_rm.py
  run_train.sh
  run_train_two_stage.sh
  run_train_gate.sh
  run_kplus_sweep_train_eval.sh
  run_eval_benchmarks.sh
  scripts/train_rm.py
  scripts/export_for_eval.py
  scripts/load_latent_rm.py
  scripts/prepare_full_data.py
  scripts/sync_git_to_github_api.py
  models/latent_reward_model.py
  models/backbone.py
  models/pooling.py
  utils/loss_functions.py
  utils/metrics.py
  utils/dataloader.py
  utils/checkpoint.py
  accel_ds2.yaml
  accel_1gpu.yaml
  ds_zero2.json
)

OPTIONAL=(
  scripts/dump_multidim_eval_samples.py
  run_dump_multidim.sh
)

missing=()
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "${f}" ]]; then
    missing+=("${f}")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[verify_repo_integrity] 缺少必需文件 (${#missing[@]}):" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "  请从 GitHub 恢复: https://github.com/21377241/latent_reward_model" >&2
  echo "  或: git checkout HEAD -- <path>" >&2
  exit 1
fi

for f in "${OPTIONAL[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[verify_repo_integrity] 可选文件缺失: ${f}" >&2
  fi
done

echo "[verify_repo_integrity] OK (${#REQUIRED[@]} required files present)"
