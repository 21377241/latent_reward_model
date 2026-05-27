#!/usr/bin/env bash
# Latent MRM — num_pos_heads ∈ {5,6,8,9} 串行「训练 + 三 benchmark 评测」一站式全流程
# 无需再单独执行 run_eval_benchmarks.sh。
#
# 默认：K=10，backbone=llama3.1_baseline；SKIP_IF_CKPT_EXISTS=1（断点续跑友好）
#   - 阶段2 gate 已有 → 跳过训练
#   - 仅阶段1 已有     → 自动 SKIP_STAGE1，只训 gate
#   - evals/<exp>_*/summary_all.json 已有 → 跳过评测（如 K+=5 已评完）
#
# 用法:
#   cd /mnt/afs/250010036/reward_model/latent_reward_model
#   CUDA_VISIBLE_DEVICES=0,1,2,3 SWANLAB_MODE=disabled bash run_kplus_sweep_train_eval.sh
#
#   # 只续跑尚未完成的 K+（例：K+=5 训练+评测均已完成则自动跳过）
#   KPLUS_VALUES="6 8 9" bash run_kplus_sweep_train_eval.sh
#
#   SKIP_IF_CKPT_EXISTS=0   # 强制重训/重评
#   SKIP_EVAL=1 / SKIP_TRAIN=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${SCRIPT_DIR}/scripts/verify_repo_integrity.sh" ]]; then
  bash "${SCRIPT_DIR}/scripts/verify_repo_integrity.sh"
fi
LATENT_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${LATENT_DIR}/.." && pwd)"

# ─── sweep 配置 ───────────────────────────────────────────────────────────
KPLUS_VALUES="${KPLUS_VALUES:-5 6 8 9}"
K_DIMENSIONS="${K_DIMENSIONS:-10}"
BACKBONE_TYPE="${BACKBONE_TYPE:-llama3.1_baseline}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_IF_CKPT_EXISTS="${SKIP_IF_CKPT_EXISTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

# 训练：不默认绑死单卡；由调用方设置 CUDA_VISIBLE_DEVICES（4 卡时用 0,1,2,3）
# 评测：默认单卡，避免与训练占用的多卡冲突
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

SWEEP_LOG_DIR="${SWEEP_LOG_DIR:-${ROOT_DIR}/evals/kplus_sweep_logs}"
mkdir -p "${SWEEP_LOG_DIR}"
SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${SWEEP_LOG_DIR}/sweep_${SWEEP_TS}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"; }

gate_experiment_name() {
  local kplus="$1"
  echo "latent_mrm_${BACKBONE_TYPE}_k${K_DIMENSIONS}_kplus${kplus}_2stage_gate"
}

stage1_experiment_name() {
  local kplus="$1"
  echo "latent_mrm_${BACKBONE_TYPE}_k${K_DIMENSIONS}_kplus${kplus}_2stage"
}

stage1_ckpt_ok() {
  local kplus="$1"
  local ckpt="${LATENT_DIR}/experiments/$(stage1_experiment_name "${kplus}")/best/latent_heads.pt"
  [[ -f "${ckpt}" ]]
}

stage2_ckpt_ok() {
  local kplus="$1"
  local ckpt="${LATENT_DIR}/experiments/$(gate_experiment_name "${kplus}")/best/latent_heads.pt"
  [[ -f "${ckpt}" ]]
}

# 返回 evals/<exp>_*/ 下已有 summary_all.json 的最新目录（无则空）
latest_eval_dir() {
  local exp="$1"
  local d best="" best_t=0 t
  shopt -s nullglob
  for d in "${ROOT_DIR}"/evals/${exp}_*/; do
    [[ -f "${d}summary_all.json" ]] || continue
    t="$(stat -c %Y "${d}summary_all.json" 2>/dev/null || echo 0)"
    if [[ "${t}" -gt "${best_t}" ]]; then
      best_t="${t}"
      best="${d%/}"
    fi
  done
  shopt -u nullglob
  [[ -n "${best}" ]] && echo "${best}"
}

eval_done_ok() {
  local kplus="$1"
  local exp
  exp="$(gate_experiment_name "${kplus}")"
  [[ -n "$(latest_eval_dir "${exp}")" ]]
}

run_one() {
  local kplus="$1"
  local exp train_log eval_log eval_out existing_eval

  exp="$(gate_experiment_name "${kplus}")"
  train_log="${SWEEP_LOG_DIR}/train_kplus${kplus}_${SWEEP_TS}.log"
  eval_log="${SWEEP_LOG_DIR}/eval_kplus${kplus}_${SWEEP_TS}.log"
  eval_out="${ROOT_DIR}/evals/${exp}_${SWEEP_TS}"
  existing_eval=""

  log "════════════════════════════════════════════════════════════"
  log " K+=${kplus}  experiment=${exp}"
  log "════════════════════════════════════════════════════════════"

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    local _skip_stage1=0
    if [[ "${SKIP_IF_CKPT_EXISTS}" == "1" ]] && stage2_ckpt_ok "${kplus}"; then
      log "[训练] 跳过：已存在 ${exp}/best/latent_heads.pt"
    else
      if [[ "${SKIP_IF_CKPT_EXISTS}" == "1" ]] && stage1_ckpt_ok "${kplus}"; then
        log "[训练] 阶段1 已有 → 仅阶段2 gate（SKIP_STAGE1=1）"
        _skip_stage1=1
      else
        log "[训练] 开始两阶段 NUM_POS_HEADS=${kplus}"
      fi
      if [[ "${DRY_RUN}" == "1" ]]; then
        if [[ "${_skip_stage1}" == "1" ]]; then
          log "  DRY_RUN: SKIP_STAGE1=1 NUM_POS_HEADS=${kplus} bash run_train_two_stage.sh"
        else
          log "  DRY_RUN: NUM_POS_HEADS=${kplus} K_DIMENSIONS=${K_DIMENSIONS} bash run_train_two_stage.sh"
        fi
      else
        (
          cd "${LATENT_DIR}"
          export NUM_POS_HEADS="${kplus}"
          export K_DIMENSIONS="${K_DIMENSIONS}"
          export BACKBONE_TYPE="${BACKBONE_TYPE}"
          if [[ "${_skip_stage1}" == "1" ]]; then
            export SKIP_STAGE1=1
          fi
          # 继承调用方的 CUDA_VISIBLE_DEVICES / SWANLAB_* 等环境变量
          bash run_train_two_stage.sh
        ) 2>&1 | tee -a "${train_log}" "${MASTER_LOG}"
        if ! stage2_ckpt_ok "${kplus}"; then
          log "[错误] K+=${kplus} 训练未产生有效 checkpoint，跳过后续评测"
          return 1
        fi
        log "[训练] 完成 → experiments/${exp}/best"
      fi
    fi
  else
    log "[训练] SKIP_TRAIN=1，跳过"
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    if [[ "${SKIP_IF_CKPT_EXISTS}" == "1" ]]; then
      existing_eval="$(latest_eval_dir "${exp}" || true)"
      if [[ -n "${existing_eval}" ]]; then
        log "[评测] 跳过：已存在 ${existing_eval}/summary_all.json"
        return 0
      fi
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      log "  DRY_RUN: EXPERIMENT_NAME=${exp} bash run_eval_benchmarks.sh → ${eval_out}"
      return 0
    fi
    if ! stage2_ckpt_ok "${kplus}"; then
      log "[错误] 无 gate checkpoint，无法评测: experiments/${exp}/best"
      return 1
    fi
    log "[评测] 开始三 benchmark → ${eval_out}"
    (
      cd "${ROOT_DIR}"
      export EXPERIMENT_NAME="${exp}"
      export OUTPUT_DIR="${eval_out}"
      export CKPT_TAG="${CKPT_TAG:-best}"
      CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
        bash latent_reward_model/run_eval_benchmarks.sh
    ) 2>&1 | tee -a "${eval_log}" "${MASTER_LOG}"
    if [[ -f "${eval_out}/summary_all.json" ]]; then
      log "[评测] 完成 → ${eval_out}/summary_all.json"
    else
      log "[警告] 未找到 summary_all.json: ${eval_out}"
      return 1
    fi
  else
    log "[评测] SKIP_EVAL=1，跳过"
  fi
  return 0
}

# ─── 入口 ─────────────────────────────────────────────────────────────────
log "K+ sweep 开始"
log "  KPLUS_VALUES=${KPLUS_VALUES}"
log "  K_DIMENSIONS=${K_DIMENSIONS}  BACKBONE_TYPE=${BACKBONE_TYPE}"
log "  SKIP_TRAIN=${SKIP_TRAIN}  SKIP_EVAL=${SKIP_EVAL}  SKIP_IF_CKPT_EXISTS=${SKIP_IF_CKPT_EXISTS}"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-（未设置，使用全部可见 GPU）}"
log "  EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES}"
log "  MASTER_LOG=${MASTER_LOG}"

FAILED=()
PASSED=()

for kplus in ${KPLUS_VALUES}; do
  if ! [[ "${kplus}" =~ ^[0-9]+$ ]] || [[ "${kplus}" -gt "${K_DIMENSIONS}" ]]; then
    log "[错误] 无效 NUM_POS_HEADS=${kplus}（K=${K_DIMENSIONS}）"
    FAILED+=("${kplus}")
    continue
  fi
  if run_one "${kplus}"; then
    PASSED+=("${kplus}")
  else
    FAILED+=("${kplus}")
  fi
done

# 汇总各实验 summary_all.json
SUMMARY_AGG="${ROOT_DIR}/evals/kplus_sweep_summary_${SWEEP_TS}.json"
if [[ "${DRY_RUN}" != "1" ]] && [[ "${SKIP_EVAL}" != "1" ]]; then
  PYTHON="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}/envs/${CONDA_ENV:-swift_env}/bin/python"
  if [[ -x "${PYTHON}" ]]; then
    KPLUS_VALUES="${KPLUS_VALUES}" SWEEP_TS="${SWEEP_TS}" ROOT_DIR="${ROOT_DIR}" LATENT_DIR="${LATENT_DIR}" \
      BACKBONE_TYPE="${BACKBONE_TYPE}" K_DIMENSIONS="${K_DIMENSIONS}" \
      "${PYTHON}" - <<'PY' > "${SUMMARY_AGG}" || true
import json
import os
from pathlib import Path

kplus_list = os.environ["KPLUS_VALUES"].split()
root = Path(os.environ["ROOT_DIR"])
latent = Path(os.environ["LATENT_DIR"])
backbone = os.environ["BACKBONE_TYPE"]
k_dim = os.environ["K_DIMENSIONS"]
ts = os.environ["SWEEP_TS"]

out = {"sweep_ts": ts, "k_dimensions": int(k_dim), "experiments": {}}
for kp in kplus_list:
    exp = f"latent_mrm_{backbone}_k{k_dim}_kplus{kp}_2stage_gate"
    candidates = sorted(
        root.glob(f"evals/{exp}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    entry = {"experiment": exp, "eval_dir": None, "metrics": None}
    for d in candidates:
        sf = d / "summary_all.json"
        if sf.is_file():
            entry["eval_dir"] = str(d)
            entry["metrics"] = json.loads(sf.read_text(encoding="utf-8"))
            break
    ckpt = latent / "experiments" / exp / "best" / "meta.json"
    if ckpt.is_file():
        entry["train_meta"] = json.loads(ckpt.read_text(encoding="utf-8"))
    out["experiments"][f"kplus_{kp}"] = entry

print(json.dumps(out, indent=2, ensure_ascii=False))
PY
    log "汇总 JSON → ${SUMMARY_AGG}"
  fi
fi

log "════════════════════════════════════════════════════════════"
log " Sweep 结束"
log "  成功 K+: ${PASSED[*]:-（无）}"
log "  失败 K+: ${FAILED[*]:-（无）}"
log "  日志目录: ${SWEEP_LOG_DIR}"
log "════════════════════════════════════════════════════════════"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
