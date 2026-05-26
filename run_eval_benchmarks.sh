#!/usr/bin/env bash
# Latent MRM — Reward-bench-2 / RM-bench / Judge-bench 评测（对齐 scripts/train_and_eval_all.sh）
#
# 用法（在 latent_reward_model/ 或 reward_model/ 下均可）：
#   bash latent_reward_model/run_eval_benchmarks.sh
#
#   EXPORT_DIR=/path/to/eval_export/best \
#   OUTPUT_DIR=/path/to/evals/latent_gate_xxx \
#   bash latent_reward_model/run_eval_benchmarks.sh
#
# 默认评测：experiments/latent_mrm_llama3.1_baseline_gate/eval_export/best

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/mnt/afs/250010036/miniconda3}"
CONDA_ENV="${CONDA_ENV:-swift_env}"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

PYTHON="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LATENT_DIR="${SCRIPT_DIR}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-latent_mrm_llama3.1_baseline_gate}"
CKPT_TAG="${CKPT_TAG:-best}"
EXPORT_DIR="${EXPORT_DIR:-${LATENT_DIR}/experiments/${EXPERIMENT_NAME}/eval_export/${CKPT_TAG}}"
CKPT_DIR="${CKPT_DIR:-${LATENT_DIR}/experiments/${EXPERIMENT_NAME}/${CKPT_TAG}}"

GLOBAL_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/evals/latent_${EXPERIMENT_NAME}_${GLOBAL_TS}}"

PARQUET="${ROOT_DIR}/data/reward-bench-2/test.parquet"
DISK_DATASET="${ROOT_DIR}/data/reward-bench-2/hf_disk"
REWARD_RUN="${ROOT_DIR}/reward-bench/scripts/run_v2.py"
RM_RUN="${ROOT_DIR}/RM-bench/scripts/run_rm.py"
RM_DATA="${ROOT_DIR}/RM-bench/data/total_dataset.json"
JUDGE_DIR="${ROOT_DIR}/Judge-bench"
GPT_PAIRS="${JUDGE_DIR}/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
CLAUDE_PAIRS="${JUDGE_DIR}/data/dataset=judgebench,response_model=claude-3-5-sonnet-20240620.jsonl"

CUDA_MULTI="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CUDA_SINGLE="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_SINGLE="${CUDA_SINGLE%%,*}"
BATCH_SIZE_EVAL=8
MAX_LENGTH=4096

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() {
  echo ""
  echo "================================================================"
  echo " $*"
  echo "================================================================"
}

# ── 确保 HF 导出目录完整 ─────────────────────────────────────────────────────
if [[ ! -f "${EXPORT_DIR}/model.safetensors" || ! -f "${EXPORT_DIR}/latent_heads.pt" ]]; then
  log "导出目录不完整，从 checkpoint 重新导出: ${CKPT_DIR} → ${EXPORT_DIR}"
  export PYTHONPATH="${LATENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  "${PYTHON}" -c "
from utils.export_for_eval import export_latent_ckpt
export_latent_ckpt('${CKPT_DIR}', '${EXPORT_DIR}')
"
fi

if [[ ! -f "${EXPORT_DIR}/model.safetensors" ]]; then
  echo "[错误] 缺少 model.safetensors: ${EXPORT_DIR}"
  exit 1
fi

EXPORT_DIR="$(cd "${EXPORT_DIR}" && pwd)"
mkdir -p "${OUTPUT_DIR}"
log "模型目录: ${EXPORT_DIR}"
log "结果目录: ${OUTPUT_DIR}"

# reward-bench parquet → disk
if [[ ! -d "${DISK_DATASET}" ]]; then
  log "转换 reward-bench-2 parquet → disk ..."
  "${PYTHON}" - <<EOF
import pandas as pd
from datasets import Dataset
df = pd.read_parquet("${PARQUET}")
ds = Dataset.from_pandas(df, preserve_index=False)
ds.save_to_disk("${DISK_DATASET}")
print(f"saved {len(ds)} rows")
EOF
fi

# ── 评测（与 train_and_eval_all.sh::run_eval 一致）────────────────────────────
eval_out_dir="${OUTPUT_DIR}"
model_dir="${EXPORT_DIR}"

section "[Reward-bench-2] ${EXPERIMENT_NAME}"
rb_out="${eval_out_dir}/reward_bench"
mkdir -p "${rb_out}"
cd "${ROOT_DIR}"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES="${CUDA_MULTI}" HF_DATASETS_OFFLINE=1 \
  "${PYTHON}" "${REWARD_RUN}" \
  --model "${model_dir}" \
  --dataset "${DISK_DATASET}" \
  --output_dir "${rb_out}" \
  --batch_size "${BATCH_SIZE_EVAL}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype bfloat16 \
  --trust_remote_code \
  --do_not_save \
  --local

export _RB_OUT="${rb_out}"
"${PYTHON}" - <<'PY'
import json, os
path = os.path.join(os.environ["_RB_OUT"], "results.json")
with open(path, encoding="utf-8") as f:
    data = json.load(f)
skip = {"model", "model_type", "chat_template", "score"}
vals = [v for k, v in data.items() if k not in skip and isinstance(v, (int, float))]
if vals:
    data["score"] = round(sum(vals) / len(vals), 6)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("[Reward-bench-2] 各维度:")
    for k, v in data.items():
        if k not in skip and isinstance(v, (int, float)):
            print(f"  {k}: {round(v, 6)}")
    print(f"  score (均值): {data['score']}")
PY

section "[RM-bench] ${EXPERIMENT_NAME}"
rm_out="${eval_out_dir}/rm_bench"
mkdir -p "${rm_out}"
rm_log="${eval_out_dir}/rm_bench.log"
cd "${ROOT_DIR}/RM-bench"
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}" \
  PYTHONPATH="${ROOT_DIR}/RM-bench:${PYTHONPATH:-}" \
  "${PYTHON}" "${RM_RUN}" \
  --model "${model_dir}" \
  --datapath "${RM_DATA}" \
  --batch_size "${BATCH_SIZE_EVAL}" \
  --max_length "${MAX_LENGTH}" \
  --torch_dtype bfloat16 \
  --trust_remote_code \
  --not_quantized \
  --chat_template tulu \
  2>&1 | tee "${rm_log}"

rm_model_sub="results/Seq_Classifier${model_dir}"
rm_src=$(find "${rm_model_sub}" -name "total_dataset_*.json" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
if [[ -z "${rm_src}" ]]; then
  rm_src=$(find results/Seq_Classifier -name "total_dataset_*.json" 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)
fi
if [[ -n "${rm_src}" && -f "${rm_src}" ]]; then
  cp "${rm_src}" "${rm_out}/total_dataset.json"
  export _RM_OUT="${rm_out}" _ROOT_DIR="${ROOT_DIR}"
  "${PYTHON}" - <<'PY'
import json, os, importlib.util
from pathlib import Path
rm_out = Path(os.environ["_RM_OUT"])
root = Path(os.environ["_ROOT_DIR"])
spec = importlib.util.spec_from_file_location("rm_utils", root / "RM-bench/scripts/utils.py")
rm_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm_utils)
data = json.loads((rm_out / "total_dataset.json").read_text())
acc = rm_utils.compute_accuracy(data)
acc_clean = {k: float(v) for k, v in acc.items()}
(rm_out / "summary.json").write_text(json.dumps(acc_clean, indent=2, ensure_ascii=False) + "\n")
print("[RM-bench] 指标:")
for k, v in acc_clean.items():
    print(f"  {k}: {v:.6f}")
PY
else
  log "[警告] 未找到 RM-bench 输出，见 ${rm_log}"
fi

section "[Judge-bench] ${EXPERIMENT_NAME}"
jb_out="${eval_out_dir}/judge_bench"
mkdir -p "${jb_out}"
cd "${JUDGE_DIR}"

run_judge_pairs() {
  local pairs_file="$1"
  local tag="$2"
  log "Judge-bench ${tag} ..."
  env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u MASTER_ADDR -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="${CUDA_SINGLE}" \
    "${PYTHON}" run_judge.py \
    --judge_name reward_model \
    --judge_model "${model_dir}" \
    --single_game \
    --pairs "${pairs_file}"

  local model_slug response_model out_file
  model_slug="${model_dir//\//_}"
  response_model="$(basename "${pairs_file}" .jsonl | sed 's/.*response_model=//')"
  out_file="outputs/dataset=judgebench,response_model=${response_model},judge_name=reward_model,judge_model=${model_slug}.jsonl"
  if [[ ! -f "${out_file}" ]]; then
    log "[警告] 未找到 Judge-bench 输出 (${tag})"
    return
  fi
  cp "${out_file}" "${jb_out}/${tag}.jsonl"
  export _JB_FILE="${jb_out}/${tag}.jsonl" _JB_TAG="${tag}" _JB_OUT="${jb_out}"
  "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
from collections import defaultdict
jb_file = Path(os.environ["_JB_FILE"])
tag = os.environ["_JB_TAG"]
jb_out = Path(os.environ["_JB_OUT"])
pairs = [json.loads(l) for l in jb_file.read_text().splitlines() if l.strip()]

def cat(src):
    if src.startswith("livebench-reasoning"): return "reasoning"
    if src.startswith("livebench-math"): return "math"
    if src.startswith("livecodebench"): return "coding"
    if src.startswith("mmlu-pro"): return "knowledge"
    return "other"

totals, corrects = defaultdict(int), defaultdict(int)
for p in pairs:
    label = p.get("label")
    judgments = p.get("judgments") or []
    if not judgments:
        continue
    decision = judgments[0].get("decision") if isinstance(judgments[0], dict) else None
    if not (label and decision):
        continue
    c = cat(p.get("source", ""))
    totals[c] += 1
    totals["overall"] += 1
    if decision == label:
        corrects[c] += 1
        corrects["overall"] += 1

result = {}
for c in ("knowledge", "reasoning", "math", "coding", "overall"):
    t, cr = totals.get(c, 0), corrects.get(c, 0)
    result[c] = round(cr / t * 100, 4) if t else None
result["detail"] = {c: {"correct": corrects.get(c, 0), "total": totals.get(c, 0)} for c in ("knowledge", "reasoning", "math", "coding", "overall")}
(jb_out / f"{tag}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(f"[Judge-bench] {tag} 指标:")
for c in ("knowledge", "reasoning", "math", "coding", "overall"):
    d = result["detail"][c]
    print(f"  {c}: {d['correct']}/{d['total']} = {result[c]}%")
PY
}

run_judge_pairs "${GPT_PAIRS}" gpt4o
run_judge_pairs "${CLAUDE_PAIRS}" claude35

# ── 汇总 summary_all.json ────────────────────────────────────────────────────
export _OUT_DIR="${OUTPUT_DIR}" _EXP="${EXPERIMENT_NAME}" _EXPORT_DIR="${EXPORT_DIR}"
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["_OUT_DIR"])
entry = {"experiment": os.environ["_EXP"], "export_dir": os.environ.get("EXPORT_DIR", "")}
rb = out / "reward_bench" / "results.json"
if rb.exists():
    d = json.loads(rb.read_text())
    skip = {"model", "model_type", "chat_template"}
    entry["reward_bench"] = {k: v for k, v in d.items() if k not in skip}
rm = out / "rm_bench" / "summary.json"
if rm.exists():
    entry["rm_bench"] = json.loads(rm.read_text())
jb = {}
for tag in ("gpt4o", "claude35"):
    p = out / "judge_bench" / f"{tag}_summary.json"
    if p.exists():
        jb[tag] = json.loads(p.read_text())
if jb:
    entry["judge_bench"] = jb
summary_path = out / "summary_all.json"
summary_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")
print("=" * 64)
print("汇总:", summary_path)
print("=" * 64)
if "reward_bench" in entry:
    rb = entry["reward_bench"]
    print("  Reward-bench-2 score:", rb.get("score"))
if "rm_bench" in entry:
    print("  RM-bench total_avg_acc:", entry["rm_bench"].get("total_avg_acc"))
for tag, m in entry.get("judge_bench", {}).items():
    print(f"  Judge-bench ({tag}) overall:", m.get("overall"), "%")
PY

log "全部完成 → ${OUTPUT_DIR}"
