#!/usr/bin/env python3
"""
在验证集上导出 Latent MRM 多维度输出（K 个 head 分数、selector、gate），
用于检查 head 高相关是否由实现错误或表示坍缩导致。

用法:
  cd /mnt/afs/250010036/reward_model/latent_reward_model
  python scripts/dump_multidim_eval_samples.py \\
    --ckpt experiments/latent_mrm_llama3.1_baseline_gate/eval_export/best \\
    --num-samples 128 --top-k 8

  # 或阶段1 checkpoint（无 gate 标量，仍有 K 维 + selector）:
  python scripts/dump_multidim_eval_samples.py \\
    --ckpt experiments/latent_mrm_llama3.1_baseline_k10_kplus6_2stage/best
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _pool_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.size(1)
    flipped = attention_mask.flip(1).long()
    last_idx = seq_len - 1 - flipped.argmax(1)
    valid = flipped.max(1).values.bool()
    last_idx = torch.where(valid, last_idx, torch.full_like(last_idx, seq_len - 1))
    b = last_hidden_state.size(0)
    device = last_hidden_state.device
    return last_hidden_state[torch.arange(b, device=device), last_idx]


def encode_messages(tokenizer, messages: List[dict], max_length: int) -> dict:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tok = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    return {k: v for k, v in tok.items()}


def pad_batch(items: List[dict], pad_id: int) -> dict:
    def pad_left(seqs):
        ml = max(s.size(1) for s in seqs)
        ids = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), ml, dtype=torch.long)
        for i, s in enumerate(seqs):
            L = s.size(1)
            ids[i, ml - L :] = s[0]
            mask[i, ml - L :] = 1
        return ids, mask

    c_ids, c_mask = pad_left([x["input_ids"] for x in items])
    r_ids, r_mask = pad_left([x["input_ids"] for x in items])
    return {
        "input_ids_c": c_ids,
        "attention_mask_c": c_mask,
        "input_ids_r": r_ids,
        "attention_mask_r": r_mask,
    }


def pearson_corr_matrix(x: torch.Tensor) -> torch.Tensor:
    """x: [N, K] -> [K, K]"""
    if x.size(0) < 2:
        return torch.eye(x.size(1))
    z = x.float() - x.float().mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True).clamp(min=1e-8)
    z = z / std
    n = z.size(0)
    return (z.t() @ z) / max(n - 1, 1)


def offdiag_mean_max(corr: torch.Tensor) -> Tuple[float, float]:
    k = corr.size(0)
    mask = ~torch.eye(k, dtype=torch.bool, device=corr.device)
    off = corr[mask].abs()
    if off.numel() == 0:
        return 0.0, 0.0
    return off.mean().item(), off.max().item()


@torch.no_grad()
def forward_pair(model, batch: dict) -> dict:
    """与训练一致：chosen/rejected 各过 backbone，selector 用拼接 hidden。"""
    out_c = model.model(
        input_ids=batch["input_ids_c"],
        attention_mask=batch["attention_mask_c"],
    )
    out_r = model.model(
        input_ids=batch["input_ids_r"],
        attention_mask=batch["attention_mask_r"],
    )
    h_c = _pool_last_hidden(out_c.last_hidden_state, batch["attention_mask_c"])
    h_r = _pool_last_hidden(out_r.last_hidden_state, batch["attention_mask_r"])

    dtype = model.reward_heads[0][0].weight.dtype
    h_c = h_c.to(dtype)
    h_r = h_r.to(dtype)

    z_c = torch.cat([head(h_c) for head in model.reward_heads], dim=-1).float()
    z_r = torch.cat([head(h_r) for head in model.reward_heads], dim=-1).float()

    combined = torch.cat([h_c, h_r], dim=-1)
    p_plus = torch.sigmoid(model.selector(combined)).float()
    p_minus = 1.0 - p_plus

    k = z_c.size(-1)
    num_pos = int(getattr(model.config, "num_pos_heads", k))
    _, topk = torch.topk(p_plus, k=min(num_pos, k), dim=-1)
    m_plus = torch.zeros_like(p_plus, dtype=torch.bool).scatter_(-1, topk, True).float()

    gate_w = None
    r_gate_c = r_gate_r = None
    if model.use_gate and model.gating_network is not None:
        gate_w = model.gating_network(h_c).float()
        r_gate_c = (gate_w * z_c).sum(-1)
        r_gate_r = model.gating_network(h_r).float()
        r_gate_r = (r_gate_r * z_r).sum(-1)

    r_pseudo_c = (z_c * m_plus).sum(-1)
    r_pseudo_r = (z_r * m_plus).sum(-1)

    return {
        "z_c": z_c,
        "z_r": z_r,
        "diff": z_c - z_r,
        "p_plus": p_plus,
        "p_minus": p_minus,
        "m_plus": m_plus,
        "r_pseudo_c": r_pseudo_c,
        "r_pseudo_r": r_pseudo_r,
        "r_gate_c": r_gate_c,
        "r_gate_r": r_gate_r,
        "gate_w": gate_w,
    }


def sample_record(
    idx: int,
    row: dict,
    out: dict,
    i: int,
    k: int,
    num_pos: int,
) -> dict:
    zc = out["z_c"][i].cpu().tolist()
    zr = out["z_r"][i].cpu().tolist()
    diff = out["diff"][i].cpu().tolist()
    pp = out["p_plus"][i].cpu().tolist()
    mp = out["m_plus"][i].cpu().tolist()
    gw = out["gate_w"][i].cpu().tolist() if out["gate_w"] is not None else None

    std_heads = float(torch.std(out["z_c"][i]).item())
    std_diff = float(torch.std(out["diff"][i]).item())
    max_spread = float((out["z_c"][i].max() - out["z_c"][i].min()).item())

    correct_dims = [j for j in range(k) if diff[j] > 0]
    wrong_dims = [j for j in range(k) if diff[j] <= 0]
    kplus_dims = [j for j in range(k) if mp[j] > 0.5]

    rec = {
        "val_idx": idx,
        "pref_correct_pseudo": bool(out["r_pseudo_c"][i] > out["r_pseudo_r"][i]),
        "pref_correct_gate": (
            bool(out["r_gate_c"][i] > out["r_gate_r"][i])
            if out["r_gate_c"] is not None
            else None
        ),
        "r_pseudo_c": float(out["r_pseudo_c"][i]),
        "r_pseudo_r": float(out["r_pseudo_r"][i]),
        "r_gate_c": float(out["r_gate_c"][i]) if out["r_gate_c"] is not None else None,
        "r_gate_r": float(out["r_gate_r"][i]) if out["r_gate_r"] is not None else None,
        "scores_chosen": [round(x, 4) for x in zc],
        "scores_rejected": [round(x, 4) for x in zr],
        "diff_c_minus_r": [round(x, 4) for x in diff],
        "p_plus": [round(x, 4) for x in pp],
        "in_k_plus_mask": [bool(x) for x in mp],
        "gate_weights_on_chosen": [round(x, 4) for x in gw] if gw else None,
        "correct_head_dims": correct_dims,
        "wrong_head_dims": wrong_dims,
        "k_plus_selector_dims": kplus_dims,
        "std_across_heads_chosen": round(std_heads, 6),
        "std_across_heads_diff": round(std_diff, 6),
        "max_minus_min_chosen": round(max_spread, 6),
        "prompt_preview": (row.get("prompt") or "")[:200],
        "chosen_preview": row["chosen"][-1]["content"][:200] if row.get("chosen") else "",
        "rejected_preview": row["rejected"][-1]["content"][:200] if row.get("rejected") else "",
    }
    return rec


def main():
    p = argparse.ArgumentParser(description="Dump multi-dim latent RM outputs on val set")
    p.add_argument(
        "--ckpt",
        default=str(_ROOT / "experiments/latent_mrm_llama3.1_baseline_gate/eval_export/best"),
    )
    p.add_argument(
        "--val-jsonl",
        default=str(_ROOT / "data/ultrafeedback_val.jsonl"),
    )
    p.add_argument("--num-samples", type=int, default=128, help="扫描验证集前 N 条")
    p.add_argument("--top-k", type=int, default=8, help="输出相关最高/最低的样例数")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "diagnostics"),
    )
    args = p.parse_args()

    ckpt = Path(args.ckpt).resolve()
    if not (ckpt / "modeling_latent_rm.py").is_file():
        alt = ckpt.parent.parent / "eval_export" / "best"
        if (alt / "modeling_latent_rm.py").is_file():
            print(f"[Resolve] using eval_export for modeling: {alt}")
            ckpt = alt
        else:
            raise FileNotFoundError(
                f"No modeling_latent_rm.py under {ckpt}; pass eval_export/best or export first."
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"multidim_samples_{ts}.json"
    out_md = out_dir / f"multidim_samples_{ts}.md"

    from transformers import AutoTokenizer

    sys.path.insert(0, str(ckpt))
    from modeling_latent_rm import LlamaForLatentRewardModel  # noqa: E402

    with open(ckpt / "latent_config.json", encoding="utf-8") as f:
        lcfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    print(f"[Load] {ckpt}")
    model = LlamaForLatentRewardModel.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
    )
    ref_w = model.reward_heads[0][0].weight.detach().float().cpu()
    head_sd = torch.load(ckpt / "latent_heads.pt", map_location="cpu", weights_only=True)
    ref_ckpt = head_sd["reward_heads.0.0.weight"].float()
    head_ok = (ref_w - ref_ckpt).abs().max().item() < 1e-5
    print(f"[Load] reward_heads match latent_heads.pt: {head_ok}")
    if not head_ok:
        raise RuntimeError("Head weights did not load; check checkpoint path and k_dimensions.")
    model.to(args.device)
    model.eval()

    k = int(lcfg.get("k_dimensions", model.k_dimensions))
    num_pos = int(lcfg.get("num_pos_heads", k))
    use_gate = bool(lcfg.get("use_gate", model.use_gate))
    model.config.num_pos_heads = num_pos

    rows: List[dict] = []
    with open(args.val_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            rows.append(json.loads(line))

    all_records: List[dict] = []
    all_diff: List[torch.Tensor] = []

    pad_id = tokenizer.pad_token_id
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        items = []
        valid_idx = []
        for j, row in enumerate(chunk):
            try:
                tc = encode_messages(tokenizer, row["chosen"], args.max_length)
                tr = encode_messages(tokenizer, row["rejected"], args.max_length)
            except Exception as e:
                print(f"[Skip] val_idx={start+j}: {e}")
                continue
            if tc["input_ids"].size(1) < 16:
                continue
            items.append(
                {
                    "input_ids": tc["input_ids"],
                    "attention_mask": tc["attention_mask"],
                    "_tr_ids": tr["input_ids"],
                    "_tr_mask": tr["attention_mask"],
                }
            )
            valid_idx.append(start + j)

        if not items:
            continue

        # repack rejected into batch format
        batch_items = []
        for it in items:
            batch_items.append(
                {"input_ids": it["input_ids"], "attention_mask": it["attention_mask"]}
            )
        batch_c = pad_batch(batch_items, pad_id)
        batch_r_items = [
            {"input_ids": it["_tr_ids"], "attention_mask": it["_tr_mask"]} for it in items
        ]
        batch_r = pad_batch(batch_r_items, pad_id)
        batch = {
            "input_ids_c": batch_c["input_ids_c"].to(args.device),
            "attention_mask_c": batch_c["attention_mask_c"].to(args.device),
            "input_ids_r": batch_r["input_ids_c"].to(args.device),
            "attention_mask_r": batch_r["attention_mask_c"].to(args.device),
        }

        out = forward_pair(model, batch)
        for bi, vi in enumerate(valid_idx):
            rec = sample_record(vi, rows[vi], out, bi, k, num_pos)
            all_records.append(rec)
            all_diff.append(out["diff"][bi].cpu())

    # 全批相关（与训练 eval diag/head_corr_* 一致：在 N 个样本上算每个 head 维的 Pearson）
    diff_mat = torch.stack(all_diff, dim=0)
    zc_mat = torch.tensor([r["scores_chosen"] for r in all_records], dtype=torch.float32)
    corr_diff = pearson_corr_matrix(diff_mat)
    corr_zc = pearson_corr_matrix(zc_mat)

    cm_diff_mean, cm_diff_max = offdiag_mean_max(corr_diff)
    cm_zc_mean, cm_zc_max = offdiag_mean_max(corr_zc)

    # 按「head 间几乎无差异」排序 — 疑似坍缩
    for rec in all_records:
        rec["collapse_score"] = 1.0 / (rec["std_across_heads_chosen"] + 1e-6)

    by_collapse = sorted(all_records, key=lambda r: r["collapse_score"], reverse=True)[: args.top_k]
    by_spread = sorted(all_records, key=lambda r: r["std_across_heads_chosen"])[: args.top_k]

    report = {
        "ckpt": str(ckpt),
        "k_dimensions": k,
        "num_pos_heads": num_pos,
        "use_gate": use_gate,
        "num_evaluated": len(all_records),
        "batch_correlation": {
            "on_diff": {
                "offdiag_abs_mean": round(cm_diff_mean, 4),
                "offdiag_abs_max": round(cm_diff_max, 4),
                "matrix": [[round(float(corr_diff[i, j]), 4) for j in range(k)] for i in range(k)],
            },
            "on_scores_chosen": {
                "offdiag_abs_mean": round(cm_zc_mean, 4),
                "offdiag_abs_max": round(cm_zc_max, 4),
                "matrix": [[round(float(corr_zc[i, j]), 4) for j in range(k)] for i in range(k)],
            },
        },
        "interpretation": {
            "high_corr_note": "训练日志 head_corr~0.99 表示 batch 内各 head 在样本维度上高度同向；"
            "若 std_across_heads≈0 则是 K 个 head 输出几乎相同（可疑 bug/坍缩）。",
            "checks": [
                "scores_chosen 各维是否数值几乎相同",
                "diff 是否各维同号同幅度",
                "gate_weights 是否近似均匀",
                "p_plus 是否饱和到 0/1",
            ],
        },
        "samples_most_collapsed": by_collapse,
        "samples_most_diverse_heads": by_spread,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Markdown 便于阅读
    lines = [
        f"# 多维度输出诊断 ({ts})",
        "",
        f"- **Checkpoint**: `{ckpt}`",
        f"- **K**={k}, **K+**={num_pos}, **use_gate**={use_gate}",
        f"- **扫描样本数**: {len(all_records)}",
        "",
        "## Batch 内 Head 相关矩阵（chosen 分数）",
        f"- 非对角 |r| 均值: **{cm_zc_mean:.4f}**, 最大: **{cm_zc_max:.4f}**",
        "",
        _matrix_md(corr_zc, k),
        "",
        "## Batch 内 Head 相关矩阵（diff = z_c - z_r）",
        f"- 非对角 |r| 均值: **{cm_diff_mean:.4f}**, 最大: **{cm_diff_max:.4f}**",
        "",
        _matrix_md(corr_diff, k),
        "",
        "## 最可疑：各 head 分数几乎相同（std 最小）",
    ]
    for rec in by_collapse[:5]:
        lines.extend(_record_md(rec, k))
    lines.append("## 对照：head 分化最明显（std 最大）")
    for rec in by_spread[:3]:
        lines.extend(_record_md(rec, k))

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Done] JSON → {out_json}")
    print(f"[Done] MD   → {out_md}")
    print(
        f"[Corr] chosen offdiag mean={cm_zc_mean:.4f} max={cm_zc_max:.4f} | "
        f"diff offdiag mean={cm_diff_mean:.4f} max={cm_diff_max:.4f}"
    )


def _matrix_md(corr: torch.Tensor, k: int) -> str:
    hdr = "| | " + " | ".join(f"h{j}" for j in range(k)) + " |"
    sep = "|---|" + "|".join("---:" for _ in range(k)) + "|"
    body = []
    for i in range(k):
        row = "| " + f"**h{i}** | " + " | ".join(f"{corr[i,j]:.3f}" for j in range(k)) + " |"
        body.append(row)
    return "\n".join([hdr, sep] + body)


def _record_md(rec: dict, k: int) -> List[str]:
    lines = [
        "",
        f"### val_idx={rec['val_idx']} | std_heads={rec['std_across_heads_chosen']} | "
        f"pref_pseudo={rec['pref_correct_pseudo']} gate={rec['pref_correct_gate']}",
        "",
        "| head | z_c | z_r | diff | p_plus | K+ mask | gate_w |",
        "|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    gw = rec.get("gate_weights_on_chosen") or [None] * k
    for j in range(k):
        lines.append(
            f"| {j} | {rec['scores_chosen'][j]} | {rec['scores_rejected'][j]} | "
            f"{rec['diff_c_minus_r'][j]} | {rec['p_plus'][j]} | {rec['in_k_plus_mask'][j]} | "
            f"{gw[j] if gw[j] is not None else '-'} |"
        )
    lines.append(
        f"- r_pseudo: chosen={rec['r_pseudo_c']:.4f} rejected={rec['r_pseudo_r']:.4f} | "
        f"r_gate: {rec['r_gate_c']} / {rec['r_gate_r']}"
    )
    lines.append(f"- prompt: {rec['prompt_preview'][:120]}...")
    return lines


if __name__ == "__main__":
    main()
