import numpy as np
import torch
from transformers import EvalPrediction


def pearson_head_offdiag_stats(
    head_scores: torch.Tensor,
    prefix: str = "diag/head_corr",
) -> dict:
    """
    K 个 head 在 batch 样本上的 Pearson 相关矩阵，返回非对角 |r| 的 mean/max。

    head_scores: [N, K]，N 为 batch 内样本数（可对 chosen / rejected / diff 分别算）。
    解读：mean/max 越高 → 各维打分越同向，分化越弱；训练后期应相对下降。
    """
    if head_scores.ndim != 2 or head_scores.size(0) < 2 or head_scores.size(1) < 2:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}

    x = head_scores.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x = x / std
    n = x.size(0)
    corr = (x.t() @ x) / max(n - 1, 1)

    k = corr.size(0)
    mask = ~torch.eye(k, dtype=torch.bool, device=corr.device)
    off_diag = corr[mask].abs()
    if off_diag.numel() == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": off_diag.mean().item(),
        f"{prefix}_max": off_diag.max().item(),
    }

def get_compute_metrics_fn(lambda_neg):
    """
    返回一个带有 lambda_neg 上下文的 metrics 计算函数
    """
    def compute_latent_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        print(type(eval_pred.predictions))
        print(len(eval_pred.predictions))
        if isinstance(preds, tuple):
            # preds = ((scores_c, scores_r, relations),)
            if len(preds) == 1 and isinstance(preds[0], tuple):
                preds = preds[0]
            scores_c, scores_r, relations = preds
        else:
            raise ValueError(
                f"Unexpected predictions type: {type(preds)}"
            )
        p_plus = relations[:, :, 0]
        p_minus = relations[:, :, 1]

        # ==========================================
        # 层级一：机制验证 (Head-Level)
        # ==========================================
        # 正向准确率
        correct_plus = (scores_c > scores_r).astype(float)
        acc_plus_heads = np.mean(np.sum(p_plus * correct_plus, axis=-1) / (np.sum(p_plus, axis=-1) + 1e-8))
        
        # 反向准确率
        correct_minus = (scores_r > scores_c).astype(float)
        acc_minus_heads = np.mean(np.sum(p_minus * correct_minus, axis=-1) / (np.sum(p_minus, axis=-1) + 1e-8))

        # ==========================================
        # 层级二：全局 BT 偏好拟合度
        # ==========================================
        weights = p_plus + lambda_neg * p_minus 
        
        reward_c = np.sum(weights * scores_c, axis=-1)
        reward_r = np.sum(weights * scores_r, axis=-1)
        
        margin_global = np.mean(reward_c - reward_r)
        acc_global = np.mean((reward_c > reward_r).astype(float))

        return {
            "acc_plus_heads": acc_plus_heads,
            "acc_minus_heads": acc_minus_heads,
            "margin_global": margin_global,
            "acc_global": acc_global
        }
        
    return compute_latent_metrics
