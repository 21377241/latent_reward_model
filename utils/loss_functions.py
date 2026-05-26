import torch
import torch.nn.functional as F

from utils.metrics import pearson_head_offdiag_stats

def compute_latent_factor_loss(
    scores_c,
    scores_r,
    relations,
    lambda_neg=1.0,
    beta_dir=0.1,
    target_tau=0.5,
    num_pos_heads=5,
    gated_score_c=None,
    gated_score_r=None,
    lambda_gate=0.0,
    lambda_latent=1.0,
    gate_weights_c=None,
):
    """
    计算基于潜在因子的 Bradley-Terry 损失
    scores_c: [batch_size, K] - chosen 得分
    scores_r: [batch_size, K] - rejected 得分
    relations: [batch_size, K, 2] - 包含 (p_plus, p_minus)
    """
    diff = scores_c - scores_r
    
    # 提取三种关系的概率
    p_plus = relations[:, :, 0]
    p_minus = relations[:, :, 1]
    
    loss_plus = -F.logsigmoid(diff)
    loss_minus = -F.logsigmoid(-diff)


    # top-k筛选
    _, topk_indices = torch.topk(p_plus, k=num_pos_heads, dim=-1) #[batch_size, num_pos_heads],表示选取为正向维度的维度索引
    mask_plus=torch.zeros_like(p_plus, dtype=torch.bool).scatter_(-1, topk_indices, True)
    mask_minus = ~mask_plus
    m_plus = mask_plus.float()
    m_minus = mask_minus.float()



    # --- 路径 A: 训练 Heads (不许 Selector 变) ---
    # 此时 p 是常量，Heads 必须根据当前分配把分数拉开
    L_heads = torch.sum(m_plus * p_plus.detach() * loss_plus +  m_minus * p_minus.detach() * loss_minus, dim=-1).mean()

    # --- 路径 B: 训练 Selector (不许 Heads 变) ---
    reward = torch.tanh(diff.detach())
    log_p_plus = torch.log(p_plus + 1e-8)
    log_p_minus = torch.log(p_minus + 1e-8)
    #若在K+组：奖励就是 reward 本身
    L_sel_plus = -torch.sum(m_plus * log_p_plus * reward, dim=-1)
    # 若在 K- 组：奖励是 -reward (因为如果 diff < 0，选 K- 才应该被奖励)
    L_sel_minus = -torch.sum(m_minus * log_p_minus * (-reward), dim=-1)

    L_selector = (L_sel_plus + L_sel_minus).mean()

    # 按维平均的 heads 损失（便于与单头 RM 量级对比、排查求和导致的 loss 飙升）
    heads_per_sample = (
        m_plus * p_plus.detach() * loss_plus
        + m_minus * p_minus.detach() * loss_minus
    ).sum(dim=-1)
    k_dim = scores_c.shape[-1]
    L_heads_mean = (heads_per_sample / max(k_dim, 1)).mean()

    # #方向性惩罚，注意允许全是正向维度
    # # 尝试一：当样本在K个维度中正向维度占比较少时惩罚
    # sample_p_plus_ratio = p_plus.mean(dim=1)#每对样本内部，K 个维度被分配为正向的平均比例[batch_size] 
    # L_dir = torch.mean(F.relu(target_tau - sample_p_plus_ratio)** 2)

    # #尝试二：每个样本中正向和反向占比尽可能五五开
    # # d_k = (p_plus - p_minus).mean(dim=1)
    # # L_dir = torch.sum(d_k ** 2)

    # #尝试三：强制正向维度的总和大于反向维度
    # # sum_p_plus = p_plus.sum(dim=-1)
    # # sum_p_minus = p_minus.sum(dim=-1)
    # # L_dir = torch.mean(torch.relu(sum_p_minus - sum_p_plus + 0.1))

    # L_selector= L_local_selector + beta_dir * L_dir

    L_gate = scores_c.new_zeros(())
    if gated_score_c is not None and lambda_gate > 0:
        L_gate = -F.logsigmoid(gated_score_c - gated_score_r).mean()

    # 总 Loss（stage1: lambda_latent=1；stage2 gate-only: lambda_latent=0）
    L_total = lambda_latent * (L_heads + L_selector) + lambda_gate * L_gate

    with torch.no_grad():
        eps = 1e-8
        k_dim = float(scores_c.shape[-1])

        # 1. 概率与 selector 饱和
        mean_p_plus = p_plus.mean().item()
        mean_p_minus = p_minus.mean().item()
        p_plus_max = p_plus.max().item()
        entropy = -torch.sum(relations * torch.log(relations + 1e-8), dim=-1).mean().item()
        avg_p_in_k_plus = (p_plus * m_plus).sum() / (m_plus.sum() + eps)
        avg_p_in_k_minus = (p_plus * m_minus).sum() / (m_minus.sum() + eps)

        den_plus = m_plus.sum(dim=-1).clamp(min=1.0)
        den_minus = m_minus.sum(dim=-1).clamp(min=1.0)

        # 2. K+/K- 上的分差与 BT 基项（未乘 p，看 loss 升是否来自 Δ 变差）
        mean_delta_kplus = ((diff * m_plus).sum(dim=-1) / den_plus).mean().item()
        mean_delta_kminus = ((diff * m_minus).sum(dim=-1) / den_minus).mean().item()
        mean_loss_plus_kplus = ((loss_plus * m_plus).sum(dim=-1) / den_plus).mean().item()
        mean_loss_minus_kminus = ((loss_minus * m_minus).sum(dim=-1) / den_minus).mean().item()

        # 3. 「方向错误」占比：K+ 上应 z_c>z_r 却失败的比例 ↑ 会直接推高 L_heads
        correct_plus = (diff > 0).float()
        correct_minus = (diff < 0).float()
        acc_plus_heads = ((m_plus * correct_plus).sum(dim=-1) / den_plus).mean().item()
        acc_minus_heads = ((m_minus * correct_minus).sum(dim=-1) / den_minus).mean().item()
        frac_wrong_kplus = ((m_plus * (1.0 - correct_plus)).sum(dim=-1) / den_plus).mean().item()
        frac_wrong_kminus = ((m_minus * (1.0 - correct_minus)).sum(dim=-1) / den_minus).mean().item()

        # 4. 加权后的分项贡献（与 L_heads 分解一致）
        weighted_plus = (m_plus * p_plus * loss_plus).sum(dim=-1).mean().item()
        weighted_minus = (m_minus * p_minus * loss_minus).sum(dim=-1).mean().item()

        # 5. 分数尺度：|z|、|Δ| 变大时 log-sigmoid 惩罚可升高
        mean_abs_sc = scores_c.abs().mean().item()
        mean_abs_sr = scores_r.abs().mean().item()
        mean_abs_diff = diff.abs().mean().item()
        scores_diversity_c = torch.std(scores_c, dim=-1).mean().item()
        scores_diversity_r = torch.std(scores_r, dim=-1).mean().item()
        scores_gap_c = (
            torch.max(scores_c, dim=-1)[0] - torch.min(scores_c, dim=-1)[0]
        ).mean().item()

        # 6. 全局偏好：有 gate 标量则用 gate，否则 K+ 伪标量求和
        reward_c_pseudo = (scores_c * m_plus).sum(dim=-1)
        reward_r_pseudo = (scores_r * m_plus).sum(dim=-1)
        if gated_score_c is not None:
            margin_global = (gated_score_c - gated_score_r).mean().item()
            acc_global = (gated_score_c > gated_score_r).float().mean().item()
        else:
            margin_global = (reward_c_pseudo - reward_r_pseudo).mean().item()
            acc_global = (reward_c_pseudo > reward_r_pseudo).float().mean().item()

        # 7. Head 分化：各维 Pearson 相关（batch 内跨样本，对齐 multihead_baseline）
        head_corr_c = pearson_head_offdiag_stats(scores_c, prefix="diag/head_corr_c")
        head_corr_r = pearson_head_offdiag_stats(scores_r, prefix="diag/head_corr_r")
        head_corr_diff = pearson_head_offdiag_stats(diff, prefix="diag/head_corr_diff")

        gate_entropy = None
        gate_max_prob = None
        if gate_weights_c is not None:
            gw = gate_weights_c.clamp(min=1e-8)
            gate_entropy = (-(gw * torch.log(gw)).sum(dim=-1).mean()).item()
            gate_max_prob = gw.max(dim=-1).values.mean().item()

        stats = {
            # --- loss 分解（排查 L_total 上升来自 heads 还是 selector）---
            "loss/L_heads_mean": L_heads_mean.item(),
            "loss/L_gate": L_gate.item() if lambda_gate > 0 else 0.0,
            "loss/L_sel_plus": L_sel_plus.mean().item(),
            "loss/L_sel_minus": L_sel_minus.mean().item(),
            "loss/weighted_plus": weighted_plus,
            "loss/weighted_minus": weighted_minus,
            # --- K+/K- 方向与 BT 基项 ---
            "diag/mean_delta_kplus": mean_delta_kplus,
            "diag/mean_delta_kminus": mean_delta_kminus,
            "diag/mean_loss_plus_kplus": mean_loss_plus_kplus,
            "diag/mean_loss_minus_kminus": mean_loss_minus_kminus,
            "diag/frac_wrong_kplus": frac_wrong_kplus,
            "diag/frac_wrong_kminus": frac_wrong_kminus,
            # --- selector / 分数尺度 ---
            "prob/p_plus": mean_p_plus,
            "prob/p_minus": mean_p_minus,
            "prob/p_plus_max": p_plus_max,
            "metrics/entropy": entropy,
            "debug/avg_p_in_k_plus": avg_p_in_k_plus.item(),
            "debug/avg_p_in_k_minus": avg_p_in_k_minus.item(),
            "diag/mean_abs_z_c": mean_abs_sc,
            "diag/mean_abs_z_r": mean_abs_sr,
            "diag/mean_abs_diff": mean_abs_diff,
            "debug/scores_std_chosen": scores_diversity_c,
            "debug/scores_std_rejected": scores_diversity_r,
            "debug/scores_gap_chosen": scores_gap_c,
            "diag/k_dimensions": k_dim,
            "diag/num_pos_heads": float(num_pos_heads),
            # --- 与偏好相关的汇总 ---
            "metrics/acc_plus_heads": acc_plus_heads,
            "metrics/acc_minus_heads": acc_minus_heads,
            "metrics/margin": margin_global,
            "metrics/accuracy": acc_global,
            "metrics/acc_pseudo_kplus": (
                (reward_c_pseudo > reward_r_pseudo).float().mean().item()
            ),
            **head_corr_c,
            **head_corr_r,
            **head_corr_diff,
        }
        if gate_entropy is not None:
            stats["gate/entropy"] = gate_entropy
            stats["gate/max_prob"] = gate_max_prob
    
    return L_total, L_heads, L_selector, L_gate, stats

