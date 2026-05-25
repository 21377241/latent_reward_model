import torch
import torch.nn.functional as F

def compute_latent_factor_loss(scores_c, scores_r, relations, lambda_neg=1.0,beta_dir=0.1,target_tau=0.5,num_pos_heads=5):
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
    # 总 Loss
    L_total = L_heads + L_selector

    with torch.no_grad():
        # 1. 概率均值监控
        mean_p_plus = p_plus.mean().item()
        mean_p_minus = p_minus.mean().item()
        entropy = -torch.sum(relations * torch.log(relations + 1e-8), dim=-1).mean().item()
        
        #p+在正向维度组和反向维度组的变化情况
        avg_p_in_k_plus = (p_plus * m_plus).sum() / (m_plus.sum() + 1e-8)
        avg_p_in_k_minus = (p_plus * m_minus).sum() / (m_minus.sum() + 1e-8)

        # 2.观测同一样本在K个维度上的打分差异
        scores_diversity_c = torch.std(scores_c, dim=-1).mean().item()
        scores_diversity_r = torch.std(scores_r, dim=-1).mean().item()
        scores_gap_c = (torch.max(scores_c, dim=-1)[0] - torch.min(scores_c, dim=-1)[0]).mean().item() #极差

        # ==================== 机制验证 ====================
        # 衡量 heads 是否听从了 selector 的指挥
        # 正向维度准确率:被划分为正向组中 z_c > z_r 的比例
        correct_plus = (scores_c > scores_r).float()
        acc_plus_heads = (torch.sum(m_plus * correct_plus, dim=-1) / (torch.sum(m_plus, dim=-1) + 1e-8)).mean().item()
        
        # 反向维度准确率: 被划分为反向组中，z_r > z_c 的比例
        correct_minus = (scores_r > scores_c).float()
        acc_minus_heads = (torch.sum(m_minus * correct_minus, dim=-1) / (torch.sum(m_minus, dim=-1) + 1e-8)).mean().item()

        # 伪全局reward
        reward_c_pseudo = (scores_c * m_plus).sum(dim=-1)
        reward_r_pseudo = (scores_r * m_plus).sum(dim=-1)

        margin_global = (reward_c_pseudo - reward_r_pseudo).mean().item()
        acc_global = (reward_c_pseudo > reward_r_pseudo).float().mean().item()
        
        stats = {
            "prob/p_plus": mean_p_plus,
            "prob/p_minus": mean_p_minus,
            "metrics/entropy": entropy,
            # "loss/L_dir": L_dir.item(),
            # 机制验证指标
            "metrics/acc_plus_heads": acc_plus_heads,
            "metrics/acc_minus_heads": acc_minus_heads,
            # 全局指标
            "metrics/margin": margin_global,
            "metrics/accuracy": acc_global,
            # 新增维度差异监控
            "debug/scores_std_chosen": scores_diversity_c,
            "debug/scores_std_rejected": scores_diversity_r,
            "debug/scores_gap_chosen": scores_gap_c,
            "debug/avg_p_in_k_plus": avg_p_in_k_plus.item(),
            "debug/avg_p_in_k_minus": avg_p_in_k_minus.item(),
        }
    
    # return L_total, L_plus, L_minus, stats
    return L_total, L_heads, L_selector, stats

