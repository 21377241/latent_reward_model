# Gate 加权一阶段训练方案（train_stage=gate_weighted）

> **状态**：设计文档；代码已从仓库回退，按本文可重新接入。  
> **动机**：单条简化损失，一阶段同时训 backbone、K 维 heads、selector、gate，替代「latent + gate」两段（原流程保留）。

---

## 1. 与现有流程对比

| 项目 | 现有默认 | gate_weighted |
|------|----------|---------------|
| 训练阶段 | latent → gate（或 joint） | 单次 gate_weighted |
| 损失 | L_heads + L_selector（+ 可选 L_gate） | 仅 gate 加权的正负维 BT |
| Gate | 阶段2 标量 L_gate | 乘在每一维 BT 上 |
| 代码 | compute_latent_factor_loss | 新增 compute_gate_weighted_bt_loss |

---

## 2. 模型前向（不变）

- scores_c, scores_r：形状 [B, K]
- relations：selector 输出的 p_plus, p_minus，形状 [B, K, 2]
- gate_weights_c：chosen 侧 gate 权重 w[k]，softmax，形状 [B, K]
- gated_score = sum_k( w[k] * z[k] )

训练：--use_gate 开启；detach_scores_for_gate=False；四部分均可训。

---

## 3. 损失定义（可读写法）

### 3.1 逐维 logistic（与阶段一相同）

```
diff[k] = z_c[k] - z_r[k]

loss_plus[k]  = -log sigmoid( diff[k] )
loss_minus[k] = -log sigmoid( -diff[k] )
```

### 3.2 Gate 加权总损失

```
L = mean_over_batch(
    sum_over_k(
        w[k] * (
            mask_plus[k]  * loss_plus[k]
          + lambda_neg * mask_minus[k] * loss_minus[k]
        )
    )
)
```

- w[k] = gate_weights_c 在 chosen 侧
- mask_plus / mask_minus：见下节
- **不包含**：L_selector、L_dir、对标量 gated_score 的单独 L_gate

### 3.3 正负维掩码

与 L_heads 一致：对每个样本按 p_plus 做 top-k，得到 0/1 掩码 m_plus、m_minus。

- **默认**：硬 top-k + STE（前向 0/1，反传经 p_plus）
- **软掩码**：mask 直接用 p_plus、p_minus
- **纯硬掩码 + 无 STE**：selector 无梯度

---

## 4. Selector 与硬 top-k

topk 不可导。原方案用 **L_heads + L_selector** 双路径；本方案用 **STE** 或软掩码：

```python
# STE（推荐）
w_plus  = m_plus_hard  + (p_plus  - p_plus.detach())
w_minus = m_minus_hard + (p_minus - p_minus.detach())
```

可选叠加：`L_total = L_gate_weighted + lambda_sel * L_selector`（复用原路径 B）。

---

## 5. 与原损失对比

| 项目 | 原阶段一 | gate_weighted |
|------|----------|---------------|
| K+ / K− | topk(p_plus) | 同左（默认 + STE） |
| Heads | L_heads：m * p.detach() * loss | w * m_ste * loss |
| Selector | L_selector | STE / 软掩码 |
| Gate | 阶段二 L_gate | 乘在逐维 loss 上 |

---

## 6. 重新接入时的改动清单

- utils/loss_functions.py：新增 compute_gate_weighted_bt_loss（勿改原函数）
- scripts/train_rm.py：train_stage=gate_weighted 及 CLI
- 可选：run_train_gate_weighted.sh
- latent_config.json：train_stage、score_mode=gated_scalar

---

## 7. 参考实现

```python
def compute_gate_weighted_bt_loss(
    scores_c, scores_r, relations, gate_weights_c,
    lambda_neg=1.0, num_pos_heads=5,
    soft_pos_mask=False, use_ste=True,
):
    diff = scores_c - scores_r
    p_plus, p_minus = relations[:, :, 0], relations[:, :, 1]
    loss_plus = -F.logsigmoid(diff)
    loss_minus = -F.logsigmoid(-diff)

    if soft_pos_mask:
        w_plus, w_minus = p_plus, p_minus
    else:
        m_plus, m_minus = topk_indicator_masks(p_plus, num_pos_heads)
        if use_ste:
            w_plus = m_plus + (p_plus - p_plus.detach())
            w_minus = m_minus + (p_minus - p_minus.detach())
        else:
            w_plus, w_minus = m_plus, m_minus

    per_sample = (gate_weights_c * (
        w_plus * loss_plus + lambda_neg * w_minus * loss_minus
    )).sum(dim=-1)
    return per_sample.mean()
```

---

## 8. 日志与评测

- 训练：loss/L_gate_weighted、gate/entropy、metrics/accuracy（gated 标量）
- best：按 eval/acc_global（gated 标量）保存

---

## 9. 注意点

1. STE 是近似，selector 梯度不精确对应「改变 top-k 成员」
2. 关注 gate 塌缩（gate/max_prob、gate/entropy）
3. 验证 acc 与阶段一 K+ 伪标量不可直接比
4. 接回代码时勿破坏 latent / joint / gate

---

## 10. 版本记录

| 日期 | 说明 |
|------|------|
| 2026-05-28 | 初版；代码曾合入后回退，设计保留于本文 |
