# 固定前缀正向维训练（无 Selector）

## 概要

- **不再使用 selector**；不构建 `selector` 模块，checkpoint 中无 `selector.*` 权重。
- **K+**：head 索引 `0, 1, …, num_pos_heads - 1`（前 `num_pos_heads` 维）。
- **K−**：其余维度（索引 `num_pos_heads … K-1`）。
- 损失：仅 `L_heads`（正负维 logistic）+ 可选 `L_gate`（与原版 `joint` / `gate` 一致）。

---

## 训练阶段一览

| train_stage   | 可训练模块              | 总损失                          |
|---------------|-------------------------|---------------------------------|
| fixed_latent  | backbone + reward_heads | L_total = L_heads               |
| fixed_joint   | 上表 + gating_network   | L_total = L_heads + λ·L_gate    |
| fixed_gate    | 仅 gate（需 resume）    | L_total = λ·L_gate              |

`λ` 即命令行 `--lambda_gate`（默认 1.0）。

---

## fixed_latent（阶段一）损失函数

### 符号

- `z_c[k]`, `z_r[k]`：chosen / rejected 在第 k 个 head 上的分数，k = 0 … K-1
- `K+` = `num_pos_heads`：正向维个数
- `diff[k] = z_c[k] - z_r[k]`

### 掩码（固定，与样本无关）

```
k 属于 K+  ⟺  k < num_pos_heads     →  m_plus[k] = 1
k 属于 K−  ⟺  k >= num_pos_heads    →  m_minus[k] = 1
```

### 逐维 BT 项

```
loss_plus[k]  = -log sigmoid( diff[k] )          # K+：希望 z_c > z_r
loss_minus[k] = -log sigmoid( -diff[k] )          # K−：希望 z_r > z_c
                = -log sigmoid( z_r[k] - z_c[k] )
```

### L_heads（训练用的唯一损失）

对 batch 中每个样本，在 K 维上求和，再对 batch 取平均：

```
L_heads = mean_over_batch(
    sum_over_k(
        m_plus[k]  * loss_plus[k]
      + lambda_neg * m_minus[k] * loss_minus[k]
    )
)
```

- `lambda_neg`：命令行 `--lambda_neg`，默认 **1.0**
- **fixed_latent 时**：`L_total = L_heads`（无 L_selector、无 L_gate）

### 代码位置

`utils/loss_functions.py` → `compute_fixed_prefix_loss`，且 `lambda_gate=0`。

---

## fixed_joint / fixed_gate 的损失（补充）

**fixed_joint**（同时训 heads + gate）：

```
L_total = L_heads + lambda_gate * L_gate

L_gate = mean_over_batch( -log sigmoid( g_c - g_r ) )

g_c = sum_k( w[k] * z_c[k] )    # w = gate 在 chosen 隐状态上的 softmax 权重
g_r = sum_k( w'[k] * z_r[k] )   # rejected 侧单独算 gate
```

**fixed_gate**（只训 gate，heads 冻结）：

```
L_total = lambda_gate * L_gate
```

gate 阶段前向会对 head 分数 `detach`，与原 `train_stage=gate` 相同。

---

## 验证指标（与训练损失不同）

| 场景 | acc_global 怎么算 |
|------|-------------------|
| 无 gate（fixed_latent） | `R_c = sum_{k=0}^{K-1} z_c[k]`，`R_r` 同理，看 `R_c > R_r` 的比例（**全 K 维**，含 K−） |
| 有 gate | `g_c > g_r`（gated 标量） |

训练时 K− **参与 L_heads**；验证 acc **不会**只加 K+ 前缀（除非看日志里的 `metrics/acc_pseudo_kplus`）。

评测导出：`score_mode = "heads_sum"`（无 gate）或 `"gated_scalar"`（有 gate）。

---

## 启动示例

### 两阶段（推荐：阶段1 head，阶段2 gate，实验名分开）

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model
bash run_train_two_stage_fixed_prefix.sh
```

默认目录与 SwanLab 名：

- 阶段1：`experiments/latent_mrm_llama3.1_baseline_k10_prefix6_fixed_2stage_latent`
- 阶段2：`experiments/latent_mrm_llama3.1_baseline_k10_prefix6_fixed_2stage_latent_gate`

自定义 K+：`NUM_POS_HEADS=6 K_DIMENSIONS=10 bash run_train_two_stage_fixed_prefix.sh`

只训阶段2：`SKIP_STAGE1=1 bash run_train_two_stage_fixed_prefix.sh`

### 单阶段

```bash
# 仅 heads
K_PLUS=6 K_DIM=10 bash run_train_fixed_prefix.sh

# heads + gate 同时
TRAIN_STAGE=fixed_joint K_PLUS=6 bash run_train_fixed_prefix.sh
```

```bash
accelerate launch --config_file accel_ds2.yaml scripts/train_rm.py \
  --train_stage fixed_latent \
  --k_dimensions 10 \
  --num_pos_heads 6 \
  ...
```

---

## 与原 selector 方案对比

| 项目 | 原 latent | fixed_latent |
|------|-----------|--------------|
| K+ 谁定 | topk(p_plus) | 固定 dim 0 … K+-1 |
| Selector | 有 + L_selector | 无 |
| 阶段一损失 | L_heads + L_selector | 仅 L_heads |
| 无 gate 时验证 acc | K+ 伪标量求和 | **全 K 维求和** |

原 `latent` / `joint` / `gate` 流程未改动。
