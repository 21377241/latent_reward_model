# Latent Reward Model 修改说明

本文档记录 `latent_reward_model` 相对初始参考实现（`reference/LatentMRM_code`）及后续迭代中的主要改动，便于复现、评审与和 `baseline/` 对齐。

**文档日期**：2026-05-25（持续更新）  
**对齐参考**：`baseline/singe_head.py`、`multihead_baseline/train_gated_multihead.py`  
**Accelerate 配置**：`latent_reward_model/accel_ds2.yaml` + `ds_zero2.json`（与 `solution1/` 同内容，本地启动）  
**远程仓库**：[21377241/latent_reward_model](https://github.com/21377241/latent_reward_model)

---

## 一、改动总览

| 类别 | 修改前（参考实现） | 当前实现 |
|------|-------------------|----------|
| Backbone | `Qwen/Qwen2.5-0.5B` | 本地 `llama3` / `llama3.1` / `armorm` |
| 对话格式 | 手写 `User: ... Assistant: ...` | Llama 3 `apply_chat_template` |
| 训练数据 | 仅 margin≥1.0；随机 9:1 划分 | 官方 train/test + 三条 score 清洗 |
| 长度过滤 | 截断至 `max_length` | 16–4096 丢弃，**不截断** |
| 特征抽取 | `sum(mask)-1` | left-padding 兼容的 last-token pooling |
| 训练框架 | HuggingFace `Trainer` | **Accelerate + DeepSpeed ZeRO-2** |
| 实验日志 | Weights & Biases | **SwanLab** |
| 优化器 | 单一 `learning_rate` | AdamW **分组 LR** + cosine warmup |
| Checkpoint | Trainer 默认格式 | `model.safetensors` + `latent_heads.pt` |

**模型结构（未改核心算法）**：K 维 `reward_heads` + `selector` + `compute_latent_factor_loss`（相对 baseline 单头 `Linear` 仍为本质差异）。

---

## 二、项目初始化与 GitHub

### 2.1 代码来源

- 自 `reference/LatentMRM_code` 拷贝核心训练代码。
- 补充 `README.md`、`.gitignore`（`data/`、`experiments/`、`swanlog/` 等）。

### 2.2 版本管理

- 本地 Git 仓库，`main` 分支；远程 [21377241/latent_reward_model](https://github.com/21377241/latent_reward_model)。
- `scripts/push_to_github.sh`：可选 GitHub API 推送脚本。

### 2.3 注意

- 训练前需 `prepare_full_data.py` 生成 jsonl，或训练时直接指定 parquet 路径。
- 勿将 Token、密钥写入仓库。

---

## 三、Backbone（Qwen → 本地 Llama / ArmoRM）

**`models/backbone.py`**

| `backbone_type` | 路径 | 加载方式 |
|-----------------|------|----------|
| `llama3_baseline` | `model/llama-3-8b-instruct` | `AutoModelForCausalLM` → `.model` |
| `llama3.1_baseline` | `model/llama-3.1-8b-instruct` | 同上 |
| `armorm_baseline` | `model/armorm` | `AutoModelForSequenceClassification` + `trust_remote_code` → `.model` |

- 环境变量 `REWARD_MODEL_ROOT` 可改仓库根路径。
- **不使用** ArmoRM 原 gating / score head。

**`models/latent_reward_model.py`**：通过 `load_backbone()` 初始化；`forward` 使用 `pool_last_hidden()`。

---

## 四、数据：清洗、格式、长度

### 4.1 Score 清洗（`utils/ultrafeedback_clean.py`）

与 baseline **训练集**默认规则一致：

1. `score_chosen - score_rejected >= 1.0`
2. `score_chosen >= 4.0`
3. 丢弃 `score_chosen == 10`（`--no_drop_score10` 可关闭）

验证集 **不** 做 score 清洗。

### 4.2 数据来源（`scripts/prepare_full_data.py` + `utils/dataloader.py`）

| 划分 | 默认来源 | 说明 |
|------|----------|------|
| 训练 | `train_prefs` parquet → `data/ultrafeedback_train.jsonl` | 清洗后导出 |
| 验证 | `test_prefs` parquet → `data/ultrafeedback_val.jsonl` | 官方 test，非随机 9:1 |

训练时二选一：

- **jsonl**：`--train_data_path` / `--eval_data_path`（`run_train.sh` 用绝对路径）
- **parquet**：`--train_data` / `--test_data`（与 baseline 相同，训练时在线清洗）

预期清洗后训练集约 **61135 → 41661** 条。

### 4.3 Llama 3 Chat Template（`utils/chat_format.py`）

- `build_conversation()` + `encode_conversation()`：`apply_chat_template`，`add_special_tokens=False`。
- jsonl 保留完整 `chosen` / `rejected` messages 列表。
- 兼容旧 jsonl（`prompt` + 字符串回复）。

### 4.4 长度过滤

- `encode_conversation()` **不截断**。
- tokenize 后：`16 <= len(chosen), len(rejected) <= 4096`，否则丢弃（train / eval 均过滤）。
- `passes_length_filter()` 在 `utils/chat_format.py`。

### 4.5 Collate

- **left padding**（`utils/dataloader.py` 中 `pad_left`），与 baseline 一致。

---

## 五、训练框架：Accelerate + DeepSpeed ZeRO-2

### 5.1 修改前

- `HuggingFace Trainer` + 多卡 DDP。
- `PairwiseDataCollator` + `LatentFactorRewardTrainer`。

### 5.2 修改后

**`scripts/train_rm.py`**：手写训练循环，与 `baseline/singe_head.py` 同架构：

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model
accelerate launch --config_file accel_ds2.yaml \
    ../latent_reward_model/scripts/train_rm.py [参数...]
```

或：`bash run_train.sh`（在 `latent_reward_model/` 下 launch）。

| 项目 | 默认 / 说明 |
|------|-------------|
| DeepSpeed | ZeRO Stage 2（`ds_zero2.json`） |
| 精度 | bf16 |
| 梯度累积 | `Accelerator(gradient_accumulation_steps=8)` + `accumulate()` |
| Grad clip | `max_grad_norm=1.0` |
| GC | `backbone.gradient_checkpointing_enable()` |
| 离线 Hub | `HF_HUB_OFFLINE=1` |

**`utils/dataloader.py`**：数据加载、tokenize、长度过滤、DataLoader。  
**`utils/checkpoint.py`**：`save_latent_ckpt()`，ZeRO-2 兼容的 `get_state_dict` 保存。

### 5.3 Checkpoint 格式

```
{output_dir}/{tag}/
  config.json
  model.safetensors      # backbone
  latent_heads.pt        # reward_heads + selector
  tokenizer*
  scheduler.pt
  meta.json
```

按 `eval/acc_global` 保存 `best/`；训练结束保存 `final/`；可选 `step_N/`。

---

## 六、优化器与学习率（对齐 Baseline）

**`utils/optimizer.py`**：`cosine_warmup()`（linear warmup + cosine，`eta_min=0.01`）。

| 参数组 | 默认 LR | 包含模块 |
|--------|---------|----------|
| backbone | `1e-6` | `backbone` |
| head | `1e-4` | `reward_heads` + `selector` |

- 优化器：**AdamW**，`weight_decay=0.01`
- Warmup：`warmup_ratio=0.05`（占总 optimizer step 比例）

**默认训练超参（`run_train.sh`）**

| 参数 | 值 |
|------|-----|
| `batch_size` | 4 |
| `grad_accum` | 8 |
| `num_epochs` | 2 |
| `eval_steps` | 50 |
| `max_eval_samples` | 2000 |

4 卡时 global batch = 4×4×8 = **128**；baseline 建议 ≥512，可增大 `grad_accum`。

---

## 七、实验日志：SwanLab

与 baseline 一致，使用 **SwanLab**（已移除 wandb）。

| 参数 | 默认 |
|------|------|
| `--swanlab_project` | `latentMRM` |
| `--swanlab_experiment_name` | `LatentRewardModel` |
| `--swanlab_mode` | `cloud`（`local` / `disabled`） |

记录指标示例：`train/loss`、`train/L_heads`、`train/L_selector`、`train/lr_*`、eval 各 `metrics/*` 与 `eval/acc_global`。

本地仍会写入：`output_dir/train_*.log`、`log.json`、`summary.json`。

---

## 八、文件清单

### 新增

| 路径 | 说明 |
|------|------|
| `models/backbone.py` | Backbone 加载 |
| `models/pooling.py` | Last-token pooling |
| `utils/chat_format.py` | Chat template 编码与长度判断 |
| `utils/ultrafeedback_clean.py` | UltraFeedback 清洗 |
| `utils/optimizer.py` | cosine_warmup 调度 |
| `utils/dataloader.py` | 数据管道与 collate |
| `utils/checkpoint.py` | Accelerate checkpoint 保存 |
| `scripts/push_to_github.sh` | GitHub 推送辅助 |
| `docs/CHANGELOG_MODIFICATIONS.md` | 本文档 |

### 核心脚本

| 路径 | 说明 |
|------|------|
| `scripts/train_rm.py` | Accelerate 训练主程序 |
| `scripts/prepare_full_data.py` | 数据导出 jsonl |
| `run_train.sh` | ZeRO-2 启动脚本 |

### 保留参考实现逻辑

| 路径 | 说明 |
|------|------|
| `utils/loss_functions.py` | Latent factor + selector 损失 |
| `utils/metrics.py` | 旧 Trainer 用指标（当前训练循环内联 eval stats） |
| `models/latent_reward_model.py` | Latent MRM 主体 |

### 已移除 / 不再使用

- HuggingFace `Trainer`、`TrainingArguments`、`PairwiseDataCollator`
- Weights & Biases

---

## 九、推荐训练流程

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model

# 1. 数据（若尚无 jsonl）
PYTHONPATH=. python scripts/prepare_full_data.py

# 2. 训练（Accelerate + ZeRO-2）
BACKBONE_TYPE=llama3_baseline bash run_train.sh
# BACKBONE_TYPE=llama3.1_baseline bash run_train.sh
# BACKBONE_TYPE=armorm_baseline bash run_train.sh

# 3. 关闭 SwanLab（可选）
# 在 run_train.sh 或命令行加：--swanlab_mode disabled
```

---

## 十、与 Baseline 对齐情况

| 项目 | Baseline | Latent RM |
|------|----------|-----------|
| 数据 train/test 划分 | train_prefs / test_prefs | 同左 |
| Score 清洗（train） | 三条规则 | 同左 |
| Score 清洗（eval） | 无 | 无 |
| Chat template | Llama 3 | 同左 |
| Padding / last hidden | left + 翻转 mask | 同左 |
| 长度过滤 | 16–4096，不截断 | 同左 |
| 优化器 / LR | 分组 AdamW + cosine | 同左（head 含 selector+reward_heads） |
| 训练框架 | Accelerate + ZeRO-2 | 同左 |
| Gradient checkpointing | 有 | 有 |
| 实验日志 | SwanLab | SwanLab |
| eval 样本上限 | 2000 | 2000（`max_eval_samples`） |
| 损失函数 | 单头 BT loss | 多维 Latent loss |
| Forward 次数 / batch | chosen∥rejected **1 次** backbone | chosen、rejected **各 1 次** |
| 两阶段 head-only | 支持 | **未实现** |
| Checkpoint 评测导出 | `reward_head.pt` + export 脚本 | `latent_heads.pt`，**未**接 reward-bench export |
| 全局 batch 默认 | 倾向 512 | 4 卡约 128，需调 `grad_accum` |

---

## 十一、变更时间线（摘要）

1. 自 `LatentMRM_code` 初始化项目并同步 GitHub。  
2. Backbone 改为本地 llama3 / llama3.1 / armorm；last-token pooling。  
3. Llama 3 chat template；数据清洗与官方 train/test 划分。  
4. 长度过滤与 baseline 对齐（不截断）。  
5. 优化器分组 LR + cosine warmup。  
6. 训练架构改为 **Accelerate + DeepSpeed ZeRO-2**；拆分 `dataloader` / `checkpoint`。  
7. 实验日志改为 **SwanLab**。

---

## 十二、模型结构文档

详见 **[MODEL_ARCHITECTURE.md](./MODEL_ARCHITECTURE.md)**（架构图、各模块、损失函数与训练解耦机制）。

---

## 十三、参考链接

- 数据集：[HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- GitHub：[21377241/latent_reward_model](https://github.com/21377241/latent_reward_model)
- SwanLab：https://swanlab.cn

后续改动请更新文首日期，并在 **第十一节** 追加条目。
