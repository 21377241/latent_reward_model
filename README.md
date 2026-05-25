# Latent Reward Model

基于多潜在维度的奖励模型（Latent MRM）：共享 backbone、K 个 reward head 与 selector 网络，对 chosen/rejected 样本进行关系分配与 Bradley-Terry 风格训练。

- 工程与对齐改动：**[docs/CHANGELOG_MODIFICATIONS.md](docs/CHANGELOG_MODIFICATIONS.md)**
- 模型结构详解：**[docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md)**

## 支持的 Backbone

与仓库 `baseline/` 一致，使用 `model/` 目录下的本地权重（默认根路径 `/mnt/afs/250010036/reward_model`）：

| `backbone_type` | 本地路径 |
|-----------------|----------|
| `llama3_baseline` | `model/llama-3-8b-instruct` |
| `llama3.1_baseline` | `model/llama-3.1-8b-instruct` |
| `armorm_baseline` | `model/armorm`（需 `trust_remote_code`） |

可通过环境变量 `REWARD_MODEL_ROOT` 修改仓库根目录。

## 项目结构

```
latent_reward_model/
├── models/
│   ├── backbone.py              # backbone 路径与加载
│   ├── pooling.py               # last-token pooling
│   └── latent_reward_model.py   # Latent MRM
├── scripts/train_rm.py
├── utils/
└── run_train.sh
```

## 快速开始

训练使用 **Accelerate launch + DeepSpeed ZeRO-2**（配置见本目录 `accel_ds2.yaml`、`ds_zero2.json`）。

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model
PYTHONPATH=. python scripts/prepare_full_data.py

# Llama3-8B（默认，4 卡 ZeRO-2）
export SWANLAB_API_KEY="你的key"   # cloud 模式；未配置可用 SWANLAB_MODE=disabled
bash run_train.sh

# Llama3.1 / ArmoRM（单 backbone）
BACKBONE_TYPE=llama3.1_baseline bash run_train.sh
BACKBONE_TYPE=armorm_baseline bash run_train.sh

# 三个 backbone 串行（各独立 output_dir + SwanLab 实验名 LatentRewardModel_<backbone>）
bash run_train_serial.sh
```

手动启动（在本项目根目录）：

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model
export PYTHONPATH=.
accelerate launch --config_file accel_ds2.yaml \
  scripts/train_rm.py \
  --backbone_type llama3_baseline \
  --train_data_path data/ultrafeedback_train.jsonl \
  --eval_data_path data/ultrafeedback_val.jsonl \
  --output_dir experiments/debug \
  --batch_size 4 --grad_accum 8
```

单卡冒烟：`accelerate launch --config_file accel_1gpu.yaml scripts/train_rm.py ...`

也可直接读 parquet（与 baseline 相同，跳过 jsonl）：

```bash
--train_data /mnt/afs/250010036/reward_model/data/ultrafeedback_binarized/parquet/train_prefs-00000-of-00001.parquet \
--test_data  .../test_prefs-00000-of-00001.parquet
```
（不要同时传 `--train_data_path`）

## 环境依赖

- Python 3.10+
- PyTorch、`transformers`, `datasets`, `accelerate`, `deepspeed`
- 默认 **4 GPU + ZeRO-2 + bf16**（见 `ds_zero2.json`）
- 建议 `global_batch_size = num_gpus × batch_size × grad_accum ≥ 512`（与 baseline 一致）

## 数据与 Chat Template

**数据源**（与 baseline 相同）：

| 划分 | 来源 | 清洗 |
|------|------|------|
| 训练 | `data/ultrafeedback_binarized/parquet/train_prefs-*.parquet` | 是 |
| 验证 | `data/ultrafeedback_binarized/parquet/test_prefs-*.parquet` | 否 |

**训练集清洗规则**（`utils/ultrafeedback_clean.py`，默认与 `baseline/singe_head.py` 一致）：

1. `score_chosen - score_rejected >= 1.0`
2. `score_chosen >= 4.0`
3. 丢弃 `score_chosen == 10`（已知标注 bug，可用 `--no_drop_score10` 关闭）

生成 jsonl：

```bash
cd /mnt/afs/250010036/reward_model/latent_reward_model
PYTHONPATH=. python scripts/prepare_full_data.py
```

- 训练使用各 backbone tokenizer 内置的 **Llama 3 `chat_template`**。
- jsonl 中保留完整 `chosen` / `rejected` **messages 列表**。
- 仍兼容旧版 jsonl（`prompt` + 字符串），会自动构造成两轮对话。

**长度过滤**（`train_rm.py`，与 baseline 一致）：tokenize 后不截断；`chosen` / `rejected` 序列长度均需在 `[min_length, max_length]`（默认 **16–4096**），否则丢弃。可通过 `--min_length`、`--max_length` 调整。

## 实验日志（SwanLab）

与 `baseline/singe_head.py` 一致，使用 [SwanLab](https://swanlab.cn) 记录训练/验证指标。

| 参数 | 默认 |
|------|------|
| `--swanlab_project` / `SWANLAB_PROJECT` | `latentMRM` |
| `--swanlab_experiment_name` / `SWANLAB_EXPERIMENT_NAME` | `LatentRewardModel_<backbone>`（`run_train.sh` 按 backbone 自动生成） |
| `--swanlab_mode` / `SWANLAB_MODE` | `cloud`（可选 `local` / `disabled`） |
| `SWANLAB_API_KEY` | 环境变量注入，`run_train.sh` 不写死在脚本里 |

本地运行日志仍会写入 `output_dir/train_*.log` 与 `log.json`。

## 优化器（与 baseline 一致）

| 参数组 | 默认学习率 | 参数 |
|--------|------------|------|
| backbone | `1e-6` | `backbone` |
| head | `1e-4` | `reward_heads` + `selector` |

- 优化器：**AdamW**，`weight_decay=0.01`
- 调度：**cosine warmup**（`warmup_ratio=0.05`，`eta_min=0.01`，同 `baseline/singe_head.py`）

可通过 `--head_lr`、`--backbone_lr`、`--weight_decay`、`--warmup_ratio` 覆盖。

## 说明

- **armorm** 通过 `AutoModelForSequenceClassification` 加载，仅使用其内部 `LlamaModel` 作为 backbone，不沿用 ArmoRM 原有 gating / score head。
- Tokenizer 使用 **left padding**，与 baseline 一致，last-token 抽取兼容 left/right padding。
