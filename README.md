# Latent Reward Model

基于多潜在维度的奖励模型（Latent MRM）：使用共享 backbone、K 个 reward head 与 selector 网络，对 chosen/rejected 样本进行关系分配与 Bradley-Terry 风格训练。

## 项目结构

```
latent_reward_model/
├── models/latent_reward_model.py   # 模型定义
├── scripts/train_rm.py             # 训练入口
├── scripts/prepare_full_data.py    # 数据准备
├── utils/loss_functions.py           # 损失函数
├── utils/metrics.py                # 评估指标
└── run_train.sh                    # 训练示例脚本
```

## 环境依赖

- Python 3.10+
- PyTorch
- `transformers`, `datasets`, `accelerate`
- 可选：`wandb`（训练日志）

## 快速开始

1. 准备偏好数据（jsonl），放入 `data/` 目录（该目录已被 `.gitignore` 忽略）。
2. 修改 `run_train.sh` 中的模型路径、数据路径与超参。
3. 运行训练：

```bash
bash run_train.sh
```

或直接调用：

```bash
PYTHONPATH=. python scripts/train_rm.py --help
```

## 许可证

请根据项目需要自行添加 LICENSE。
