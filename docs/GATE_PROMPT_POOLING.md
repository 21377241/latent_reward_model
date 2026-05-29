# Gate Prompt-End Pooling

## 动机

默认 `gate_pooling_mode=sequence_end` 时，gate 使用 **整条序列末 token** 的 hidden（含 assistant 回复），权重可能随 response 内容「投机」变化。

`gate_pooling_mode=prompt_end` 时，gate **仅**使用 **prompt / instruction 末 token** 的 hidden，按问题类型选择评价维度，与具体回答内容解耦。

Reward head 仍使用序列末 token pooling（不变）。

## 实现要点

1. **数据**：`utils/dataloader.py` 在 tokenize 时记录 `prompt_len`（`prompt_token_length`），collate 得到 `prompt_end_pos`（left-pad 下标）。
2. **训练**：`LatentRewardModel.forward(..., prompt_end_pos=...)`；gate 从 `out_c.last_hidden_state[b, prompt_end_pos[b]]` 取向量。
3. **因果 LM**：整段 `prompt+response` 前向后，prompt 末位置 hidden 与只前向 prompt 一致。
4. **chosen / rejected**：同一 instruction 下 **共享** 一组 `gate_weights`（由 chosen 序列上的 prompt 末 hidden 计算）。

## 用法

```bash
# 两阶段阶段二示例
accelerate launch ... scripts/train_rm.py \
  --train_stage fixed_gate \
  --use_gate \
  --gate_pooling_mode prompt_end \
  ...
```

`run_train_two_stage_fixed_prefix.sh` 可通过追加参数启用：

```bash
GATE_POOLING_MODE=prompt_end bash run_train_two_stage_fixed_prefix.sh --gate_pooling_mode prompt_end
```

或在脚本 `EXTRA` 中默认传入（按需修改）。

## 评测（`modeling_latent_rm.py`）

`gate_pooling_mode=prompt_end` 时，forward 需传入：

```python
prompt_end_positions: LongTensor [B]   # 每条序列 prompt 末 token 下标
```

未传入时回退为 `sequence_end`（与旧 checkpoint 兼容）。

`latent_config.json` / `config.json` 字段：`gate_pooling_mode`: `"sequence_end"` | `"prompt_end"`。

## 与默认行为对比

| 项目 | `sequence_end`（默认） | `prompt_end` |
|------|------------------------|--------------|
| Head pooling | 序列末 token | 序列末 token |
| Gate 输入 hidden | chosen/rejected 各自序列末 | **prompt 末**（同一样本 c/r 共享 w） |
| 依赖 response | 是 | 否（仅 instruction） |
