"""
HuggingFace 兼容的 Latent Reward Model（单序列 → 标量 logits）。

供 reward-bench / RM-bench 等通过 AutoModelForSequenceClassification 加载。
权重目录需含：model.safetensors、latent_heads.pt、latent_config.json、本文件。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class GatingNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_dims: int,
        gate_hidden_size: int = 1024,
        gate_num_layers: int = 3,
        gate_temperature: float = 10.0,
    ):
        super().__init__()
        self.temperature = gate_temperature
        layers: list[nn.Module] = []
        in_dim = hidden_size
        for _ in range(gate_num_layers):
            layers.append(nn.Linear(in_dim, gate_hidden_size))
            in_dim = gate_hidden_size
        layers.append(nn.Linear(in_dim, num_dims))
        self.layers = nn.ModuleList(layers)
        last = self.layers[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.to(self.layers[0].weight.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return F.softmax(x / self.temperature, dim=-1)


def _pool_last_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    seq_len = attention_mask.size(1)
    flipped = attention_mask.flip(1).long()
    last_idx = seq_len - 1 - flipped.argmax(1)
    valid = flipped.max(1).values.bool()
    last_idx = torch.where(
        valid,
        last_idx,
        torch.full_like(last_idx, seq_len - 1),
    )
    batch_size = last_hidden_state.size(0)
    device = last_hidden_state.device
    return last_hidden_state[torch.arange(batch_size, device=device), last_idx]


class LlamaForLatentRewardModel(LlamaPreTrainedModel):
    """Latent MRM：K 个 MLP head + 可选 gate 聚合为标量 reward。"""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        hidden_size = config.hidden_size
        self.k_dimensions = int(getattr(config, "k_dimensions", 8))
        self.use_gate = bool(getattr(config, "use_gate", True))
        gate_hidden_size = int(getattr(config, "gate_hidden_size", 1024))
        gate_num_layers = int(getattr(config, "gate_num_layers", 3))
        gate_temperature = float(getattr(config, "gate_temperature", 10.0))
        self.gate_pooling_mode = getattr(config, "gate_pooling_mode", "sequence_end")
        # gated_scalar | heads_sum | heads_mean（K 维等权平均，即 sum/K）
        self.score_mode = getattr(config, "score_mode", "gated_scalar")
        self.num_pos_heads = int(getattr(config, "num_pos_heads", self.k_dimensions))
        self.use_selector = bool(getattr(config, "use_selector", True))
        self.pos_dim_mode = getattr(config, "pos_dim_mode", "selector")

        dtype = torch.bfloat16
        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(self.k_dimensions)
        ])
        for head in self.reward_heads:
            head.to(dtype)

        if self.use_selector:
            self.selector = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.k_dimensions),
            ).to(dtype)
        else:
            self.selector = None

        if self.use_gate:
            self.gating_network = GatingNetwork(
                hidden_size=hidden_size,
                num_dims=self.k_dimensions,
                gate_hidden_size=gate_hidden_size,
                gate_num_layers=gate_num_layers,
                gate_temperature=gate_temperature,
            )
        else:
            self.gating_network = None

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        ckpt_dir = str(pretrained_model_name_or_path)
        heads_file = os.path.join(ckpt_dir, "latent_heads.pt")
        latent_cfg_file = os.path.join(ckpt_dir, "latent_config.json")

        if os.path.isfile(latent_cfg_file):
            with open(latent_cfg_file, encoding="utf-8") as f:
                lcfg = json.load(f)
            model.use_gate = bool(lcfg.get("use_gate", model.use_gate))
            model.score_mode = lcfg.get("score_mode", model.score_mode)
            model.num_pos_heads = int(
                lcfg.get("num_pos_heads", model.num_pos_heads)
            )
            model.use_selector = bool(lcfg.get("use_selector", model.use_selector))
            model.pos_dim_mode = lcfg.get("pos_dim_mode", model.pos_dim_mode)
            model.gate_pooling_mode = lcfg.get(
                "gate_pooling_mode", getattr(model, "gate_pooling_mode", "sequence_end")
            )
            model.score_mode = lcfg.get("score_mode", model.score_mode)

        if os.path.isfile(heads_file):
            head_state = torch.load(heads_file, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(head_state, strict=False)
            if missing:
                print(
                    f"[LatentRM] latent_heads missing keys: {len(missing)} "
                    f"(first: {missing[:3]})"
                )
            if unexpected:
                print(f"[LatentRM] latent_heads unexpected keys: {len(unexpected)}")

        return model

    def _scalar_reward(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor,
        gate_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = getattr(self, "score_mode", "gated_scalar")
        if mode == "heads_mean":
            return scores.mean(dim=-1)
        if mode == "heads_sum":
            return scores.sum(dim=-1)
        # 兼容旧配置名；语义同 heads_sum（全部维求和）
        if mode == "fixed_prefix_sum":
            mode = "heads_sum"
        if self.use_gate and self.gating_network is not None:
            h_gate = gate_hidden if gate_hidden is not None else hidden
            gate_w = self.gating_network(h_gate)
            return (gate_w * scores).sum(dim=-1)
        # 兼容旧导出：无 gate 时默认各 head 求和
        return scores.sum(dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[tuple, list]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = next(self.parameters()).device
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden.shape[:2], dtype=torch.long, device=hidden.device
            )

        last_h = _pool_last_hidden(hidden, attention_mask)
        head_dtype = self.reward_heads[0][0].weight.dtype
        last_h = last_h.to(head_dtype)

        gate_hidden = None
        prompt_end_positions = kwargs.get("prompt_end_positions")
        if (
            self.use_gate
            and self.gating_network is not None
            and getattr(self, "gate_pooling_mode", "sequence_end") == "prompt_end"
            and prompt_end_positions is not None
        ):
            pos = prompt_end_positions.to(hidden.device, dtype=torch.long)
            batch_size = hidden.size(0)
            pos = pos.clamp(min=0, max=hidden.size(1) - 1)
            gate_hidden = hidden[torch.arange(batch_size, device=hidden.device), pos]
            gate_hidden = gate_hidden.to(head_dtype)

        scores = torch.cat([head(last_h) for head in self.reward_heads], dim=-1).float()
        reward = self._scalar_reward(last_h, scores, gate_hidden=gate_hidden)
        logits = reward.unsqueeze(-1)

        if not return_dict:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
