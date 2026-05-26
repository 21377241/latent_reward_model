import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import load_backbone
from models.pooling import pool_last_hidden


class GatingNetwork(nn.Module):
    """将 K 维 head 分数加权聚合成标量 reward（对齐 multihead_baseline）。"""

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


class LatentRewardModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        backbone_type: str,
        k_dimensions: int = 4,
        torch_dtype: torch.dtype = torch.float32,
        use_gate: bool = False,
        gate_hidden_size: int = 1024,
        gate_num_layers: int = 3,
        gate_temperature: float = 10.0,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.use_gate = use_gate
        self.backbone, self.config = load_backbone(
            model_path, backbone_type, torch_dtype=torch_dtype
        )
        hidden_size = self.config.hidden_size
        self.k = k_dimensions

        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )
            for _ in range(self.k)
        ])

        self.selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.k),
        )
        self.reward_heads = self.reward_heads.to(torch_dtype)
        self.selector = self.selector.to(torch_dtype)

        if use_gate:
            self.gating_network = GatingNetwork(
                hidden_size=hidden_size,
                num_dims=self.k,
                gate_hidden_size=gate_hidden_size,
                gate_num_layers=gate_num_layers,
                gate_temperature=gate_temperature,
            ).to(torch_dtype)
        else:
            self.gating_network = None

    def aggregate_scores(
        self, hidden: torch.Tensor, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """hidden [B,H], scores [B,K] -> scalar [B], gate_weights [B,K]."""
        gate_w = self.gating_network(hidden)
        scalar = (gate_w * scores).sum(dim=-1)
        return scalar, gate_w

    def forward(
        self,
        input_ids_c,
        attention_mask_c,
        input_ids_r,
        attention_mask_r,
        detach_scores_for_gate: bool = False,
        **kwargs,
    ):
        out_c = self.backbone(
            input_ids=input_ids_c, attention_mask=attention_mask_c
        )
        out_r = self.backbone(
            input_ids=input_ids_r, attention_mask=attention_mask_r
        )

        h_c = pool_last_hidden(out_c.last_hidden_state, attention_mask_c)
        h_r = pool_last_hidden(out_r.last_hidden_state, attention_mask_r)

        scores_c = torch.cat([head(h_c) for head in self.reward_heads], dim=-1)
        scores_r = torch.cat([head(h_r) for head in self.reward_heads], dim=-1)

        combined_features = torch.cat([h_c, h_r], dim=-1)
        p_plus = torch.sigmoid(self.selector(combined_features))
        p_minus = 1.0 - p_plus
        relations = torch.stack([p_plus, p_minus], dim=-1)

        gated_c, gated_r, gate_w_c, gate_w_r = None, None, None, None
        if self.use_gate:
            sc = scores_c.detach() if detach_scores_for_gate else scores_c
            sr = scores_r.detach() if detach_scores_for_gate else scores_r
            gated_c, gate_w_c = self.aggregate_scores(h_c, sc)
            gated_r, gate_w_r = self.aggregate_scores(h_r, sr)

        return scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, gate_w_r
