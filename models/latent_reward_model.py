import torch
import torch.nn as nn

from models.backbone import load_backbone
from models.pooling import pool_last_hidden


class LatentRewardModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        backbone_type: str,
        k_dimensions: int = 4,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.backbone_type = backbone_type
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

    def forward(
        self,
        input_ids_c,
        attention_mask_c,
        input_ids_r,
        attention_mask_r,
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

        return scores_c, scores_r, relations
