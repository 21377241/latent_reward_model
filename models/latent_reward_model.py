import torch
import torch.nn as nn
from transformers import AutoModel

class LatentRewardModel(nn.Module):
    def __init__(self, model_name, k_dimensions=4,torch_dtype=torch.float32):
        super().__init__()
        # 加载基础模型 (如 Qwen2.5-0.5B)
        self.backbone = AutoModel.from_pretrained(model_name,dtype=torch_dtype)
        self.config = self.backbone.config
        hidden_size = self.backbone.config.hidden_size
        self.k = k_dimensions
        
        # 定义 K 个独立的潜在评价维度头
        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(self.k)
        ])
        
        # 定义关系分配网络 (Selector)
        # 输入维度是 hidden_size * 2，因为我们要拼接 chosen 和 rejected 的特征
        self.selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.k) 
        )
        self.reward_heads = self.reward_heads.to(torch_dtype)
        self.selector = self.selector.to(torch_dtype)

    def forward(self, input_ids_c, attention_mask_c, input_ids_r, attention_mask_r,**kwargs):
        # 1. 过 Backbone 提取特征
        out_c = self.backbone(input_ids=input_ids_c, attention_mask=attention_mask_c)
        
        out_r = self.backbone(input_ids=input_ids_r, attention_mask=attention_mask_r)
        
        batch_size = input_ids_c.shape[0]
        device = out_c.last_hidden_state.device
        
        # 2. 正确提取特征：取最后一个有效 token，而不是 mean pooling
        seq_lengths_c = attention_mask_c.sum(dim=1) - 1
        seq_lengths_r = attention_mask_r.sum(dim=1) - 1
        h_c = out_c.last_hidden_state[torch.arange(batch_size, device=device), seq_lengths_c]
        h_r = out_r.last_hidden_state[torch.arange(batch_size, device=device), seq_lengths_r]
        
        # 2. 计算 K 个维度的分数 (z_k)
        scores_c = torch.cat([head(h_c) for head in self.reward_heads], dim=-1) # [batch, K]
        scores_r = torch.cat([head(h_r) for head in self.reward_heads], dim=-1) # [batch, K]
        
        # 3. 关系分配网络预测 (p+)预测K个维度中属于正向维度的概率
        combined_features = torch.cat([h_c, h_r], dim=-1)
        p_plus = torch.sigmoid(self.selector(combined_features)) # [batch, K]
        p_minus = 1.0 - p_plus # [batch, K]
        relations = torch.stack([p_plus, p_minus], dim=-1) # [batch, K, 2]
        
        return scores_c, scores_r, relations
