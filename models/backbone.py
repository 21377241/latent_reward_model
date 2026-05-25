"""本地 backbone 路径与加载逻辑（与 baseline / multihead_baseline 对齐）。"""

import gc
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification

MODEL_ROOT = os.environ.get("REWARD_MODEL_ROOT", "/mnt/afs/250010036/reward_model")

BACKBONE_PATHS = {
    "llama3_baseline": os.path.join(MODEL_ROOT, "model", "llama-3-8b-instruct"),
    "llama3.1_baseline": os.path.join(MODEL_ROOT, "model", "llama-3.1-8b-instruct"),
    "armorm_baseline": os.path.join(MODEL_ROOT, "model", "armorm"),
}

BACKBONE_CHOICES = tuple(BACKBONE_PATHS.keys())


def resolve_model_path(
    backbone_type: str,
    model_name_or_path: Optional[str] = None,
) -> str:
    """解析模型路径：显式传入 model_name_or_path 时优先使用，否则按 backbone_type 查表。"""
    if model_name_or_path:
        return model_name_or_path
    if backbone_type not in BACKBONE_PATHS:
        raise ValueError(
            f"未知 backbone_type={backbone_type!r}，可选: {list(BACKBONE_CHOICES)}"
        )
    path = BACKBONE_PATHS[backbone_type]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"本地模型目录不存在: {path}")
    return path


def load_backbone(
    model_path: str,
    backbone_type: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, AutoConfig]:
    """
    加载 backbone：
    - llama3 / llama3.1：AutoModelForCausalLM → .model (LlamaModel)
    - armorm：AutoModelForSequenceClassification (trust_remote_code) → .model
    """
    if backbone_type == "armorm_baseline":
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    elif backbone_type in ("llama3_baseline", "llama3.1_baseline"):
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(f"不支持的 backbone_type: {backbone_type}")

    backbone = base.model
    config = base.config
    del base
    gc.collect()
    return backbone, config
