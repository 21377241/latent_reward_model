#!/usr/bin/env python3
"""
从 checkpoint 目录加载 LatentRewardModel（训练同款 forward，用于偏好对推理）。

用法:
  PYTHONPATH=. python scripts/load_latent_rm.py /path/to/experiments/xxx/best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.backbone import resolve_model_path
from models.latent_reward_model import LatentRewardModel
from utils.checkpoint import load_latent_ckpt, resolve_ckpt_dir


def load_from_ckpt(ckpt_dir: str, device: str = "cuda") -> LatentRewardModel:
    ckpt_dir = resolve_ckpt_dir(ckpt_dir)
    cfg_path = Path(ckpt_dir) / "latent_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"需要 latent_config.json: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        lcfg = json.load(f)

    model_path = resolve_model_path(lcfg["backbone_type"])
    model = LatentRewardModel(
        model_path=model_path,
        backbone_type=lcfg["backbone_type"],
        k_dimensions=lcfg["k_dimensions"],
        torch_dtype=torch.bfloat16,
        use_gate=lcfg.get("use_gate", False),
        use_selector=lcfg.get("use_selector", True),
        gate_hidden_size=lcfg.get("gate_hidden_size", 1024),
        gate_num_layers=lcfg.get("gate_num_layers", 3),
        gate_temperature=lcfg.get("gate_temperature", 10.0),
        gate_pooling_mode=lcfg.get("gate_pooling_mode", "sequence_end"),
    )
    load_latent_ckpt(model, ckpt_dir, load_gate=lcfg.get("use_gate", True))
    model.eval()
    return model.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="含 latent_config.json 的 checkpoint 目录")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    model = load_from_ckpt(args.ckpt_dir, args.device)
    print(
        f"loaded k={model.k} use_gate={model.use_gate} "
        f"gate_pooling={model.gate_pooling_mode} "
        f"from {resolve_ckpt_dir(args.ckpt_dir)}"
    )


if __name__ == "__main__":
    main()
