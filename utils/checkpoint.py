"""Accelerate + DeepSpeed 兼容的 checkpoint 保存与加载。"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CKPT_BACKBONE_SAFE = "model.safetensors"
CKPT_BACKBONE_BIN = "pytorch_model.bin"
CKPT_HEADS = "latent_heads.pt"
CKPT_META = "meta.json"
CKPT_LATENT_CONFIG = "latent_config.json"


def build_latent_config(args, resume_from: str | None = None, eval_acc_global: float | None = None) -> Dict[str, Any]:
    """从训练 args 生成 latent_config.json 内容（评测/导出必读）。"""
    use_gate = getattr(args, "use_gate", False)
    cfg: Dict[str, Any] = {
        "model_type": "latent_reward_model",
        "backbone_type": args.backbone_type,
        "k_dimensions": args.k_dimensions,
        "num_pos_heads": args.num_pos_heads,
        "use_gate": use_gate,
        "gate_hidden_size": getattr(args, "gate_hidden_size", 1024),
        "gate_num_layers": getattr(args, "gate_num_layers", 3),
        "gate_temperature": getattr(args, "gate_temperature", 10.0),
        "train_stage": getattr(args, "train_stage", "latent"),
        "lambda_neg": getattr(args, "lambda_neg", 1.0),
        "score_mode": "gated_scalar" if use_gate else "heads_sum",
    }
    if resume_from:
        cfg["resume_from"] = os.path.abspath(os.path.expanduser(resume_from))
    if eval_acc_global is not None:
        cfg["eval_acc_global"] = float(eval_acc_global)
    return cfg


def resolve_ckpt_dir(resume_from: str) -> str:
    """
    解析 checkpoint 目录。支持：
      - 绝对/相对路径到含 latent_heads.pt 的目录（如 .../experiments/xxx/best）
      - output_dir + 子目录名（若 resume_from 为 best/final/step_N）
    """
    path = os.path.abspath(os.path.expanduser(resume_from))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"checkpoint 目录不存在: {path}")

    has_heads = os.path.isfile(os.path.join(path, CKPT_HEADS))
    has_backbone = os.path.isfile(os.path.join(path, CKPT_BACKBONE_SAFE)) or os.path.isfile(
        os.path.join(path, CKPT_BACKBONE_BIN)
    )
    if not has_heads and not has_backbone:
        raise FileNotFoundError(
            f"目录中未找到 {CKPT_HEADS} 或 backbone 权重: {path}"
        )
    return path


def _load_backbone_state_dict(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    safe_path = os.path.join(ckpt_dir, CKPT_BACKBONE_SAFE)
    bin_path = os.path.join(ckpt_dir, CKPT_BACKBONE_BIN)
    if os.path.isfile(safe_path):
        try:
            from safetensors.torch import load_file

            return load_file(safe_path)
        except ImportError as e:
            raise ImportError("加载 model.safetensors 需要 pip install safetensors") from e
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"未找到 backbone 权重: {safe_path} 或 {bin_path}")


def load_latent_ckpt(
    model: nn.Module,
    resume_from: str,
    *,
    load_backbone: bool = True,
    load_heads: bool = True,
    load_gate: Optional[bool] = None,
    strict_backbone: bool = False,
) -> Dict[str, Any]:
    """
    将 checkpoint 载入未 wrap 的 LatentRewardModel（须在 accelerator.prepare 之前调用）。

    Args:
        load_gate: None=若 ckpt 含 gating_network 且模型 use_gate 则加载；
                   True/False 强制是否加载 gate 权重。
    Returns:
        meta 信息（step、tag、missing/unexpected keys 统计等）
    """
    ckpt_dir = resolve_ckpt_dir(resume_from)
    info: Dict[str, Any] = {"ckpt_dir": ckpt_dir}

    meta_path = os.path.join(ckpt_dir, CKPT_META)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            info["meta"] = json.load(f)

    if load_backbone:
        backbone_sd = _load_backbone_state_dict(ckpt_dir)
        missing, unexpected = model.backbone.load_state_dict(
            backbone_sd, strict=strict_backbone
        )
        info["backbone_missing"] = list(missing)
        info["backbone_unexpected"] = list(unexpected)
        logger.info(
            "[Resume] backbone ← %s  missing=%d unexpected=%d",
            ckpt_dir,
            len(missing),
            len(unexpected),
        )

    if load_heads:
        heads_path = os.path.join(ckpt_dir, CKPT_HEADS)
        if not os.path.isfile(heads_path):
            raise FileNotFoundError(f"未找到 {heads_path}")
        heads_sd: Dict[str, torch.Tensor] = torch.load(
            heads_path, map_location="cpu", weights_only=True
        )

        use_gate = getattr(model, "use_gate", False)
        has_gate_in_ckpt = any(k.startswith("gating_network.") for k in heads_sd)
        if load_gate is None:
            load_gate = use_gate and has_gate_in_ckpt
        if not load_gate:
            heads_sd = {
                k: v for k, v in heads_sd.items() if not k.startswith("gating_network.")
            }
            if use_gate and not has_gate_in_ckpt:
                logger.info("[Resume] checkpoint 无 gate 权重，gating_network 保持随机初始化")

        missing, unexpected = model.load_state_dict(heads_sd, strict=False)
        info["heads_missing"] = list(missing)
        info["heads_unexpected"] = list(unexpected)
        info["loaded_gate"] = load_gate and has_gate_in_ckpt
        logger.info(
            "[Resume] latent_heads ← %s  load_gate=%s  missing=%d unexpected=%d",
            heads_path,
            info["loaded_gate"],
            len(missing),
            len(unexpected),
        )

    return info


def save_latent_ckpt(
    accelerator,
    model,
    scheduler,
    step,
    tag,
    output_dir,
    tokenizer=None,
    latent_config: Optional[Dict[str, Any]] = None,
    eval_acc_global: Optional[float] = None,
):
    """
    保存 LatentRewardModel checkpoint：
      {output_dir}/{tag}/
        config.json
        model.safetensors      ← backbone
        latent_heads.pt        ← reward_heads + selector + gating_network（若启用）
        tokenizer*
        scheduler.pt
        meta.json
    """
    save_dir = os.path.join(output_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        backbone_sd = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        heads_sd = {
            k: v
            for k, v in state_dict.items()
            if k.startswith(("reward_heads.", "selector.", "gating_network."))
        }

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.config.save_pretrained(save_dir)

        try:
            from safetensors.torch import load_file  # noqa: F401
            from safetensors.torch import save_file

            save_file(backbone_sd, os.path.join(save_dir, CKPT_BACKBONE_SAFE))
        except ImportError:
            torch.save(backbone_sd, os.path.join(save_dir, CKPT_BACKBONE_BIN))

        torch.save(heads_sd, os.path.join(save_dir, CKPT_HEADS))

        if latent_config is not None:
            with open(os.path.join(save_dir, CKPT_LATENT_CONFIG), "w", encoding="utf-8") as f:
                json.dump(latent_config, f, indent=2)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        meta = {"step": step, "tag": tag}
        if latent_config is not None:
            meta.update(
                {
                    k: latent_config[k]
                    for k in (
                        "backbone_type",
                        "k_dimensions",
                        "use_gate",
                        "train_stage",
                        "score_mode",
                    )
                    if k in latent_config
                }
            )
        if eval_acc_global is not None:
            meta["eval_acc_global"] = float(eval_acc_global)
        with open(os.path.join(save_dir, CKPT_META), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("[Ckpt] saved '%s' → %s  step=%s", tag, save_dir, step)
