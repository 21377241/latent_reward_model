"""Accelerate + DeepSpeed 兼容的 checkpoint 保存。"""

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


def save_latent_ckpt(accelerator, model, scheduler, step, tag, output_dir, tokenizer=None):
    """
    保存 LatentRewardModel checkpoint：
      {output_dir}/{tag}/
        config.json
        model.safetensors      ← backbone
        latent_heads.pt        ← reward_heads + selector
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
            if k.startswith("reward_heads.") or k.startswith("selector.")
        }

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.config.save_pretrained(save_dir)

        try:
            from safetensors.torch import save_file
            save_file(backbone_sd, os.path.join(save_dir, "model.safetensors"))
        except ImportError:
            torch.save(backbone_sd, os.path.join(save_dir, "pytorch_model.bin"))

        torch.save(heads_sd, os.path.join(save_dir, "latent_heads.pt"))

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)

        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"step": step, "tag": tag}, f)

        logger.info("[Ckpt] saved '%s' → %s  step=%s", tag, save_dir, step)
