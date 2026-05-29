"""将 LatentRewardModel checkpoint 导出为 reward-bench / RM-bench 可加载格式。"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file, save_file

from utils.checkpoint import CKPT_BACKBONE_BIN, CKPT_BACKBONE_SAFE, CKPT_HEADS, CKPT_META

logger = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parents[1]


def _maybe_fix_pad_token(ckpt_tag_dir: Path, out_dir: Path) -> Optional[int]:
    src_tok_cfg_path = ckpt_tag_dir / "tokenizer_config.json"
    if not src_tok_cfg_path.exists():
        return None

    src_tok_cfg = json.loads(src_tok_cfg_path.read_text())
    pad_tok = src_tok_cfg.get("pad_token")
    eos_tok = src_tok_cfg.get("eos_token")
    if not (pad_tok and eos_tok and pad_tok == eos_tok):
        return None

    used_in_conv = {128000, 128001, 128006, 128007, 128008, 128009}
    atd = src_tok_cfg.get("added_tokens_decoder", {})
    new_pad_id = None
    new_pad_token = None
    for k, v in sorted(atd.items(), key=lambda x: int(x[0])):
        kid = int(k)
        if kid not in used_in_conv and v.get("special", False):
            new_pad_id = kid
            new_pad_token = v["content"]
            break
    if new_pad_id is None:
        return None

    tok_cfg_path = out_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
        tok_cfg["padding_side"] = "left"
        tok_cfg["pad_token"] = new_pad_token
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

    stm_path = out_dir / "special_tokens_map.json"
    if stm_path.exists():
        stm = json.loads(stm_path.read_text())
        if "pad_token" in stm:
            if isinstance(stm["pad_token"], dict):
                stm["pad_token"]["content"] = new_pad_token
            else:
                stm["pad_token"] = new_pad_token
            stm_path.write_text(json.dumps(stm, indent=2) + "\n")

    logger.info(
        "[export] pad_token fix: '%s' -> '%s' (id %s)",
        pad_tok,
        new_pad_token,
        new_pad_id,
    )
    return new_pad_id


def _copy_tokenizer_files(ckpt_tag_dir: Path, out_dir: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        src = ckpt_tag_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    tok_cfg_path = out_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
        tok_cfg["padding_side"] = "left"
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")


def _infer_latent_config(heads_path: Path) -> dict:
    head_keys = list(
        torch.load(str(heads_path), map_location="cpu", weights_only=True).keys()
    )
    k_indices = [
        int(k.split(".")[1])
        for k in head_keys
        if k.startswith("reward_heads.") and k.split(".")[1].isdigit()
    ]
    k_dim = max(k_indices) + 1 if k_indices else 8
    use_gate = any(k.startswith("gating_network.") for k in head_keys)
    return {
        "model_type": "latent_reward_model",
        "k_dimensions": k_dim,
        "num_pos_heads": k_dim,
        "use_gate": use_gate,
        "gate_hidden_size": 1024,
        "gate_num_layers": 3,
        "gate_temperature": 10.0,
        "train_stage": "unknown",
        "score_mode": "gated_scalar" if use_gate else "heads_sum",
    }


def export_latent_ckpt(
    ckpt_tag_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    pkg_root: Optional[Path] = None,
    score_mode: Optional[str] = None,
) -> Path:
    """
    导出评测目录。

    Args:
        ckpt_tag_dir: 含 model.safetensors + latent_heads.pt 的 checkpoint 目录
        out_dir: 输出目录；默认 <ckpt_parent>/eval_export/<tag>/
        pkg_root: modeling_latent_rm.py 所在项目根（默认 latent_reward_model/）

    Returns:
        导出目录 Path
    """
    ckpt_tag_dir = Path(ckpt_tag_dir).resolve()
    if out_dir is None:
        out_dir = ckpt_tag_dir.parent / "eval_export" / ckpt_tag_dir.name
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pkg_root = (pkg_root or _PKG_ROOT).resolve()

    latent_cfg_path = ckpt_tag_dir / "latent_config.json"
    heads_path = ckpt_tag_dir / CKPT_HEADS

    if latent_cfg_path.exists():
        latent_cfg = json.loads(latent_cfg_path.read_text())
    else:
        logger.warning(
            "[export] 无 latent_config.json，从 %s 推断结构",
            CKPT_HEADS,
        )
        if not heads_path.exists():
            raise FileNotFoundError(f"{CKPT_HEADS} not found: {heads_path}")
        latent_cfg = _infer_latent_config(heads_path)

    if score_mode:
        latent_cfg["score_mode"] = score_mode

    src_cfg_path = ckpt_tag_dir / "config.json"
    if not src_cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {src_cfg_path}")
    cfg = json.loads(src_cfg_path.read_text())

    if not heads_path.exists():
        raise FileNotFoundError(f"{CKPT_HEADS} not found: {heads_path}")

    src_st = ckpt_tag_dir / CKPT_BACKBONE_SAFE
    src_bin = ckpt_tag_dir / CKPT_BACKBONE_BIN
    if src_st.exists():
        raw_state = load_file(str(src_st))
        first_key = next(iter(raw_state))
        if first_key.startswith("model."):
            state = dict(raw_state)
        else:
            state = {f"model.{k}": v for k, v in raw_state.items()}
        save_file(state, str(out_dir / CKPT_BACKBONE_SAFE))
        logger.info("[export] backbone tensors: %d", len(state))
    elif src_bin.exists():
        state = torch.load(str(src_bin), map_location="cpu", weights_only=True)
        first_key = next(iter(state))
        if not first_key.startswith("model."):
            state = {f"model.{k}": v for k, v in state.items()}
        torch.save(state, out_dir / CKPT_BACKBONE_BIN)
        logger.info("[export] backbone tensors(bin): %d", len(state))
    else:
        raise FileNotFoundError(f"backbone weights not found in {ckpt_tag_dir}")

    shutil.copy2(heads_path, out_dir / CKPT_HEADS)
    (out_dir / "latent_config.json").write_text(
        json.dumps(latent_cfg, indent=2) + "\n"
    )

    modeling_src = pkg_root / "modeling_latent_rm.py"
    if not modeling_src.exists():
        raise FileNotFoundError(f"modeling_latent_rm.py not found: {modeling_src}")
    shutil.copy2(modeling_src, out_dir / "modeling_latent_rm.py")

    _copy_tokenizer_files(ckpt_tag_dir, out_dir)
    new_pad_id = _maybe_fix_pad_token(ckpt_tag_dir, out_dir)

    for key in (
        "k_dimensions",
        "num_pos_heads",
        "use_gate",
        "gate_hidden_size",
        "gate_num_layers",
        "gate_temperature",
        "gate_pooling_mode",
        "backbone_type",
        "score_mode",
        "train_stage",
    ):
        if key in latent_cfg:
            cfg[key] = latent_cfg[key]

    cfg["architectures"] = ["LlamaForLatentRewardModel"]
    cfg["num_labels"] = 1
    cfg["id2label"] = {"0": "LABEL_0"}
    cfg["label2id"] = {"LABEL_0": 0}
    cfg["use_cache"] = False
    cfg["reward_model_type"] = "latent_reward_model"
    cfg["auto_map"] = {
        "AutoModelForSequenceClassification": (
            "modeling_latent_rm.LlamaForLatentRewardModel"
        )
    }
    if new_pad_id is not None:
        cfg["pad_token_id"] = new_pad_id

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    meta = {}
    meta_f = ckpt_tag_dir / CKPT_META
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())

    done = {
        "source": str(ckpt_tag_dir),
        "format": "LlamaForLatentRewardModel",
        "train_meta": meta,
        "latent_config": latent_cfg,
    }
    (out_dir / "export_done.json").write_text(json.dumps(done, indent=2) + "\n")
    logger.info("[export] done -> %s", out_dir)
    return out_dir

# 兼容旧名称
export = export_latent_ckpt
