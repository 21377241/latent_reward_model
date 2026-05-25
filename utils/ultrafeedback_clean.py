"""
ultrafeedback_binarized 数据清洗逻辑（与 baseline/singe_head.py、multihead_baseline 对齐）。
"""

from dataclasses import dataclass
from typing import List, Tuple

from datasets import Dataset


@dataclass
class UltrafeedbackCleanConfig:
    """与 baseline 默认参数一致。"""

    min_score_margin: float = 1.0
    min_chosen_score: float = 4.0
    drop_score10: bool = True


def passes_ultrafeedback_filters(
    item: dict,
    cfg: UltrafeedbackCleanConfig,
) -> bool:
    """
    单条样本是否保留（训练集用）：
      1. score_chosen - score_rejected >= min_score_margin
      2. score_chosen >= min_chosen_score
      3. 默认丢弃 score_chosen == 10（标注 bug）
    """
    score_chosen = float(item["score_chosen"])
    score_rejected = float(item["score_rejected"])

    if cfg.min_score_margin > 0:
        if (score_chosen - score_rejected) < cfg.min_score_margin:
            return False

    if cfg.min_chosen_score > 0:
        if score_chosen < cfg.min_chosen_score:
            return False

    if cfg.drop_score10:
        if score_chosen >= 10.0:
            return False

    return True


def _active_rules(cfg: UltrafeedbackCleanConfig) -> List[str]:
    rules = []
    if cfg.min_score_margin > 0:
        rules.append(f"margin≥{cfg.min_score_margin}")
    if cfg.min_chosen_score > 0:
        rules.append(f"chosen≥{cfg.min_chosen_score}")
    if cfg.drop_score10:
        rules.append("drop_score10")
    return rules


def clean_ultrafeedback_dataset(
    dataset: Dataset,
    cfg: UltrafeedbackCleanConfig,
    split: str = "train",
    num_proc: int = 4,
) -> Tuple[Dataset, str]:
    """
    对 HuggingFace Dataset 做清洗。
    仅 split=='train' 时过滤；验证/测试集保持完整（与 baseline 一致）。
    返回 (dataset, log_message)。
    """
    if split != "train":
        return dataset, f"[Clean] {split}: skip (no filtering)"

    before = len(dataset)
    reasons = _active_rules(cfg)

    if cfg.min_score_margin > 0:
        dataset = dataset.filter(
            lambda ex: (ex["score_chosen"] - ex["score_rejected"]) >= cfg.min_score_margin,
            num_proc=num_proc,
            desc=f"margin >= {cfg.min_score_margin}",
        )

    if cfg.min_chosen_score > 0:
        dataset = dataset.filter(
            lambda ex: ex["score_chosen"] >= cfg.min_chosen_score,
            num_proc=num_proc,
            desc=f"score_chosen >= {cfg.min_chosen_score}",
        )

    if cfg.drop_score10:
        dataset = dataset.filter(
            lambda ex: ex["score_chosen"] < 10.0,
            num_proc=num_proc,
            desc="drop score_chosen==10 (annotation bug)",
        )

    msg = (
        f"[Clean] train: {before} → {len(dataset)}  "
        f"dropped={before - len(dataset)}  "
        f"rules=[{', '.join(reasons) or 'none'}]"
    )
    return dataset, msg
