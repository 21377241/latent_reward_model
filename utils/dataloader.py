"""数据加载：tokenize、长度过滤、left-pad collate（与 baseline 对齐）。"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.chat_format import build_conversation, encode_conversation, passes_length_filter
from utils.ultrafeedback_clean import UltrafeedbackCleanConfig, clean_ultrafeedback_dataset

logger = logging.getLogger(__name__)

MODEL_ROOT = os.environ.get("REWARD_MODEL_ROOT", "/mnt/afs/250010036/reward_model")
DEFAULT_TRAIN_PARQUET = os.path.join(
    MODEL_ROOT, "data/ultrafeedback_binarized/parquet/train_prefs-00000-of-00001.parquet"
)
DEFAULT_TEST_PARQUET = os.path.join(
    MODEL_ROOT, "data/ultrafeedback_binarized/parquet/test_prefs-00000-of-00001.parquet"
)


def _load_raw_dataset(args):
    if getattr(args, "train_data_path", None):
        logger.info("加载 jsonl: train=%s eval=%s", args.train_data_path, args.eval_data_path)
        train_ds = load_dataset("json", data_files=args.train_data_path, split="train")
        eval_path = args.eval_data_path or args.train_data_path
        eval_ds = load_dataset("json", data_files=eval_path, split="train")
        return train_ds, eval_ds, False

    train_path = args.train_data or DEFAULT_TRAIN_PARQUET
    test_path = args.test_data or DEFAULT_TEST_PARQUET
    logger.info("加载 parquet: train=%s test=%s", train_path, test_path)
    train_ds = load_dataset("parquet", data_files=train_path, split="train")
    test_ds = load_dataset("parquet", data_files=test_path, split="train")
    return train_ds, test_ds, True


def _tokenize_dataset(ds, tokenizer, max_length: int, min_length: int, split: str):
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer 未配置 chat_template")

    def tokenize_batch(examples):
        n = len(examples["chosen"])
        prompts = examples.get("prompt", [""] * n)
        out = {"c_ids": [], "c_mask": [], "r_ids": [], "r_mask": []}
        for i in range(n):
            conv_c = build_conversation(examples["chosen"][i], prompts[i])
            conv_r = build_conversation(examples["rejected"][i], prompts[i])
            tok_c = encode_conversation(tokenizer, conv_c)
            tok_r = encode_conversation(tokenizer, conv_r)
            out["c_ids"].append(tok_c["input_ids"])
            out["c_mask"].append(tok_c["attention_mask"])
            out["r_ids"].append(tok_r["input_ids"])
            out["r_mask"].append(tok_r["attention_mask"])
        return out

    ds = ds.map(tokenize_batch, batched=True, num_proc=4, desc=f"tokenize {split}")
    before = len(ds)
    ds = ds.filter(
        lambda ex: passes_length_filter(
            len(ex["c_ids"]), len(ex["r_ids"]), min_length, max_length
        ),
        num_proc=4,
        desc=f"length filter {split}",
    )
    logger.info(
        "[Data] %s: %d → %d  length-dropped=%d  rules=[%d≤len≤%d]",
        split, before, len(ds), before - len(ds), min_length, max_length,
    )
    return ds


def build_loaders(args, tokenizer) -> Tuple[DataLoader, DataLoader]:
    train_ds, eval_ds, has_scores = _load_raw_dataset(args)

    if has_scores:
        clean_cfg = UltrafeedbackCleanConfig(
            min_score_margin=args.min_score_margin,
            min_chosen_score=args.min_chosen_score,
            drop_score10=args.drop_score10,
        )
        train_ds, msg = clean_ultrafeedback_dataset(train_ds, clean_cfg, split="train")
        logger.info(msg)

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    train_ds = _tokenize_dataset(
        train_ds, tokenizer, args.max_length, args.min_length, "train"
    )
    eval_ds = _tokenize_dataset(
        eval_ds, tokenizer, args.max_length, args.min_length, "eval"
    )
    logger.info("[Data] final  train=%d  eval=%d", len(train_ds), len(eval_ds))

    pad_id = tokenizer.pad_token_id

    def collate(batch):
        def pad_left(seqs):
            max_len = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
                mask[i, max_len - len(s):] = 1
            return ids, mask

        c_ids, c_mask = pad_left([x["c_ids"] for x in batch])
        r_ids, r_mask = pad_left([x["r_ids"] for x in batch])
        return {
            "input_ids_c": c_ids,
            "attention_mask_c": c_mask,
            "input_ids_r": r_ids,
            "attention_mask_r": r_mask,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size * 2
    eval_loader = DataLoader(
        eval_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    return train_loader, eval_loader
