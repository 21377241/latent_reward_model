"""
导出 latent_reward_model 训练用 jsonl。

数据与清洗逻辑与 baseline/singe_head.py 对齐：
  - 训练：train_prefs parquet + 三条 score 过滤
  - 验证：test_prefs parquet，不做 score 过滤
"""

import argparse
import json
import os

from datasets import load_dataset

from utils.ultrafeedback_clean import UltrafeedbackCleanConfig, clean_ultrafeedback_dataset

MODEL_ROOT = os.environ.get("REWARD_MODEL_ROOT", "/mnt/afs/250010036/reward_model")
DEFAULT_TRAIN_PARQUET = os.path.join(
    MODEL_ROOT,
    "data/ultrafeedback_binarized/parquet/train_prefs-00000-of-00001.parquet",
)
DEFAULT_TEST_PARQUET = os.path.join(
    MODEL_ROOT,
    "data/ultrafeedback_binarized/parquet/test_prefs-00000-of-00001.parquet",
)


def load_prefs_split(parquet_path: str, hf_split: str):
    """优先本地 parquet，否则从 HuggingFace 拉取对应 split。"""
    if os.path.isfile(parquet_path):
        print(f"加载本地 parquet: {parquet_path}")
        return load_dataset("parquet", data_files=parquet_path, split="train")
    print(f"本地文件不存在 ({parquet_path})，从 HF 加载 split={hf_split}")
    return load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=hf_split)


def to_jsonl_record(item: dict) -> dict:
    return {
        "prompt": item["prompt"].strip(),
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    }


def save_jsonl(dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(to_jsonl_record(item), ensure_ascii=False) + "\n")


def prepare_ultrafeedback_binarized(
    train_parquet: str = DEFAULT_TRAIN_PARQUET,
    test_parquet: str = DEFAULT_TEST_PARQUET,
    train_output_path: str = "data/ultrafeedback_train.jsonl",
    val_output_path: str = "data/ultrafeedback_val.jsonl",
    min_score_margin: float = 1.0,
    min_chosen_score: float = 4.0,
    drop_score10: bool = True,
    num_proc: int = 4,
):
    clean_cfg = UltrafeedbackCleanConfig(
        min_score_margin=min_score_margin,
        min_chosen_score=min_chosen_score,
        drop_score10=drop_score10,
    )

    train_ds = load_prefs_split(train_parquet, "train_prefs")
    test_ds = load_prefs_split(test_parquet, "test_prefs")

    train_ds, clean_log = clean_ultrafeedback_dataset(
        train_ds, clean_cfg, split="train", num_proc=num_proc
    )
    print(clean_log)
    print(f"[Data] eval (test_prefs): {len(test_ds)} samples, no score filtering")

    save_jsonl(train_ds, train_output_path)
    save_jsonl(test_ds, val_output_path)

    print(f"训练集已保存: {train_output_path} ({len(train_ds)} 条)")
    print(f"验证集已保存: {val_output_path} ({len(test_ds)} 条)")

    if len(train_ds) > 0:
        sample = to_jsonl_record(train_ds[0])
        print("\n训练集示例（首条）:")
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:2000])

    return train_ds, test_ds


def parse_args():
    p = argparse.ArgumentParser(description="准备 UltraFeedback Binarized jsonl（清洗逻辑同 baseline）")
    p.add_argument("--train_parquet", default=DEFAULT_TRAIN_PARQUET)
    p.add_argument("--test_parquet", default=DEFAULT_TEST_PARQUET)
    p.add_argument("--train_output", default="data/ultrafeedback_train.jsonl")
    p.add_argument("--val_output", default="data/ultrafeedback_val.jsonl")
    p.add_argument("--min_score_margin", type=float, default=1.0,
                   help="score_chosen - score_rejected 下限，0 表示不过滤")
    p.add_argument("--min_chosen_score", type=float, default=4.0,
                   help="score_chosen 下限，0 表示不过滤")
    p.add_argument("--no_drop_score10", action="store_true",
                   help="保留 score_chosen==10 的样本（baseline 默认会丢弃）")
    p.add_argument("--num_proc", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_ultrafeedback_binarized(
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        train_output_path=args.train_output,
        val_output_path=args.val_output,
        min_score_margin=args.min_score_margin,
        min_chosen_score=args.min_chosen_score,
        drop_score10=not args.no_drop_score10,
        num_proc=args.num_proc,
    )
