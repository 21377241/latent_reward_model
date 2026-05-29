#!/usr/bin/env python3
"""
CLI：将 LatentRewardModel checkpoint 导出为评测格式。

用法:
  cd latent_reward_model
  PYTHONPATH=. python scripts/export_for_eval.py experiments/xxx/best
  PYTHONPATH=. python scripts/export_for_eval.py experiments/xxx/best \\
      --out_dir experiments/xxx/eval_export/best
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.export_for_eval import export_latent_ckpt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="导出 Latent RM 评测目录")
    parser.add_argument(
        "ckpt_tag_dir",
        type=Path,
        help="checkpoint 目录（best/final）",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="导出目录（默认: <parent>/eval_export/<tag>）",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default=None,
        help="覆盖聚合方式，如 heads_mean（K 维等权平均）",
    )
    args = parser.parse_args()
    export_latent_ckpt(
        args.ckpt_tag_dir, args.out_dir, score_mode=args.score_mode
    )


if __name__ == "__main__":
    main()
