"""
Latent Reward Model 训练脚本（Accelerate + DeepSpeed ZeRO-2）

启动示例（在仓库根目录下，复用 solution1/accel_ds2.yaml）：
  cd /mnt/afs/250010036/reward_model/solution1
  accelerate launch --config_file accel_ds2.yaml \\
      ../latent_reward_model/scripts/train_rm.py \\
      --backbone_type llama3_baseline \\
      --output_dir ../latent_reward_model/experiments/test \\
      --batch_size 4 --grad_accum 8

或使用 run_train.sh。
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from models.backbone import BACKBONE_CHOICES, resolve_model_path
from models.latent_reward_model import LatentRewardModel
from utils.checkpoint import save_latent_ckpt
from utils.dataloader import build_loaders
from utils.loss_functions import compute_latent_factor_loss
from utils.optimizer import cosine_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import swanlab
    _SWANLAB = True
except ImportError:
    _SWANLAB = False


def get_args():
    p = argparse.ArgumentParser(description="Latent MRM (Accelerate + ZeRO-2)")

    p.add_argument("--backbone_type", default="llama3_baseline", choices=list(BACKBONE_CHOICES))
    p.add_argument("--model_name_or_path", default=None)

    p.add_argument("--train_data_path", default=None, help="jsonl 训练集（prepare_full_data 输出）")
    p.add_argument("--eval_data_path", default=None, help="jsonl 验证集")
    p.add_argument("--train_data", default=None, help="parquet 训练集（与 baseline 相同）")
    p.add_argument("--test_data", default=None, help="parquet 验证集")

    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--min_length", type=int, default=16)
    p.add_argument("--min_score_margin", type=float, default=1.0)
    p.add_argument("--min_chosen_score", type=float, default=4.0)
    p.add_argument("--no_drop_score10", action="store_true")

    p.add_argument("--k_dimensions", type=int, default=8)
    p.add_argument("--lambda_neg", type=float, default=1.0)
    p.add_argument("--beta_dir", type=float, default=0.2)
    p.add_argument("--target_tau", type=float, default=0.8)
    p.add_argument("--num_pos_heads", type=int, default=5)

    p.add_argument("--head_lr", type=float, default=1e-5)
    p.add_argument("--backbone_lr", type=float, default=1e-7)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", default="./experiments/latent_mrm")
    p.add_argument("--run_name", default=None, help="本地输出目录命名等，不用于 SwanLab 实验名")
    p.add_argument("--swanlab_experiment_name", default="LatentRewardModel")
    p.add_argument("--swanlab_project", default="latentMRM")
    p.add_argument(
        "--swanlab_mode",
        default="cloud",
        choices=["cloud", "local", "disabled"],
        help="与 baseline 一致；disabled 时不记录 SwanLab",
    )

    args = p.parse_args()
    args.drop_score10 = not args.no_drop_score10
    return args


def swan_init(args):
    if not _SWANLAB or args.swanlab_mode == "disabled":
        return None
    return swanlab.init(
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment_name,
        config=vars(args),
        mode=args.swanlab_mode,
    )


def swan_log(run, metrics, step):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        logger.warning("[SwanLab] %s", e)


def swan_finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


@torch.no_grad()
def evaluate(model, eval_loader, accelerator, args):
    model.eval()
    loss_sum = 0.0
    metric_sums = {}
    n_batches = 0

    try:
        for batch in eval_loader:
            scores_c, scores_r, relations = model(
                input_ids_c=batch["input_ids_c"],
                attention_mask_c=batch["attention_mask_c"],
                input_ids_r=batch["input_ids_r"],
                attention_mask_r=batch["attention_mask_r"],
            )
            loss, _, _, stats = compute_latent_factor_loss(
                scores_c,
                scores_r,
                relations,
                lambda_neg=args.lambda_neg,
                beta_dir=args.beta_dir,
                target_tau=args.target_tau,
                num_pos_heads=args.num_pos_heads,
            )
            loss_sum += loss.item()
            for k, v in stats.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            n_batches += 1
    finally:
        model.train()

    n_batches = max(n_batches, 1)
    out = {"eval/loss": loss_sum / n_batches}
    for k, v in metric_sums.items():
        out[f"eval/{k.replace('/', '_')}"] = v / n_batches
    if "eval/metrics_accuracy" in out:
        out["eval/acc_global"] = out["eval/metrics_accuracy"]
    return out


def main():
    args = get_args()
    os.environ["HF_HUB_OFFLINE"] = "1"

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        gbs = accelerator.num_processes * args.grad_accum * args.batch_size
        log_file = os.path.join(
            args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
        logger.info(vars(args))
        logger.info(
            "[Config] num_gpus=%s  global_batch_size=%s  %s",
            accelerator.num_processes,
            gbs,
            "OK" if gbs >= 512 else "WARNING: < 512，建议增大 grad_accum",
        )

    model_path = resolve_model_path(args.backbone_type, args.model_name_or_path)
    trust_remote_code = args.backbone_type == "armorm_baseline"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with accelerator.main_process_first():
        train_loader, eval_loader = build_loaders(args, tokenizer)

    model = LatentRewardModel(
        model_path=model_path,
        backbone_type=args.backbone_type,
        k_dimensions=args.k_dimensions,
        torch_dtype=torch.bfloat16,
    )
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    head_params = list(model.reward_heads.parameters()) + list(model.selector.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone.parameters()), "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = cosine_warmup(optimizer, total_steps, warmup_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    swan_run = swan_init(args) if accelerator.is_main_process else None
    global_step = 0
    best_acc = 0.0
    log_history = []

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                scores_c, scores_r, relations = model(
                    input_ids_c=batch["input_ids_c"],
                    attention_mask_c=batch["attention_mask_c"],
                    input_ids_r=batch["input_ids_r"],
                    attention_mask_r=batch["attention_mask_r"],
                )
                loss, l_heads, l_sel, stats = compute_latent_factor_loss(
                    scores_c,
                    scores_r,
                    relations,
                    lambda_neg=args.lambda_neg,
                    beta_dir=args.beta_dir,
                    target_tau=args.target_tau,
                    num_pos_heads=args.num_pos_heads,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (
                        accelerator.is_main_process
                        and global_step % args.logging_steps == 0
                    ):
                        lrs = scheduler.get_last_lr()
                        train_metrics = {
                            "train/loss": loss.item(),
                            "train/L_heads": l_heads.item(),
                            "train/L_selector": l_sel.item(),
                            "train/lr_backbone": lrs[0],
                            "train/lr_head": lrs[-1],
                        }
                        for k, v in stats.items():
                            train_metrics[f"train/{k}"] = v
                        logger.info(
                            "[E%d S%d] loss=%.4f  lr_bb=%.2e  lr_h=%.2e",
                            epoch + 1,
                            global_step,
                            train_metrics["train/loss"],
                            lrs[0],
                            lrs[-1],
                        )
                        swan_log(swan_run, train_metrics, global_step)

                    if global_step % args.eval_steps == 0:
                        eval_stats = evaluate(model, eval_loader, accelerator, args)
                        if accelerator.is_main_process:
                            acc = eval_stats.get("eval/acc_global", eval_stats.get("eval/metrics_accuracy", 0.0))
                            logger.info(
                                "  ► eval loss=%.4f  acc_global=%.4f",
                                eval_stats.get("eval/loss", 0.0),
                                acc,
                            )
                            log_history.append({"step": global_step, **eval_stats})
                            with open(os.path.join(args.output_dir, "log.json"), "w", encoding="utf-8") as f:
                                json.dump(log_history, f, indent=2)
                            swan_log(swan_run, eval_stats, global_step)

                            if acc > best_acc:
                                best_acc = acc
                                save_latent_ckpt(
                                    accelerator,
                                    model,
                                    scheduler,
                                    global_step,
                                    "best",
                                    args.output_dir,
                                    tokenizer,
                                )
                                logger.info("  ★ new best acc_global=%.4f", best_acc)

                    if global_step % args.save_steps == 0 and accelerator.is_main_process:
                        save_latent_ckpt(
                            accelerator,
                            model,
                            scheduler,
                            global_step,
                            f"step_{global_step}",
                            args.output_dir,
                            tokenizer,
                        )

    if accelerator.is_main_process:
        final_stats = evaluate(model, eval_loader, accelerator, args)
        save_latent_ckpt(
            accelerator, model, scheduler, global_step, "final", args.output_dir, tokenizer
        )
        summary = {"best_acc_global": best_acc, **final_stats}
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("[Done] best_acc_global=%.4f", best_acc)
        swan_log(swan_run, {"eval/final_acc_global": best_acc}, global_step)
        swan_finish(swan_run)


if __name__ == "__main__":
    main()
