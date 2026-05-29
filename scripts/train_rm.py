"""
Latent Reward Model 训练脚本（Accelerate + DeepSpeed ZeRO-2）

启动示例（在本项目根目录下）：
  cd /mnt/afs/250010036/reward_model/latent_reward_model
  accelerate launch --config_file accel_ds2.yaml \\
      scripts/train_rm.py \\
      --backbone_type llama3_baseline \\
      --output_dir ../latent_reward_model/experiments/test \\
      --batch_size 4 --grad_accum 8

或使用 run_train.sh。
"""

import sys
from pathlib import Path

# 从任意 cwd 启动时，需把 latent_reward_model 根目录加入 path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from utils.checkpoint import (
    build_latent_config,
    load_latent_ckpt,
    resolve_ckpt_dir,
    save_latent_ckpt,
)
from utils.export_for_eval import export_latent_ckpt
from utils.dataloader import build_loaders
from utils.loss_functions import (
    compute_fixed_prefix_loss,
    compute_latent_factor_loss,
)

FIXED_PREFIX_STAGES = ("fixed_latent", "fixed_joint", "fixed_gate")


def _uses_fixed_prefix(args) -> bool:
    return args.train_stage in FIXED_PREFIX_STAGES
from utils.optimizer import cosine_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def _setup_main_logging(log_file: str | None) -> None:
    """主进程：同时写终端与文件，避免多卡时只有文件、终端长时间无输出。"""
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file and not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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

    p.add_argument("--use_gate", action="store_true", help="启用 Gate 将 K 维分数聚合成标量")
    p.add_argument(
        "--train_stage",
        default="latent",
        choices=[
            "latent",
            "joint",
            "gate",
            "fixed_latent",
            "fixed_joint",
            "fixed_gate",
        ],
        help="latent/joint/gate=原 selector 方案；"
        "fixed_*=无 selector，维度 0..num_pos_heads-1 为 K+",
    )
    p.add_argument("--lambda_gate", type=float, default=1.0, help="gate BT 损失权重（joint/gate 阶段）")
    p.add_argument("--gate_hidden_size", type=int, default=1024)
    p.add_argument("--gate_num_layers", type=int, default=3)
    p.add_argument("--gate_temperature", type=float, default=10.0)
    p.add_argument("--gate_lr", type=float, default=0.0, help="0 表示与 head_lr 相同")

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
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_dir", default="./experiments/latent_mrm")
    p.add_argument(
        "--resume_from",
        default=None,
        help="阶段1 checkpoint 目录，如 experiments/xxx/best（含 model.safetensors + latent_heads.pt）",
    )
    p.add_argument(
        "--resume_load_gate",
        action="store_true",
        help="从 checkpoint 加载 gating_network（阶段2 gate 训练默认不加载，随机初始化 gate）",
    )
    p.add_argument(
        "--resume_strict_backbone",
        action="store_true",
        help="backbone load_state_dict strict=True（默认 False）",
    )
    p.add_argument(
        "--no_export_eval",
        action="store_true",
        help="保存 checkpoint 后不自动导出 eval_export（默认 best/final 会导出）",
    )
    p.add_argument("--run_name", default=None, help="本地 output 目录命名等，不用于 SwanLab 实验名")
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
    if args.train_stage == "gate" and not args.use_gate:
        p.error("train_stage=gate 需要同时指定 --use_gate")
    if args.train_stage == "fixed_gate" and not args.use_gate:
        p.error("train_stage=fixed_gate 需要同时指定 --use_gate")
    if args.train_stage in ("joint", "fixed_joint") and not args.use_gate:
        args.use_gate = True
    if args.train_stage in ("gate", "fixed_gate") and not args.resume_from:
        p.error(f"train_stage={args.train_stage} 必须指定 --resume_from")
    if args.num_pos_heads > args.k_dimensions:
        p.error(
            f"num_pos_heads({args.num_pos_heads}) 不能大于 k_dimensions({args.k_dimensions})"
        )
    args.use_selector = not _uses_fixed_prefix(args)
    args.pos_dim_mode = "selector" if args.use_selector else "fixed_prefix"
    return args


def _resume_load_gate_flag(args):
    """阶段2 默认不加载 gate；joint/latent 续训则尽量加载。"""
    if args.resume_load_gate:
        return True
    if args.train_stage in ("gate", "fixed_gate"):
        return False
    return None


def _loss_lambdas(args):
    if args.train_stage in ("gate", "fixed_gate"):
        return 0.0, args.lambda_gate
    if args.train_stage in ("joint", "fixed_joint"):
        return 1.0, args.lambda_gate
    return 1.0, 0.0


def _configure_requires_grad(model, args):
    """按 train_stage 冻结/解冻子模块（在 accelerator.prepare 之前调用）。"""
    train_latent = args.train_stage in (
        "latent",
        "joint",
        "fixed_latent",
        "fixed_joint",
    )
    train_gate = args.use_gate and args.train_stage in (
        "joint",
        "gate",
        "fixed_joint",
        "fixed_gate",
    )

    for p in model.backbone.parameters():
        p.requires_grad = train_latent
    for p in model.reward_heads.parameters():
        p.requires_grad = train_latent
    if model.selector is not None:
        for p in model.selector.parameters():
            p.requires_grad = train_latent and args.use_selector
    if model.use_gate and model.gating_network is not None:
        for p in model.gating_network.parameters():
            p.requires_grad = train_gate


def _forward_batch(model, batch, detach_scores_for_gate: bool):
    return model(
        input_ids_c=batch["input_ids_c"],
        attention_mask_c=batch["attention_mask_c"],
        input_ids_r=batch["input_ids_r"],
        attention_mask_r=batch["attention_mask_r"],
        detach_scores_for_gate=detach_scores_for_gate,
    )


def _make_latent_config(args, eval_acc_global=None):
    return build_latent_config(
        args,
        resume_from=args.resume_from,
        eval_acc_global=eval_acc_global,
    )


def _export_eval_ckpt(ckpt_dir: str) -> None:
    """导出 eval_export/<tag>/ 供 reward-bench 等加载。"""
    ckpt_path = Path(ckpt_dir)
    heads_ok = (ckpt_path / "latent_heads.pt").is_file()
    cfg_ok = (ckpt_path / "latent_config.json").is_file()
    if not heads_ok:
        logger.warning("[Export] 跳过 %s：无 latent_heads.pt", ckpt_dir)
        return
    if not cfg_ok:
        logger.info("[Export] %s 无 latent_config.json，将从权重推断", ckpt_dir)
    try:
        out_dir = export_latent_ckpt(ckpt_path, pkg_root=_ROOT)
        logger.info("[Export] eval → %s", out_dir)
    except Exception as e:
        logger.exception("[Export] 失败 %s: %s", ckpt_dir, e)


def _compute_loss(
    scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, args
):
    lambda_latent, lambda_gate = _loss_lambdas(args)
    if _uses_fixed_prefix(args):
        return compute_fixed_prefix_loss(
            scores_c,
            scores_r,
            num_pos_heads=args.num_pos_heads,
            lambda_neg=args.lambda_neg,
            gated_score_c=gated_c,
            gated_score_r=gated_r,
            lambda_gate=lambda_gate,
            lambda_latent=lambda_latent,
            gate_weights_c=gate_w_c,
        )
    return compute_latent_factor_loss(
        scores_c,
        scores_r,
        relations,
        lambda_neg=args.lambda_neg,
        beta_dir=args.beta_dir,
        target_tau=args.target_tau,
        num_pos_heads=args.num_pos_heads,
        gated_score_c=gated_c,
        gated_score_r=gated_r,
        lambda_gate=lambda_gate,
        lambda_latent=lambda_latent,
        gate_weights_c=gate_w_c,
    )


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
    detach_gate = args.train_stage in ("gate", "fixed_gate")

    try:
        for batch in eval_loader:
            scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, _ = _forward_batch(
                model, batch, detach_scores_for_gate=detach_gate
            )
            loss, _, _, _, stats = _compute_loss(
                scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, args
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

    log_file = None
    if accelerator.is_main_process:
        gbs = accelerator.num_processes * args.grad_accum * args.batch_size
        log_file = os.path.join(
            args.output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        _setup_main_logging(log_file)
        logger.info(vars(args))
        logger.info(
            "[Config] num_gpus=%s  global_batch_size=%s  %s",
            accelerator.num_processes,
            gbs,
            "OK" if gbs >= 512 else "WARNING: < 512，建议增大 grad_accum",
        )
        logger.info(
            "[Config] logging_steps=%d  eval_steps=%d  grad_accum=%d  "
            "（训练日志每 %d 个 optimizer step 打印一次）",
            args.logging_steps,
            args.eval_steps,
            args.grad_accum,
            args.logging_steps,
        )

    model_path = resolve_model_path(args.backbone_type, args.model_name_or_path)
    trust_remote_code = args.backbone_type == "armorm_baseline"

    tokenizer_path = model_path
    if args.resume_from:
        ckpt_dir = resolve_ckpt_dir(args.resume_from)
        if os.path.isfile(os.path.join(ckpt_dir, "tokenizer_config.json")):
            tokenizer_path = ckpt_dir
        if accelerator.is_main_process:
            logger.info("[Resume] checkpoint 目录: %s", ckpt_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, trust_remote_code=trust_remote_code
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if accelerator.is_main_process:
        logger.info("[Data] 开始加载/清洗/tokenize 数据集（仅主进程，可能较久）…")
    with accelerator.main_process_first():
        train_loader, eval_loader = build_loaders(args, tokenizer)

    if accelerator.is_main_process:
        logger.info("[Model] 加载 backbone: %s", model_path)
    model = LatentRewardModel(
        model_path=model_path,
        backbone_type=args.backbone_type,
        k_dimensions=args.k_dimensions,
        torch_dtype=torch.bfloat16,
        use_gate=args.use_gate,
        use_selector=args.use_selector,
        gate_hidden_size=args.gate_hidden_size,
        gate_num_layers=args.gate_num_layers,
        gate_temperature=args.gate_temperature,
    )
    _configure_requires_grad(model, args)

    resume_info = None
    if args.resume_from:
        resume_info = load_latent_ckpt(
            model,
            args.resume_from,
            load_gate=_resume_load_gate_flag(args),
            strict_backbone=args.resume_strict_backbone,
        )

    if accelerator.is_main_process:
        logger.info(
            "[Model] use_gate=%s  use_selector=%s  pos_dim_mode=%s  "
            "train_stage=%s  k=%d  k+_prefix=%d  resume=%s",
            args.use_gate,
            args.use_selector,
            args.pos_dim_mode,
            args.train_stage,
            args.k_dimensions,
            args.num_pos_heads,
            args.resume_from or "none",
        )
        if resume_info is not None:
            logger.info("[Resume] meta=%s", resume_info.get("meta"))

    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    param_groups = []
    train_latent = args.train_stage in (
        "latent",
        "joint",
        "fixed_latent",
        "fixed_joint",
    )
    train_gate = args.use_gate and args.train_stage in (
        "joint",
        "gate",
        "fixed_joint",
        "fixed_gate",
    )
    if train_latent:
        head_params = list(model.reward_heads.parameters())
        if model.selector is not None:
            head_params += list(model.selector.parameters())
        param_groups.append(
            {"params": list(model.backbone.parameters()), "lr": args.backbone_lr}
        )
        param_groups.append({"params": head_params, "lr": args.head_lr})
    if train_gate:
        gate_lr = args.gate_lr if args.gate_lr > 0 else args.head_lr
        param_groups.append(
            {"params": list(model.gating_network.parameters()), "lr": gate_lr}
        )
    if not param_groups:
        raise RuntimeError("无待训练参数，请检查 --use_gate 与 --train_stage")

    if accelerator.is_main_process:
        group_names = []
        if train_latent:
            head_label = "heads+selector" if model.selector is not None else "heads"
            group_names.extend(["backbone", head_label])
        if train_gate:
            group_names.append("gating_network")
        logger.info(
            "[Optim] 可训练参数组: %s  lrs=%s",
            group_names,
            [g["lr"] for g in param_groups],
        )
        if args.train_stage in ("gate", "fixed_gate"):
            frozen = "backbone / heads" + (" / selector" if model.selector else "")
            logger.info(
                "[Optim] train_stage=%s：%s 已冻结，仅 gate_lr 生效",
                args.train_stage,
                frozen,
            )
        if _uses_fixed_prefix(args) and args.train_stage == "fixed_latent":
            logger.info(
                "[Optim] fixed_latent：K+ = 维度 [0, %d)，无 selector",
                args.num_pos_heads,
            )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = cosine_warmup(optimizer, total_steps, warmup_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    if accelerator.is_main_process:
        logger.info(
            "[Train] 开始训练  epochs=%d  steps/epoch≈%d  total_steps≈%d  "
            "micro_batches/epoch=%d",
            args.num_epochs,
            steps_per_epoch,
            total_steps,
            len(train_loader),
        )
        sys.stdout.flush()

    swan_run = swan_init(args) if accelerator.is_main_process else None
    global_step = 0
    if resume_info and resume_info.get("meta"):
        global_step = int(resume_info["meta"].get("step", 0))
    best_acc = 0.0
    log_history = []

    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            logger.info("[Train] ===== Epoch %d/%d =====", epoch + 1, args.num_epochs)
            sys.stdout.flush()
        model.train()
        detach_gate = args.train_stage in ("gate", "fixed_gate")
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, _ = (
                    _forward_batch(model, batch, detach_scores_for_gate=detach_gate)
                )
                loss, l_heads, l_sel, l_gate, stats = _compute_loss(
                    scores_c, scores_r, relations, gated_c, gated_r, gate_w_c, args
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if accelerator.is_main_process and (
                        global_step == 1
                        or global_step % args.logging_steps == 0
                    ):
                        lrs = scheduler.get_last_lr()
                        train_metrics = {
                            "train/loss": loss.item(),
                            "train/L_heads": l_heads.item(),
                            "train/L_selector": l_sel.item(),
                            "train/L_gate": l_gate.item(),
                            "train/lr_backbone": lrs[0] if len(lrs) > 1 else lrs[0],
                            "train/lr_head": lrs[-1],
                        }
                        for k, v in stats.items():
                            train_metrics[f"train/{k}"] = v
                        logger.info(
                            "[E%d S%d] loss=%.4f  L_h=%.3f  L_h_mean=%.3f  L_sel=%.4f  "
                            "L_gate=%.4f  acc=%.3f  wrong_k+=%.3f  "
                            "head_r|c|Δ=%.3f/%.3f/%.3f",
                            epoch + 1,
                            global_step,
                            train_metrics["train/loss"],
                            train_metrics["train/L_heads"],
                            train_metrics.get("train/loss/L_heads_mean", 0.0),
                            train_metrics["train/L_selector"],
                            train_metrics["train/L_gate"],
                            train_metrics.get("train/metrics/accuracy", 0.0),
                            train_metrics.get("train/diag/frac_wrong_kplus", 0.0),
                            train_metrics.get("train/diag/head_corr_r_mean", 0.0),
                            train_metrics.get("train/diag/head_corr_c_mean", 0.0),
                            train_metrics.get("train/diag/head_corr_diff_mean", 0.0),
                        )
                        sys.stdout.flush()
                        swan_log(swan_run, train_metrics, global_step)

                    if global_step % args.eval_steps == 0:
                        eval_stats = evaluate(model, eval_loader, accelerator, args)
                        if accelerator.is_main_process:
                            acc = eval_stats.get("eval/acc_global", eval_stats.get("eval/metrics_accuracy", 0.0))
                            logger.info(
                                "  ► eval loss=%.4f  L_h_mean=%.3f  acc=%.3f  "
                                "wrong_k+=%.3f  head_r|c|Δ=%.3f/%.3f/%.3f",
                                eval_stats.get("eval/loss", 0.0),
                                eval_stats.get("eval/loss_L_heads_mean", 0.0),
                                acc,
                                eval_stats.get("eval/diag/frac_wrong_kplus", 0.0),
                                eval_stats.get("eval/diag/mean_delta_kplus", 0.0),
                                eval_stats.get("eval/diag/head_corr_r_mean", 0.0),
                                eval_stats.get("eval/diag/head_corr_c_mean", 0.0),
                                eval_stats.get("eval/diag/head_corr_diff_mean", 0.0),
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
                                    latent_config=_make_latent_config(
                                        args, eval_acc_global=best_acc
                                    ),
                                    eval_acc_global=best_acc,
                                )
                                logger.info("  ★ new best acc_global=%.4f", best_acc)
                                if not args.no_export_eval:
                                    _export_eval_ckpt(
                                        os.path.join(args.output_dir, "best")
                                    )

                    if global_step % args.save_steps == 0 and accelerator.is_main_process:
                        save_latent_ckpt(
                            accelerator,
                            model,
                            scheduler,
                            global_step,
                            f"step_{global_step}",
                            args.output_dir,
                            tokenizer,
                            latent_config=_make_latent_config(args),
                        )

    if accelerator.is_main_process:
        final_stats = evaluate(model, eval_loader, accelerator, args)
        save_latent_ckpt(
            accelerator,
            model,
            scheduler,
            global_step,
            "final",
            args.output_dir,
            tokenizer,
            latent_config=_make_latent_config(args, eval_acc_global=best_acc),
            eval_acc_global=best_acc,
        )
        summary = {"best_acc_global": best_acc, **final_stats}
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("[Done] best_acc_global=%.4f", best_acc)
        if not args.no_export_eval:
            best_dir = os.path.join(args.output_dir, "best")
            if os.path.isdir(best_dir):
                _export_eval_ckpt(best_dir)
            else:
                _export_eval_ckpt(os.path.join(args.output_dir, "final"))
        swan_log(swan_run, {"eval/final_acc_global": best_acc}, global_step)
        swan_finish(swan_run)


if __name__ == "__main__":
    main()
