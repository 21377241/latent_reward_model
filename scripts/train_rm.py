import torch
import os
import wandb
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments
from datasets import load_dataset

from models.latent_reward_model import LatentRewardModel
from utils.loss_functions import compute_latent_factor_loss
from utils.metrics import get_compute_metrics_fn

# ==========================================
# 1. 定义自定义参数类 (加入 max_length)
# ==========================================
@dataclass
class ScriptArguments:
    """
    除了 TrainingArguments 自带的训练参数外，定义我们特有的模型与数据参数
    """
    model_name_or_path: str = field(default="Qwen/Qwen2.5-0.5B", metadata={"help": "基础模型的路径或名称"})
    train_data_path: str = field(default="data/dummy_preference.json", metadata={"help": "训练集路径"})
    eval_data_path: Optional[str] = field(default=None, metadata={"help": "验证集路径，如果不传则使用训练集"})
    max_length: int = field(default=1024, metadata={"help": "文本截断最大长度"})
    
    k_dimensions: int = field(default=4, metadata={"help": "潜在评价维度的数量 K"})
    lambda_neg: float = field(default=1.0, metadata={"help": "反向维度损失的权重 \lambda_-"})

    num_pos_heads: int = field(default=4, metadata={"help": "每对样本中被划入正向集合的维度数量"})
    target_tau: float = field(default=0.5, metadata={"help": "样本级正向维度的最小比例阈值 \tau"})
    beta_dir: float = field(default=1.0, metadata={"help": "方向性惩罚 L_dir 的权重"})

# ==========================================
# 2. 自定义 Data Collator (处理 Pairwise 动态 Padding)
# ==========================================
class PairwiseDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分离 chosen 和 rejected 的特征
        chosen_features = [{"input_ids": f["input_ids_chosen"], "attention_mask": f["attention_mask_chosen"]} for f in features]
        rejected_features = [{"input_ids": f["input_ids_rejected"], "attention_mask": f["attention_mask_rejected"]} for f in features]

        # 利用 tokenizer 分别进行动态 padding
        batch_chosen = self.tokenizer.pad(chosen_features, return_tensors="pt")
        batch_rejected = self.tokenizer.pad(rejected_features, return_tensors="pt")

        # 拼接成模型 forward 所需的格式
        return {
            "input_ids_c": batch_chosen["input_ids"],
            "attention_mask_c": batch_chosen["attention_mask"],
            "input_ids_r": batch_rejected["input_ids"],
            "attention_mask_r": batch_rejected["attention_mask"],
            "labels": torch.zeros(len(features), dtype=torch.long)
        }

# ==========================================
# 3. 继承基础 Trainer
# ==========================================
class LatentFactorRewardTrainer(Trainer):
    def __init__(self, lambda_neg=1.0, beta_dir=0.1,target_tau=0.5,num_pos_heads=4,**kwargs):
        super().__init__(**kwargs)
        self.lambda_neg = lambda_neg
        self.beta_dir = beta_dir
        self.target_tau=target_tau
        self.num_pos_heads=num_pos_heads

    def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None,
    ):
        with torch.no_grad():
            scores_c, scores_r, relations = model(
                input_ids_c=inputs["input_ids_c"],
                attention_mask_c=inputs["attention_mask_c"],
                input_ids_r=inputs["input_ids_r"],
                attention_mask_r=inputs["attention_mask_r"]
            )

            loss, _, _, _ = compute_latent_factor_loss(
                scores_c,
                scores_r,
                relations,
                lambda_neg=self.lambda_neg,
                beta_dir=self.beta_dir,
                target_tau=self.target_tau
            )

        logits = (
            scores_c.detach(),
            scores_r.detach(),
            relations.detach(),
        )

        labels = inputs["labels"]
        return (loss.detach(), logits, labels)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs 字典的 key 已经由 data_collator 对齐到了模型的 forward 参数名
        scores_c, scores_r, relations = model(
            input_ids_c=inputs["input_ids_c"],
            attention_mask_c=inputs["attention_mask_c"],
            input_ids_r=inputs["input_ids_r"],
            attention_mask_r=inputs["attention_mask_r"]
        )

        loss,L_heads, L_selector, stats = compute_latent_factor_loss(
            scores_c, scores_r, relations, 
            lambda_neg=self.lambda_neg, 
            beta_dir=self.beta_dir,
            target_tau=self.target_tau
        )

        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "loss/L_heads": L_heads.item(),
                "loss/L_selector": L_selector.item(),
                "loss/total": loss.item(),
            }
            log_dict.update(stats)
            self.log(log_dict)

        return loss


# ==========================================
# 4. 主入口逻辑
# ==========================================
if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments)) # 替换为 TrainingArguments
    script_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.eval_strategy == "no":
        # 如果你没在命令行传参数，默认让它每个 epoch 验证一次，或者设置 steps
        training_args.eval_strategy = "steps"
        training_args.eval_steps = 100 # 每 100 步在验证集上跑一次

    training_args.report_to = ["wandb"]
    training_args.remove_unused_columns = False
    training_args.prediction_loss_only = False
    if training_args.run_name is None:
        # 如果你运行 bash 脚本时忘了加 --run_name，就给它一个动态的默认后备名称
        model_short = script_args.model_name_or_path.split("/")[-1]
        training_args.run_name = f"{model_short}-K{script_args.k_dimensions}-lamNeg{script_args.lambda_neg}"


    # 必须设为 False，否则基础 Trainer 会把不属于模型基础 forward 参数列的纯文本给丢弃，导致 tokenize 失败
    training_args.remove_unused_columns = False 

    print(f" 初始化 Tokenizer 和模型: {script_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    target_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    
    model = LatentRewardModel(
        model_name=script_args.model_name_or_path, 
        k_dimensions=script_args.k_dimensions,
        torch_dtype=target_dtype
    )

    custom_config = dataclasses.asdict(script_args)
    wandb.init(
        project="latentMRM", 
        name=training_args.run_name, 
        config=custom_config 
    )

    print(f"加载数据集: {script_args.train_data_path}")
    train_dataset = load_dataset("json", data_files=script_args.train_data_path, split="train")
    if script_args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=script_args.eval_data_path, split="train")
    else:
        print("未提供 eval_data_path，使用 train_dataset 作为 eval_dataset")
        eval_dataset = train_dataset

    # --- 将纯文本转换为 token_ids ---
    def preprocess_function(examples):
        text_chosen = [f"User: {p}\n\nAssistant: {c}" for p, c in zip(examples["prompt"], examples["chosen"])]
        text_rejected = [f"User: {p}\n\nAssistant: {r}" for p, r in zip(examples["prompt"], examples["rejected"])]

        # 这里不加 padding，由 collator 在 batch 级别动态 padding 节省显存
        tokenized_chosen = tokenizer(text_chosen, truncation=True, max_length=script_args.max_length)
        tokenized_rejected = tokenizer(text_rejected, truncation=True, max_length=script_args.max_length)

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    print("正在对数据集进行 Tokenize 处理...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=4)

    trainer = LatentFactorRewardTrainer(
        lambda_neg=script_args.lambda_neg,
        beta_dir=script_args.beta_dir,
        target_tau=script_args.target_tau,
        num_pos_heads=script_args.num_pos_heads,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairwiseDataCollator(tokenizer=tokenizer) ,# 传入自定义的批处理逻辑
        compute_metrics=get_compute_metrics_fn(script_args.lambda_neg)
    )

    trainer.train()

    wandb.finish()
