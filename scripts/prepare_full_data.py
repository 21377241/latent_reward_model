import json
import random
from datasets import load_dataset
from collections import defaultdict

def prepare_ultrafeedback_binarized(val_ratio=0.1, seed=42, margin_threshold=1.0,
    train_output_path="data/ultrafeedback_train.jsonl",
    val_output_path="data/ultrafeedback_val.jsonl"):
    print("正在加载 UltraFeedback Binarized 数据集...")

    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs"
    )

    pairwise_data = []

    for item in dataset:

        # prompt
        prompt = item["prompt"].strip()
        chosen = item["chosen"][-1]["content"].strip()
        rejected = item["rejected"][-1]["content"].strip()

        # score
        score_chosen = float(item["score_chosen"])
        score_rejected = float(item["score_rejected"])

        # margin
        margin = score_chosen - score_rejected

        # 过滤低质量 pair
        if margin < margin_threshold:
            continue

        pairwise_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    # 随机打乱
    random.shuffle(pairwise_data)

    # 划分 train / val
    split_idx = int(len(pairwise_data) * (1 - val_ratio))

    train_data = pairwise_data[:split_idx]
    val_data = pairwise_data[split_idx:]

    # 保存 jsonl
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    save_jsonl(train_data, train_output_path)
    save_jsonl(val_data, val_output_path)

    print(f"总样本数: {len(pairwise_data)}")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")

    # 打印一个样本看看
    if len(train_data) > 0:
        print("\n示例数据:")
        print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

    return train_data, val_data

def prepare_full_helpsteer(val_ratio=0.1, seed=42):
    random.seed(seed)
    print("正在加载全量 ultrafeedback数据集...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    # 1. 按 prompt 对 response 进行分组
    prompt_groups = defaultdict(list)
    for item in dataset:
        # 计算综合分用于构建 BT Pair
        total_score = (item['helpfulness'] + item['correctness'] + 
                       item['coherence'] + item['complexity'] + item['verbosity'])
        prompt_groups[item['prompt']].append({
            'response': item['response'],
            'score': total_score
        })

    pairwise_data = []
    for prompt, responses in prompt_groups.items():
        if len(responses) < 2:
            continue
            
        # 排序并选取分数最高的为 chosen，最低的为 rejected
        sorted_res = sorted(responses, key=lambda x: x['score'], reverse=True)
        chosen = sorted_res[0]
        rejected = sorted_res[-1]
        
        # 仅保留有明显分差的样本以保证监督信号质量
        if chosen['score'] - rejected['score'] > 1:
            pairwise_data.append({
                "prompt": prompt,
                "chosen": chosen['response'],
                "rejected": rejected['response']
            })

    # 2. 随机打乱并切分
    random.shuffle(pairwise_data)
    split_idx = int(len(pairwise_data) * (1 - val_ratio))
    train_data = pairwise_data[:split_idx]
    val_data = pairwise_data[split_idx:]

    # 3. 保存文件
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    save_jsonl(train_data, "data/helpsteer_full_train.jsonl")
    save_jsonl(val_data, "data/helpsteer_full_val.jsonl")
            
    print(f"训练集: {len(train_data)} 条 | 验证集: {len(val_data)} 条")

if __name__ == "__main__":
    # prepare_full_helpsteer()
    prepare_ultrafeedback_binarized()
