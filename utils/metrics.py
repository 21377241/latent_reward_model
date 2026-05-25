import numpy as np
from transformers import EvalPrediction

def get_compute_metrics_fn(lambda_neg):
    """
    返回一个带有 lambda_neg 上下文的 metrics 计算函数
    """
    def compute_latent_metrics(eval_pred: EvalPrediction):
        preds = eval_pred.predictions
        print(type(eval_pred.predictions))
        print(len(eval_pred.predictions))
        if isinstance(preds, tuple):
            # preds = ((scores_c, scores_r, relations),)
            if len(preds) == 1 and isinstance(preds[0], tuple):
                preds = preds[0]
            scores_c, scores_r, relations = preds
        else:
            raise ValueError(
                f"Unexpected predictions type: {type(preds)}"
            )
        p_plus = relations[:, :, 0]
        p_minus = relations[:, :, 1]

        # ==========================================
        # 层级一：机制验证 (Head-Level)
        # ==========================================
        # 正向准确率
        correct_plus = (scores_c > scores_r).astype(float)
        acc_plus_heads = np.mean(np.sum(p_plus * correct_plus, axis=-1) / (np.sum(p_plus, axis=-1) + 1e-8))
        
        # 反向准确率
        correct_minus = (scores_r > scores_c).astype(float)
        acc_minus_heads = np.mean(np.sum(p_minus * correct_minus, axis=-1) / (np.sum(p_minus, axis=-1) + 1e-8))

        # ==========================================
        # 层级二：全局 BT 偏好拟合度
        # ==========================================
        weights = p_plus + lambda_neg * p_minus 
        
        reward_c = np.sum(weights * scores_c, axis=-1)
        reward_r = np.sum(weights * scores_r, axis=-1)
        
        margin_global = np.mean(reward_c - reward_r)
        acc_global = np.mean((reward_c > reward_r).astype(float))

        return {
            "acc_plus_heads": acc_plus_heads,
            "acc_minus_heads": acc_minus_heads,
            "margin_global": margin_global,
            "acc_global": acc_global
        }
        
    return compute_latent_metrics
