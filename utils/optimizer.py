"""优化器与学习率调度（与 baseline/singe_head.py 一致）。"""

import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(optimizer, total_steps: int, warmup_steps: int, eta_min: float = 0.01):
    """Cosine decay + linear warmup，同 baseline.cosine_warmup。"""

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
