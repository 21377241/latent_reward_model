import torch


def pool_last_hidden(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    取每条序列最后一个有效 token 的 hidden state。
    兼容 left-padding（baseline / ArmoRM）与 right-padding。
    """
    seq_len = attention_mask.size(1)
    flipped = attention_mask.flip(1).long()
    last_idx = seq_len - 1 - flipped.argmax(1)
    valid = flipped.max(1).values.bool()
    last_idx = torch.where(
        valid,
        last_idx,
        torch.full_like(last_idx, seq_len - 1),
    )
    batch_size = last_hidden_state.size(0)
    device = last_hidden_state.device
    return last_hidden_state[torch.arange(batch_size, device=device), last_idx]
