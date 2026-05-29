import torch


def pool_hidden_at_positions(
    last_hidden_state: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    按每条序列的 token 下标取向量。
    positions: [B] long，每条序列一个下标（如 prompt 末 token）。
    last_hidden_state: [B, L, H]
    """
    batch_size = last_hidden_state.size(0)
    device = last_hidden_state.device
    positions = positions.to(device=device, dtype=torch.long)
    positions = positions.clamp(min=0, max=last_hidden_state.size(1) - 1)
    return last_hidden_state[torch.arange(batch_size, device=device), positions]


def prompt_end_positions_left_pad(
    seq_lengths: torch.Tensor,
    prompt_token_lengths: torch.Tensor,
    padded_seq_len: int,
) -> torch.Tensor:
    """
    left-pad 后 prompt 最后一个 token 的下标。
    seq_lengths / prompt_token_lengths: [B]
    """
    offsets = padded_seq_len - seq_lengths
    return offsets + prompt_token_lengths - 1


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
