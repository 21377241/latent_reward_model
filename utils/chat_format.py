"""Llama 3 chat template 编码（与 baseline/singe_head.py 对齐）。"""

from typing import Any, Dict, List, Union

Message = Dict[str, str]
Conversation = List[Message]


def is_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], dict)
        and "role" in value[0]
        and "content" in value[0]
    )


def build_conversation(
    chosen_or_rejected: Union[str, Conversation],
    prompt: str = "",
) -> Conversation:
    """
    将样本转为 apply_chat_template 所需的 messages。
    - 新格式：直接使用完整 chosen / rejected 对话列表
    - 旧格式（prompt + 字符串）：构造成 user + assistant 两轮
    """
    if is_message_list(chosen_or_rejected):
        return chosen_or_rejected
    return [
        {"role": "user", "content": (prompt or "").strip()},
        {"role": "assistant", "content": str(chosen_or_rejected).strip()},
    ]


def encode_conversation(tokenizer, conversation: Conversation) -> Dict[str, List[int]]:
    """
    apply_chat_template + tokenize（add_special_tokens=False）。
    不做截断；过长样本由后续长度 filter 丢弃（与 baseline 一致）。
    """
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    return tokenizer(text, add_special_tokens=False)


def passes_length_filter(
    len_chosen: int,
    len_rejected: int,
    min_length: int,
    max_length: int,
) -> bool:
    """chosen / rejected 均需在 [min_length, max_length] 内。"""
    return (
        min_length <= len_chosen <= max_length
        and min_length <= len_rejected <= max_length
    )
