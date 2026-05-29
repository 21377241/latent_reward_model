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


def prompt_token_length(tokenizer, conversation: Conversation) -> int:
    """
    prompt / instruction 部分的 token 数（不含 assistant 回复正文）。
    与完整序列拼接后 left-pad 时，prompt 结束下标 = offset + 返回值 - 1。
    """
    if not conversation:
        return 0
    if conversation[-1].get("role") == "assistant":
        prefix_messages = conversation[:-1]
        add_generation_prompt = True
    else:
        prefix_messages = conversation
        add_generation_prompt = False
    text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


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
