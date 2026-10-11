from transformers import PreTrainedTokenizerBase
import torch

def tokenize_prompt_and_output(
    prompt_strs:list[str],
    output_strs:list[str],
    tokenizer:PreTrainedTokenizerBase,
) -> dict[str,torch.Tensor]:
    # 1. tokenize prompt
    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens = False,
        padding = False,
        truncation = False,
        return_attention_mask = False,
    )["input_ids"]

    # 2. tokenize output
    output_ids = tokenizer(
        output_strs,
        add_special_tokens = False,
        padding = False,
        truncation = False,
        return_attention_mask = False,
    )["input_ids"]

    # 3. 对每个样本拼接：
    # prompt_ids + output_ids
    prompt_and_output_ids = [
        prompt + output
        for prompt, output in zip(prompt_ids, output_ids)
    ]

    # 4. 找到 batch 中最大序列长度
    max_len = max(
        len(ids) for ids in prompt_and_output_ids
    )

    # 5. 获取 tokenizer 的 pad_token_id
    pad_token_id = tokenizer.pad_token_id

    # 6. 对 token ids 进行 padding
    # 并构造完整 response mask
    padded_ids = []
    response_masks = []

    for prompt, output, ids in zip(prompt_ids, output_ids, prompt_and_output_ids):
        pad_len = max_len - len(ids)
        padded_ids.append(
            ids + [pad_token_id] * pad_len
        )
        response_masks.append(
            [False] * len(prompt)
            + [True] * len(output)
            + [False] * pad_len
        )

    # 7. list -> torch.Tensor
    full_ids = torch.tensor(
        padded_ids,
        dtype = torch.long
    )

    full_response_mask = torch.tensor(
        response_masks,
        dtype = torch.long
    )

    # 8. next-token prediction
    input_ids = full_ids[:, :-1]

    labels = full_ids[:, 1:]

    # 9. 让 response_mask 与 labels 对齐
    response_mask = full_response_mask[:, 1:]

    return {
        "input_ids" : input_ids,
        "labels" : labels,
        "response_mask" : response_mask
    }