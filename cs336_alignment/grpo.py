from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F

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


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    # 1. 模型前向传播
    output = model(input_ids)
    # 2. 取出模型输出的 logits [B, seqlen, vocablen]
    logits = output.logits
    # 3. 在词表维度计算 log softmax
    all_log_probs = F.log_softmax(logits, dim = -1)
    # 4. 根据真实标签取出对应 token 的 log probability [B, seqlen]
    log_probs = torch.gather(
        all_log_probs,
        dim = -1,
        index = labels.unsqueeze(-1)
    ).squeeze(-1)
    # 5. 创建返回字典
    result = {"log_probs" : log_probs}
    # 6. 如果要求计算 entropy：
    #    将 log probability 转换成 probability
    #    根据熵公式计算每个位置的 entropy
    #    加入返回字典
    if return_token_entropy:
        probs = F.softmax(logits, dim = -1)
        token_entropy = -torch.sum(probs * all_log_probs, dim = -1)
        result["token_entropy"] = token_entropy
    # 7. 返回结果
    return result