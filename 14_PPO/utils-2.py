import torch 
import torch.nn.functional as F
from typing import Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl.core import masked_mean

def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def batched_forward_pass(model_actor,model_value, input_ids, attention_mask):
    last_hidden_state = model_actor,model_value.transformer(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    logits = model_actor,model_value.lm_head(last_hidden_state)
    value = model_value.score(last_hidden_state)
    prob_log = logprobs_from_logits(
        logits, input_ids
    )  # 进行对齐，最后一个token的不要，input_ids去掉开头的特殊token，因为logits是从开头特殊token的后一个token开始的
    value = value.squeeze(-1)
    return prob_log, value

def compute_advantages(value, reward_kl):
    advantages = []
    for i in reversed(range(reward_kl.shape[1])):
        value_next = 0
        if i < reward_kl.shape[1] - 1:
            value_next = value[:, i + 1]

        delta = reward_kl[:, i] + value_next - value[:, i]
        adv_last = 0
        if advantages:
            adv_last = advantages[-1]
        advantages.append(delta + 0.95 * adv_last)
    advantages = torch.stack(advantages[::-1]).transpose(0, 1)
    return advantages

@torch.no_grad()
def generate_response(model, inputs, pad_token_id, eos_token_id):
    # querys: {"input_ids":tensor, "attention_mask":tensor}
    device = model.device
    response = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=50,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    attention_mask = (response != pad_token_id).long()
    answer_mask = torch.zeros_like(attention_mask)
    answer_mask[:, inputs["input_ids"].shape[1] :] = 1
    return response, attention_mask, answer_mask
    # answer = response[:, querys["input_ids"].shape[1]:] # 取问题后面的那一部分
    
@torch.no_grad()
def get_scores(model_critic, inputs_qa):
    scores = model_critic(
        inputs_qa.input_ids.to("cuda"), inputs_qa.attention_mask.to("cuda")
    ).logits.squeeze(
        1
    )  # [b]
    return scores

def manual_pad_batch_tensors(
    tensors_list,
    padding_value=0,
    padding_side: Literal["left", "right"] = "left",
):
    """手动实现批次张量的填充"""
    # 获取最大序列长度
    max_seq_len = max(t.shape[1] for t in tensors_list)

    # 创建填充后的张量列表
    padded_tensors = []
    for tensor in tensors_list:
        b, s = tensor.shape
        # 创建填充张量
        padding = torch.full(
            (b, max_seq_len - s),
            padding_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        # 拼接原始张量和填充部分
        if padding_side == "left":
            padded = torch.cat([padding, tensor], dim=1)
        elif padding_side == "right":
            # 如果是右侧填充，则将填充部分放在后面
            padded = torch.cat([tensor, padding], dim=1)
        padded_tensors.append(padded)

    return torch.cat(padded_tensors, dim=0)

from trl.core import masked_whiten


@torch.no_grad()
def get_data(
    dataset,
    model_actor,
    model_ref,
    model_critic,
    model_value,
    tokenizer_actor,
    tokenizer_critic,
    batch_size,
    device,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [t["query"] for t in x],
    )
    input_ids_all = []
    attention_mask_all = []
    answer_mask_all = []
    prob_log_old_all = []
    value_old_all = []
    advantages_all = []
    returns_all = []
    for batch_queries in tqdm(dataloader):
        # get responses
        queries_inputs = tokenizer_actor(
            batch_queries, padding=True, return_tensors="pt", padding_side="left"
        )
        response, attention_mask, answer_mask = generate_response(
            model_actor, queries_inputs
        )
        response_text = tokenizer_actor.batch_decode(
            response, skip_special_tokens=False
        )
        input_ids_all.append(response)
        attention_mask_all.append(attention_mask)
        answer_mask_all.append(answer_mask)
        # get rewards
        inputs_qa = tokenizer_critic(
            response, padding=True, return_tensors="pt", padding_side="right"
        )
        scores = get_scores(model_critic, inputs_qa)  # [b]
        # get advantages
        prob_log_old, value_old = batched_forward_pass(
            model_actor, response, attention_mask
        )  # [b,s], [b,s]
        prob_log_ref, _ = batched_forward_pass(model_ref, model_value, response, attention_mask)
        kl = (prob_log_old - prob_log_ref) * -0.2  # reward = kl + score
        scores_pad = torch.zeros_like(prob_log_old)
        scores_pad[:, scores_pad.shape[-1] - 1] = scores
        rewards = kl + scores_pad  # [b,s]
        value_old = value_old * answer_mask
        rewards = rewards * answer_mask  # [b,s]
        advantages = compute_advantages(value_old, rewards)  # [b,s]
        returns = advantages + value_old  # [b,s]
        advantages = masked_whiten(advantages, answer_mask)
        prob_log_old_all.append(prob_log_old)
        value_old_all.append(value_old)
        advantages_all.append(advantages)
        returns_all.append(returns)
    input_ids_all = manual_pad_batch_tensors(
        input_ids_all,
        padding_value=tokenizer_actor.pad_token_id,
        padding_side="left",
    ).to(device)
    attention_mask_all = manual_pad_batch_tensors(
        attention_mask_all,
        padding_value=0,
        padding_side="left",
    ).to(device)
    answer_mask_all = manual_pad_batch_tensors(
        answer_mask_all,
        padding_value=0,
        padding_side="left",
    ).to(device)
    prob_log_old_all = manual_pad_batch_tensors(
        prob_log_old_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    ).to(device)
    value_old_all = manual_pad_batch_tensors(
        value_old_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    ).to(device)
    advantages_all = manual_pad_batch_tensors(
        advantages_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    ).to(device)
    returns_all = manual_pad_batch_tensors(
        returns_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    ).to(device)
    return {
        "input_ids": input_ids_all,
        "attention_mask": attention_mask_all,
        "answer_mask": answer_mask_all,
        "prob_log_old": prob_log_old_all,
        "value_old": value_old_all,
        "advantages": advantages_all,
        "returns": returns_all,
    }

def train_step(
    model_actor, model_value, tokenizer_actor, one_step_dataset, optimizer, epochs=1
):
    # Backward pass and optimization
    model_actor.zero_grad()
    model_value.zero_grad()
    dataloader = torch.utils.data.DataLoader(
        one_step_dataset, batch_size=16, shuffle=True
    )
    skip = 0
    total = 0
    loss_total = 0
    for _ in range(epochs):
        loss_total = 0
        for batch in tqdm(dataloader):
            (
                input_ids,
                attention_mask,
                answer_mask,
                prob_log_old,
                value_old,
                advantages,
                returns,
            ) = batch
            # Forward pass
            prob_log_new, value_new = batched_forward_pass(
                model_actor, input_ids, attention_mask
            )
            # 重要性采样
            ratio = torch.exp(prob_log_new - prob_log_old)
            # 策略梯度损失
            loss_pg1 = ratio * advantages
            loss_pg2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            loss_pg = masked_mean(-torch.min(loss_pg1, loss_pg2), answer_mask)

            # value函数损失
            loss_vf1 = (value_new - returns) ** 2
            loss_vf2 = torch.clamp(value_new, value_old - 0.2, value_old + 0.2)
            loss_vf2 = (loss_vf2 - returns) ** 2
            loss_vf = masked_mean(torch.max(loss_vf1, loss_vf2), answer_mask)
            loss = loss_pg + 0.05 * loss_vf
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch Loss: {loss_total / len(dataloader)}")