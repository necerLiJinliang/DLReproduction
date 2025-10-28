# %%
import torch
import random
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
import bitsandbytes as bnb
from tqdm import tqdm
from typing import Literal
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel,
)
from copy import deepcopy

# 提前预处理数据

# %%
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

# %%
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# %%
# from modelscope.msdatasets import MsDataset

# ds = MsDataset.load(
#     "Anthropic/hh-rlhf",
#     subset_name="default",
# )
# ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
dataset = load_dataset(
    "json", data_files={"train": "dataset/train.jsonl", "test": "dataset/test.jsonl"}
)


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
name_actor_model = "./model_save/base_model/llama-3-8b-instruct"
# name_critic_model = "distilbert/distilbert-base-uncased"
name_critic_model = "./model_save/base_model/llama-3.2-1b"
tokenizer_actor = AutoTokenizer.from_pretrained(
    name_actor_model,
)
tokenizer_actor.pad_token = tokenizer_actor.eos_token
tokenizer_critic = AutoTokenizer.from_pretrained(name_critic_model)
tokenizer_critic.pad_token = tokenizer_critic.eos_token


# %%
class ValueModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super(ValueModel, self).__init__()
        self.drop = torch.nn.Dropout(0)
        self.linear = bnb.nn.Linear4bit(
            hidden_size, 1, quant_type="fp4", compute_dtype=torch.float16
        )

        self.group_infer_linear = bnb.nn.Linear4bit(
            hidden_size, 1, quant_type="fp4", compute_dtype=torch.float16
        )

    def forward(self, x):
        return self.linear(self.drop(x))

    def score(self, x):
        return self.linear(self.drop(x))

    def group_infer(self, x):
        """
        进行分组推理，返回分组后的结果
        """
        x = self.group_infer_linear(x.mean(dim=1))
        return torch.sigmoid(x)  # [b,1]


# %%
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4bit 加载
    bnb_4bit_quant_type="nf4",  # 使用 NormalFloat4 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算 dtype 使用 float16
    bnb_4bit_use_double_quant=True,  # 使用二次量化进一步压缩
)
peft_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05,
)
model_actor = AutoModelForCausalLM.from_pretrained(
    name_actor_model,
    quantization_config=bnb_config,
).to("cuda")
model_actor = prepare_model_for_kbit_training(model_actor)  # 准备模型以支持kbit训练
model_actor = get_peft_model(model_actor, peft_config)  # 应用LoRA配置
model_actor.config.use_cache = False  # 确保禁用缓存以支持LoRA
model_ref = AutoModelForCausalLM.from_pretrained(
    name_actor_model,
    quantization_config=bnb_config,
).to("cuda")
model_actor.config.pad_token_id = tokenizer_actor.pad_token_id
model_ref.config.pad_token_id = tokenizer_actor.pad_token_id
model_critic = AutoModelForSequenceClassification.from_pretrained(
    name_critic_model,
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=1,
).to("cuda")
model_critic = PeftModel.from_pretrained(
    model_critic,
    "model_save/reward_model/llama-3.2-1b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model_critic.config.pad_token_id = tokenizer_critic.pad_token_id
model_value = ValueModel(hidden_size=model_actor.config.hidden_size).to(
    device, dtype=torch.bfloat16
)

# %%
model_value

# %%
import torch.nn.functional as F


def logprobs_from_logits(logits, labels, gather=True):
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def batched_forward_pass(
    model_actor,
    model_value,
    input_ids,
    attention_mask,
    ref=False,
    group_infer=False,
):
    if ref:
        model_actor.eval()
    else:
        model_actor.train()
    if isinstance(model_actor, PeftModel):
        last_hidden_state = model_actor.base_model.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    else:
        last_hidden_state = model_actor.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    logits = model_actor.lm_head(last_hidden_state)
    value = model_value.score(last_hidden_state.detach())
    prob_log = logprobs_from_logits(
        logits, input_ids
    )  # 进行对齐，最后一个token的不要，input_ids去掉开头的特殊token，因为logits是从开头特殊token的后一个token开始的
    value = value.squeeze(-1)
    if group_infer:
        group_prob = model_value.group_infer(last_hidden_state.detach())
        return prob_log, value, group_prob
    return prob_log, value


# %%
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


# %%
advantages = compute_advantages(
    torch.randn(4, 35).to("cuda"),
    torch.randn(4, 35).to("cuda"),
)
advantages.shape


# %%
@torch.no_grad()
def generate_response(model, inputs, pad_token_id, eos_token_id):
    # querys: {"input_ids":tensor, "attention_mask":tensor}
    model.eval()
    device = model.device
    response = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_new_tokens=50,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        temperature=0.7,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    attention_mask = (response != pad_token_id).long()
    answer_mask = torch.zeros_like(attention_mask)
    answer_mask[:, inputs["input_ids"].shape[1] :] = 1
    return response, attention_mask, answer_mask
    # answer = response[:, querys["input_ids"].shape[1]:] # 取问题后面的那一部分


# %%
@torch.no_grad()
def get_scores(model_critic, inputs_qa):
    scores = model_critic(
        inputs_qa["input_ids"].to("cuda"), inputs_qa["attention_mask"].to("cuda")
    ).logits.squeeze(
        1
    )  # [b]
    return scores


# %%
train_dataset = dataset["train"]
test_dataset = dataset["test"]


def process_func(example):
    text = example["chosen"]
    query = "\n".join(text.split("\n")[:-1])  # 去掉最后一行
    return {"query": query}


train_dataset = train_dataset.map(
    process_func, batched=False, remove_columns=["chosen", "rejected"]
)


# Filter the dataset to keep only queries with length < 400
def filter_by_query_length(example):
    return len(example["query"]) < 400


train_dataset = train_dataset.filter(filter_by_query_length)


def text2tensor(sample, tokenizer_actor, tokenizer_critic):
    text = sample["query"]
    inputs_actor = tokenizer_actor(
        text, padding=True, return_tensors="pt", padding_side="left"
    )
    return {
        "input_ids": inputs_actor["input_ids"],
        "attention_mask": inputs_actor["attention_mask"],
    }


def collator(batch, tokenizer_actor):
    # inputs_actor = [item["inputs"] for item in batch]
    inputs_actor = [
        {
            "input_ids": torch.LongTensor(item["input_ids"][0]),
            "attention_mask": torch.tensor(item["attention_mask"][0]),
        }
        for item in batch
    ]
    max_length = max(len(inputs["input_ids"]) for inputs in inputs_actor)
    inputs_actor = tokenizer_actor.pad(
        inputs_actor,
        padding=True,
        return_tensors="pt",
        padding_side="left",
        max_length=max_length,
        # truncation=True,
    )
    return {
        "inputs": inputs_actor,
    }


train_dataset = train_dataset.map(
    lambda x: text2tensor(x, tokenizer_actor, tokenizer_critic),
    batched=False,
    remove_columns=["query"],
)


print(
    f"Original train size: {len(dataset['train'])}, filtered train size: {len(train_dataset)}"
)
print(
    f"Original test size: {len(dataset['test'])}, filtered test size: {len(test_dataset)}"
)


# %%
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


seq1 = torch.randint(20, [2, 5])
seq2 = torch.randint(20, [2, 7])
seq3 = torch.randint(20, [2, 6])
# 示例使用
seq = manual_pad_batch_tensors([seq1, seq2, seq3])
print(seq.shape)  # 输出: [2, 7]

# %%
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
    batch_size=16,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=lambda x: [t["query"] for t in x],
        collate_fn=lambda x: collator(x, tokenizer_actor),
    )
    input_ids_all = []
    attention_mask_all = []
    answer_mask_all = []
    prob_log_old_all = []
    value_old_all = []
    advantages_all = []
    returns_all = []
    for batch in tqdm(dataloader, desc="Generate data"):
        # get responses
        # queries_inputs = tokenizer_actor(
        #     batch_queries, padding=True, return_tensors="pt", padding_side="left"
        # )
        response, attention_mask, answer_mask = generate_response(
            model_actor,
            batch["inputs"],
            pad_token_id=tokenizer_actor.pad_token_id,
            eos_token_id=tokenizer_actor.eos_token_id,
        )
        response_text = tokenizer_actor.batch_decode(
            response, skip_special_tokens=False
        )
        input_ids_all.append(response.cpu())
        attention_mask_all.append(attention_mask.cpu())
        answer_mask_all.append(answer_mask.cpu())
        # get rewards
        inputs_qa = tokenizer_critic(
            response_text, padding=True, return_tensors="pt", padding_side="left"
        )
        scores = get_scores(
            model_critic,
            {
                "input_ids": inputs_qa["input_ids"],
                "attention_mask": inputs_qa["attention_mask"],
            },
        )  # [b]
        # get advantages
        prob_log_old, value_old = batched_forward_pass(
            model_actor, model_value, response, attention_mask
        )  # [b,s], [b,s]
        with torch.no_grad():
            prob_log_ref, _ = batched_forward_pass(
                model_ref, model_value, response, attention_mask, ref=True
            )
        kl = (prob_log_old - prob_log_ref) * -0.2  # reward = kl + score
        scores_pad = torch.zeros_like(prob_log_old)
        scores_pad[:, scores_pad.shape[-1] - 1] = scores
        rewards = kl + scores_pad  # [b,s]
        # rewards = kl + scores.unsqueeze(dim=-1)
        value_old = value_old * answer_mask
        rewards = rewards * answer_mask  # [b,s]
        advantages = compute_advantages(value_old, rewards)  # [b,s]
        returns = advantages + value_old  # [b,s]
        advantages = masked_whiten(advantages, answer_mask)
        prob_log_old_all.append(prob_log_old.cpu())
        value_old_all.append(value_old.cpu())
        advantages_all.append(advantages.cpu())
        returns_all.append(returns.cpu())
    del rewards, advantages, returns, prob_log_old, value_old, scores, kl
    torch.cuda.empty_cache()
    input_ids_all = manual_pad_batch_tensors(
        input_ids_all,
        padding_value=tokenizer_actor.pad_token_id,
        padding_side="left",
    )
    attention_mask_all = manual_pad_batch_tensors(
        attention_mask_all,
        padding_value=0,
        padding_side="left",
    )
    answer_mask_all = manual_pad_batch_tensors(
        answer_mask_all,
        padding_value=0,
        padding_side="left",
    )
    prob_log_old_all = manual_pad_batch_tensors(
        prob_log_old_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    )
    value_old_all = manual_pad_batch_tensors(
        value_old_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    )
    advantages_all = manual_pad_batch_tensors(
        advantages_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    )
    returns_all = manual_pad_batch_tensors(
        returns_all,
        padding_value=0.0,  # 使用0.0作为填充值
        padding_side="left",
    )
    return {
        "input_ids": input_ids_all,
        "attention_mask": attention_mask_all,
        "answer_mask": answer_mask_all,
        "prob_log_old": prob_log_old_all,
        "value_old": value_old_all,
        "advantages": advantages_all,
        "returns": returns_all,
    }


# %%
# one_step_data = get_data(
#     train_dataset.select(range(50)),  # 仅选择前1000条数据进行测试
#     model_actor,
#     model_ref,
#     model_critic,
#     model_value,
#     tokenizer_actor,
#     tokenizer_critic,
#     batch_size=16
# )

# %%
torch.cuda.empty_cache()

# %%
# one_step_dataset = torch.utils.data.TensorDataset(
#     one_step_data["input_ids"],
#     one_step_data["attention_mask"],
#     one_step_data["answer_mask"],
#     one_step_data["prob_log_old"],
#     one_step_data["value_old"],
#     one_step_data["advantages"],
#     one_step_data["returns"],
# )

# %%
from trl.core import masked_mean


def train_step(
    model_actor,
    model_value,
    tokenizer_actor,
    one_step_dataset,
    optimizer,
    epochs=1,
    device="cuda",
    batch_size=16,
    lr_scheduler=None,
):
    # Backward pass and optimization
    model_actor.train()
    model_value.train()
    model_actor.zero_grad()
    model_value.zero_grad()
    dataloader = torch.utils.data.DataLoader(
        one_step_dataset, batch_size=batch_size, shuffle=True
    )
    skip = 0
    total = 0
    loss_total = 0
    mean_return = 0
    for _ in range(epochs):
        loss_total = 0
        group_inference_loss = 0
        for batch in tqdm(dataloader, desc="Training step"):
            (
                input_ids,
                attention_mask,
                answer_mask,
                prob_log_old,
                value_old,
                advantages,
                returns,
            ) = batch
            input_ids = (input_ids.to(device),)
            attention_mask = attention_mask.to(device)
            answer_mask = answer_mask.to(device)
            prob_log_old = prob_log_old.to(device)
            value_old = value_old.to(device)
            advantages = advantages.to(device)
            returns = returns.to(device)
            # Forward pass
            prob_log_new, value_new, group_prob = batched_forward_pass(
                model_actor,
                model_value,
                input_ids[0],
                attention_mask,
                ref=False,
                group_infer=True,
            )
            ########## 分组推理的损失函数（最大化方差）
            group1_return = group_prob * (
                returns.detach() * prob_log_new.detach()
            ).mean(
                dim=-1, keepdim=True
            )  # b,1
            group2_retrun = (1 - group_prob) * (
                returns.detach() * prob_log_new.detach()
            ).mean(
                dim=-1, keepdim=True
            )  # b,1
            group1_return = group1_return.squeeze(-1)
            group2_retrun = group2_retrun.squeeze(-1)
            group_var_inference = torch.var(
                torch.cat([group1_return, group2_retrun], dim=-1), dim=0
            )
            ############
            group1_return = group_prob.detach() * (returns * prob_log_new).mean(
                dim=-1, keepdim=True
            )  # b,1
            group2_retrun = (1 - group_prob).detach() * (returns * prob_log_new).mean(
                dim=-1, keepdim=True
            )  # b,1
            group1_return = group1_return.squeeze(-1)
            group2_retrun = group2_retrun.squeeze(-1)
            group_var_policy = torch.var(
                torch.cat([group1_return, group2_retrun], dim=-1), dim=0
            )
            #####

            # 重要性采样
            ratio = torch.exp(prob_log_new - prob_log_old)
            if masked_mean(ratio, answer_mask).item() > 10:
                skip += 1
                continue
            # 策略梯度损失
            loss_pg1 = ratio * advantages
            loss_pg2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            loss_pg = masked_mean(-torch.min(loss_pg1, loss_pg2), answer_mask)

            # value函数损失
            loss_vf1 = (value_new - returns) ** 2
            loss_vf2 = torch.clamp(value_new, value_old - 0.2, value_old + 0.2)
            loss_vf2 = (loss_vf2 - returns) ** 2
            loss_vf = masked_mean(torch.max(loss_vf1, loss_vf2), answer_mask)
            loss = (
                loss_pg + 0.1 * loss_vf - group_var_policy * 1 - group_var_inference * 1
            )
            # group_inference_loss = group_inference_loss - group_var_in
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            mean_return += masked_mean(returns, answer_mask).item()
        print(
            f"Epoch Loss: {loss_total / len(dataloader)}, Return Mean: {mean_return / len(dataloader)}"
        )


# %%
# 训练
model_actor.train()
model_value.train()
model_ref.eval()
model_critic.eval()
# train_datasize = len(train_dataset.select(range(5000)))  # 仅选择前10000条数据进行测试
train_datasize = len(train_dataset.select(range(2000)))  # 仅选择前10000条数据进行测试
num_epochs_data = 100
epochs = train_datasize // num_epochs_data + 1
pbar = tqdm(range(epochs), desc="Training epochs")
optimizer = torch.optim.AdamW(
    list(model_actor.parameters()) + list(model_value.parameters()),
    lr=1e-5,
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(epochs * 0.1),
    num_training_steps=epochs,
)
try:
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        pbar.set_postfix(epoch=f"Epoch {i+1}/{epochs}", step=i + 1)
        pbar.update(1)
        start_idx = i * num_epochs_data
        end_idx = start_idx + num_epochs_data
        if end_idx >= len(train_dataset):
            end_idx = len(train_dataset) - 1
        one_step_data = get_data(
            train_dataset.select(
                range(start_idx, end_idx)
            ),  # 仅选择前1000条数据进行测试
            model_actor,
            model_ref,
            model_critic,
            model_value,
            tokenizer_actor,
            tokenizer_critic,
            batch_size=32,
        )
        one_step_dataset = torch.utils.data.TensorDataset(
            one_step_data["input_ids"],
            one_step_data["attention_mask"],
            one_step_data["answer_mask"],
            one_step_data["prob_log_old"],
            one_step_data["value_old"],
            one_step_data["advantages"],
            one_step_data["returns"],
        )
        train_step(
            model_actor,
            model_value,
            tokenizer_actor,
            one_step_dataset,
            optimizer,
            epochs=1,
            device="cuda",
            batch_size=32,
            lr_scheduler=lr_scheduler,
        )
        # model_ref = model_actor
        # del model_ref
        # torch.cuda.empty_cache()
        # model_ref = deepcopy(model_actor)
        # model_ref.eval()
        # model_ref.requires_grad_(False)
        # model_ref.to("cuda")
finally:
    model_actor.save_pretrained(
        "model_save/ppo_model/llama-3-8b-instruct-gil", safe_serialization=True
    )
# %%
torch.cuda.empty_cache()

# %%


# %%
test_data = dataset["test"]
sample = test_data[5]
query = "\n".join(sample["chosen"].split("\n")[:-1])  # 去掉最后一行

# %%
print(sample["chosen"])

# %%
inputs = tokenizer_actor(query, padding=True, return_tensors="pt", padding_side="left")
model_actor.eval()
with torch.no_grad():
    response = model_actor.generate(
        input_ids=inputs["input_ids"].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda"),
        pad_token_id=tokenizer_actor.pad_token_id,
        eos_token_id=tokenizer_actor.eos_token_id,
        max_new_tokens=50,
        temperature=0.7,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    response_text = tokenizer_actor.decode(response[0], skip_special_tokens=True)
print(response_text)

# %%
inputs = tokenizer_actor(query, padding=True, return_tensors="pt", padding_side="left")
model_ref.eval()
with torch.no_grad():
    response = model_ref.generate(
        input_ids=inputs["input_ids"].to("cuda"),
        attention_mask=inputs["attention_mask"].to("cuda"),
        pad_token_id=tokenizer_actor.pad_token_id,
        eos_token_id=tokenizer_actor.eos_token_id,
        max_new_tokens=50,
        temperature=0.7,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    response_text = tokenizer_actor.decode(response[0], skip_special_tokens=True)
print(response_text)

# %%


# %%
