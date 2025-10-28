# %%
from datasets import load_dataset, Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Literal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
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

# %%
model_path = "../model_save/base_model/qwen-1.5-1.8b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model_policy = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model_policy = prepare_model_for_kbit_training(model_policy)
model_policy = get_peft_model(model_policy, peft_config)
model_policy.print_trainable_parameters()
model_ref = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# %%
dataset = load_dataset("json", data_files="./dataset/ultrafeedback/flan.jsonl")

# %%
dataset


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


def sample_trans(sample):
    question = sample["instruction"]
    completions = sample["completions"]
    scores_list = [[0, 0], [0, 1], [0, 2], [0, 3]]
    responses = []
    for i, annotation in enumerate(completions):
        responses.append(annotation["response"])
        scores_list[i][0] = float(annotation["overall_score"])
    scores_list = sorted(scores_list, key=lambda x: x[0], reverse=True)
    results = []
    for i in range(1, 4):
        results.append(
            {
                "question": question,
                "chosen": responses[scores_list[0][1]],
                "rejected": responses[scores_list[i][1]],
            }
        )
    return results


def process_function(sample, tokenizer):
    query = sample["question"]
    chosen = sample["chosen"]
    rejected = sample["rejected"]
    chosen_inputs = tokenizer(
        "\n".join([query, chosen]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    query_len = len(tokenizer(query)["input_ids"])
    chosen_answer_len = chosen_inputs["input_ids"][0].shape[0] - query_len
    chosen_answer_len = max(chosen_answer_len, 0)
    rejected_inputs = tokenizer(
        "\n".join([query, rejected]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    rejected_answer_len = rejected_inputs["input_ids"][0].shape[0] - query_len
    rejected_answer_len = max(rejected_answer_len, 0)
    return {
        "chosen": chosen_inputs,
        "rejected": rejected_inputs,
        "chosen_answer_len": chosen_answer_len,
        "rejected_answer_len": rejected_answer_len,
    }


def collate_fn(batch, tokenizer):
    chosen_input_ids = [torch.LongTensor(s["chosen"]["input_ids"][0]) for s in batch]
    chosen_attention_mask = [
        torch.tensor(s["chosen"]["attention_mask"][0]) for s in batch
    ]
    rejected_input_ids = [
        torch.LongTensor(s["rejected"]["input_ids"][0]) for s in batch
    ]
    rejected_attention_mask = [
        torch.tensor(s["rejected"]["attention_mask"][0]) for s in batch
    ]
    chosen_inputs = tokenizer.pad(
        {
            "input_ids": chosen_input_ids,
            "attention_mask": chosen_attention_mask,
        },
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    rejected_inputs = tokenizer.pad(
        {
            "input_ids": rejected_input_ids,
            "attention_mask": rejected_attention_mask,
        },
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    answer_mask_chosen = torch.zeros_like(chosen_inputs["input_ids"])
    answer_mask_rejected = torch.zeros_like(rejected_inputs["input_ids"])
    for i, s in enumerate(batch):
        answer_mask_chosen[
            i, answer_mask_chosen.shape[-1] - 1 - s["chosen_answer_len"] :
        ] = 1
        answer_mask_rejected[
            i, answer_mask_chosen.shape[1] - 1 - s["rejected_answer_len"] :
        ] = 1
    return {
        "chosen_inputs": chosen_inputs,
        "rejected_inputs": rejected_inputs,
        "answer_mask_chosen": answer_mask_chosen,
        "answer_mask_rejected": answer_mask_rejected,
    }


# %%
text = "The capital of France is"
tokenizer(text, padding=True, return_tensors="pt")

# %%
train_dataset = dataset["train"].filter(lambda x: len(x["instruction"]) < 200)

# %%
processed_dataset = []
for sample in tqdm(train_dataset.select(range(1000))):
    processed_dataset.extend(sample_trans(sample))

# %%
new_dataset = Dataset.from_list(processed_dataset)

# %%
new_dataset = new_dataset.map(
    lambda sample: process_function(sample, tokenizer),
    batched=False,
    remove_columns=["question"],
)

# %%


# %%
def calculate_seq_log_prob(model, input_ids, attention_mask, answer_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    answer_mask = answer_mask[:, 1:]
    target_log_probs = target_log_probs * answer_mask
    return target_log_probs.sum(dim=-1)


# %%
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    pi_ratio = policy_chosen - policy_rejected
    ref_ratio = ref_chosen - ref_rejected
    loss = -torch.log(torch.sigmoid(beta * (pi_ratio - ref_ratio))).mean()
    return loss


# %%
def train_llm_dpo(
    policy_model,
    ref_model,
    train_data,
    optimizer,
    batch_size=8,
    epochs=3,
):
    dataloader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer)
    )
    pbar = tqdm(total=len(dataloader) * epochs, desc="Training DPO")
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * epochs,
    )
    policy_model.train()
    ref_model.eval()
    steps = 0
    total_loss = 0.0
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            chosen_ids = batch["chosen_inputs"]["input_ids"].to(device)
            rejected_ids = batch["rejected_inputs"]["input_ids"].to(device)
            attention_mask_chosen = batch["chosen_inputs"]["attention_mask"].to(device)
            attention_mask_rejected = batch["rejected_inputs"]["attention_mask"].to(
                device
            )
            answer_mask_chosen = batch["answer_mask_chosen"].to(device)
            answer_mask_rejected = batch["answer_mask_rejected"].to(device)
            policy_chosen_log_probs = calculate_seq_log_prob(
                policy_model, chosen_ids, attention_mask_chosen, answer_mask_chosen
            )
            policy_rejected_log_probs = calculate_seq_log_prob(
                policy_model,
                rejected_ids,
                attention_mask_rejected,
                answer_mask_rejected,
            )
            with torch.no_grad():
                ref_chosen_log_probs = calculate_seq_log_prob(
                    ref_model, chosen_ids, attention_mask_chosen, answer_mask_chosen
                )
                ref_rejected_log_probs = calculate_seq_log_prob(
                    ref_model,
                    rejected_ids,
                    attention_mask_rejected,
                    answer_mask_rejected,
                )
            loss = dpo_loss(
                policy_chosen=policy_chosen_log_probs,
                policy_rejected=policy_rejected_log_probs,
                ref_chosen=ref_chosen_log_probs,
                ref_rejected=ref_rejected_log_probs,
            )
            loss.backward()
            optimizer.step()
            del policy_chosen_log_probs, policy_rejected_log_probs
            del ref_chosen_log_probs, ref_rejected_log_probs
            torch.cuda.empty_cache()
            lr_scheduler.step()
            total_loss += loss.item()
            steps += 1
            pbar.update(1)
            if steps % 20 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps:.4f}")
        # print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}")


# %%
optimizer = torch.optim.AdamW(model_policy.parameters(), lr=1e-5)
epochs = 3
try:
    train_llm_dpo(model_policy, model_ref, new_dataset, optimizer, epochs)
finally:
    model_policy.save_pretrained("../model_save/dpo_model/qwen-1.5-1.8b-dpo")

# %%
