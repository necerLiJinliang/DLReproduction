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
sample = dataset["train"][0]

# %%
sample

# %%
sample["completions"][0]["annotations"]["helpfulness"]


# %%
def sample_trans2(sample, tokenizer):
    query = sample["instruction"]
    completions = sample["completions"]
    score_dims = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
    inputs_list = []
    scores = []
    query_len = len(tokenizer(query)["input_ids"])
    answers_len = []
    for completion in completions:
        input_text = "\n".join([query, completion["response"]])
        inputs = tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            padding=True,
            padding_side="left",
        )
        inputs_list.append(inputs)
        answers_len.append(len(inputs["input_ids"]) - query_len)
        score_4 = [
            (
                float(completion["annotations"][score_dim]["Rating"])
                if completion["annotations"][score_dim]["Rating"] != "N/A"
                else 0
            )
            for score_dim in score_dims
        ]
        scores.append(score_4)
    return {
        "inputs_list": inputs_list,
        "scores_list": scores,
        "answers_len": answers_len,
    }


# %%
dataset["train"]


# %%
def collate_fn(batch, tokenizer):
    inputs_ids = []
    attention_mask = []
    scores = []
    answers_len = []
    for item in batch:
        item_inputs_ids = [
            torch.LongTensor(inputs["input_ids"]) for inputs in item["inputs_list"]
        ]
        inputs_ids.extend(item_inputs_ids)
        item_attention_mask = [
            torch.tensor(inputs["attention_mask"]) for inputs in item["inputs_list"]
        ]
        attention_mask.extend(item_attention_mask)  # [b*a, l]
        answers_len.extend(item["answers_len"])  # [b*a]
        scores.append(item["scores_list"])  # [b,a,4]
    inputs = tokenizer.pad(
        {"input_ids": inputs_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    answer_mask = torch.zeros_like(inputs["input_ids"])
    for i, length in enumerate(answers_len):
        answer_mask[i, answer_mask.shape[-1] - 1 - length :] = 1
    scores = torch.tensor(scores, dtype=torch.float32)
    return {
        "inputs": inputs,
        "answer_mask": answer_mask,
        "scores": scores,
    }


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
train_dataset = (
    dataset["train"].filter(lambda x: len(x["instruction"]) < 200).select(range(1000))
)
train_dataset = train_dataset.map(
    lambda sample: sample_trans2(sample, tokenizer),
    remove_columns=train_dataset.column_names,
)


# %%
def calculate_seq_log_prob(model, input_ids, attention_mask):
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits  # [b*a, l, v]
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # [b*a, l-1]
    target_log_probs = target_log_probs
    return target_log_probs  # [b*a,l]


# %%
def dpo_loss(
    log_probs_policy,
    log_probs_ref,
    answer_mask,
    scores,  # [b,a,4]
    lambda_vector,  # [1,1,4]
    listwise_size=4,
    beta=0.1,
):
    answer_mask = answer_mask.reshape(-1, listwise_size, answer_mask.shape[-1])[
        :, :, :-1
    ]  # [b,a,l]
    log_probs_policy = (
        log_probs_policy.reshape(-1, listwise_size, log_probs_policy.shape[-1])
        * answer_mask
    )  # [b,a,l]
    log_probs_ref = (
        log_probs_ref.reshape(-1, listwise_size, log_probs_ref.shape[-1]) * answer_mask
    )  # [b,a,l]
    log_probs_policy = (log_probs_policy * answer_mask).sum(dim=-1)
    log_probs_ref = (log_probs_ref * answer_mask).sum(dim=-1)  # [b,a]
    pi_ratio = beta * (log_probs_policy - log_probs_ref)  # [b,a]
    pi_ratio = pi_ratio.unsqueeze(dim=-1).expand(-1, -1, scores.shape[-1])  # [b,a,4]
    log_softmax_pi_ratio = pi_ratio - pi_ratio.logsumexp(dim=1, keepdim=True)  # [b,a,4]
    target_pi = scores / (scores.sum(dim=1, keepdim=True) + 1e-10)  # [b,a,4]
    loss = -log_softmax_pi_ratio * target_pi  # [b,a,4]  [b,a,4]
    loss = loss.mean(dim=1)  # [b,4]
    loss = loss * lambda_vector.reshape(1, -1)  # [b,4] * [1,4]
    loss = loss.sum(dim=1).mean()
    return loss


# %%
def train_llm_dpo(
    policy_model, ref_model, train_data, optimizer, epochs=3, batch_size=1
):
    lambda_vector = torch.tensor([0.25, 0.25, 0.25, 0.25]).to(device)
    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer),
    )
    pbar = tqdm(total=len(dataloader) * epochs, desc="Training lambda-DPO")
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
            inputs = batch["inputs"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            scores = batch["scores"]
            log_prob_policy = calculate_seq_log_prob(
                policy_model,
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            with torch.no_grad():
                log_prob_ref = calculate_seq_log_prob(
                    ref_model,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )
            loss = dpo_loss(
                log_prob_policy,
                log_prob_ref,
                answer_mask,
                scores.to(device),
                lambda_vector.to(device),
            )
            loss.backward()
            optimizer.step()
            del log_prob_policy, log_prob_ref
            del inputs, answer_mask, scores
            torch.cuda.empty_cache()
            lr_scheduler.step()
            steps += 1
            total_loss += loss.item()
            pbar.update(1)
            if steps % 20 == 0:
                print(f"Step {steps}, Loss: {total_loss/steps:.4f}")


# %%
optimizer = torch.optim.AdamW(model_policy.parameters(), lr=5e-6)
epochs = 3
try:
    train_llm_dpo(
        model_policy, model_ref, train_dataset, optimizer, epochs, batch_size=1
    )
finally:
    model_policy.save_pretrained(
        "../model_save/lambda_dpo_model/qwen-1.5-1.8b-dpo-lambda"
    )

# %%
