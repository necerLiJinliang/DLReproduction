import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from typing import Optional, Union, Dict, List
from dataclasses import dataclass
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)

model_path = "model_save/base_model/llama-3.2-1b"
# dataset = load_dataset(
#     "Anthropic/hh-rlhf",
#     data_dir="harmless-base",
#     cache_dir="./dataset/hh-rlhf",
#     # data_dir="./dataset/hh-rlhf",
# )
# dataset = load_dataset(path="./dataset/hh-rlhf")
dataset = load_dataset(
    "json", data_files={"train": "dataset/train.jsonl", "test": "dataset/test.jsonl"}
)
# Filter dataset to keep only entries where both chosen and rejected are less than 400 characters
print(
    f"Original dataset sizes: train={len(dataset['train'])}, test={len(dataset['test'])}"
)


# def is_short_enough(example):
#     return len(example["chosen"]) < 512 and len(example["rejected"]) < 512


# dataset = dataset.filter(is_short_enough)
print(
    f"Filtered dataset sizes: train={len(dataset['train'])}, test={len(dataset['test'])}"
)


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


def process_function(examples):
    chosen_encodings = tokenizer(
        examples["chosen"], truncation=True, padding="max_length", max_length=512
    )
    rejected_encodings = tokenizer(
        examples["rejected"], truncation=True, padding="max_length", max_length=512
    )
    inputs = {
        "input_ids_chosen": chosen_encodings["input_ids"],
        "attention_mask_chosen": chosen_encodings["attention_mask"],
        "input_ids_rejected": rejected_encodings["input_ids"],
        "attention_mask_rejected": rejected_encodings["attention_mask"],
    }
    return inputs


@dataclass
class RewardDataCollator:
    tokenizer: PreTrainedTokenizerBase = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 提取chosen和rejected的输入
        chosen_inputs = [
            {
                "input_ids": f["input_ids_chosen"],
                "attention_mask": f["attention_mask_chosen"],
            }
            for f in features
        ]
        rejected_inputs = [
            {
                "input_ids": f["input_ids_rejected"],
                "attention_mask": f["attention_mask_rejected"],
            }
            for f in features
        ]

        # 分别对chosen和rejected进行填充
        chosen_batch = self.tokenizer.pad(
            chosen_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        rejected_batch = self.tokenizer.pad(
            rejected_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # 返回处理后的批次
        return {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
        }


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    quantization_config=bnb_config,
)
model.config.pad_token_id = tokenizer.pad_token_id
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
    task_type=TaskType.SEQ_CLS,
    lora_alpha=16,
    lora_dropout=0.05,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

dataset = dataset.map(process_function, remove_columns=["chosen", "rejected"])
config = RewardConfig(
    output_dir="./model_save/reward_model/llama-3.2-1b",
    save_strategy="no",  # 完全不保存中间 checkpoint
    # bf16=False,
    # fp16=False,
)
config.num_train_epochs = 5
config.per_device_train_batch_size = 32

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=RewardDataCollator(
        tokenizer=tokenizer, max_length=512
    ),  # 添加数据处理类
    peft_config=peft_config,  # 添加LoRA配置
)
try:
    trainer.train()
finally:
    trainer.save_model("./model_save/reward_model/llama-3.2-1b")
