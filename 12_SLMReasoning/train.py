import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import GSM8KDateset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import datasets
import os


def train(
    args,
    model,
    train_dataset,
    dev_dataset,
    tokenizer,
    device,
):
    data_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=GSM8KDateset.collate_fn
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_loss = 0.0
    total_steps = len(data_loader) * args.num_epochs
    step = 0
    pbar = tqdm(total=total_steps)
    best_acc = -1
    for epoch in range(args.num_epochs):
        for batch in data_loader:
            inputs = {
                "input_ids": batch["question_ids"].to(device),
                "attention_mask": batch["question_attention_mask"].to(device),
                "labels": batch["answer_ids"].to(device),
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch + 1}/{args.num_epochs}")
        if step % len(data_loader) == 0:
            acc = evaluate(args, model, dev_dataset, tokenizer, device)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    model,
                    args.checkpoint_path,
                    "best_model.pth",
                    optimizer=optimizer,
                    epoch=epoch,
                    step=step,
                )
            print(
                f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss / step}, Accuracy: {acc}"
            )
            total_loss = 0.0


def evaluate(args, model, dataset, tokenizer, device):
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=GSM8KDateset.collate_fn,
    )
    model.eval()
    results_true = []
    results_pred = []
    for batch in data_loader:
        inputs = {
            "input_ids": batch["question_ids"].to(device),
            "attention_mask": batch["question_attention_mask"].to(device),
            "labels": batch["answer_ids"].to(device),
        }

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
            )
        decode_pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decode_label = tokenizer.batch_decode(
            inputs["labels"], skip_special_tokens=True
        )
        decode_pred = [s.split("### ")[-1] for s in decode_pred]
        decode_label = [s.split("### ")[-1] for s in decode_label]
        results_pred.extend(decode_pred)
        results_true.extend(decode_label)
    acc = sum([1 for i, j in zip(results_true, results_pred) if i == j]) / len(
        results_true
    )
    return acc


def save_checkpoint(
    model,
    checkpoint_path,
    checkpoint_name,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
    step=None,
):
    checkpoint = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_name))


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SLM Reasoning")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="google-t5/t5-small",
        help="Path to a pretrained model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints",
        help="Directory to save trained model",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Example usage of parsed arguments
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Pretrained Model: {args.pretrained_model}")

    # Initialize your model, optimizer, and other components here
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.pretrained_model, force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model, force_download=True
    )
    gsm8k_dataset = datasets.load_dataset("openai/gsm8k", "main", cache_dir="./dataset")
    num_train_samples = gsm8k_dataset["train"].num_rows
    train_size = int(0.9 * num_train_samples)
    dev_size = num_train_samples - train_size
    train_data = GSM8KDateset(gsm8k_dataset["train"][:train_size], tokenizer)
    dev_data = GSM8KDateset(gsm8k_dataset["train"][train_size:], tokenizer)
    test_dataset = GSM8KDateset(gsm8k_dataset["test"], tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(args, model, train_data, dev_data, tokenizer, device)


if __name__ == "__main__":
    main()
