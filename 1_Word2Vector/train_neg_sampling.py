import torch
from torch.utils.data import DataLoader
from my_utils import CBOWNegativeSamplingDataset
import argparse
from model import Word2VectorNegSampling
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR

device = None


def train(
    args,
    model: Word2VectorNegSampling,
    train_dataset: CBOWNegativeSamplingDataset,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs
    )
    epoch_num = args.epochs
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CBOWNegativeSamplingDataset.collate_fn,
    )
    for epoch in range(epoch_num):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = {
                "inputs_vector": batch[0].to(device),
                "chosen_ids": batch[1].to(device),
                "labels": batch[2].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        torch.save(model.state_dict(), args.save_path)
        print(f"Epoch{epoch+1}, Loss: {total_loss/len(train_dataloader)}")


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to training model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="1_Word2Vector/model_save/model_neg.pth",
        help="Path to save model",
    )
    # parser.add_argument(
    #     "--warmup_rate",
    #     type=float,
    #     default=0.06,
    #     help="Warm up rate.",
    # )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="1_Word2Vector/data/processed_data.json",
        help="",
    )
    parser.add_argument(
        "--word2id_file",
        type=str,
        default="1_Word2Vector/data/word2id.json",
        help="",
    )
    parser.add_argument(
        "--huffman_code_file",
        type=str,
        default="1_Word2Vector/data/huffman_codes.json",
        help="",
    )
    parser.add_argument(
        "--word_freq_file",
        type=str,
        default="1_Word2Vector/data/words_freq.json",
        help="File path of wrod frequence.",
    )
    parser.add_argument(
        "--no_leaf_code2index_file",
        type=str,
        default="1_Word2Vector/data/no_leaf_code2index.json",
        help="",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="",
    )

    parser.add_argument("--info", type=str, default="Classification bert base model.")
    args = parser.parse_args()
    global device
    device = torch.device(args.device)

    word2id = json.load(open(args.word2id_file))
    words_num = len(word2id)
    train_dataset = CBOWNegativeSamplingDataset(
        args.train_data_file,
        word2id_file=args.word2id_file,
        word_freq_file=args.word_freq_file,
        window_size=5,
        neg_sample_size=10,
        p_index=0.75,
    )
    model = Word2VectorNegSampling(words_num=words_num, embedding_dim=100).to(device)
    train(
        args,
        model,
        train_dataset,
    )


if __name__ == "__main__":
    main()
