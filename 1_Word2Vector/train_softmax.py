import torch
from torch.utils.data import DataLoader
from my_utils import CBOWSoftmaxDataset
import argparse
from model import Word2VectorModelSoftmax
import json
from tqdm import tqdm

device = None


def train(
    args,
    model: Word2VectorModelSoftmax,
    train_dataset: CBOWSoftmaxDataset,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    epoch_num = args.epochs
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CBOWSoftmaxDataset.collate_fn,
    )
    for epoch in range(epoch_num):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = {
                "inputs_vector": batch[0].to(device),
                "target_ids": batch[1].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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
        default=1000,
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
        default="1_Word2Vector/model_save/model_softmax.pth",
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
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--word2id_file",
        type=str,
        default="1_Word2Vector/data/word2id.json",
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--huffman_code_file",
        type=str,
        default="1_Word2Vector/data/huffman_codes.json",
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--no_leaf_code2index_file",
        type=str,
        default="1_Word2Vector/data/no_leaf_code2index.json",
        help="File path of train dataset.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=100,
        help="File path of train dataset.",
    )

    parser.add_argument("--info", type=str, default="Classification bert base model.")
    args = parser.parse_args()
    global device
    device = torch.device(args.device)

    word2id = json.load(open(args.word2id_file))
    words_num = len(word2id)
    train_dataset = CBOWSoftmaxDataset(
        args.train_data_file,
        word2id_file=args.word2id_file,
        window_size=5,
    )
    model = Word2VectorModelSoftmax(
        words_num=words_num, embedding_dim=100, no_leaf_nodes_num=words_num - 1
    ).to(device)
    train(
        args,
        model,
        train_dataset,
    )


if __name__ == "__main__":
    main()
