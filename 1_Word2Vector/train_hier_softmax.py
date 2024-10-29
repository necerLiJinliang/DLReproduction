import torch
from torch.utils.data import DataLoader
from my_utils import CBOWDataset
import argparse
from model import Word2VectorModelHierarchicalSoftmax
import json
from tqdm import tqdm

device = None


def train(
    args,
    model: Word2VectorModelHierarchicalSoftmax,
    train_dataset: CBOWDataset,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs
    )
    epoch_num = args.epochs
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CBOWDataset.collate_fn,
    )
    for epoch in range(epoch_num):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = {
                "inputs_vector": batch[0].to(device),
                "path_nodes_indices": batch[1],
                "huffman_codes": batch[2],
                "mask": batch[3].to(device),
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
    parser.add_argument("--lr", type=float, default=2, help="Learning rate.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
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
        default="1_Word2Vector/model_save/model_hier.pth",
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
    train_dataset = CBOWDataset(
        args.train_data_file,
        word2id_file=args.word2id_file,
        huffman_code_file=args.huffman_code_file,
        no_leaf_code2index_file=args.no_leaf_code2index_file,
        window_size=10,
    )
    model = Word2VectorModelHierarchicalSoftmax(
        words_num=words_num, embedding_dim=100, no_leaf_nodes_num=words_num
    ).to(device)
    train(
        args,
        model,
        train_dataset,
    )


if __name__ == "__main__":
    main()
