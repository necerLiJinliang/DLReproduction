import torch
import pandas as pd
from torch.utils.data import DataLoader
from my_utils import LINEDataset
from tqdm import tqdm
import argparse
from model import LINEModel

device = None


def train(args, model: LINEModel, train_dataset):
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LINEDataset.collate_fn,
        num_workers=6,
    )
    epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=0,
        total_iters=epochs * len(dataloader),
    )
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(dataloader, desc="Training"):
            inputs = {
                "source_nodes": batch[0].to(device),
                "sample_nodes": batch[1].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        torch.save(model.state_dict(), args.save_path)
        print(f"Epoch{epoch+1}, Loss: {total_loss/len(dataloader)}")


def eval():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--num_neg_sample",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="3_LINE/model_save/model.pth",
    )
    parser.add_argument(
        "--nodes_file",
        type=str,
        default="3_LINE/dataset/Flickr/csv/users.csv",
    )
    parser.add_argument(
        "--edges_file",
        type=str,
        default="3_LINE/dataset/Flickr/csv/links.csv",
    )
    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    nodes = pd.read_csv(args.nodes_file, names=["node"])
    num_nodes = nodes["node"].max() + 1
    train_dataset = LINEDataset(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        neg_sample_num=args.num_neg_sample,
        edge_weight_by_degree=False,
        sample_num=100000000,
    )
    model = LINEModel(
        num_nodes=num_nodes,
        embedding_dim=args.num_dim,
        order=2,
    )
    model.to(device)
    train(
        args,
        model,
        train_dataset,
    )


if __name__ == "__main__":
    main()
