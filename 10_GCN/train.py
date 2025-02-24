import torch
import torch.nn as nn
from utils.data import DataPre
from tqdm import tqdm
from model import GCN
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model: GCN, data: DataPre, num_epochs=100):
    epochs = args.epochs
    supervised_nodes, labels = data.get_supervised_sample()
    test_nodes, test_labels = data.get_test_sample()
    x = data.get_features().to(device)
    sparse_adj = data.get_sparse_adj_matrix().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, loss = model(x, sparse_adj, supervised_nodes, labels)
        loss.backward()
        optimizer.step()
        acc = evaluate(model, sparse_adj, x, test_nodes, test_labels)
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {acc.item()}")
    end_time = time.time()
    print(f"Time: {end_time - start_time}")


def evaluate(model: GCN, sparse_adj, x, test_nodes, labels):
    model.eval()
    logits = model(x, sparse_adj, None, None)[0]
    test_logits = logits[test_nodes]
    pred = torch.argmax(test_logits, dim=1)
    acc = torch.sum(pred == labels) / len(labels)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cites_file_path",
        type=str,
        default="10_GCN/dataset/cora/cora.cites",
    )
    parser.add_argument(
        "--content_file_path",
        type=str,
        default="10_GCN/dataset/cora/cora.content",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    data = DataPre(args.cites_file_path, args.content_file_path)
    model = GCN(1433, 7, hidden_dim=100)
    model.to(device)
    train(args, model, data)


if __name__ == "__main__":
    main()
