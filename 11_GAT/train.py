import torch
import torch.nn as nn
from utils.data import DataPre, set_random_seed
from tqdm import tqdm
from model import GAT
import argparse
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_random_seed(24)


def train(args, model, data: DataPre, num_epochs=100):
    epochs = args.epochs
    supervised_nodes, labels = data.get_supervised_sample()
    test_nodes, test_labels = data.get_test_sample()
    x = data.get_features().to(device)
    edge_index = data.get_edge_index().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0005,
    )
    max_acc = -1
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, loss = model(x, edge_index, supervised_nodes, labels)
        # l2_reg = sum(
        #     torch.norm(param, 2) ** 2
        #     for param in model.parameters()
        #     if param.requires_grad
        # )
        # loss = loss + 0.0005 * l2_reg
        loss.backward()
        optimizer.step()
        acc = evaluate(model, edge_index, x, test_nodes, test_labels)
        max_acc = max(acc, max_acc)
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {acc.item()}")
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
    print("Max Accuracy: ", max_acc)


def evaluate(model, edge_index, x, test_nodes, labels):
    model.eval()
    logits = model(x, edge_index, None, None)[0]
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
    model = GAT(1433, 7, hidden_dim=64, dropout=0.6)
    model.to(device)
    train(args, model, data)


if __name__ == "__main__":
    main()
