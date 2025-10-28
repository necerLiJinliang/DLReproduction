from my_utils import SkipGramKLDataset, LogRecorder
import torch
from torch.utils.data import DataLoader
from model import SkipGramKLModel
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime

device = None


def train(
    args,
    model: SkipGramKLModel,
    train_dataset: SkipGramKLDataset,
    log_recorder,
):
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=SkipGramKLDataset.collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_num = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=steps_num
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            inputs = {
                "input_nodes": batch[0].to(device),
                "prob_dist": batch[1].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss
            scheduler.step()
        torch.save(model.state_dict(), args.save_path)
        f1_micro, f1_macor = eval(args, model)
        print(
            f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, F1_micro: {f1_micro}, F1_marcor: {f1_macor}"
        )
        log_recorder.add_log(
            f1_micor=f1_micro,
            f1_macor=f1_macor,
            epoch=epoch + 1,
        )


def eval(
    args,
    model: SkipGramKLDataset,
):
    nodes_features = model.embedding.weight.detach().numpy()
    nodes = pd.read_csv(args.nodes_file, names=["node"])
    groups = pd.read_csv(args.groups_file, names=["group"])
    group_edges = pd.read_csv(args.group_edges_file, names=["node", "label"])
    labels = np.zeros([nodes.shape[0], groups.shape[0]])
    for i in range(group_edges.shape[0]):
        node = group_edges["node"].iloc[i]
        group = group_edges["label"].iloc[i]
        labels[node - 1][group - 1] = 1
    X = nodes_features
    Y = labels
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    base_classifier = LogisticRegression(solver="liblinear")
    ovr_classifier = OneVsRestClassifier(base_classifier)
    ovr_classifier.fit(X_train, Y_train)
    y_pred = ovr_classifier.predict(X_test)
    f1_micro = f1_score(Y_test, y_pred, average="micro")
    f1_macro = f1_score(Y_test, y_pred, average="macro")
    return f1_micro, f1_macro


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
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
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="2_DeepWalk/model_save/model_kl.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--nodes_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/nodes.csv",
        help="Nodes file path.",
    )
    parser.add_argument(
        "--edges_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/edges.csv",
        help="Edges file path.",
    )
    parser.add_argument(
        "--groups_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/groups.csv",
        help="Groups file path.",
    )
    parser.add_argument(
        "--group_edges_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/group-edges.csv",
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--walks_num",
        type=int,
        default=2,
        help="Num of random walks",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=40,
        help="Length of random walk",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Length of random walk",
    )

    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    dataset = SkipGramKLDataset(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        window_size=args.window_size,
        walks_num=args.walks_num,
        t=args.t,
        bias=1,
    )

    nodes = pd.read_csv(args.nodes_file, names=["node"])
    nodes_num = nodes.shape[0]
    model = SkipGramKLModel(
        nodes_num=nodes_num,
        embedding_dim=args.embedding_dim,
    )
    model.to(device)
    args_dict = vars(args)
    log_recorder = LogRecorder(config=args_dict, info="Deep Walk", verbose=False)
    try:
        train(args=args, model=model, train_dataset=dataset, log_recorder=log_recorder)
    except:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_recorder.save(f"2_DeepWalk/log/{time_str}.json")


if __name__ == "__main__":
    main()
