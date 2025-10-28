from my_utils import SkipGramSoftmaxDataset, LogRecorder
import torch
from torch.utils.data import DataLoader
from model import SkipGramSoftmaxModel
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
    model: SkipGramSoftmaxModel,
    train_dataset: SkipGramSoftmaxDataset,
    log_recorder: LogRecorder,
):
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=SkipGramSoftmaxDataset.collate_fn,
        num_workers=4,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_num = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0, total_iters=steps_num
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=args.epochs,eta_min=0)
    best_f1_micro = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            inputs = {
                "input_nodes": batch[0].to(device),
                "context_nodes": batch[1].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
        f1_micro, f1_macor = eval(args, model)
        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro
            if args.save_path is not None:
                torch.save(model.state_dict(), args.save_path)
        print(
            f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}, F1_micro: {f1_micro}, F1_macro: {f1_macor}"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log_recorder.add_log(
            f1_micor=f1_micro,
            f1_macor=f1_macor,
            epoch=epoch + 1,
            lr=current_lr,
        )
        log_recorder.best_score = best_f1_micro


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def eval(
    args,
    model: SkipGramSoftmaxModel,
    test_size: float = 0.3,
):
    model.eval()
    nodes_features = model.embedding.weight.detach().cpu().numpy()
    nodes = pd.read_csv(args.nodes_file, names=["node"])
    groups = pd.read_csv(args.groups_file, names=["group"])
    group_edges = pd.read_csv(args.group_edges_file, names=["node", "label"])
    labels = np.zeros([nodes.shape[0], groups.shape[0]])
    for i in range(group_edges.shape[0]):
        node = group_edges["node"].iloc[i]
        group = group_edges["label"].iloc[i]
        labels[node - 1][group - 1] = 1
    X = nodes_features
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    base_classifier = LogisticRegression(solver="liblinear")
    ovr_classifier = TopKRanker(base_classifier)
    top_k_list = y_test.sum(axis=1).astype(int)
    ovr_classifier.fit(X_train, y_train)
    y_pred_list = ovr_classifier.predict(X_test, top_k_list)
    y_pred = np.zeros(y_test.shape)
    for i in range(len(y_pred_list)):
        for j in y_pred_list[i]:
            y_pred[i][j] = 1
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    return f1_micro, f1_macro


def test(args, model):
    test_size_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    score_list = [eval(args, model, test_size) for test_size in test_size_list]
    print(score_list)


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8000,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used to training model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="2_DeepWalk/model_save/model_softmax.pth",
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
        default=10,
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--walks_num",
        type=int,
        default=80,
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
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Length of random walk",
    )
    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    dataset = SkipGramSoftmaxDataset(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        window_size=args.window_size,
        walks_num=args.walks_num,
        t=args.t,
        bias=1,
        save_path=None,
        load_path="2_DeepWalk/BlogCatalog-dataset/random_walk_80.pkl",
    )

    nodes = pd.read_csv(args.nodes_file, names=["node"])
    nodes_num = nodes.shape[0]
    model = SkipGramSoftmaxModel(
        nodes_num=nodes_num,
        embedding_dim=args.embedding_dim,
    )
    args_dict = vars(args)
    log_recorder = LogRecorder(config=args_dict, info="Deep Walk", verbose=False)
    if args.load_path is not None:
        print(f"Load model parameters from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    model.to(device)
    if args.test == True:
        print("Evaluating")
        test(args, model)
    else:
        try:
            train(
                args=args, model=model, train_dataset=dataset, log_recorder=log_recorder
            )
        except:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_recorder.save(f"2_DeepWalk/log/s{time_str}.json")
        finally:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_recorder.save(f"2_DeepWalk/log/s{time_str}.json")


if __name__ == "__main__":
    main()
