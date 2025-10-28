from my_utils import Node2vecDataset, LogRecorder
import torch
from torch.utils.data import DataLoader
from model import SkipGramNegativeSamplingModel
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer
import numpy as np
from datetime import datetime
import gc

device = None
# 由于blog数据量较大，所以要对数据加载做一些处理，将所有的随机游走数据分成了10块，每次加载一块，训练完10块算完成一个epoch


def train(
    args,
    model: SkipGramNegativeSamplingModel,
    train_dataset: Node2vecDataset,
    log_recorder: LogRecorder,
):
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=Node2vecDataset.collate_fn,
        num_workers=10,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    steps_num = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.00001, total_iters=args.epochs
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=args.epochs,eta_min=0)
    switch = 1
    best_f1_micro = 0
    del_flag = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            inputs = {
                "source_nodes": batch[0].to(device),
                "context_nodes": batch[1].to(device),
                "labels": batch[2].to(device),
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
    model: SkipGramNegativeSamplingModel,
    test_size: float = 0.5,
):
    def scatter(x: list):
        vector = np.zeros(groups.shape[0])
        vector[x] = 1
        return vector

    model.eval()
    nodes_features = model.embedding.weight.detach().cpu().numpy()
    nodes = pd.read_csv(args.nodes_file, names=["node"])
    nodes_list = nodes["node"].tolist()
    groups = pd.read_csv(args.groups_file, names=["group"])
    group_edges = pd.read_csv(args.group_edges_file, names=["node", "label"])
    group_edges["label"] = group_edges["label"] - 1
    group_edges = group_edges.groupby(by=["node"])["label"].apply(list).reset_index()
    group_edges["label"] = group_edges["label"].apply(scatter)
    nodes_list = group_edges["node"].tolist()
    labels = group_edges["label"].tolist()
    labels = np.stack(labels)
    # labels = np.zeros([nodes.shape[0], groups.shape[0]])
    X = nodes_features[nodes_list]
    X = Normalizer(norm="l2").fit_transform(X)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    base_classifier = LogisticRegression(solver="liblinear")
    ovr_classifier = TopKRanker(base_classifier, n_jobs=-1)
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
    score_list = []
    for test_size in tqdm(test_size_list, desc="Test"):
        score = eval(args, model, test_size)
        score_list.append(score)
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
        default=1,
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
        default=97,
        help="Learning rate.",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        # default="4_Node2Vec/model_save/model_blog_neg_sampling2.pth",
        default=None,
        help="Path to load model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="4_Node2Vec/model_save/model_blog_neg_sampling3.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--nodes_file",
        type=str,
        default="4_Node2Vec/BlogCatalog-dataset/nodes.csv",
        help="Nodes file path.",
    )
    parser.add_argument(
        "--edges_file",
        type=str,
        default="4_Node2Vec/BlogCatalog-dataset/edges.csv",
        help="Edges file path.",
    )
    parser.add_argument(
        "--groups_file",
        type=str,
        default="4_Node2Vec/BlogCatalog-dataset/groups.csv",
        help="Groups file path.",
    )
    parser.add_argument(
        "--group_edges_file",
        type=str,
        default="4_Node2Vec/BlogCatalog-dataset/group-edges.csv",
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=10,
        help="Num of random walks",
    )
    parser.add_argument(
        "--l",
        type=int,
        default=80,
        help="Length of random walk",
    )
    parser.add_argument(
        "--num_negative_samples",
        type=int,
        default=5,
        help="Number of negative samples.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Length of random walk",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=0.25,
        help="Length of random walk",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=0.25,
        help="Length of random walk",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Length of random walk",
    )
    parser.add_argument(
        "--walks_load_path",
        type=str,
        default="4_Node2Vec/walks/walks-25-25.pkl",
        help="Length of random walk",
    )
    parser.add_argument(
        "--walks_save_path",
        type=str,
        default="4_Node2Vec/walks/walks-25-25.pkl",
        # default=None,
        help="Length of random walk",
    )
    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    nodes = pd.read_csv(args.nodes_file, names=["node"])
    nodes_num = nodes["node"].max() + 1
    model = SkipGramNegativeSamplingModel(
        nodes_num=nodes_num,
        embedding_dim=args.embedding_dim,
    )
    train_dataset = Node2vecDataset(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        num_walks=args.num_walks,
        length_walk=args.l,
        num_negtive_samples=args.num_negative_samples,
        windom_size=args.window_size,
        p=args.p,
        q=args.q,
        # walks_load_path=args.walks_load_path,
        walks_load_path=None,
        walks_save_path=args.walks_save_path,
        # walks_save_path=None,
    )

    args_dict = vars(args)
    log_recorder = LogRecorder(config=args_dict, info="Deep Walk", verbose=False)
    if args.load_path is not None:
        print(f"Load model parameters from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    model.to(device)
    if args.test == True:
        # print("Evaluating")
        # test(args, model)
        score = eval(args=args, model=model, test_size=0.5)
        print(score)
    else:
        try:
            log_recorder.info = "Negative sampling traing. Dataset blog."
            train(
                args=args,
                model=model,
                train_dataset=train_dataset,
                log_recorder=log_recorder,
            )
        except:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_recorder.save(f"4_Node2Vec/log/f{time_str}.json")
        finally:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_recorder.save(f"4_Node2Vec/log/f{time_str}.json")


if __name__ == "__main__":
    main()
