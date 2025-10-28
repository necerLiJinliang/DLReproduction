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
import pickle
from gensim.models import Word2Vec
from train_blog import TopKRanker
from node2vec import Node2Vec


def train(model, walks):
    model.train


def eval(
    args,
    model: Word2Vec,
    test_size: float = 0.5,
):
    def scatter(x: list):
        vector = np.zeros(groups.shape[0])
        vector[x] = 1
        return vector

    # model.eval()
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
    X = [model.wv[str(node)] for node in nodes_list]
    X = np.stack(X, axis=0)
    # X = nodes_features[nodes_list]
    # X = Normalizer(norm="l2").fit_transform(X)
    y = labels
    f1_micro_list = []
    f1_macro_list = []
    for i in tqdm(range(10), desc="Evaluating"):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=i,
        )
        base_classifier = LogisticRegression(solver="liblinear", random_state=i)
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
        f1_micro_list.append(f1_micro)
        f1_macro_list.append(f1_macro)
    f1_micro = np.array(f1_micro_list).mean()
    f1_macro = np.array(f1_macro_list).mean()
    return f1_micro, f1_macro


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
        default="4_Node2Vec/model_save/word2vec-blog.model",
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
        type=float,
        default=4,
        help="Length of random walk",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=4,
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
    parser.add_argument(
        "--alias_path",
        type=str,
        default="4_Node2Vec/walks/alias.pkl",
        # default=None,
        help="Length of random walk",
    )
    args = parser.parse_args()
    # with open(args.walks_load_path, "rb") as f:
    #     walks = pickle.load(f)
    nodes = pd.read_csv(args.nodes_file, names=["node"])["node"].tolist()
    total_words = max(nodes)
    nodes = [str(node) for node in nodes]
    node2vec = Node2Vec(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        num_walks=args.num_walks,
        num_negative_samples=args.num_negative_samples,
        window_size=args.window_size,
        p=args.p,
        q=args.q,
        # alias_path=args.alias_path,
        length_walk=args.l,
    )
    walks = node2vec.random_walk()
    walks = [list(map(str, s)) for s in walks]
    model = Word2Vec(
        # sentences=walks,
        vector_size=128,
        window=10,
        workers=8,
        sg=1,
        epochs=5,
        min_count=0,
        negative=0,
        hs=1
    )
    model.build_vocab(corpus_iterable=[nodes])
    model.train(
        walks,
        total_words=total_words,
        total_examples=len(walks),
        epochs=2,
        # start_alpha=0.25,
        # end_alpha=0.0001,
    )

    # print(f"Save model in {args.save_path}.")
    # model.save(args.save_path)
    scores = eval(args, model=model, test_size=0.5)
    print(scores)


if __name__ == "__main__":
    main()
