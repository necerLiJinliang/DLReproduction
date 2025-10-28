from graph import MultiLayerWeightedGraph
from gensim.models import Word2Vec
import argparse
import networkx as nx
from utils import build_graph_from_edgelist
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def eval(model: Word2Vec, labels: pd.DataFrame):
    nodes = sorted(labels["node"].tolist())
    nodes_str = [str(node) for node in nodes]
    embeddings = [model.wv[node] for node in nodes_str]
    # embeddings = np.stack(embeddings, axis=0)
    embeddings = np.random.randn(len(nodes), 128)
    X = embeddings
    y = np.array(labels["label"].tolist())
    # Split the data into training and testing sets
    # Standardize the embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train a OneVsRest logistic regression classifier with L2 regularization
    classifier = OneVsRestClassifier(SVC())

    # Perform 5-fold cross-validation
    scores = cross_val_score(classifier, X, y, cv=5, scoring="recall_macro")

    # Calculate and print the mean accuracy
    mean_accuracy = np.mean(scores)
    print(f"Mean Accuracy (5-fold CV): {mean_accuracy}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a model on the karate dataset using struc2vec."
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=10,
        help="Number of walks per node",
    )
    parser.add_argument(
        "--length_walk",
        type=int,
        default=80,
        help="Length of each walk",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="Return parameter",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of layers",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for Word2Vec",
    )
    parser.add_argument(
        "--graph_file_path",
        type=str,
        # default="5_struc2vec/graph/brazil-airports.edgelist",
        default="5_struc2vec/graph/brazil-airports.edgelist",
        help="Path to the graph file",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        # default="5_struc2vec/graph/labels-brazil-airports.txt",
        default="5_struc2vec/graph/labels-brazil-airports.txt",
        help="Path to the labels file",
    )
    parser.add_argument(
        "--multilayer_graph_object_path",
        type=str,
        # default="5_struc2vec/graph_object/brazil.pkl",
        default="5_struc2vec/graph_object/brazil.pkl",
        help="Path to the labels file",
    )
    args = parser.parse_args()
    model = Word2Vec.load("5_struc2vec/model_save/brazil-airports.emb")

    # model.save("5_struc2vec/model_save/karate_word2vec.model")
    labels = pd.read_csv(args.labels_path, delimiter=" ")
    acc = eval(model, labels)


if __name__ == "__main__":
    main()
