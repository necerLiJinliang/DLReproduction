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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from node2vec import Node2Vec
from sklearn.metrics import make_scorer
from sklearn.svm import SVC


def eval(model: Word2Vec, labels: pd.DataFrame):
    nodes = sorted(labels["node"].tolist())
    nodes_str = [str(node) for node in nodes]
    embeddings = [model.wv[node] for node in nodes_str]
    embeddings = np.stack(embeddings, axis=0)
    X = embeddings
    y = np.array(labels["label"].tolist())
    # Split the data into training and testing sets
    # Standardize the embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train a OneVsRest logistic regression classifier with L2 regularization
    classifier = OneVsRestClassifier(LogisticRegression())
    # classifier = OneVsRestClassifier(SVC())

    # Perform 5-fold cross-validation
    mean_accuracies = []
    for i in range(10):
        b_ac = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(classifier, X, y, cv=5, scoring="accuracy")
        mean_accuracy = np.mean(scores)
        mean_accuracies.append(mean_accuracy)

    # Calculate and print the overall mean accuracy
    overall_mean_accuracy = np.mean(mean_accuracies)
    print(f"Overall Mean Accuracy (5 runs of 5-fold CV): {overall_mean_accuracy}")
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
        # default="5_struc2vec/graph/europe-airports.edgelist",
        # default="5_struc2vec/graph/europe-airports.edgelist",
        default="5_struc2vec/graph/karate-mirrored.edgelist",
        help="Path to the graph file",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        # default="5_struc2vec/graph/labels-europe-airports.txt",
        default="5_struc2vec/graph/labels-europe-airports.txt",
        help="Path to the labels file",
    )
    parser.add_argument(
        "--multilayer_graph_object_path",
        type=str,
        # default="5_struc2vec/graph_object/europe.pkl",
        default="5_struc2vec/graph_object/europe.pkl",
        help="Path to the labels file",
    )
    args = parser.parse_args()
    base_graph = build_graph_from_edgelist(args.graph_file_path)
    nodes = list(base_graph.nodes())
    edges = list(base_graph.edges())
    node2vec_obj = Node2Vec(
        num_walks=args.num_walks,
        length_walk=args.length_walk,
        num_negative_samples=5,
        window_size=10,
        p=1,
        q=1,
        nodes=nodes,
        edges=edges,
        alias_path=None,
        alias_save_path=None,
    )
    walks = node2vec_obj.random_walk()
    print(len(walks[0]))
    walks = [[str(s) for s in w] for w in walks]
    model = Word2Vec(
        sentences=walks,
        vector_size=128,
        window=args.window_size,
        min_count=0,
        sg=1,
        hs=1,
        workers=4,
        epochs=20,
    )

    model.save("5_struc2vec/model_save/karate_word2vec_deepwalk.model")
    labels = pd.read_csv(args.labels_path, delimiter=" ")
    # acc = eval(model, labels)


if __name__ == "__main__":
    main()
