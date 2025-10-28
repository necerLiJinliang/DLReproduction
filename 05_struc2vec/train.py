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
        default="5_struc2vec/graph/usa-airports.edgelist",
        help="Path to the graph file",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        # default="5_struc2vec/graph/labels-brazil-airports.txt",
        default="5_struc2vec/graph/labels-usa-airports.txt",
        help="Path to the labels file",
    )
    parser.add_argument(
        "--multilayer_graph_object_path",
        type=str,
        # default="5_struc2vec/graph_object/brazil.pkl",
        default="5_struc2vec/graph_object/usa.pkl",
        help="Path to the labels file",
    )
    args = parser.parse_args()
    base_graph = build_graph_from_edgelist(args.graph_file_path)
    if nx.is_connected(base_graph):
        diameter = nx.diameter(base_graph)
    else:
        # 如果图不连通，可以选择计算每个连通子图的直径，或者采取其他措施
        largest_cc = max(nx.connected_components(base_graph), key=len)
        subgraph = base_graph.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        print("Graph is not connected. Using the largest connected component.")
    # diameter = nx.diameter(base_graph)
    load_graph_object = False
    if load_graph_object:
        print("loading graph object.")
        with open(args.multilayer_graph_object_path, "rb") as f:
            multilayer_graph = pickle.load(f)
    else:
        multilayer_graph = MultiLayerWeightedGraph(base_graph=base_graph, k=diameter)
        with open(args.multilayer_graph_object_path, "wb") as f:
            pickle.dump(multilayer_graph, f)
    walks = multilayer_graph.random_walks(
        num_walks=args.num_walks,
        length_walk=args.length_walk,
        p=args.p,
    )
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

    # model.save("5_struc2vec/model_save/karate_word2vec.model")
    labels = pd.read_csv(args.labels_path, delimiter=" ")
    acc = eval(model, labels)


if __name__ == "__main__":
    main()
