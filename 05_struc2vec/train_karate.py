from graph import MultiLayerWeightedGraph
from gensim.models import Word2Vec
import argparse
import networkx as nx
from utils import build_graph_from_edgelist


def main():
    parser = argparse.ArgumentParser(
        description="Train a model on the karate dataset using struc2vec."
    )
    parser.add_argument(
        "--num_walks",
        type=int,
        default=5,
        help="Number of walks per node",
    )
    parser.add_argument(
        "--length_walk",
        type=int,
        default=15,
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
        default=3,
        help="Window size for Word2Vec",
    )
    parser.add_argument(
        "--graph_file_path",
        type=str,
        default="5_struc2vec/graph/karate-mirrored.edgelist",
        help="Path to the graph file",
    )
    args = parser.parse_args()
    base_graph = build_graph_from_edgelist(args.graph_file_path)

    # karate_club_graph_1 = nx.karate_club_graph()
    # karate_club_graph_2 = nx.karate_club_graph()

    # combined_graph = nx.disjoint_union(karate_club_graph_1, karate_club_graph_2)
    # combined_graph.add_edge(0, len(karate_club_graph_1))

    # base_graph = combined_graph

    multilayer_graph = MultiLayerWeightedGraph(base_graph=base_graph, k=args.k)
    walks = multilayer_graph.random_walks(
        num_walks=args.num_walks,
        length_walk=args.length_walk,
        p=args.p,
    )
    print(len(walks))
    walks = [[str(s) for s in w] for w in walks]
    model = Word2Vec(
        sentences=walks,
        vector_size=128,
        window=args.window_size,
        min_count=0,
        sg=1,
        hs=1,
        workers=4,
    )
    model.save("5_struc2vec/model_save/karate_word2vec.model")


if __name__ == "__main__":
    main()
