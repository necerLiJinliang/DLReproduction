import pandas as pd
import networkx as nx


def build_graph_from_edgelist(file_path):
    edges = pd.read_csv(file_path, delimiter=" ", names=["node1", "node2"])
    nodes = set(edges["node1"]).union(set(edges["node2"]))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges.values.tolist())
    return g
