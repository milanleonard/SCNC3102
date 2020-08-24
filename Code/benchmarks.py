#%% Imports
from pennylane import numpy as np # get pennylane's numpy wrapper
import qiskit
import pennylane as qml
from itertools import combinations, groupby
import random
import networkx as nx
# -----------------
# SECTION 1: QAOA FOR MAX CUT
#%% CONSTRUCTING CONNECTED GRAPHS
"""
Need to setup a way to construct connected graphs
"""
def gnp_random_connected_graph(n, p, seed):
    """Generate a random connected graph
    n     : int, number of nodes
    p     : float in [0,1]. Probability of creating an edge
    seed  : int for initialising randomness
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


# %% Set-up VQE problems
