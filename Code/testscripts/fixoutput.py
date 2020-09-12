import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pennylane as qml
from pennylane import numpy
import random
from functools import partial
import networkx as nx
from itertools import combinations, groupby
from collections import defaultdict
def strdefaultdict(): # to understand maxcutbenchmark dict
    return defaultdict(str)

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
def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)
# %%
# Leaving this as a function for multiprocessing speedup

def qaoa_costland(graph, n_layers, init_params, final_params):
    # SETUP PARAMETERS
    n_wires = len(graph.nodes)
    edges = graph.edges
    grid_size = 50
    X, Y = np.meshgrid(np.linspace(-np.pi,np.pi,grid_size),np.linspace(-np.pi,np.pi,grid_size))
    def U_B(beta):
        for wire in range(n_wires):
            qml.RX(2 * beta, wires=wire)
    def U_C(gamma):
        for edge in edges:
            wire1 = edge[0]
            wire2 = edge[1]
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(gamma, wires=wire2)
            qml.CNOT(wires=[wire1, wire2])

    dev = qml.device("default.qubit", wires=n_wires, analytic=True, shots=1)
    
    @qml.qnode(dev)
    def circuit(gammas, betas, edge=None, n_layers=1, n_wires=1):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        for i in range(n_layers):
            U_C(gammas[i])
            U_B(betas[i])
        if edges is None:
            # measurement phase
            return qml.sample(comp_basis_measurement(range(n_wires)))
        return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))
    
    def obj_wrapper(params):
        objstart = partial(objective, params, True, False)
        objend = partial(objective, params, False, True)
        return np.vectorize(objstart), np.vectorize(objend)
    
    def objective(params, start=False, end=False, X=None, Y=None):
        gammas = params[0]
        betas = params[1]
        if start:
            gammas[0] = X
            betas[0] = Y
        elif end:
            gammas[-1] = X
            betas[-1] = Y 
        neg_obj = 0
        for edge in edges:
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers, n_wires=n_wires))
        return neg_obj
    
    objstart, objend = obj_wrapper(init_params)
    meshgridfirststartparams = objstart(X, Y)
    meshgridfirstlastparams = objend(X,Y)
    objstart, objend = obj_wrapper(final_params)
    meshgridendfirstparams = objstart(X, Y)
    meshgridendlastparams = objend(X,Y)
   
    return [i.tolist() for i in (meshgridfirststartparams, meshgridfirstlastparams, meshgridendfirstparams, meshgridendlastparams)]

Ns = (4,   8,   12,  12)
Ps = (0.2, 0.2, 0.3, 0.1)

GRAPHS = [gnp_random_connected_graph(n,p, 42) for n,p in zip(Ns, Ps)]
GRAPH_NAMES = ["4, 0.2", "8, 0.2", "12, 0.3", "12, 0.1"]
graphnames_to_idx = {"4, 0.2":0,"8, 0.2":1,"12, 0.3":2,  "12, 0.1":3 }
OPTIM_NAMES = ["adam", "gd", "roto"]
arr_map_over = [(i,j,str(k)) for i in OPTIM_NAMES for j in GRAPH_NAMES for k in range(1,3)]
num_layers = [str(i) for i in range(3)]
with open("../datafiles/output.pkl", "rb") as f:
    maxcutbenchmark = pkl.load(f)

for idx, (i,j,k) in enumerate(arr_map_over):
    print(f"At {idx} of {len(arr_map_over)}")
    init_params, final_params = np.array(maxcutbenchmark[i][j][k]['params'][0]), np.array(maxcutbenchmark[i][j][k]['params'][-1])
    maxcutbenchmark[i][j][k]["meshgrids"] = qaoa_costland(GRAPHS[graphnames_to_idx[j]], int(k), init_params, final_params)

with open("../datafiles/output.pkl", "wb") as f:
    pkl.dump(maxcutbenchmark, f)