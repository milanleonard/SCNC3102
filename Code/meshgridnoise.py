#%% Imports
from pennylane import numpy as pnp # get pennylane's numpy wrapper
import numpy as np
import pennylane as qml
from itertools import combinations, groupby
import qiskit
import random
import networkx as nx
from pennylane import expval, var
from functools import partial
from collections import defaultdict
import qiskit.providers.aer.noise as noise

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

def qaoa_maxcut_grid_noise(graph, n_layers, shots=5000, NoiseModel=None):

    n_wires = len(graph.nodes)
    edges = graph.edges

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
    
    dev = qml.device("qiskit.aer", wires=n_wires, shots=shots, noise_model=NoiseModel)

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
    
    init_params = 0.01 * np.random.rand(2, n_layers)
    
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
    
    grid_size = 50
    X, Y = np.meshgrid(np.linspace(-np.pi,np.pi,grid_size),np.linspace(-np.pi,np.pi,grid_size))
    objstart, objend = obj_wrapper(init_params)
    meshgridfirststartparams = objstart(X, Y)
    meshgridfirstlastparams = objend(X,Y)

    return meshgridfirststartparams, meshgridfirstlastparams


if __name__ == "__main__":
    import multiprocessing
    noise_args = np.linspace(0,0.15,15)
    Noise_Models = [noise.NoiseModel() for i in range(10)]
    for noise_arg, noisemodel in zip(noise_args, Noise_Models):
        noisemodel.add_all_qubit_quantum_error(noise.depolarizing_error(noise_arg ,1), ['u1','u2','u3'])

    TEST_G = gnp_random_connected_graph(4,0.2,42)
    args = [(TEST_G, 3, 5000, noisemodel) for noisemodel in Noise_Models]
    with multiprocessing.Pool(15) as p:
        results = p.starmap(qaoa_maxcut_grid_noise, args)
    
    OUTPUT_ARR = np.zeros(10,50,50)
    for idx, result in enumerate(results):
        OUTPUT_ARR[idx] = result
    np.save("./datafiles/meshgridsnoisy.npy", OUTPUT_ARR)
