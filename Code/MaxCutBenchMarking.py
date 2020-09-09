#%% Imports
from pennylane import numpy as np # get pennylane's numpy wrapper
import pennylane as qml
from itertools import combinations, groupby
import random
import networkx as nx
from pennylane import expval, var
from functools import partial
from collections import defaultdict
def strdefaultdict():
    return defaultdict(str)

# -----------------
# SECTION 2: QAOA FOR MAX CUT
#%% CONSTRUCTING CONNECTED GRAPHs
"""
Setup a way to construct connected graph instance problems
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
#%%

def comp_basis_measurement(wires):
    n_wires = len(wires)
    return qml.Hermitian(np.diag(range(2 ** n_wires)), wires=wires)

pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z)
# %%
# Leaving this as a function for multiprocessing speedup

def qaoa_maxcut(opt, graph, n_layers, verbose=False):
    if opt == "adam":
        opt = qml.AdamOptimizer(0.1)
    elif opt == "gd":
        opt = qml.GradientDescentOptimizer(0.1)
    elif opt == "qng":
        opt = qml.QNGOptimizer(0.1)
    elif opt == "roto":
        opt = qml.RotosolveOptimizer()
    # SETUP PARAMETERS
    losses = []
    n_wires = len(graph.nodes)
    edges = graph.edges
    grid_size = 100
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



    paramsrecord = [init_params.tolist()]
    print(f"Start objective fn {objective(init_params)}")
    params = init_params

    
    steps = 100
    
    for i in range(steps):
        params = opt.step(objective, params)
        paramsrecord.append(params.tolist())
        losses.append(objective(params))
        print(objective(params))
        if verbose:
            if i % 5 == 0: print(f"Objective at step {i} is {losses[-1]}")
    
    
    objstart, objend = obj_wrapper(init_params)
    meshgridfirststartparams = objstart(X, Y)
    meshgridfirstlastparams = objend(X,Y)
    objstart, objend = obj_wrapper(params)
    meshgridendfirstparams = objstart(X, Y)
    meshgridendlastparams = objend(X,Y)
   
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))

    return {"losses":losses, "params":paramsrecord,\
        "MeshGridStartFirstParams":meshgridfirststartparams, "MeshGridStartLastParams":meshgridfirstlastparams, \
        "MeshGridEndFirstParams":meshgridendfirstparams, "MeshGridEndLastParams":meshgridendlastparams}
'''
Dictionary structure of output will look like
{Optimizer:
  {Graph_Instace:
   {Num_Layers [1,2,3]:
    {losses: 1-dim np.array (iter, loss)}
     params: 2 dim np array (gammas and betas at each iteration)
     MeshGridStartFirstParams: Look at loss landscape at start of network at start of training
     MeshGridStartLastParams : "" End of network
     MeshGridEndFirstParams: ^^ end of training ^^
     MeshGridEndLastParams: ^^ end of training ^^
    }
   }
  }
}
'''

#%%
if __name__ == "__main__":
    
    

    output = defaultdict(strdefaultdict)

    Ns = (4,   8,   12,  12)
    Ps = (0.2, 0.2, 0.3, 0.1)

    GRAPHS = [gnp_random_connected_graph(n,p, 42) for n,p in zip(Ns, Ps)]
    GRAPH_NAMES = ["4, 0.2", "8, 0.2", "12, 0.3", "12, 0.1"]

    OPTIM_NAMES = ["adam", "gd", "roto"]

    arr_map_over = [(i,j,str(k)) for i in OPTIM_NAMES for j in GRAPH_NAMES for k in range(1,3)]
    args_map = [(i,j,k) for i in OPTIM_NAMES for j in GRAPHS for k in range(1,3)]

    import multiprocessing
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.starmap(qaoa_maxcut, args_map)

    for idx, (optim_name, graphname, layerno) in enumerate(arr_map_over):
        output.setdefault(optim_name, {})
        output.get(optim_name).setdefault(graphname, {})
        output.get(optim_name).get(graphname).setdefault(str)
        output[optim_name][graphname][layerno] = results[idx]
    
    import pickle as pkl
    with open("output.pkl", "wb") as f:
        pkl.dump(output,f)

# %%
'''
Old code
'''

''' 
(sequential train)
for opt, name in zip(OPTIMIZERS, OPTIM_NAMES):
    output.setdefault(name, {})
    for graph, graphname in zip(GRAPHS, GRAPH_NAMES):
        output.get(name).setdefault(graphname, {})
        for num_layers in range(1,3):
            print(f"Optimizer {name} is optimizing QAOA with {num_layers} layers on graph with n,p = {graphname}")
            
            output.get(name).get(graphname).setdefault(str)
            output[name][graphname][str(num_layers)] = qaoa_maxcut(graph, num_layers, opt, verbose=True)
            
'''
#%%
# %%
