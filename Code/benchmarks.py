#%% Imports
from pennylane import numpy as np # get pennylane's numpy wrapper
import qiskit
import pennylane as qml
from itertools import combinations, groupby
import matplotlib.pyplot as plt
import random
import networkx as nx
from pennylane import expval, var

#%%
# -----------------
## COMPUTE TRUE MINIMUM
# SECTION 1: SINGLE VQE PROBLEM - QNG, SGD, ADAM etc. GO THROUGH EACH OF THE GRADIENT BASED UPDATE METHODS
""" SETUP VARIATONAL CIRCUIT: TAKEN FROM https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html"""
dev = qml.device("default.qubit", wires=3)
@qml.qnode(dev)
def circuit(params):
    # |psi_0>: state preparation
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 7, wires=2)

    # V0(theta0, theta1): Parametrized layer 0
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)

    # W1: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    # V_1(theta2, theta3): Parametrized layer 1
    qml.RY(params[2], wires=1)
    qml.RX(params[3], wires=2)

    # W2: non-parametrized gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])

    return qml.expval(qml.PauliY(0)) 
# %% GLOBAL PARAMS FOR EACH TEST
steps = 400
init_params = np.array([0.432, -0.123, 0.543, 0.233])

#%%

GradientDescentCost = [circuit(init_params)]
opt = qml.GradientDescentOptimizer(0.01)

thetagd = init_params
for _ in range(steps):
    thetagd = opt.step(circuit, thetagd)
    GradientDescentCost.append(circuit(thetagd))
#%% Quantum natural gradient

Quantum_natural_GD_Cost = [circuit(init_params)]

opt = qml.QNGOptimizer(0.01)

thetaqng = init_params
for _ in range(steps):
    thetaqng = opt.step(circuit, thetaqng)
    Quantum_natural_GD_Cost.append(circuit(thetaqng))

#%% ROTOSOLVE

Rotosolve_Cost = [circuit(init_params)]

opt = qml.RotosolveOptimizer()

thetart = init_params
for _ in range(steps):
    thetart = opt.step(circuit, thetart)
    Rotosolve_Cost.append(circuit(thetart))

#%% ADAM OPTIMIZER
Adam_Cost = [circuit(init_params)]

opt = qml.AdamOptimizer(0.01)

thetaadam = init_params
for _ in range(steps):
    thetaadam = opt.step(circuit, thetaadam)
    Adam_Cost.append(circuit(thetaadam))

#%% Plotting
def plot_descents(save=False):
    plt.xlabel("Number of _iterations_")
    plt.ylabel("Cost functions")
    plt.title("Optimizer performance")
    plt.plot(GradientDescentCost, label="Standard gradient descent")
    plt.plot(Quantum_natural_GD_Cost, label="Quantum Natural Gradient")
    plt.plot(Rotosolve_Cost, label="Rotosolve analytic minimum")
    plt.plot(Adam_Cost, label="ADAM optimizer")
    plt.legend()
    if save:
        plt.savefig("./Images/descentcomparisons.png")
    plt.show()
#%% Running the plot
plot_descents(save=False)
2#%% Let's have a look at the cost landscape
from functools import partial
def wrap_circuit(param1,param2):
    def circuit_wrap(theta0,theta1,theta2,theta3):
        return circuit(np.array([theta0,theta1,theta2,theta3]))
    circuitwrapvec = np.vectorize(partial(circuit_wrap,param1,param2))
    return circuitwrapvec
#circuitwrapvec = wrap_circuit(theta[0],theta[1])
circuitwrapvec = wrap_circuit(thetart[0],thetart[1])
grid_size = 30
X, Y = np.meshgrid(np.linspace(-np.pi,np.pi,grid_size),np.linspace(-np.pi,np.pi,grid_size))
Z = circuitwrapvec(X,Y)

#%%
# Plot this stuff
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
ax.set_title('Cost landscape for two parameters')
ax.set_xlabel("$\theta_1$")
ax.set_ylabel("$\theta_2$")
ax.set_zlabel("$\mathcal{L}(\mathbb{\theta})")
plt.show()

# -----------------
# SECTION 2: QAOA FOR MAX CUT
#%% CONSTRUCTING CONNECTED GRAPHS
"""
COMING TO THIS LATER -- THIS IS FOR A BENCHMARK PROBLEM
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
