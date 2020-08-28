#%%
from pennylane import numpy as np # get pennylane's numpy wrapper
import pennylane as qml
from pennylane import expval, var
dev = qml.device("default.qubit", wires=3, analytic=False, shots=20)
@qml.qnode(dev)
def circuit(params):
    qml.RY(np.pi / 4, wires=0)
    qml.RY(np.pi / 3, wires=1)
    qml.RY(np.pi / 7, wires=2)
    qml.RZ(params[0], wires=0)
    qml.RZ(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RY(params[2], wires=1)
    qml.RX(params[3], wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliY(0))

init_params = np.array([0.432, -0.123, 0.543, 0.233])
opt = qml.QNGOptimizer(0.01)
#%%
thetaqng = init_params
for _ in range(100):
    working = False:
    while not working:
        try:
            thetaqng = opt.step(circuit, thetaqng)
            working = True
        except:
            pass
    working = False
    circuit(thetaqng)
    print(thetaqng)
    print(circuit(thetaqng))
# %%

# %%
