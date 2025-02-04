#%% Imports
from pennylane import numpy as np # get pennylane's numpy wrapper
import pennylane as qml
from pennylane import expval, var
from multiprocessing import Process
#%%
# -----------------
## COMPUTE TRUE MINIMUM
# SECTION 1: SINGLE VQE PROBLEM - QNG, SGD, ADAM etc. GO THROUGH EACH OF THE GRADIENT BASED UPDATE METHODS
""" SETUP VARIATONAL CIRCUIT: TAKEN FROM https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html"""
DENSITY = 100
SHOTRANGE = np.linspace(10,1000,DENSITY)
STEPS = 500
init_params = np.array([0.432, -0.123, 0.543, 0.233])
GD_COST = np.zeros((DENSITY,STEPS))
ROTO_COST = np.zeros((DENSITY,STEPS))
ADAM_COST = np.zeros((DENSITY,STEPS))
QNG_COST = np.zeros((DENSITY,STEPS))

def initialize(shotnum):
    dev = qml.device("default.qubit", wires=3, analytic=False, shots=shotnum)
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
    return circuit
if __name__ == "__main__":
    def GDOPT():
        for x_idx, shotnum in enumerate(SHOTRANGE):
            circuit = initialize(shotnum)
            print(f"round {x_idx} of {DENSITY} with {shotnum} shots")
            
            opt = qml.GradientDescentOptimizer(0.01)
            print("Doing the gradient descent")
            thetagd = init_params
            for y_idx in range(STEPS):
                thetagd = opt.step(circuit, thetagd)
                GD_COST[x_idx,y_idx] = circuit(thetagd)
        print("finally save GD cost file")
        np.save("./datafiles/gdshotcosttoy.npy",GD_COST)
        return # exit for GC 
    
    def QNGOPT():
        for x_idx, shotnum in enumerate(SHOTRANGE):
            #%% Quantum natural gradient
            circuit = initialize(shotnum)
            print("Doing the quantum natural gradient")
            thetaqng = init_params
            opt = qml.QNGOptimizer(0.01)

            for y_idx in range(STEPS):
                working = False
                while not working:
                    try:
                        thetaqng = opt.step(circuit, thetaqng)
                        QNG_COST[x_idx,y_idx] = circuit(thetaqng)
                        working = True
                    except:
                        pass
                working = False
        np.save("./datafiles/qngshotcosttoy.npy",QNG_COST)

        #%% ROTOSOLVE
    def ROTOOPT():
        for x_idx, shotnum in enumerate(SHOTRANGE):
            circuit = initialize(shotnum)
            print("Doing rotosolve")
            opt = qml.RotosolveOptimizer()

            thetart = init_params
            for y_idx in range(STEPS):
                thetart = opt.step(circuit, thetart)
                ROTO_COST[x_idx,y_idx] = circuit(thetart)
        np.save("./datafiles/rotoshotcosttoy.npy",ROTO_COST)


        print("Doing adam")
    def ADAMOPT():
        for x_idx, shotnum in enumerate(SHOTRANGE):
            circuit = initialize(shotnum)
            opt = qml.AdamOptimizer(0.01)
            thetaadam = init_params
            for y_idx in range(STEPS):
                thetaadam = opt.step(circuit, thetaadam)
                ADAM_COST[x_idx,y_idx] = circuit(thetaadam)
        np.save("./datafiles/adamshotcosttoy.npy",ADAM_COST)

if __name__ == "__main__":
    gdop = Process(target=GDOPT)
    adamop = Process(target=ADAMOPT)
    qngop = Process(target=QNGOPT)
    rotoop = Process(target=ROTOOPT)
    gdop.start()
    adamop.start()
    qngop.start()
    rotoop.start()
    gdop.join()
    rotoop.join()
    qngop.join()
    adamop.join()
    print("Finished all, exiting")

# %%
