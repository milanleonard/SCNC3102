""" PennyLane implementation of Multiobjective genetic variational quantum eigensolver
https://arxiv.org/abs/2007.04424 D. Chivilikhin, A. Samarin ...
"""
import pennylane as qml
from pennylane import numpy as np

class MoGOptimizer:
    def __init__(mutation_rate:float):
        raise NotImplemented



    def _block_1(params:[float], wires:[int]):
        """ Block (a) as defined in Fig. 2 of the reference paper
            Args:
                params [float]: the parameters of this layer, do not need to be trainable
                wires (int,int): the two wires to apply this circuit to
        """

        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[0])
        qml.RZ(params[2], wires=wires[1])
        qml.RY(params[3], wires=wires[1])
        qml.CNOT(wires=wires)
        qml.RY(-params[0], wires=wires[0])
        qml.RZ(-params[1], wires=wires[0])
        qml.RY(-params[2], wires=wires[1])
        qml.RZ(-params[3], wires=wires[1])
    
    def _block_2(params, wires:[int], identity=False):
        """ Block (b) as defined in Fig. 2 of the reference paper
            Args:
                params [float]: the parameters of this layer, do not need to be trainable
                wires (int,int): the two wires to apply this circuit to
                init (bool): if True the block will initially act as an identity matrix
        """
        if identity==True:
            params[3] = np.pi
            params[4] = 2*np.pi - params[2]
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[0])
        qml.RZ(params[2], wires=wires[1])
        qml.RY(params[3], wires=wires[1])
        qml.CNOT(wires=wires)
        qml.RY(-params[3], wires=wires[1])
        qml.RZ(-params[2]/2 - params[4]/2, wires=wires[1])
        qml.CNOT(wires=wires)
        qml.RY(-params[1], wires=wires[0])
        qml.RZ(-params[0], wires=wires[0])
        qml.RZ(-params[2]/2 + params[4]/2)

    def _initial_circuit(num_qubits:int):
        """ Creates the initial circuit, 1/2 the time goes through each qubit and randomly picks block one or 2 for this wire
        and the next. Entangles all qubits. Alternatively, Choose a random number from N_init uniformly from [N,4N] and add
        N_init blocks randomly targeting (in my implementation, adjacent due to efficiency constraints)
        """
        first_choice = np.random.uniform(0,1)
        if first_choice <= 0.5:
            for i in range(0,num_qubits-1):
                second_choice = np.random.uniform(0,1)
                if second_choice <= 0.5:
                    params = np.random.uniform(-np.pi, np.pi, size=4)
                    _block_1(params, wires=[i,i+1])
                else:
                    params = np.random.uniform(-np.pi, np.pi, size=5)
                    _block_2(params, wires=[i,i+1])
        else:
            N_init = np.random.uniform(num_qubits, 4*num_qubits)
            for i in range(N_init):
                wire1 = np.random.randint(0,num_qubits-1)
                second_choice = np.random.uniform(0,1)
                if second_choice <= 0.5:
                    params = np.random.uniform(-np.pi, np.pi, size=4)
                    _block_1(params, wires=[wire1,wire1+1])
                else:
                    params = np.random.uniform(-np.pi, np.pi, size=5)
                    _block_2(params, wires=[wire1,wire1+1])








