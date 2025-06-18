import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # Pennylane often uses its own numpy wrapper

# Define n_qubits globally or pass it
n_qubits = 3 # This must match what you used in Colab

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    The quantum circuit for the QNN layer.
    """
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    """
    PyTorch wrapper for the PennyLane quantum circuit.
    """
    def __init__(self):
        super().__init__()
        # weight_shapes must match the weights argument in quantum_circuit
        weight_shapes = {"weights": (1, n_qubits, 3)} # 1 layer, n_qubits, 3 params per qubit
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)

class HybridQuantumModel(nn.Module):
    """
    The full hybrid quantum-classical neural network model.
    """
    def __init__(self):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(3, 3),  # Input features (MolLogP, MolWt, TPSA) -> 3
            nn.Tanh()
        )
        self.q_layer = QuantumLayer()
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 1), # Output from QNN (n_qubits expectation values) -> 1
            nn.Sigmoid()            # For binary classification (probability)
        )

    def forward(self, x):
        x = self.pre_net(x)
        x = self.q_layer(x)
        x = self.post_net(x)
        return x