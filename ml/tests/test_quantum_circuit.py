"""
Unit tests for quantum circuit construction and EstimatorQNN execution.
"""

import numpy as np
import pytest
from src.models.quantum_circuit import build_qmedcam_circuit, create_qmedcam_qnn

def test_circuit_construction():
    circuit, feature_map, ansatz = build_qmedcam_circuit(
        num_qubits=4,
        feature_map_reps=2,
        ansatz_reps=3,
        entanglement="full"
    )
    assert circuit.num_qubits == 4
    # ZZFeatureMap with 4 qubits has 4 input parameters
    assert len(feature_map.parameters) == 4
    # RealAmplitudes with 4 qubits and reps=3 has 4 * (3 + 1) = 16 trainable weights
    assert len(ansatz.parameters) == 16
    assert len(circuit.parameters) == 20

def test_qnn_forward_and_backward():
    qnn, circuit = create_qmedcam_qnn(
        num_qubits=4,
        feature_map_reps=2,
        ansatz_reps=3,
        observable_qubit=0
    )

    batch_size = 2
    inputs = np.random.uniform(-1, 1, size=(batch_size, 4))
    weights = np.random.uniform(-np.pi, np.pi, size=(len(qnn.weight_params),))

    # Forward pass
    forward_out = qnn.forward(inputs, weights)
    assert forward_out.shape == (batch_size, 1)
    # Pauli-Z expectation values are bounded in [-1, 1]
    assert np.all(forward_out >= -1.0 - 1e-5)
    assert np.all(forward_out <= 1.0 + 1e-5)

    # Backward pass (parameter-shift rule)
    input_grads, weight_grads = qnn.backward(inputs, weights)
    assert input_grads.shape == (batch_size, 1, 4)
    assert weight_grads.shape == (batch_size, 1, len(weights))
    assert not np.isnan(input_grads).any()
    assert not np.isnan(weight_grads).any()
