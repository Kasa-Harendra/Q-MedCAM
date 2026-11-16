"""
Quantum circuit architecture and EstimatorQNN implementation for Q-MedCAM.
Implements:
- 4-qubit ZZFeatureMap (reps=2)
- Trainable RealAmplitudes ansatz (reps=3, full entanglement)
- Pauli-Z expectation value measurement
- EstimatorQNN with exact Parameter-Shift gradient rule and input_gradients=True
"""

from typing import Tuple, Optional
import warnings
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    HAS_AER_V2 = True
except ImportError:
    HAS_AER_V2 = False

from qiskit.primitives import StatevectorEstimator

def build_qmedcam_circuit(
    num_qubits: int = 4,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    entanglement: str = "full"
) -> Tuple[QuantumCircuit, ZZFeatureMap, RealAmplitudes]:
    """
    Constructs the hybrid feature encoding and variational ansatz circuit.
    
    Args:
        num_qubits: Dimension of quantum state (matches 4D classical bottleneck).
        feature_map_reps: Number of ZZFeatureMap entangling layers.
        ansatz_reps: Number of variational ansatz layers.
        entanglement: Entanglement topology ('full' or 'linear').
        
    Returns:
        composite_circuit: Feature map composed with ansatz.
        feature_map: Parameterized feature encoding circuit.
        ansatz: Variational ansatz with trainable parameters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=feature_map_reps,
            entanglement=entanglement,
            parameter_prefix="x"
        )

        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=ansatz_reps,
            entanglement=entanglement,
            parameter_prefix="theta"
        )

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    return circuit, feature_map, ansatz

def create_qmedcam_qnn(
    num_qubits: int = 4,
    feature_map_reps: int = 2,
    ansatz_reps: int = 3,
    entanglement: str = "full",
    observable_qubit: int = 0,
    multi_observable: bool = False
) -> Tuple[EstimatorQNN, QuantumCircuit]:
    """
    Creates an EstimatorQNN that computes Pauli-Z expectation values using
    the parameter-shift rule (REQ-6) and enables input_gradients for autograd routing.
    If multi_observable=True, returns 4 observables measuring <Z_i> on each individual qubit.
    """
    circuit, feature_map, ansatz = build_qmedcam_circuit(
        num_qubits=num_qubits,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        entanglement=entanglement
    )

    if multi_observable:
        # Measure Pauli-Z on all individual qubits: [IIIZ, IIZI, IZII, ZIII]
        observables = []
        for q in range(num_qubits):
            pauli_chars = ["I"] * num_qubits
            pauli_chars[num_qubits - 1 - q] = "Z"
            observables.append(SparsePauliOp.from_list([("".join(pauli_chars), 1.0)]))
    else:
        # Observable: Pauli-Z measurement on designated qubit (e.g. qubit 0)
        pauli_chars = ["I"] * num_qubits
        pauli_chars[num_qubits - 1 - observable_qubit] = "Z"
        pauli_string = "".join(pauli_chars)
        observables = SparsePauliOp.from_list([(pauli_string, 1.0)])

    # Statevector estimator for exact analytical parameter-shift gradient computation
    estimator = StatevectorEstimator()

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        input_gradients=True
    )

    return qnn, circuit
