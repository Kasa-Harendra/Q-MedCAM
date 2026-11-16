"""
Hardware acceleration and device management for PyTorch and Qiskit.
"""

from typing import Tuple
import torch

def get_torch_device() -> torch.device:
    """
    Selects the optimal compute device for classical deep learning.
    Prefers CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_quantum_backend_name() -> str:
    """
    Determines simulator mode for Qiskit Aer (GPU if available, otherwise CPU).
    """
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        available_methods = sim.available_methods()
        if "statevector_gpu" in available_methods and torch.cuda.is_available():
            return "statevector_gpu"
        return "statevector"
    except Exception:
        return "statevector"
