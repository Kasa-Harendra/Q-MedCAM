"""
Q-MedCAM: Hybrid Quantum-Classical Explainable Skin Lesion Screener.
"""

import ctypes

# Pre-load libgomp globally to prevent glibc static TLS block exhaustion
# across PyTorch OpenMP and Qiskit Rust accelerate native binaries.
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import qiskit  # noqa: F401
import torch  # noqa: F401

__version__ = "2.0.0"
