"""
Pytest configuration for Q-MedCAM.
Pre-loads libgomp to avoid Linux glibc static TLS exhaustion when loading PyTorch and Qiskit C/Rust extensions.
"""

import ctypes

try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

import qiskit  # noqa: F401
import torch  # noqa: F401
