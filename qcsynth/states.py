"""
Target quantum states.

Everything that produces a *target statevector* lives here:

* the n-qubit families used as training targets (``W``, ``dicke`` with k=2),
* random states and the Schmidt decomposition used by
  ``prepare_curriculum.py`` to build the top-down curriculum chain for random
  targets (the stages of that chain are then trained as ``custom`` targets).

All statevectors are ``numpy`` arrays of dtype ``complex128`` and length
``2**n_qubits``. Basis-index convention: PennyLane's default, i.e. wire 0 is
the most-significant bit of the basis index.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "random_state",
    "dicke_state",
    "w_state",
    "schmidt_seed",
]


# ---------------------------------------------------------------------------
# n-qubit families
# ---------------------------------------------------------------------------
def random_state(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """Random statevector with i.i.d. complex Gaussian amplitudes, normalised.

    Uses numpy's *global* RNG; if ``seed`` is given, the global RNG is seeded
    first, so the same seed always reproduces the same state.

    Args:
        n_qubits (int): number of qubits; the state has ``2**n_qubits`` amplitudes.
        seed (Optional[int]): seed for numpy's global RNG. ``None`` leaves the RNG
            untouched (non-reproducible draw).

    Returns:
        np.ndarray: normalised statevector, shape ``(2**n_qubits,)``, dtype ``complex128``.
    """
    if seed is not None:
        np.random.seed(seed)
    dim = 2 ** n_qubits
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    return (psi / np.linalg.norm(psi)).astype(np.complex128)


def dicke_state(n_qubits: int, k: int = 2) -> np.ndarray:
    """Dicke state |D^n_k>: equal superposition of all basis states with ``k`` ones.

    Excitation positions ``p`` are mapped to basis indices via ``1 << p``. The
    ``dicke`` target family of this codebase uses ``k = 2``.

    Args:
        n_qubits (int): number of qubits ``n``.
        k (int): number of excitations (ones) in every basis state of the superposition.

    Returns:
        np.ndarray: statevector with ``C(n, k)`` non-zero amplitudes equal to
        ``1 / sqrt(C(n, k))``, shape ``(2**n_qubits,)``, dtype ``complex128``.
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    num_terms = math.comb(n_qubits, k)
    amplitude = 1 / np.sqrt(num_terms)
    for positions in combinations(range(n_qubits), k):
        basis_state = sum(1 << p for p in positions)
        state[basis_state] = amplitude
    return state


def w_state(n_qubits: int) -> np.ndarray:
    """W state: equal superposition of all basis states with exactly one 1
    (a Dicke state with ``k = 1``).

    Args:
        n_qubits (int): number of qubits ``n``.

    Returns:
        np.ndarray: statevector with ``n`` non-zero amplitudes equal to
        ``1 / sqrt(n)``, shape ``(2**n_qubits,)``, dtype ``complex128``.
    """
    state = np.zeros(2 ** n_qubits, dtype=np.complex128)
    amplitude = 1 / np.sqrt(n_qubits)
    for i in range(n_qubits):
        state[1 << i] = amplitude
    return state


# ---------------------------------------------------------------------------
# Schmidt decomposition (curriculum chain for random targets)
# ---------------------------------------------------------------------------
def schmidt_seed(psi: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Schmidt decomposition of an n-qubit state across the cut (qubits 0..n-2) | (qubit n-1).

    Wire 0 is the most-significant bit (PennyLane ordering), so a plain reshape
    to ``(2**(n-1), 2)`` puts the last qubit on the trailing axis.

    Args:
        psi (np.ndarray): n-qubit statevector, shape ``(2**n_qubits,)``.
        n_qubits (int): number of qubits ``n`` of ``psi``.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: ``(phi_A, phi_B, lambda0)`` where

        * ``phi_A`` (np.ndarray, shape ``(2**(n-1),)``): dominant Schmidt vector
          on qubits 0..n-2, i.e. the target of the previous curriculum stage;
        * ``phi_B`` (np.ndarray, shape ``(2,)``): the matching 1-qubit factor on
          qubit n-1;
        * ``lambda0`` (float): largest fidelity any product state
          ``|phi_A> ⊗ |phi_B>`` can reach against ``psi`` (``S[0]**2``).
    """
    dim_A = 2 ** (n_qubits - 1)
    M = np.asarray(psi, dtype=np.complex128).reshape(dim_A, 2)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    phi_A = U[:, 0]
    phi_B = Vh[0, :]  # M = U @ diag(S) @ Vh, so the rank-1 term is U[:,0] ⊗ Vh[0,:]
    lambda0 = float(S[0] ** 2)
    return phi_A, phi_B, lambda0
