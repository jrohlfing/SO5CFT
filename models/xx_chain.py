"""
XX spin chain via Jordan-Wigner -> free fermion exact diagonalisation.

H_XX = -sum_i (S^+_i S^-_{i+1} + h.c.) = -(1/2) sum_i (c^dag_i c_{i+1} + h.c.)

(plus a Jordan-Wigner string that's trivial for nearest-neighbour hopping).

This is c = 1 (free Dirac fermion / compact boson). The ground state at
half-filling has correlation matrix

  C[i, j] = sin(pi (i - j) / 2) / (pi (i - j))           for PBC, half-filling

or, more practically, build C from the single-particle eigenstates and the
fermi-Dirac occupation theta(-eps_k).

OBC version (we use OBC for direct comparison with TFIM, Potts):
  H_ij = -(1/2) delta_{|i-j|, 1}, eigenvalues eps_k, eigenvectors u_k(i).
  Fill the lowest L_chain/2 modes. Then
    C[i, j] = sum_{k filled} u_k(i) u_k(j)
"""

from __future__ import annotations

import numpy as np


def build_correlation_matrix(L: int, hopping: float = 0.5,
                              half_filled: bool = True) -> np.ndarray:
    """
    OBC XX chain. Returns L x L single-particle correlation matrix
    C[i,j] = <c^dag_i c_j> in the ground state at half filling.
    """
    H = np.zeros((L, L), dtype=np.float64)
    for i in range(L - 1):
        H[i, i + 1] = -hopping
        H[i + 1, i] = -hopping
    eps, U = np.linalg.eigh(H)
    n_fill = L // 2 if half_filled else L // 2  # default half-filling
    U_fill = U[:, :n_fill]                       # lowest n_fill modes
    C = U_fill @ U_fill.T.conj()                 # (L, L)
    return C.real
