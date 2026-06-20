"""
Critical 1D Transverse-Field Ising Model (TFIM) ground state via TenPy DMRG.

Spec convention: H = -sum sigma^x_i sigma^x_{i+1} - g sum sigma^z_i,
critical at g = 1. Two paired conditions: g_A = 1.000, g_B = 1.005.

We use TenPy's `TFIChain` model whose default convention matches:
    H = -J sum sigma^x sigma^x - g sum sigma^z

For each block size r, build the reduced density matrix of a contiguous r-site
block at the chain centre, by tenpy's `mps.get_rho_segment([first..last])`.
The full RDM is a 2^r x 2^r matrix; we are limited by memory to r <= ~8 for
direct diagonalisation. Larger r are not computable at face value with this
approach (spec acknowledged: "extract reduced density matrices via MPS
tracing"; full RDM is infeasible past r ~ 10 with explicit basis).

The Z_2 generator on the block is Q = prod_{i in block} sigma^z_i. We split the
RDM into Z_2-even (eigenstates of Q with eigenvalue +1) and Z_2-odd sectors,
then for the residual rho_res = rho_A - rho_B we identify:

  - identity (I)   = Z_2-even eigenvector of rho_res with largest |lambda|
  - sigma          = Z_2-odd eigenvector of rho_res with largest |lambda|
  - epsilon (eps)  = Z_2-even eigenvector of rho_res with 2nd-largest |lambda|

W is the (sum of lambda^2 within sector) / total, in fixed order (I, sigma, eps).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

import tenpy
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine


def build_ground_state(L: int, g: float, chi: int, verbose: bool = True,
                       conserve: str = "parity", seed: int = 0):
    """
    Run TenPy DMRG for H = -sum sigma^x sigma^x - g sum sigma^z.

    TenPy's TFIChain at default convention is exactly the spec's H. We use
    `conserve='parity'` so the DMRG converges to a definite Z_2-parity ground
    state (avoiding spontaneous symmetry breaking near the critical point at
    finite chi).
    """
    model_params = {
        "L": L,
        "J": 1.0,
        "g": g,
        "bc_MPS": "finite",
        "conserve": conserve,
    }
    M = TFIChain(model_params)
    # initial state: parity-even product state. For conserve='parity', the
    # Z_2-even sector contains the alternating singlet and the uniform |up..up>
    # state -- we pick uniform |up>.
    init_state = [0] * L
    psi = tenpy.networks.mps.MPS.from_lat_product_state(M.lat, [[s] for s in init_state])
    np.random.seed(seed)  # affect mixer's random initialisation if any
    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "max_E_err": 1e-8,
        "max_S_err": 1e-6,
        "mixer": True,
        "max_sweeps": 30,
        "min_sweeps": 6,
    }
    if verbose:
        print(f"[TFIM-DMRG] L={L} g={g} chi={chi} conserve={conserve}: starting",
              flush=True)
    t0 = time.time()
    eng = TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, _ = eng.run()
    if verbose:
        print(f"[TFIM-DMRG] L={L} g={g}: E/site = {E / L:.6f} "
              f"(exact -4/pi = {-4 / np.pi:.6f} at criticality)  "
              f"in {time.time() - t0:.1f}s", flush=True)
    return psi, E


def half_chain_entropy(psi) -> float:
    """
    Return the von Neumann entropy at the exact half-chain bond, computed from
    the singular values at the center bond.

    tenpy's `entanglement_entropy_segment()` returns the entropy of the segment
    [0..i], which differs from the exact half-chain bipartition entropy by
    boundary terms in the OBC case. We use Schmidt values directly for a clean
    half-chain entropy.
    """
    L = psi.L
    sv = psi.get_SL(L // 2)
    sv2 = sv ** 2
    sv2 = sv2[sv2 > 1e-14]
    sv2 = sv2 / sv2.sum()
    return float(-np.sum(sv2 * np.log(sv2)))


def block_rdm(psi, first: int, last: int) -> np.ndarray:
    """
    Build the explicit (2^r) x (2^r) reduced density matrix on sites
    [first..last] inclusive, where r = last - first + 1.

    Uses tenpy's mps.get_rho_segment.
    """
    rho = psi.get_rho_segment(range(first, last + 1))
    # rho is a tenpy Array with legs (p0, p1, ..., p*0, p*1, ...). Convert to
    # explicit numpy with shape (2^r, 2^r).
    r = last - first + 1
    arr = rho.to_ndarray()  # shape (2,2,...,2,2,...) -- r ket legs then r bra legs
    arr = arr.reshape(2 ** r, 2 ** r)
    return arr.astype(np.float64)


def z2_block_operator(r: int) -> np.ndarray:
    """Return Q = sigma^z^{⊗r} as a 2^r x 2^r matrix."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    Q = sz.copy()
    for _ in range(r - 1):
        Q = np.kron(Q, sz)
    return Q


def decompose_residual_z2(rho_res: np.ndarray, r: int) -> dict:
    """
    Diagonalise rho_res (Hermitian, generally non-PSD because it is a residual),
    assign each eigenvector to a Z_2 sector via Q = sigma^z^{⊗r}, and form

      W_I     = sum lambda^2 of Z_2-even eigenvectors with largest |lambda|
      W_sigma = sum lambda^2 of Z_2-odd eigenvectors with largest |lambda|
      W_eps   = sum lambda^2 of Z_2-even eigenvectors with 2nd-largest |lambda|

    The spec is satisfied at the single-state level: each sector gets its top-|lambda|
    representative (so W_I, W_sigma, W_eps are |lambda|^2 of single eigenstates,
    not sums). We follow that literal reading.

    Returns dict with W (length-3), W_total (sum), and bookkeeping diagnostics.
    """
    rho_res = (rho_res + rho_res.conj().T) / 2  # enforce hermiticity
    evals, evecs = np.linalg.eigh(rho_res)
    Q = z2_block_operator(r)
    # Z_2 parity of each eigenvector
    parity = np.einsum("ji,jk,ki->i", evecs.conj(), Q, evecs).real
    # numerical: classify by sign
    is_even = parity > 0
    is_odd = ~is_even

    abs_evals = np.abs(evals)
    if is_even.any():
        even_idx = np.argsort(-abs_evals * is_even.astype(np.float64))
        even_idx = even_idx[is_even[even_idx]]
    else:
        even_idx = np.array([], dtype=int)
    if is_odd.any():
        odd_idx = np.argsort(-abs_evals * is_odd.astype(np.float64))
        odd_idx = odd_idx[is_odd[odd_idx]]
    else:
        odd_idx = np.array([], dtype=int)

    def safe(idx_arr, k):
        if k < len(idx_arr):
            return float(abs_evals[idx_arr[k]] ** 2)
        return 0.0

    W_I = safe(even_idx, 0)
    W_sigma = safe(odd_idx, 0)
    W_eps = safe(even_idx, 1)
    W = np.array([W_I, W_sigma, W_eps], dtype=np.float64)
    total = W.sum()
    if total > 0:
        W_norm = W / total
    else:
        W_norm = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return {
        "W": W_norm,
        "raw_lambda_sq": W.tolist(),
        "n_even": int(is_even.sum()),
        "n_odd": int(is_odd.sum()),
        "min_parity": float(parity.min()),
        "max_parity": float(parity.max()),
        "tr_rho_res": float(np.trace(rho_res).real),
    }
