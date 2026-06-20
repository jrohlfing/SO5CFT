"""
Effective central charge estimator from log-scaling of block entanglement
(quantum pure states) or mutual information (classical multivariate data).

Two estimator families, both reduce to a slope fit against the Calabrese-Cardy
chord length:

    chord(L; N) = (N / pi) * sin(pi * L / N)

For QUANTUM pure states (TenPy MPS, free-fermion correlation matrix):
    S(L) = (c/6) ln(chord(L; N)) + const     [OBC single-cut convention]
    c_eff_quantum = 6 * slope

For CLASSICAL / GAUSSIAN multivariate signals (N channels x T samples):
    I(A:B) = (c/3) ln(chord(L; N)) + const   [PBC two-cut on classical]
    c_eff_classical = 3 * slope

For a pure quantum state these agree exactly (I = 2S, so slopes differ by 2,
and the prefactors differ by 2, giving the same c).

Fit quality is gated by R^2 >= R2_MIN (default 0.9). The scaling window is
specified explicitly (L_min, L_max as fraction of N); we exclude the smallest
L (lattice artefact) and largest L (saturation). Documented in every result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


R2_MIN = 0.9


def chord_length(L: np.ndarray, N: int) -> np.ndarray:
    """Calabrese-Cardy chord length. L can be array; returns array."""
    return (N / np.pi) * np.sin(np.pi * L / N)


@dataclass
class CCFit:
    slope: float
    intercept: float
    r2: float
    c_eff: float
    L_used: np.ndarray
    y_used: np.ndarray
    x_used: np.ndarray         # ln(chord)
    window: tuple[float, float]
    convention: str            # "quantum" or "classical"
    n_pts: int
    accepted: bool             # r2 >= R2_MIN

    def to_dict(self):
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r2": self.r2,
            "c_eff": self.c_eff,
            "L_used": self.L_used.tolist(),
            "window": list(self.window),
            "convention": self.convention,
            "n_pts": self.n_pts,
            "accepted": self.accepted,
        }


def fit_central_charge(L: np.ndarray, y: np.ndarray, N: int,
                       convention: str,
                       window: tuple[float, float] = (0.10, 0.50),
                       r2_min: float = R2_MIN) -> CCFit:
    """
    Fit y(L) = a * ln(chord(L;N)) + b for L in the scaling window
    [window[0]*N, window[1]*N]. Convention determines the prefactor:
      "quantum"   -> c_eff = 6 * a   (S(L) for OBC pure state, single cut)
      "classical" -> c_eff = 3 * a   (I(A:B) for classical, two-cut)

    y is the quantity being fit (S(L) or I(L)). Returns a CCFit dataclass
    with accepted=True iff r^2 >= r2_min and >= 4 points in the window.
    """
    L = np.asarray(L, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Lmin = window[0] * N
    Lmax = window[1] * N
    mask = (L >= Lmin) & (L <= Lmax) & np.isfinite(y)
    if mask.sum() < 4:
        return CCFit(slope=np.nan, intercept=np.nan, r2=np.nan, c_eff=np.nan,
                     L_used=L[mask], y_used=y[mask], x_used=np.array([]),
                     window=window, convention=convention,
                     n_pts=int(mask.sum()), accepted=False)
    x = np.log(chord_length(L[mask], N))
    yy = y[mask]
    slope, intercept = np.polyfit(x, yy, 1)
    yhat = slope * x + intercept
    ss_res = np.sum((yy - yhat) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    prefactor = 6.0 if convention == "quantum" else 3.0
    c_eff = prefactor * slope
    return CCFit(slope=float(slope), intercept=float(intercept), r2=float(r2),
                 c_eff=float(c_eff),
                 L_used=L[mask], y_used=yy, x_used=x,
                 window=window, convention=convention,
                 n_pts=int(mask.sum()),
                 accepted=bool(r2 >= r2_min))


# ----- Gaussian mutual information estimator -------------------------------

def block_mi_gaussian(X: torch.Tensor | np.ndarray,
                      L_values: list[int],
                      shrink: float = 1e-4,
                      device: str | torch.device | None = None) -> np.ndarray:
    """
    Compute I(A:B) for A = first L channels, B = remaining N-L channels,
    using the Gaussian covariance formula:

      I(A:B) = 0.5 * (logdet(Cov_AA) + logdet(Cov_BB) - logdet(Cov_full))

    X: (N, T) multivariate signal (channels x time).
    L_values: list of block sizes to evaluate.
    shrink: Ledoit-Wolf-like shrinkage toward diagonal for numerical stability:
            Cov_used = (1-shrink) * Cov_sample + shrink * diag(Cov_sample).

    Returns I(L) as a length-len(L_values) numpy array.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if device is None:
        device = X.device if X.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device).to(torch.float64)
    N, T = X.shape
    Xc = X - X.mean(dim=1, keepdim=True)
    Cov = (Xc @ Xc.T) / max(T - 1, 1)
    diag = torch.diag(torch.diagonal(Cov))
    Cov_s = (1.0 - shrink) * Cov + shrink * diag
    # logdet of full once
    sign_full, logdet_full = torch.linalg.slogdet(Cov_s)
    I_list = []
    for L in L_values:
        if L < 1 or L > N - 1:
            I_list.append(np.nan)
            continue
        Cov_AA = Cov_s[:L, :L]
        Cov_BB = Cov_s[L:, L:]
        s1, ld_A = torch.linalg.slogdet(Cov_AA)
        s2, ld_B = torch.linalg.slogdet(Cov_BB)
        if s1.item() <= 0 or s2.item() <= 0 or sign_full.item() <= 0:
            I_list.append(np.nan)
            continue
        I = 0.5 * (ld_A.item() + ld_B.item() - logdet_full.item())
        I_list.append(I)
    return np.array(I_list, dtype=np.float64)


# ----- Quantum entanglement from MPS Schmidt values -----------------------

def mps_block_entropy(psi, L_values: list[int]) -> np.ndarray:
    """
    Entanglement entropy at bond L (between sites L-1 and L) of a TenPy MPS.
    Returns S(L) as length-len(L_values) numpy array.
    """
    S_list = []
    for L in L_values:
        sv = psi.get_SL(L)
        sv2 = sv ** 2
        sv2 = sv2[sv2 > 1e-14]
        sv2 = sv2 / sv2.sum()
        S_list.append(float(-np.sum(sv2 * np.log(sv2))))
    return np.array(S_list, dtype=np.float64)


# ----- Free-fermion entanglement from correlation matrix ------------------

def free_fermion_block_entropy(C: np.ndarray, L_values: list[int]) -> np.ndarray:
    """
    For a free-fermion ground state with single-particle correlation matrix
    C[i,j] = <c^dag_i c_j>, the entanglement entropy of a contiguous block
    [0..L-1] is:

      S(L) = -sum_k [eta_k log(eta_k) + (1-eta_k) log(1-eta_k)]

    where eta_k are eigenvalues of C_{AA} = C[:L, :L].
    """
    S_list = []
    for L in L_values:
        if L < 1 or L >= C.shape[0]:
            S_list.append(np.nan)
            continue
        CAA = C[:L, :L]
        eta = np.linalg.eigvalsh((CAA + CAA.conj().T) / 2.0)
        eta = np.clip(eta, 1e-14, 1.0 - 1e-14)
        s = -(eta * np.log(eta) + (1.0 - eta) * np.log(1.0 - eta)).sum()
        S_list.append(float(s.real))
    return np.array(S_list, dtype=np.float64)


# ----- Standard L grid (geometric) -----------------------------------------

def geometric_L_grid(N: int, n_points: int = 12,
                     L_min_frac: float = 0.02,
                     L_max_frac: float = 0.85) -> np.ndarray:
    """Geometric grid of integer block sizes covering [L_min_frac*N, L_max_frac*N]."""
    L_min = max(2, int(L_min_frac * N))
    L_max = max(L_min + 1, int(L_max_frac * N))
    Ls = np.unique(np.round(np.geomspace(L_min, L_max, n_points)).astype(int))
    return Ls
