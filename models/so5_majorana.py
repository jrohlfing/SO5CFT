"""
SO(5)_1 lattice candidate: five critical Majorana (Kitaev) chains.

See MODEL-DERIVATION.md (committed before any numerics) for the physics. In
brief: N free Majorana fermions realize SO(N)_1 WZW (c = N/2). We take five
critical Kitaev (Majorana) chains; the SO(5) symmetry rotates the five flavors;
the free fixed point is SO(5)_1 with primaries {1, v, s} at chiral weights
{0, 1/2, 5/16}.

This file supplies ONLY the model layer. It does not touch analysis/c_eff.py;
the validated estimator (slope -> c, R^2 >= 0.9 gate) is reused unchanged. For
the free point we build the Majorana covariance matrix of the BdG ground state
(Peschel's method) and feed the resulting S(L) to fit_central_charge, exactly
as models/xx_chain.py supplies a correlation matrix.

Three free-fermion routines (T1 entanglement, T2 finite-size spectrum, T3 edge
parity) plus one interacting TeNPy model (T4 Gross-Neveu scan).

Conventions
-----------
Single critical Kitaev chain: H = sum_i [ -t(c+_i c_{i+1}+h.c.)
+ Delta(c_i c_{i+1}+h.c.) ] - mu sum_i (c+_i c_i - 1/2). Majorana
c_j = (a_j + i b_j)/2. We write H = (i/4) gamma^T M gamma with M real
antisymmetric, gamma = [a_0,b_0,a_1,b_1,...]. Critical point of one chain
(t=Delta): |mu| = 2t (here mu=2 with t=Delta=1). Topological phase: |mu| < 2t.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Free-fermion BdG / Majorana machinery
# ============================================================================

def _add(M: np.ndarray, p: int, q: int, lam: float) -> None:
    """Add the term (i/2) lam gamma_p gamma_q to H = (i/4) gamma^T M gamma."""
    M[p, q] += lam
    M[q, p] -= lam


def kitaev_majorana_matrix(N: int, t: float, Delta: float, mu: float,
                           bc: str = "open") -> np.ndarray:
    """
    Real antisymmetric 2N x 2N matrix M for a single Kitaev chain,
    H = (i/4) gamma^T M gamma, gamma = [a_0,b_0,...,a_{N-1},b_{N-1}].

    Onsite:  -mu (i/2) a_j b_j
    Bond:    (i/2)[ (Delta-t) a_i b_{i+1} - (t+Delta) a_{i+1} b_i ]
    bc in {"open","periodic","antiperiodic"} sets the wrap bond.
    """
    M = np.zeros((2 * N, 2 * N), dtype=np.float64)
    a = lambda j: 2 * j
    b = lambda j: 2 * j + 1
    for j in range(N):
        _add(M, a(j), b(j), -mu)
    for i in range(N - 1):
        _add(M, a(i), b(i + 1), (Delta - t))
        _add(M, a(i + 1), b(i), -(t + Delta))
    if bc in ("periodic", "antiperiodic"):
        s = 1.0 if bc == "periodic" else -1.0
        _add(M, a(N - 1), b(0), s * (Delta - t))
        _add(M, a(0), b(N - 1), -s * (t + Delta))
    elif bc != "open":
        raise ValueError(bc)
    return M


def single_particle_min_abs_energy(M: np.ndarray) -> float:
    """Lowest |single-particle energy| = min |eig(iM)|. iM is Hermitian."""
    w = np.linalg.eigvalsh(1j * M)
    return float(np.min(np.abs(w)))


def ground_state_covariance(M: np.ndarray) -> np.ndarray:
    """
    Majorana covariance Gamma_{jk} = (i/2)<[gamma_j,gamma_k]> of the BdG ground
    state of H = (i/4) gamma^T M gamma. Using iM = U diag(w) U^dag,
    Gamma = -i U sign(w) U^dag (real antisymmetric; sign(0)=0 handles exact
    zero modes as maximally-mixed, i.e. they do not bias the entropy). The
    overall sign convention does not affect entanglement entropy.
    """
    w, U = np.linalg.eigh(1j * M)
    Gamma = (-1j) * (U * np.sign(w)) @ U.conj().T
    return Gamma.real


def _binary_entropy(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)


def majorana_block_entropy(Gamma: np.ndarray,
                           L_sites: list[int]) -> np.ndarray:
    """
    Entanglement entropy of the contiguous block of sites [0..L-1] (the first
    2L Majoranas) for a Gaussian Majorana state with covariance Gamma.
    Eigenvalues of i*Gamma_A come in +-nu pairs, nu in [0,1];
    S = sum_m H2((1+nu_m)/2).
    """
    S = []
    for L in L_sites:
        GA = Gamma[:2 * L, :2 * L]
        ev = np.linalg.eigvalsh(1j * GA)        # real, +- pairs in [-1,1]
        nu = np.clip(ev[ev >= 0.0], 0.0, 1.0)   # nonneg half: L values
        S.append(float(np.sum(_binary_entropy((1.0 + nu) / 2.0))))
    return np.array(S, dtype=np.float64)


def n_flavor_block_entropy(N: int, n_flavor: int, L_sites: list[int],
                           t: float = 1.0, Delta: float = 1.0,
                           mu: float = 2.0) -> np.ndarray:
    """
    S(L) for n decoupled identical critical Kitaev chains. Because the flavors
    are decoupled, the covariance is block-diagonal and the entropy is additive:
    S_total(L) = n_flavor * S_single(L). We return the additive total. (See
    MODEL-DERIVATION.md sec 3; verified against the n=1 -> c=0.5 and
    n=2 -> c=1.0 self-checks in the driver.)
    """
    M = kitaev_majorana_matrix(N, t, Delta, mu, bc="open")
    Gamma = ground_state_covariance(M)
    S_single = majorana_block_entropy(Gamma, L_sites)
    return n_flavor * S_single


# ============================================================================
# T2: finite-size conformal spectrum of the critical chain on a ring
# ============================================================================
#
# Critical TFIM/Majorana dispersion eps(k) = 4|sin(k/2)|, spin-wave velocity
# v = 2 (lattice units, t=Delta=h=J=1). NS (antiperiodic fermions) <-> integer
# parity sector with half-integer momenta; R (periodic) <-> momenta incl 0, pi.
# Scaling dimensions: E_n - E_0 = (2 pi v / N) x_n.

ISING_VELOCITY = 2.0


def _ising_dispersion(k: np.ndarray) -> np.ndarray:
    return 4.0 * np.abs(np.sin(k / 2.0))


def _sector_momenta(N: int, sector: str) -> np.ndarray:
    m = np.arange(N)
    if sector == "NS":
        k = (2 * m + 1) * np.pi / N
    elif sector == "R":
        k = 2 * m * np.pi / N
    else:
        raise ValueError(sector)
    return (k + np.pi) % (2 * np.pi) - np.pi


def ising_sector_gs_energy(N: int, sector: str) -> float:
    """Ground-state energy of one critical Majorana chain in a BC sector."""
    k = _sector_momenta(N, sector)
    return float(-0.5 * np.sum(_ising_dispersion(k)))


def finite_size_dimensions(N: int) -> dict:
    """
    Conformal scaling dimensions read off the critical chain on a ring of N
    sites, in units of (2 pi v / N):

      x_sigma : single-chain disorder field (Ising^5 op, NOT an SO(5)_1
                primary)            -> expect 1/8
      x_eps   : single-chain energy / SO(5) vector v -> expect 1
      x_spinor: JOINT twist of all 5 chains = SO(5)_1 spinor s
                = 5 * x_sigma       -> expect 5/8
    Returns the absolute dimensions and the v-independent ratios.
    """
    unit = 2.0 * np.pi * ISING_VELOCITY / N
    E_NS = ising_sector_gs_energy(N, "NS")
    E_R = ising_sector_gs_energy(N, "R")
    x_sigma = (E_R - E_NS) / unit
    kpos = np.sort(np.abs(_sector_momenta(N, "NS")))
    x_eps = (_ising_dispersion(kpos[0]) + _ising_dispersion(kpos[1])) / unit
    x_spinor = 5.0 * x_sigma
    return {
        "N": N,
        "x_sigma_single": x_sigma,          # ~1/8
        "x_vector": x_eps,                  # ~1
        "x_spinor_SO5": x_spinor,           # ~5/8
        "ratio_spinor_over_vector": x_spinor / x_eps,   # ~5/8
        "ratio_singleSigma_over_vector": x_sigma / x_eps,  # ~1/8
    }


# ============================================================================
# T3: unpaired-Majorana parity signature (odd vs even species)
# ============================================================================

def multi_flavor_topological_matrix(n_flavor: int, N: int,
                                     t: float = 1.0, Delta: float = 1.0,
                                     mu: float = 0.5, lam: float = 0.3,
                                     seed: int = 0) -> np.ndarray:
    """
    n_flavor topological Kitaev chains (|mu|<2t), open BC, plus a SMALL generic
    SO(n)-direction antisymmetric inter-flavor Majorana coupling lam * A_{ab}
    (A real antisymmetric, fixed random seed). This is a fermion-parity-
    preserving perturbation: the n left-edge Majoranas couple through the
    antisymmetric n x n matrix A, so the number of surviving zero modes is
    n mod 2 (odd -> one protected unpaired Majorana; even -> none).
    """
    dim = 2 * n_flavor * N
    M = np.zeros((dim, dim), dtype=np.float64)
    idx = lambda fl, site, comp: ((fl * N) + site) * 2 + comp  # comp 0=a,1=b
    for fl in range(n_flavor):
        for j in range(N):
            _add(M, idx(fl, j, 0), idx(fl, j, 1), -mu)
        for i in range(N - 1):
            _add(M, idx(fl, i, 0), idx(fl, i + 1, 1), (Delta - t))
            _add(M, idx(fl, i + 1, 0), idx(fl, i, 1), -(t + Delta))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_flavor, n_flavor))
    A = (A - A.T) / 2.0
    for a in range(n_flavor):
        for b in range(a + 1, n_flavor):
            for i in range(N):
                _add(M, idx(a, i, 0), idx(b, i, 0), lam * A[a, b])
                _add(M, idx(a, i, 1), idx(b, i, 1), lam * A[a, b])
    return M


def edge_mode_splitting(n_flavor: int, N: int, **kw) -> float:
    """Lowest |single-particle energy| of the open n-flavor topological chain."""
    M = multi_flavor_topological_matrix(n_flavor, N, **kw)
    return single_particle_min_abs_energy(M)


# ============================================================================
# T4: interacting SO(5) Gross-Neveu lattice model (TeNPy DMRG)
# ============================================================================
#
# Five-leg critical transverse-field Ising ladder. Each rung is a GroupedSite
# of 5 spin-1/2. Per leg a: critical Ising  -J Sigmax_a Sigmax_a(+1) - h Sigmaz_a
# (J=h=1 critical). Gross-Neveu energy-energy rung coupling (MODEL-DERIVATION.md
# sec 4):  g sum_{a<b} Sigmaz_a Sigmaz_b. At g=0 -> five decoupled critical
# Ising chains -> c = 5/2. g != 0 is the marginal current-current perturbation.

def build_ising_ladder_model(n_flavor: int, L: int, g: float,
                             J: float = 1.0, h: float = 1.0):
    """
    TeNPy model for the n-leg critical Ising ladder on a Square lattice
    (Lx=L rungs, Ly=n legs), single spin-1/2 per site (d=2), snake MPS order.
    Per leg (x-bonds): -J Sigmax Sigmax;  transverse field: -h Sigmaz;
    Gross-Neveu energy-energy coupling between ALL leg pairs within a rung
    (y-displacements dy=1..n-1):  g Sigmaz Sigmaz.

    This d=2 formulation is far cheaper than a composite d=2^n rung site (the
    two-site SVD is 2*chi x 2*chi rather than 2^n*chi x 2^n*chi).
    """
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.lattice import Square
    from tenpy.models.model import CouplingMPOModel

    class _IsingLadder(CouplingMPOModel):
        def init_sites(self, mp):
            return SpinHalfSite(conserve=None)

        def init_lattice(self, mp):
            site = self.init_sites(mp)
            return Square(mp["L"], mp["n"], site,
                          bc=["open", "open"], bc_MPS="finite", order="default")

        def init_terms(self, mp):
            n = mp["n"]
            JJ = mp["J"]
            hh = mp["h"]
            gg = mp["g"]
            # Ising bond along the chain (x), per leg (same y)
            self.add_coupling(-JJ, 0, "Sigmax", 0, "Sigmax", [1, 0],
                              plus_hc=False)
            self.add_onsite(-hh, 0, "Sigmaz")
            # Gross-Neveu all-pairs energy-energy coupling within a rung (along y)
            if gg != 0.0:
                for dy in range(1, n):
                    self.add_coupling(gg, 0, "Sigmaz", 0, "Sigmaz", [0, dy],
                                      plus_hc=False)

    return _IsingLadder({"n": n_flavor, "L": L, "J": J, "h": h, "g": g})


def run_ladder_dmrg(n_flavor: int, L: int, g: float, chi: int,
                    J: float = 1.0, h: float = 1.0,
                    max_sweeps: int = 24, verbose: bool = False) -> dict:
    """
    DMRG ground state of the n-leg Ising ladder. Returns energy, bond
    entanglement entropies, and the mid-chain correlation length (a gapping
    diagnostic). The c_eff fit is done by the caller via the unchanged
    fit_central_charge.
    """
    import time
    from tenpy.networks.mps import MPS
    from tenpy.algorithms.dmrg import TwoSiteDMRGEngine

    n_sites = n_flavor * L
    M = build_ising_ladder_model(n_flavor, L, g, J=J, h=h)
    psi = MPS.from_product_state(M.lat.mps_sites(), [0] * n_sites, bc="finite")
    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-9},
        "max_E_err": 1e-7,
        "max_S_err": 1e-5,
        "mixer": True,
        "max_sweeps": max_sweeps,
        "min_sweeps": 6,
    }
    t0 = time.time()
    eng = TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, _ = eng.run()
    SvN = psi.entanglement_entropy()      # length n_sites-1 (all MPS bonds)
    try:
        xi = float(psi.correlation_length())
    except Exception:
        xi = float("nan")
    return {
        "n_flavor": n_flavor, "L": L, "g": g, "chi": chi,
        "E": float(E), "E_per_site": float(E / n_sites),
        "SvN": SvN.tolist(),
        "S_mid": float(SvN[n_sites // 2]),
        "max_chi": int(max(psi.chi)),
        "corr_length": xi,
        "walltime_s": float(time.time() - t0),
    }
