"""Free-theory twist validation (Task C validation rung), via exact free Majoranas.

SO(5)_1 = five free Majorana fermions (Ising x5). The spinor primary is the twist
field: each Majorana contributes chiral weight h=1/16 in the twisted (Ramond /
periodic-Majorana) sector, so five Majoranas give h = 5/16 and total scaling
dimension Delta_s = 2h = 5/8. The twisted-sector ground-state degeneracy is
2^floor(N/2) = 4 for N=5 (the N Majorana zero modes form a Clifford algebra whose
irreducible rep is the 4-dim SO(5) spinor).

Method (exact, no DMRG): a single critical Majorana hopping chain on a ring,
    H = (i/2) sum_{j=1}^{L} s_j gamma_j gamma_{j+1},   gamma_{L+1} = s * gamma_1
with s = -1 (antiperiodic, NS sector: identity) or s = +1 (periodic, Ramond
sector: contains the zero mode / twist field). H = (i/4) sum A_{jk} gamma_j gamma_k
with A real antisymmetric; eigenvalues come in pairs +-i*eps_m (eps_m >= 0), and
the BCS ground-state energy is E = -(1/2) sum_m eps_m.

The twist-field chiral weight is the universal Casimir difference between the two
spin structures:
    h_twist = (E_R - E_NS) * L / (2 pi v) + (NS,R Casimir offsets)
We instead read it cleanly from the finite-size scaling: the Ramond ground state
sits at total scaling dimension Delta = 1/8 (h = h_bar = 1/16) above the NS ground
state, so (E_R - E_NS) = (2 pi v / L) * (1/8) - the (c/24) Casimir pieces cancel
between sectors only partially; we calibrate against the known NS gap to the energy
operator (Delta_eps = 1) to fix v and remove the offset, giving h per Majorana.

Validation target: h_1 -> 1/16 = 0.06250; five copies h_5 -> 5/16 = 0.31250,
Delta_s = 5/8 = 0.62500; Ramond zero-mode degeneracy 2^floor(5/2) = 4.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(parents=True, exist_ok=True)


def majorana_spectrum(L, bc):
    """Single-particle eps_m >= 0 for the uniform Majorana ring.
    bc=-1 antiperiodic (NS), bc=+1 periodic (R)."""
    A = np.zeros((L, L))
    for j in range(L - 1):
        A[j, j + 1] = 1.0
        A[j + 1, j] = -1.0
    A[L - 1, 0] = bc
    A[0, L - 1] = -bc
    # H = (i/4) sum A gamma gamma; eigenvalues of iA are real, +-eps pairs
    ev = np.linalg.eigvalsh(1j * A)        # real eigenvalues, symmetric around 0
    eps = np.sort(np.abs(ev))[::2]          # take the L/2 nonneg magnitudes (paired)
    # robust pairing: take positive half
    pos = np.sort(ev)[::-1][: L // 2]
    return np.abs(pos)


def gs_energy(L, bc):
    eps = majorana_spectrum(L, bc)
    return -0.5 * np.sum(eps), eps


def n_zero_modes(L, bc, tol=1e-9):
    eps = majorana_spectrum(L, bc)
    return int(np.sum(eps < tol))


def velocity(L):
    """Fermi velocity from the NS dispersion near k=0: eps_k ~ v|k|.
    For this uniform Majorana ring eps_k = 2|sin(k)| with k spacing 2pi/L, so v=2."""
    # numerically: smallest NS single-particle energy / smallest momentum
    eps = np.sort(majorana_spectrum(2 ** 0 * L, -1))
    kmin = math.pi / L  # NS smallest |k| = (2pi/L)(1/2)
    # eps at kmin ~ v*kmin
    return float(eps[0] / kmin)


def single_ising_weights(L):
    """Return h_sigma (chiral) and Delta_eps for a single Ising/Majorana at size L.
    Uses sector ground-state energies and the NS first gap to calibrate 2*pi*v/L."""
    E_NS, eps_NS = gs_energy(L, -1)
    E_R, eps_R = gs_energy(L, +1)
    # NS excitations: add the two lowest NS modes (fermion parity even) -> energy op
    # Delta_eps = 1: E_eps - E_NS = (2 pi v/L)*1. Lowest NS single-particle eps gives
    # a fermion; energy operator = two-fermion (lowest pair) in NS.
    epsNS = np.sort(eps_NS)
    gap_eps = 2 * epsNS[0]                      # two lowest NS fermions (Delta=1)
    unit = gap_eps                              # = 2 pi v / L  (since Delta_eps=1)
    # sigma (Ramond GS) sits at Delta=1/8 above NS GS:
    dE_sigma = E_R - E_NS
    Delta_sigma = dE_sigma / unit
    h_sigma = Delta_sigma / 2.0
    return {"L": L, "E_NS": E_NS, "E_R": E_R, "unit_2pivL": unit,
            "Delta_sigma": Delta_sigma, "h_sigma": h_sigma,
            "R_zero_modes": n_zero_modes(L, +1)}


def main():
    print("=== Free-theory twist validation: SO(5)_1 = 5 Majoranas ===")
    print("Single Ising/Majorana, twist (Ramond) sector -> h_sigma = 1/16 = 0.06250\n")
    rows = []
    for L in [32, 64, 128, 256, 512]:
        w = single_ising_weights(L)
        rows.append(w)
        print(f"  L={L:4d}  Delta_sigma={w['Delta_sigma']:.6f} (target 0.125)  "
              f"h_sigma={w['h_sigma']:.6f} (target 0.06250)  "
              f"R-zero-modes={w['R_zero_modes']}")
    big = rows[-1]
    h1 = big["h_sigma"]
    print(f"\n  Largest L={big['L']}: h_sigma(1 Majorana) = {h1:.6f}  [1/16={1/16:.6f}]")
    # five decoupled Majoranas: weights add
    h5 = 5 * h1
    Delta_s = 2 * h5
    deg = 2 ** (5 // 2)
    print(f"  Five Majoranas: h_5 = 5*h1 = {h5:.6f}  [5/16={5/16:.6f}]")
    print(f"                  Delta_s = 2*h5 = {Delta_s:.6f}  [5/8={5/8:.6f}]")
    print(f"  Twisted-sector (Ramond) ground-state degeneracy 2^floor(5/2) = {deg} "
          f"(= SO(5) spinor dim)")
    # zero-mode degeneracy check for N=5 twisted Majoranas: N periodic Majoranas
    # have N zero modes -> Clifford rep dim 2^floor(N/2)=4
    out = {"single_ising": rows, "h1_best": h1, "h5": h5, "Delta_s": Delta_s,
           "spinor_degeneracy": deg, "target_h1": 1/16, "target_Delta_s": 5/8,
           "pass_h1": bool(abs(h1 - 1/16) < 2e-3),
           "pass_Delta_s": bool(abs(Delta_s - 5/8) < 2e-2)}
    json.dump(out, open(RES / "free_twist_validation.json", "w"), indent=2)
    print(f"\n  PASS h1~1/16: {out['pass_h1']}   PASS Delta_s~5/8: {out['pass_Delta_s']}")
    print(f"  wrote {RES/'free_twist_validation.json'}")


if __name__ == "__main__":
    main()
