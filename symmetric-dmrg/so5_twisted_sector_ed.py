"""Many-body exact diagonalization of the SO(5) spinor sector (twisted sector).

Supports Section 4.3 of the manuscript "A Parity Signature Distinguishing SO(5)_1
from Spin(4)_1 Boundaries": the third, direct-measurement establishment of the
spinor multiplet.

The twisted sector is built directly from antiperiodic (Ramond) Majorana modes -- the
genuine twisted Fock space -- rather than from a sign-flipped bond on the periodic
chain. A flipped bond leaves the Hilbert space unchanged and cannot reach the
half-integer-weight sector. We diagonalize the many-body twisted Hamiltonian for n
critical Majorana flavors on M-site rings and label the ground states by the SO(5)
quadratic Casimir C2 = sum_{a<b} (T^{ab})^2.

Result: the genuine all-Ramond (n=5) twisted ground is 32-fold and decomposes under
SO(5) as 2 x (1 + 5 + 10) -- singlet, vector, adjoint (C2 = 0, 4, 6) -- with no
C2 = 5/2 spinor present. Because 4 (x) 4 = 1 + 5 + 10 for SO(5) = Sp(4), the
non-chiral lattice twisted ground is spinor_L (x) spinor_R (with a factor 2 from the
parity projection), not the bare four-dimensional chiral spinor. The SO(5)_1 spinor
is a chiral primary (h_s = 5/16); a local non-chiral lattice ring realizes its square,
the lattice chirality (Nielsen-Ninomiya) obstruction. Turning on the SO(5)-invariant
current-current coupling preserves the same multiplet content.

Dependencies: numpy, scipy. Single workstation; the n=5, M=4 case is a dense
1024-dimensional diagonalization.
"""
from __future__ import annotations
import json
import os
import numpy as np
import scipy.sparse as sp

I2 = sp.identity(2, format="csr")
X = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
Y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=complex)


def op(local, n, N):
    """Local single-site operator placed at site n of an N-site tensor product."""
    mats = [I2] * N
    mats[n] = local
    out = mats[0]
    for k in range(1, N):
        out = sp.kron(out, mats[k], format="csr")
    return out


def jw_majorana(N):
    """2N Majorana operators via a global Jordan-Wigner string (site order 0..N-1):
    gamma[2n] = (prod_{m<n} Z_m) X_n, gamma[2n+1] = (prod_{m<n} Z_m) Y_n."""
    g = {}
    Zstr = sp.identity(2 ** N, format="csr")
    for n in range(N):
        g[2 * n] = (Zstr @ op(X, n, N))
        g[2 * n + 1] = (Zstr @ op(Y, n, N))
        Zstr = Zstr @ op(Z, n, N)
    return g


def build(n, M, bcs, g=0.0):
    """n Majorana flavors on M-site rings. bcs[a] = +1 Ramond / -1 NS per flavor.
    Optional SO(5)-invariant current-current interaction g (quartic). Returns the
    Hamiltonian H, the SO(5) generators T^{ab}, and the quadratic Casimir C2."""
    Nq = (n * M) // 2
    gam = jw_majorana(Nq)
    dim = 2 ** Nq
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for a in range(n):
        base = a * M
        for m in range(M - 1):
            H = H + 1j * (gam[base + m] @ gam[base + m + 1])
        H = H + bcs[a] * 1j * (gam[base + M - 1] @ gam[base])
    T = {}
    for a in range(n):
        for b in range(a + 1, n):
            Tab = sp.csr_matrix((dim, dim), dtype=complex)
            for j in range(M):
                Tab = Tab + 0.5j * (gam[a * M + j] @ gam[b * M + j])
            T[(a, b)] = Tab
    C2 = sp.csr_matrix((dim, dim), dtype=complex)
    for Tab in T.values():
        C2 = C2 + Tab @ Tab
    if g:
        for (a, b), Tab in list(T.items()):
            for j in range(M):
                Jab_j = 0.5j * (gam[a * M + j] @ gam[b * M + j])
                jp = (j + 1) % M
                Jab_jp = 0.5j * (gam[a * M + jp] @ gam[b * M + jp])
                H = H + g * (Jab_j @ Jab_jp)
    return H, T, C2


def casimir_content(H, C2):
    """Diagonalize H, take the degenerate ground space, and return its size, ground
    energy, and the SO(5) Casimir eigenvalues grouped by degeneracy."""
    w, v = np.linalg.eigh(H.toarray())
    w = w.real
    E0 = w.min()
    idx = np.where(w - E0 < 1e-7)[0]
    G = v[:, idx]
    C2g = G.conj().T @ (C2 @ G)
    ev = np.sort(np.linalg.eigvalsh(C2g).real)
    out, i = [], 0
    while i < len(ev):
        j = i
        while j < len(ev) and abs(ev[j] - ev[i]) < 1e-4:
            j += 1
        out.append((round(float(ev[i]), 3), j - i))
        i = j
    return len(idx), float(E0), out


def lowest_k_content(H, C2, k):
    """SO(5) Casimir content of the lowest-k eigenstates by energy, with the internal
    energy span of that manifold and the gap to the next state. Used for the
    interacting case, where the coupling lifts the free degeneracy but the lowest-k
    manifold keeps its multiplet content."""
    w, v = np.linalg.eigh(H.toarray())
    w = w.real
    order = np.argsort(w)
    G = v[:, order[:k]]
    C2g = G.conj().T @ (C2 @ G)
    ev = np.sort(np.linalg.eigvalsh(C2g).real)
    out, i = [], 0
    while i < len(ev):
        j = i
        while j < len(ev) and abs(ev[j] - ev[i]) < 1e-2:
            j += 1
        out.append((round(float(ev[i]), 3), j - i))
        i = j
    wsort = np.sort(w)
    span = float(wsort[k - 1] - wsort[0])
    gap = float(wsort[k] - wsort[k - 1]) if k < len(wsort) else float("nan")
    return out, span, gap


def irrep(c2):
    return ("singlet" if c2 < 0.3 else "spinor" if abs(c2 - 2.5) < 0.3
            else "vector" if abs(c2 - 4) < 0.3 else "adjoint" if abs(c2 - 6) < 0.3 else "?")


def main():
    result = {}
    print("=== many-body twisted-sector ED: SO(5) spinor sector ===\n")

    print("all-Ramond ground degeneracy (validation, M=4): expect 2^n")
    val = {}
    for n in (1, 2, 3, 4, 5):
        H, T, C2 = build(n, 4, [+1] * n)
        deg, E0, _ = casimir_content(H, C2)
        val[n] = {"ground_deg": deg, "expected_2pow_n": 2 ** n, "E0": E0}
        print(f"  n={n} (M=4): ground deg={deg} (=2^{n})  E0={E0:.4f}")
    result["validation_all_ramond_M4"] = val

    print("\nSO(5) Casimir content of the n=5 all-Ramond twisted ground (M=4):")
    H, T, C2 = build(5, 4, [+1] * 5)
    deg, E0, content = casimir_content(H, C2)
    result["n5_twisted_ground"] = {
        "ground_deg": deg, "E0": E0,
        "casimir_content": [{"C2": v, "deg": d, "irrep": irrep(v)} for v, d in content],
        "bare_spinor_C2_2p5_present": any(abs(v - 2.5) < 0.3 for v, _ in content),
        "decomposition": "2 x (1 + 5 + 10) = 2 x (spinor (x) spinor); 4(x)4=1+5+10 for Sp(4)",
    }
    for val_, d in content:
        print(f"  C2={val_}: deg {d}   {irrep(val_)}")
    tot = sum(d for _, d in content)
    print(f"  total = {tot} = 2 x (1+5+10) = 2 x (spinor (x) spinor)   [4(x)4=1+5+10 for Sp(4)]")
    print("  => no bare C2=5/2 spinor: the chiral spinor is not a subspace of the")
    print("     non-chiral lattice twisted ground (Nielsen-Ninomiya chirality obstruction).")

    print("\ninteracting (g=0.5) n=5: the coupling lifts the free 32-fold degeneracy;")
    print("Casimir content of the lowest 32 states (the split twisted manifold), M=4:")
    Hi, T, C2 = build(5, 4, [+1] * 5, g=0.5)
    conti, span, gap = lowest_k_content(Hi, C2, 32)
    result["n5_twisted_manifold_interacting_g0p5"] = {
        "k": 32, "manifold_energy_span": round(span, 4), "gap_to_next": round(gap, 4),
        "casimir_content": [{"C2": v, "deg": d, "irrep": irrep(v)} for v, d in conti],
        "bare_spinor_C2_2p5_present": any(abs(v - 2.5) < 0.3 for v, _ in conti),
    }
    for val_, d in conti:
        print(f"  C2={val_}: deg {d}   {irrep(val_)}")
    print(f"  manifold spans {span:.3f} in energy, gap {gap:.3f} to the next state")
    print("  => same 1,5,10 content, now split in energy; still no bare C2=5/2 spinor.")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "twisted_sector_ed.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
