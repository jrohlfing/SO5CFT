"""SO(4) bilinear-biquadratic chain at the Reshetikhin point -- CHARGE-CONSERVING
DMRG. The EVEN-parity control for the matched SO(5)_1 vs Spin(4)_1 entanglement
tower comparison.

This is the n=4 sibling of so5_bbq_dmrg_sym.py, built on the identical
charge-conserving framework so the two entanglement spectra come off the SAME
pipeline, same boundary conditions, same cut.

Why theta = 0 for n=4
----------------------
The SO(n) bilinear-biquadratic chain  H = sum_i [cos(t) B_i + sin(t) B_i^2],
B = sum_{a<b} L^{ab}_i L^{ab}_{i+1}  (SO(n)-invariant bilinear), has its
Reshetikhin integrable point (log-derivative of the Zamolodchikov-Fateev O(n)
R-matrix  R(u)=I+uP-(u/(u+kappa))K, kappa=(n-2)/2) at

    tan(theta_R) = (n-4)/(n-2)^2 .

n=5 -> 1/9 (the established SO(5)_1 point used by the n=5 runs). n=4 -> 0.
So the Spin(4)_1 / SO(4)_1 point is the PURE BILINEAR chain, theta = 0.

The 4-dim vector of SO(4) is the (1/2,1/2) of SU(2)xSU(2)=Spin(4); the SO(4)
bilinear splits as B = S^A_i.S^A_{i+1} + S^B_i.S^B_{i+1}, i.e. TWO DECOUPLED
spin-1/2 Heisenberg chains = SU(2)_1 x SU(2)_1 = SO(4)_1, c = 2. This is the
exact even analog of the n=5 Reshetikhin point and the piece neither Alet 2011
nor Wu-Tu 2022 put beside the odd case.

The two Cartan charges
----------------------
The Cartan generators T^{01}, T^{23} are off-diagonal in the real Cartesian basis
|0..3>; diagonal in the COMPLEX weight basis
    |+1,0> = (|0> + i|1>)/sqrt2     charge (q1,q2) = (+1, 0)
    |-1,0> = (|0> - i|1>)/sqrt2                      (-1, 0)
    |0,+1> = (|2> + i|3>)/sqrt2                      ( 0,+1)
    |0,-1> = (|2> - i|3>)/sqrt2                      ( 0,-1)
There is NO (0,0) weight -- that is precisely the even/odd distinction vs SO(5),
which carries the extra |4> = (0,0) vector state. We build B (real basis), rotate
to the weight basis with the unitary U, and decompose into charge-graded
matrix-unit couplings E^{mn}_i E^{pq}_{i+1}; TeNPy conserves total (q1,q2).

Usage:
    python so4_bbq_dmrg_sym.py --mode validate
    python so4_bbq_dmrg_sym.py --L 64 --chi 800 --mode production --tag so4_L64_chi800 --chi-ramp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

THREADS = int(os.environ.get("DMRG_THREADS_PER_WORKER", "30"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import numpy as np
from scipy import stats

import tenpy
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site
from tenpy.algorithms import dmrg as dmrg_mod
from tenpy.models.model import CouplingMPOModel
from tenpy.linalg.charges import ChargeInfo, LegCharge

N = 4  # SO(4) vector rep dimension

# Weight (Cartan) charges of the 4 vector-rep states, in WEIGHT-basis order.
# NOTE: no (0,0) state -- the even-parity signature.
SO4_WEIGHTS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# SO(4) generators (real basis) and the real->weight unitary
# --------------------------------------------------------------------------- #

def _so4_generators_real():
    """List of (a,b,T^{ab}) with a<b in {0..3}; (T^{ab})_{cd}=-i(d^a_c d^b_d - d^a_d d^b_c).
    Hermitian generators of so(4) in the 4-dim vector rep (6 of them)."""
    gens = []
    for a in range(N):
        for b in range(a + 1, N):
            T = np.zeros((N, N), dtype=complex)
            T[a, b] = -1j
            T[b, a] = 1j
            gens.append((a, b, T))
    assert len(gens) == 6
    return gens


def _weight_unitary():
    """U: columns are weight-basis eigenvectors expressed in the real basis.
    |real> = U |weight>, so an operator O_real becomes O_weight = U^dag O_real U."""
    s2 = math.sqrt(2.0)
    U = np.zeros((N, N), dtype=complex)
    e = np.eye(N, dtype=complex)
    U[:, 0] = (e[:, 0] + 1j * e[:, 1]) / s2   # (+1, 0)
    U[:, 1] = (e[:, 0] - 1j * e[:, 1]) / s2   # (-1, 0)
    U[:, 2] = (e[:, 2] + 1j * e[:, 3]) / s2   # ( 0,+1)
    U[:, 3] = (e[:, 2] - 1j * e[:, 3]) / s2   # ( 0,-1)
    return U


def _verify_cartan_diagonal(U):
    """T^{01} -> diag(1,-1,0,0), T^{23} -> diag(0,0,1,-1) in the weight basis."""
    gens = {(a, b): T for a, b, T in _so4_generators_real()}
    T01_w = U.conj().T @ gens[(0, 1)] @ U
    T23_w = U.conj().T @ gens[(2, 3)] @ U
    assert np.allclose(T01_w, np.diag([1, -1, 0, 0]), atol=1e-12), T01_w
    assert np.allclose(T23_w, np.diag([0, 0, 1, -1]), atol=1e-12), T23_w


# --------------------------------------------------------------------------- #
# Bond Hamiltonian: B = sum_{a<b} T^{ab} x T^{ab}; H_bond = cos(t) B + sin(t) B^2
# --------------------------------------------------------------------------- #

def _bilinear_real():
    """B = sum_{a<b} T^{ab} (x) T^{ab}  (16x16, real basis). Hermitian.
    Eigenvalues on the three SO(4) irreps of V x V: singlet -3, antisym(adjoint) -1,
    sym-traceless +1 (the SO(n) values -(n-1), -1, +1)."""
    B = np.zeros((N * N, N * N), dtype=complex)
    for _, _, T in _so4_generators_real():
        B += np.kron(T, T)
    return B


def _bond_real(theta):
    """h_bond = cos(theta) B + sin(theta) B^2 (16x16), real Cartesian basis."""
    B = _bilinear_real()
    h = math.cos(theta) * B + math.sin(theta) * (B @ B)
    return h, B


def _bond_weight(theta):
    """Rotate the bond Hamiltonian into the weight basis; return 16x16 (complex)."""
    U = _weight_unitary()
    _verify_cartan_diagonal(U)
    h_real, B = _bond_real(theta)
    UU = np.kron(U, U)
    h_w = UU.conj().T @ h_real @ UU
    # spectrum must be invariant under the unitary rotation
    ev_r = np.sort(np.linalg.eigvalsh(h_real).real)
    ev_w = np.sort(np.linalg.eigvalsh(h_w).real)
    assert np.allclose(ev_r, ev_w, atol=1e-10), (ev_r, ev_w)
    return h_w, ev_w


def _charge_conservation_report(h_w, tol=1e-12):
    """Check every nonzero matrix-unit term E^{mn}_i E^{pq}_{i+1} conserves total
    Cartan charge: q[m]+q[p] == q[n]+q[q]. Returns (n_terms, n_violations)."""
    h4 = h_w.reshape(N, N, N, N)  # [m_i, p_{i+1}, n_i, q_{i+1}]
    n_terms = 0
    n_viol = 0
    for m in range(N):
        for p in range(N):
            for n in range(N):
                for q in range(N):
                    c = h4[m, p, n, q]
                    if abs(c) > tol:
                        n_terms += 1
                        dq = (SO4_WEIGHTS[m] + SO4_WEIGHTS[p]) - (SO4_WEIGHTS[n] + SO4_WEIGHTS[q])
                        if not np.all(dq == 0):
                            n_viol += 1
    return n_terms, n_viol


# --------------------------------------------------------------------------- #
# Charge-conserving site + model
# --------------------------------------------------------------------------- #

def make_so4_site_sym():
    chinfo = ChargeInfo([1, 1], ["q1", "q2"])
    leg = LegCharge.from_qflat(chinfo, SO4_WEIGHTS)
    labels = ["+1.0", "-1.0", "0.+1", "0.-1"]
    site = Site(leg, state_labels=labels)
    for m in range(N):
        for n in range(N):
            M = np.zeros((N, N), dtype=complex)
            M[m, n] = 1.0
            site.add_op(f"E{m}{n}", M, hc=f"E{n}{m}")
    return site


class SO4BBQSym(CouplingMPOModel):
    """SO(4) BBQ chain, weight basis, two Cartan U(1) charges conserved."""

    def init_sites(self, mp):
        return make_so4_site_sym()

    def init_terms(self, mp):
        theta = mp.get("theta", 0.0)   # Reshetikhin point for n=4
        tol = mp.get("term_tol", 1e-12)
        h_w, _ = _bond_weight(theta)
        h4 = h_w.reshape(N, N, N, N)  # [m_i, p_{i+1}, n_i, q_{i+1}]
        for m in range(N):
            for n in range(N):
                for p in range(N):
                    for q in range(N):
                        c = h4[m, p, n, q]
                        if abs(c) > tol:
                            self.add_coupling(c, 0, f"E{m}{n}", 0, f"E{p}{q}", 1,
                                              plus_hc=False)


# --------------------------------------------------------------------------- #
# Charge-neutral initial product state (total charge (0,0))
# --------------------------------------------------------------------------- #

def _neutral_init(L):
    """Product state with total Cartan charge (0,0). Block [0,1,2,3] is neutral
    ((+1,0)+(-1,0)+(0,1)+(0,-1)=0). The SO(4) vector chain admits a neutral total
    charge only for EVEN L (every single-site charge is nonzero -- itself a parity
    fact); we use L divisible by 4 so the block tiles exactly, with a [0,1] pad for
    L % 4 == 2."""
    pad = {0: [], 2: [0, 1]}
    r = L % 4
    if r not in pad:
        raise ValueError(f"L={L}: SO(4) neutral state needs L even; use L%4 in (0,2)")
    base = [0, 1, 2, 3] * (L // 4)
    init = base + pad[r]
    assert len(init) == L
    tot = SO4_WEIGHTS[init].sum(axis=0)
    assert np.all(tot == 0), (init, tot)
    return init


# --------------------------------------------------------------------------- #
# Calabrese-Cardy fit (OBC) + smoothed variant   (identical to n=5 pipeline)
# --------------------------------------------------------------------------- #

def fit_cc_obc(ents, L, ell_lo_frac=0.25, ell_hi_frac=0.75):
    lo = max(2, int(L * ell_lo_frac))
    hi = min(L - 2, int(L * ell_hi_frac))
    xs, ys, S_vs_l = [], [], []
    for l in range(1, L):
        S_vs_l.append((l, float(ents[l - 1])))
        if lo <= l <= hi:
            xs.append(math.log((2 * L / math.pi) * math.sin(math.pi * l / L)))
            ys.append(float(ents[l - 1]))
    out = {"S_vs_l": S_vs_l, "c_fit": None, "R_squared": None}
    if len(xs) >= 4:
        res = stats.linregress(np.array(xs), np.array(ys))
        out.update(c_fit=float(6 * res.slope), c_se=float(6 * res.stderr),
                   R_squared=float(res.rvalue ** 2), intercept=float(res.intercept))
    sm = [(S_vs_l[i][1] + S_vs_l[i + 1][1]) / 2 for i in range(len(S_vs_l) - 1)]
    xs2, ys2 = [], []
    for i, l in enumerate(range(1, L - 1)):
        if lo <= l <= hi:
            xs2.append(math.log((2 * L / math.pi) * math.sin(math.pi * l / L)))
            ys2.append(sm[i])
    if len(xs2) >= 4:
        res2 = stats.linregress(np.array(xs2), np.array(ys2))
        out.update(c_fit_smoothed=float(6 * res2.slope),
                   R2_smoothed=float(res2.rvalue ** 2))
    return out


# --------------------------------------------------------------------------- #
# Charge-resolved entanglement spectrum (the tower)
# --------------------------------------------------------------------------- #

def entanglement_spectrum_by_charge(psi, bond):
    """Return list of {q1, q2, eps[], lambda[]} at the given bond.
    eps = -2 ln(lambda) (Li-Haldane), lambda = Schmidt value."""
    spec = psi.entanglement_spectrum(by_charge=True)[bond]
    out = []
    for charge, ent_energies in spec:
        eps = np.sort(np.asarray(ent_energies, dtype=float))
        lam = np.exp(-0.5 * eps)
        out.append({"q1": int(charge[0]), "q2": int(charge[1]),
                    "eps": eps.tolist(), "lambda": lam.tolist()})
    out.sort(key=lambda d: (min(d["eps"]) if d["eps"] else 1e9))
    return out


# --------------------------------------------------------------------------- #
# DMRG runner   (identical structure to n=5 pipeline)
# --------------------------------------------------------------------------- #

def _chi_ramp_schedule(chi_max):
    rungs = []
    c = 200
    while c < chi_max:
        rungs.append(c)
        c *= 2
    rungs.append(chi_max)
    return {3 * i: chi for i, chi in enumerate(rungs)}


def run_dmrg(L, theta, chi_max, max_sweeps=40, tol=1e-9, progress_log=None,
             chi_ramp=False):
    t0 = time.time()
    mp = {"L": L, "bc_MPS": "finite", "theta": theta}
    model = SO4BBQSym(mp)
    sites = model.lat.mps_sites()
    init = _neutral_init(L)
    psi = MPS.from_product_state(sites, init, bc="finite")
    if progress_log:
        with open(progress_log, "a", buffering=1) as fh:
            fh.write(f"[so4 L={L} chi={chi_max}] start t={time.time()-t0:.1f}s "
                     f"MPO bond dim={max(model.H_MPO.chi)} ramp={chi_ramp}\n")
    dmrg_params = {
        "trunc_params": {"chi_max": chi_max, "svd_min": 1e-11},
        "mixer": True,
        "mixer_params": {"amplitude": 1e-5, "decay": 2.0, "disable_after": 12},
        "max_sweeps": max_sweeps, "min_sweeps": 6, "max_E_err": tol,
        "combine": True, "active_sites": 2,
        "lanczos_params": {"N_max": 16},
    }
    if chi_ramp:
        dmrg_params["chi_list"] = _chi_ramp_schedule(chi_max)
    info = dmrg_mod.run(psi, model, dmrg_params)
    E0 = float(info["E"])
    psi.canonical_form()
    ents = psi.entanglement_entropy()
    fit = fit_cc_obc(ents, L)
    S_half = float(ents[L // 2 - 1])
    tot_charge = psi.get_total_charge(True)
    es = entanglement_spectrum_by_charge(psi, L // 2)
    res = {
        "model": "SO4_BBQ_sym", "N": N,
        "L": L, "theta": theta, "chi_max": chi_max,
        "E0": E0, "E0_per_bond": E0 / (L - 1),
        "S_Lhalf": S_half,
        "c_fit": fit.get("c_fit"), "c_se": fit.get("c_se"),
        "fit_R2": fit.get("R_squared"),
        "c_fit_smoothed": fit.get("c_fit_smoothed"),
        "R2_smoothed": fit.get("R2_smoothed"),
        "S_vs_l": fit["S_vs_l"],
        "total_charge": [int(x) for x in tot_charge],
        "max_chi_used": int(max(psi.chi)),
        "es_mid_by_charge": es,
        "runtime_sec": time.time() - t0,
    }
    return res, psi, model


# --------------------------------------------------------------------------- #
# Validation mode
# --------------------------------------------------------------------------- #

def validate():
    theta = 0.0
    tanR = (N - 4) / (N - 2) ** 2
    print(f"[so4] n={N}  Reshetikhin tan(theta_R)=(n-4)/(n-2)^2 = {tanR:.6f} "
          f"-> theta_R = {math.atan(tanR):.6f} rad", flush=True)
    h_w, ev = _bond_weight(theta)
    uniq = np.round(ev, 8)
    vals, counts = np.unique(uniq, return_counts=True)
    print(f"[so4] bond spectrum at theta=0 (val x mult): "
          f"{list(zip(vals.tolist(), counts.tolist()))}", flush=True)
    print(f"[so4]   expected (B eigenvalues): -3 x1 (singlet), -1 x6 (antisym), "
          f"+1 x9 (sym-traceless)", flush=True)
    n_terms, n_viol = _charge_conservation_report(h_w)
    print(f"[so4] matrix-unit terms (|c|>1e-12): {n_terms}, "
          f"charge-violating: {n_viol}", flush=True)
    assert n_viol == 0, "charge conservation broken in weight basis!"
    # tiny DMRG sanity (L=12: two decoupled L=12 Heisenberg chains)
    print("[so4] tiny DMRG L=12 chi=64 ...", flush=True)
    res, psi, model = run_dmrg(12, theta, 64, max_sweeps=24)
    print(f"[so4] MPO bond dim = {max(model.H_MPO.chi)}", flush=True)
    print(f"[so4] E0={res['E0']:.6f}  E0/bond={res['E0_per_bond']:.6f}  "
          f"total_charge={res['total_charge']}  S(L/2)={res['S_Lhalf']:.4f}  "
          f"c_sm={res['c_fit_smoothed']}", flush=True)
    # cross-check: E0 should equal 2x a single L=12 spin-1/2 Heisenberg chain.
    # single open Heisenberg L=12 ground energy (J=1, H=sum S.S) ~ -5.0937...
    print(f"[so4] (decoupled cross-check) E0/2 per chain = {res['E0']/2:.6f} "
          f"[open spin-1/2 Heisenberg L=12 ~ -5.0937]", flush=True)
    print(f"[so4] ES at mid bond, lowest sectors:", flush=True)
    allmin = min(min(d["eps"]) for d in res["es_mid_by_charge"] if d["eps"])
    for d in res["es_mid_by_charge"][:6]:
        lo = sorted(d["eps"])[:3]
        print(f"        q=({d['q1']:+d},{d['q2']:+d})  n={len(d['eps'])}  "
              f"eps-min={[round(x-allmin,3) for x in lo]}", flush=True)
    with open(OUT / "validate_so4.json", "w") as fh:
        json.dump({k: v for k, v in res.items() if k != "es_mid_by_charge"}, fh, indent=2)
    print("[so4] validation OK", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "production"], default="validate")
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--chi", type=int, default=400)
    ap.add_argument("--theta", type=float, default=0.0)  # Reshetikhin point for n=4
    ap.add_argument("--max-sweeps", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chi-ramp", action="store_true")
    args = ap.parse_args()

    if args.mode == "validate":
        validate()
        return

    assert args.L > 0
    tag = args.tag or f"so4_L{args.L}_chi{args.chi}"
    progress_log = str(OUT / f"progress_{tag}.log")
    print(f"[so4] production L={args.L} chi={args.chi} theta={args.theta} tag={tag}",
          flush=True)
    res, psi, model = run_dmrg(args.L, args.theta, args.chi,
                               max_sweeps=args.max_sweeps, progress_log=progress_log,
                               chi_ramp=args.chi_ramp)
    print(f"[so4] L={args.L} chi={args.chi} E0={res['E0']:.6f} "
          f"S(L/2)={res['S_Lhalf']:.4f} c_sm={res['c_fit_smoothed']} "
          f"R2_sm={res['R2_smoothed']} q={res['total_charge']} "
          f"chi_used={res['max_chi_used']} ({res['runtime_sec']:.0f}s)", flush=True)
    out_path = OUT / f"{tag}.json"
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[so4] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
