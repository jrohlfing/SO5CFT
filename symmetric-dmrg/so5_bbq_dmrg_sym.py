"""SO(5) bilinear-biquadratic chain at the Reshetikhin point -- CHARGE-CONSERVING DMRG.

This is the symmetric counterpart of so5-primary/scripts/so5_bbq_dmrg.py. The
prior (non-symmetric) campaign used LegCharge.from_trivial(5) and watched the
extracted central charge drift downward from c~2.1 to c~1.48 (smoothed) as L
grew -- the truncation at finite chi splits the SO(5)_1 entanglement
degeneracies because nothing protects them. The fix (Staedert handoff): impose
the two abelian Cartan U(1) charges of SO(5) so the degeneracies are protected by
construction.

The subtlety the prior summary flagged: the Cartan generators T^{01}, T^{23} are
OFF-diagonal in the real Cartesian basis |0..4> in which the matrix units E^{ab}
and the validated bond Hamiltonian are written. They are diagonal only in the
COMPLEX weight basis
    |+1,0> = (|0> + i|1>)/sqrt2     charge (q1,q2) = (+1, 0)
    |-1,0> = (|0> - i|1>)/sqrt2                      (-1, 0)
    |0,+1> = (|2> + i|3>)/sqrt2                      ( 0,+1)
    |0,-1> = (|2> - i|3>)/sqrt2                      ( 0,-1)
    |0, 0> = |4>                                     ( 0, 0)
So we build the validated bond Hamiltonian (projector form, identical to the
prior runs) in the real basis, rotate it into the weight basis with the unitary
U whose columns are those eigenvectors, and decompose the rotated 25x25 bond
matrix into a sum of charge-graded matrix-unit couplings E^{mn}_i E^{pq}_{i+1}.
Each such term carries definite (q1,q2) on each site, and TeNPy enforces total
charge conservation -- protecting the degeneracies.

Bond Hamiltonian (projector form, per-bond constant dropped, matches prior runs):
    h_bond = alpha P_singlet + beta P_antisym
    alpha  = -5 cos(theta) + 15 sin(theta)
    beta   = -2 cos(theta)
    theta  = arctan(1/9)  (Reshetikhin, SO(5)_1, c = 5/2 predicted)

Usage:
    python so5_bbq_dmrg_sym.py --mode validate
    python so5_bbq_dmrg_sym.py --L 64 --chi 1200 --mode production --tag L64_chi1200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

THREADS = int(os.environ.get("STAEDERT_THREADS_PER_WORKER", "30"))
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

# Weight (Cartan) charges of the 5 vector-rep states, in WEIGHT-basis order.
SO5_WEIGHTS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=int)

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# SO(5) generators (real basis) and the real->weight unitary
# --------------------------------------------------------------------------- #

def _so5_generators_real():
    """List of (a,b,T^{ab}) with a<b in {0..4}; (T^{ab})_{cd}=-i(d^a_c d^b_d - d^a_d d^b_c)."""
    gens = []
    for a in range(5):
        for b in range(a + 1, 5):
            T = np.zeros((5, 5), dtype=complex)
            T[a, b] = -1j
            T[b, a] = 1j
            gens.append((a, b, T))
    assert len(gens) == 10
    return gens


def _weight_unitary():
    """U: columns are weight-basis eigenvectors expressed in the real basis.
    |real> = U |weight>, so an operator O_real becomes O_weight = U^dag O_real U."""
    s2 = math.sqrt(2.0)
    U = np.zeros((5, 5), dtype=complex)
    e = np.eye(5, dtype=complex)
    U[:, 0] = (e[:, 0] + 1j * e[:, 1]) / s2   # (+1, 0)
    U[:, 1] = (e[:, 0] - 1j * e[:, 1]) / s2   # (-1, 0)
    U[:, 2] = (e[:, 2] + 1j * e[:, 3]) / s2   # ( 0,+1)
    U[:, 3] = (e[:, 2] - 1j * e[:, 3]) / s2   # ( 0,-1)
    U[:, 4] = e[:, 4]                          # ( 0, 0)
    return U


def _verify_cartan_diagonal(U):
    """T^{01} -> diag(1,-1,0,0,0), T^{23} -> diag(0,0,1,-1,0) in the weight basis."""
    gens = {(a, b): T for a, b, T in _so5_generators_real()}
    T01_w = U.conj().T @ gens[(0, 1)] @ U
    T23_w = U.conj().T @ gens[(2, 3)] @ U
    assert np.allclose(T01_w, np.diag([1, -1, 0, 0, 0]), atol=1e-12), T01_w
    assert np.allclose(T23_w, np.diag([0, 0, 1, -1, 0]), atol=1e-12), T23_w


# --------------------------------------------------------------------------- #
# Bond Hamiltonian: projector form in real basis, rotated to weight basis
# --------------------------------------------------------------------------- #

def _bond_real(theta):
    """h_bond = alpha P_singlet + beta P_antisym (25x25), real Cartesian basis.
    Returns also the spectrum check tuple."""
    ct, st = math.cos(theta), math.sin(theta)
    alpha = -5 * ct + 15 * st
    beta = -2 * ct
    # matrix units E^{ab} = |a><b| (real)
    def E(a, b):
        M = np.zeros((5, 5), dtype=complex)
        M[a, b] = 1.0
        return M
    P_sing = np.zeros((25, 25), dtype=complex)
    SWAP = np.zeros((25, 25), dtype=complex)
    for a in range(5):
        for b in range(5):
            P_sing += np.kron(E(a, b), E(a, b)) / 5.0      # (1/5) sum E^{ab} x E^{ab}
            SWAP += np.kron(E(a, b), E(b, a))              # sum E^{ab} x E^{ba}
    I25 = np.eye(25, dtype=complex)
    P_anti = 0.5 * (I25 - SWAP)
    h = alpha * P_sing + beta * P_anti
    return h, alpha, beta


def _bond_weight(theta):
    """Rotate the bond Hamiltonian into the weight basis; return 25x25 (complex)."""
    U = _weight_unitary()
    _verify_cartan_diagonal(U)
    h_real, alpha, beta = _bond_real(theta)
    UU = np.kron(U, U)
    h_w = UU.conj().T @ h_real @ UU
    # spectrum must be invariant under the unitary rotation
    ev_r = np.sort(np.linalg.eigvalsh(h_real).real)
    ev_w = np.sort(np.linalg.eigvalsh(h_w).real)
    assert np.allclose(ev_r, ev_w, atol=1e-10), (ev_r, ev_w)
    return h_w, alpha, beta, ev_w


def _charge_conservation_report(h_w, tol=1e-12):
    """Check every nonzero matrix-unit term E^{mn}_i E^{pq}_{i+1} conserves total
    Cartan charge: q[m]+q[p] == q[n]+q[q]. Returns (n_terms, n_violations)."""
    h4 = h_w.reshape(5, 5, 5, 5)  # [m_i, p_{i+1}, n_i, q_{i+1}]  (out_i, out_{i+1}, in_i, in_{i+1})
    n_terms = 0
    n_viol = 0
    for m in range(5):
        for p in range(5):
            for n in range(5):
                for q in range(5):
                    c = h4[m, p, n, q]
                    if abs(c) > tol:
                        n_terms += 1
                        dq = (SO5_WEIGHTS[m] + SO5_WEIGHTS[p]) - (SO5_WEIGHTS[n] + SO5_WEIGHTS[q])
                        if not np.all(dq == 0):
                            n_viol += 1
    return n_terms, n_viol


# --------------------------------------------------------------------------- #
# Charge-conserving site + model
# --------------------------------------------------------------------------- #

def make_so5_site_sym():
    chinfo = ChargeInfo([1, 1], ["q1", "q2"])
    leg = LegCharge.from_qflat(chinfo, SO5_WEIGHTS)
    labels = ["+1.0", "-1.0", "0.+1", "0.-1", "0.0"]
    site = Site(leg, state_labels=labels)
    # All 25 matrix units in the weight basis. |m><n| has definite charge q[m]-q[n].
    for m in range(5):
        for n in range(5):
            M = np.zeros((5, 5), dtype=complex)
            M[m, n] = 1.0
            site.add_op(f"E{m}{n}", M, hc=f"E{n}{m}")
    return site


class SO5BBQSym(CouplingMPOModel):
    """SO(5) BBQ chain, weight basis, two Cartan U(1) charges conserved."""

    def init_sites(self, mp):
        return make_so5_site_sym()

    def init_terms(self, mp):
        theta = mp.get("theta", math.atan(1.0 / 9.0))
        tol = mp.get("term_tol", 1e-12)
        h_w, _, _, _ = _bond_weight(theta)
        h4 = h_w.reshape(5, 5, 5, 5)  # [m_i, p_{i+1}, n_i, q_{i+1}]
        # term coefficient for E^{mn}_i E^{pq}_{i+1} is <m p|h|n q> = h4[m,p,n,q]
        for m in range(5):
            for n in range(5):
                for p in range(5):
                    for q in range(5):
                        c = h4[m, p, n, q]
                        if abs(c) > tol:
                            self.add_coupling(c, 0, f"E{m}{n}", 0, f"E{p}{q}", 1,
                                              plus_hc=False)


# --------------------------------------------------------------------------- #
# Charge-neutral initial product state (total charge (0,0))
# --------------------------------------------------------------------------- #

def _neutral_init(L):
    """Product state with total Cartan charge (0,0). Blocks of [0,1,2,3,4] are
    neutral; the remainder is filled with a neutral pad."""
    pad = {0: [], 1: [4], 2: [0, 1], 3: [0, 1, 4], 4: [0, 1, 2, 3]}
    base = ([0, 1, 2, 3, 4] * ((L // 5) + 1))[: (L // 5) * 5]
    r = L - len(base)
    init = base + pad[r]
    assert len(init) == L
    tot = SO5_WEIGHTS[init].sum(axis=0)
    assert np.all(tot == 0), (init, tot)
    return init


# --------------------------------------------------------------------------- #
# Calabrese-Cardy fit (OBC) + smoothed variant
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
    # smoothed: average consecutive l to kill even-odd dimerization, then fit
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
# Charge-resolved entanglement spectrum (Task 3, the tower)
# --------------------------------------------------------------------------- #

def entanglement_spectrum_by_charge(psi, bond):
    """Return list of {q1, q2, eps[], lambda[]} at the given bond.
    eps = -2 ln(lambda) (entanglement 'energy', Li-Haldane), lambda = Schmidt value."""
    spec = psi.entanglement_spectrum(by_charge=True)[bond]
    out = []
    for charge, ent_energies in spec:
        # TeNPy returns -2*log(schmidt) already as 'entanglement energies'
        eps = np.sort(np.asarray(ent_energies, dtype=float))
        lam = np.exp(-0.5 * eps)
        out.append({"q1": int(charge[0]), "q2": int(charge[1]),
                    "eps": eps.tolist(), "lambda": lam.tolist()})
    out.sort(key=lambda d: (min(d["eps"]) if d["eps"] else 1e9))
    return out


# --------------------------------------------------------------------------- #
# DMRG runner
# --------------------------------------------------------------------------- #

def _chi_ramp_schedule(chi_max):
    """chi_list {sweep: chi} ramping 200->...->chi_max, ~3 sweeps per rung, so the
    state is built cheaply at low chi before paying for the top chi. Converges far
    faster than starting cold at chi_max from a product state."""
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
    model = SO5BBQSym(mp)
    sites = model.lat.mps_sites()
    init = _neutral_init(L)
    psi = MPS.from_product_state(sites, init, bc="finite")
    if progress_log:
        with open(progress_log, "a", buffering=1) as fh:
            fh.write(f"[sym L={L} chi={chi_max}] start t={time.time()-t0:.1f}s "
                     f"MPO bond dim={max(model.H_MPO.chi)} "
                     f"ramp={chi_ramp}\n")
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
    theta = math.atan(1.0 / 9.0)
    print(f"[sym] theta_R = {theta:.6f} rad ({math.degrees(theta):.3f} deg), "
          f"tan = {math.tan(theta):.6f} (1/9 = {1/9:.6f})", flush=True)
    h_w, alpha, beta, ev = _bond_weight(theta)
    print(f"[sym] alpha={alpha:.6f} beta={beta:.6f}", flush=True)
    uniq = np.round(ev, 8)
    vals, counts = np.unique(uniq, return_counts=True)
    print(f"[sym] bond spectrum (val x mult): "
          f"{list(zip(vals.tolist(), counts.tolist()))}", flush=True)
    print(f"[sym]   expected: alpha={alpha:.6f} x1 (singlet), "
          f"beta={beta:.6f} x10 (antisym), 0 x14 (sym-traceless)", flush=True)
    n_terms, n_viol = _charge_conservation_report(h_w)
    print(f"[sym] matrix-unit terms (|c|>1e-12): {n_terms}, "
          f"charge-violating: {n_viol}", flush=True)
    assert n_viol == 0, "charge conservation broken in weight basis!"
    # tiny DMRG sanity
    print("[sym] tiny DMRG L=10 chi=64 ...", flush=True)
    res, psi, model = run_dmrg(10, theta, 64, max_sweeps=20)
    print(f"[sym] MPO bond dim = {max(model.H_MPO.chi)}", flush=True)
    print(f"[sym] E0={res['E0']:.6f}  E0/bond={res['E0_per_bond']:.6f}  "
          f"total_charge={res['total_charge']}  S(L/2)={res['S_Lhalf']:.4f}  "
          f"c_sm={res['c_fit_smoothed']}", flush=True)
    print(f"[sym] ES at mid bond, lowest sectors:", flush=True)
    for d in res["es_mid_by_charge"][:6]:
        lo = sorted(d["eps"])[:3]
        print(f"        q=({d['q1']:+d},{d['q2']:+d})  n={len(d['eps'])}  "
              f"eps_low={[round(x,3) for x in lo]}", flush=True)
    with open(OUT / "validate.json", "w") as fh:
        json.dump({k: v for k, v in res.items() if k != "es_mid_by_charge"}, fh, indent=2)
    print("[sym] validation OK", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "production"], default="validate")
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--chi", type=int, default=400)
    ap.add_argument("--theta", type=float, default=math.atan(1.0 / 9.0))
    ap.add_argument("--max-sweeps", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chi-ramp", action="store_true",
                    help="ramp chi 200->...->chi over early sweeps (faster convergence)")
    args = ap.parse_args()

    if args.mode == "validate":
        validate()
        return

    assert args.L > 0
    tag = args.tag or f"L{args.L}_chi{args.chi}"
    progress_log = str(OUT / f"progress_{tag}.log")
    print(f"[sym] production L={args.L} chi={args.chi} tag={tag}", flush=True)
    res, psi, model = run_dmrg(args.L, args.theta, args.chi,
                               max_sweeps=args.max_sweeps, progress_log=progress_log,
                               chi_ramp=args.chi_ramp)
    print(f"[sym] L={args.L} chi={args.chi} E0={res['E0']:.6f} "
          f"S(L/2)={res['S_Lhalf']:.4f} c_sm={res['c_fit_smoothed']} "
          f"R2_sm={res['R2_smoothed']} q={res['total_charge']} "
          f"chi_used={res['max_chi_used']} ({res['runtime_sec']:.0f}s)", flush=True)
    out_path = OUT / f"so5sym_{tag}.json"
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[sym] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
