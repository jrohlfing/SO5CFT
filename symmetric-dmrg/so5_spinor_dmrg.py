"""SO(5) = Sp(4) chain in the 4-dim SPINOR (Sp(4)-fundamental) representation --
charge-conserving DMRG. Task A of the spinor-sector handoff: feasibility benchmark.

Unlike the 5-dim VECTOR chain (integer Cartan weights, cannot expose the spinor
in its entanglement spectrum), the spinor site carries HALF-INTEGER Cartan weights
(+-1/2,+-1/2) -- the spinor sector lives in the local Hilbert space directly. If a
critical SO(5)_1 point is reachable in this chain, the spinor multiplet appears in
the ES with no twist needed.

Construction (no basis rotation needed -- Cartans are already diagonal):
  SO(5) gamma matrices (4x4, {Gamma^a,Gamma^b}=2 delta^{ab}):
    G1=sx(x)I, G2=sy(x)I, G3=sz(x)sx, G4=sz(x)sy, G5=sz(x)sz
  Generators  Sigma^{ab} = -(i/2) G^a G^b  (a<b), 10 of them, Hermitian.
  Cartans  Sigma^{12}=(1/2) sz(x)I,  Sigma^{34}=(1/2) I(x)sz  -> diagonal.
  Comp-basis states (up-up,up-dn,dn-up,dn-dn) have weights
    (+1/2,+1/2),(+1/2,-1/2),(-1/2,+1/2),(-1/2,-1/2)  [stored x2 as integer charges].
  Casimir  sum_{a<b} (Sigma^{ab})^2 = 5/2 (the spinor C2 of SO(5)).

Bond Hamiltonian (BBQ family, same as the vector chain):
    h_bond = cos(theta) B + sin(theta) B^2,   B = sum_{a<b} Sigma^{ab}_i Sigma^{ab}_{i+1}
theta=0 = bare bilinear (SO(5)-Heisenberg in the spinor rep; predicted dimerized).

Usage:
    python so5_spinor_dmrg.py --mode validate
    python so5_spinor_dmrg.py --mode production --L 64 --chi 800 --theta 0.0 --chi-ramp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

THREADS = int(os.environ.get("STAEDERT_THREADS_PER_WORKER", "16"))
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

D = 4  # spinor rep dimension
# Cartan weights x2 (integer), comp-basis order uu, ud, du, dd:
SPINOR_W2 = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=int)

THIS_DIR = Path(__file__).resolve().parent
OUT = THIS_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _gammas():
    G = [np.kron(_sx, _I2), np.kron(_sy, _I2), np.kron(_sz, _sx),
         np.kron(_sz, _sy), np.kron(_sz, _sz)]
    # sanity: Clifford algebra
    for a in range(5):
        for b in range(5):
            anti = G[a] @ G[b] + G[b] @ G[a]
            assert np.allclose(anti, 2 * (a == b) * np.eye(4)), (a, b)
    return G


def _generators():
    G = _gammas()
    gens = []
    for a in range(5):
        for b in range(a + 1, 5):
            S = -0.5j * (G[a] @ G[b])
            assert np.allclose(S, S.conj().T), (a, b)  # Hermitian
            gens.append((a, b, S))
    assert len(gens) == 10
    return gens


def _check_cartans(gens):
    d = {(a, b): S for a, b, S in gens}
    S12 = d[(0, 1)]
    S34 = d[(2, 3)]
    # diagonal with eigenvalues +-1/2 matching SPINOR_W2/2
    assert np.allclose(S12, np.diag([0.5, 0.5, -0.5, -0.5])), S12
    assert np.allclose(S34, np.diag([0.5, -0.5, 0.5, -0.5])), S34


def _bilinear():
    gens = _generators()
    _check_cartans(gens)
    B = np.zeros((D * D, D * D), dtype=complex)
    for _, _, S in gens:
        B += np.kron(S, S)
    # Casimir check: B is sum Sigma(x)Sigma; on-site sum_(a<b) Sigma^2 = 5/2 * I
    C2 = sum(S @ S for _, _, S in gens)
    assert np.allclose(C2, 2.5 * np.eye(D)), np.round(C2, 4)
    return B


def _bond(theta):
    B = _bilinear()
    h = math.cos(theta) * B + math.sin(theta) * (B @ B)
    # already in weight (Cartan-diagonal) basis; no rotation
    return h, B


def _charge_violations(h, tol=1e-12):
    h4 = h.reshape(D, D, D, D)  # [m_i,p_{i+1},n_i,q_{i+1}]
    nt = nv = 0
    for m in range(D):
        for p in range(D):
            for n in range(D):
                for q in range(D):
                    if abs(h4[m, p, n, q]) > tol:
                        nt += 1
                        dq = (SPINOR_W2[m] + SPINOR_W2[p]) - (SPINOR_W2[n] + SPINOR_W2[q])
                        if not np.all(dq == 0):
                            nv += 1
    return nt, nv


def make_spinor_site():
    chinfo = ChargeInfo([1, 1], ["q1", "q2"])
    leg = LegCharge.from_qflat(chinfo, SPINOR_W2)
    site = Site(leg, state_labels=["uu", "ud", "du", "dd"])
    for m in range(D):
        for n in range(D):
            M = np.zeros((D, D), dtype=complex)
            M[m, n] = 1.0
            site.add_op(f"E{m}{n}", M, hc=f"E{n}{m}")
    return site


class SO5SpinorBBQ(CouplingMPOModel):
    def init_sites(self, mp):
        return make_spinor_site()

    def init_terms(self, mp):
        theta = mp.get("theta", 0.0)
        tol = mp.get("term_tol", 1e-12)
        h, _ = _bond(theta)
        h4 = h.reshape(D, D, D, D)
        for m in range(D):
            for n in range(D):
                for p in range(D):
                    for q in range(D):
                        c = h4[m, p, n, q]
                        if abs(c) > tol:
                            self.add_coupling(c, 0, f"E{m}{n}", 0, f"E{p}{q}", 1,
                                              plus_hc=False)


def _neutral_init(L):
    """Total charge (0,0). Block [uu,dd] is neutral ((+1,+1)+(-1,-1)=0); also
    [ud,du]. Use pairs; needs even L."""
    assert L % 2 == 0, "use even L for neutral total charge"
    init = (["uu", "dd"] * (L // 2))
    w = SPINOR_W2[[ {"uu":0,"ud":1,"du":2,"dd":3}[s] for s in init ]].sum(axis=0)
    assert np.all(w == 0), w
    return init


def fit_cc_obc(ents, L):
    lo, hi = max(2, L // 4), min(L - 2, 3 * L // 4)
    S_vs_l = [(l, float(ents[l - 1])) for l in range(1, L)]
    sm = [(S_vs_l[i][1] + S_vs_l[i + 1][1]) / 2 for i in range(len(S_vs_l) - 1)]
    xs, ys = [], []
    for i, l in enumerate(range(1, L - 1)):
        if lo <= l <= hi:
            xs.append(math.log((2 * L / math.pi) * math.sin(math.pi * l / L)))
            ys.append(sm[i])
    out = {"S_vs_l": S_vs_l}
    if len(xs) >= 4:
        r = stats.linregress(np.array(xs), np.array(ys))
        out.update(c_fit_smoothed=float(6 * r.slope), R2_smoothed=float(r.rvalue ** 2))
    return out


def bond_energies(psi, theta):
    """<h_{i,i+1}> per bond, summing matrix-unit coupling expectations. Returns
    list over bonds. Used for the dimerization order parameter."""
    h4 = _bond(theta)[0].reshape(D, D, D, D)
    terms = [(m, n, p, q, h4[m, p, n, q]) for m in range(D) for n in range(D)
             for p in range(D) for q in range(D) if abs(h4[m, p, n, q]) > 1e-12]
    es = []
    for i in range(psi.L - 1):
        e = 0.0
        for (m, n, p, q, c) in terms:
            ev = psi.expectation_value_term([(f"E{m}{n}", i), (f"E{p}{q}", i + 1)])
            e += (c * ev).real
        es.append(float(e))
    return es


def entanglement_spectrum_by_charge(psi, bond):
    """ES at `bond` organized by Cartan charge (stored x2). For the spinor rep the
    leading shell, if critical SO(5)_1, should be the 4-fold spinor multiplet with
    weights (+-1,+-1) [= (+-1/2,+-1/2)]."""
    spec = psi.entanglement_spectrum(by_charge=True)[bond]
    out = []
    for charge, ent in spec:
        eps = np.sort(np.asarray(ent, dtype=float))
        out.append({"q1": int(charge[0]), "q2": int(charge[1]), "eps": eps.tolist()})
    out.sort(key=lambda d: (min(d["eps"]) if d["eps"] else 1e9))
    return out


def run(L, theta, chi, max_sweeps=40, chi_ramp=False, want_gap=False, measure_bonds=True,
        want_es=False):
    t0 = time.time()
    model = SO5SpinorBBQ({"L": L, "bc_MPS": "finite", "theta": theta})
    sites = model.lat.mps_sites()
    psi = MPS.from_product_state(sites, _neutral_init(L), bc="finite")
    dp = {"trunc_params": {"chi_max": chi, "svd_min": 1e-11},
          "mixer": True, "mixer_params": {"amplitude": 1e-5, "decay": 2.0, "disable_after": 12},
          "max_sweeps": max_sweeps, "min_sweeps": 6, "max_E_err": 1e-9,
          "combine": True, "active_sites": 2, "lanczos_params": {"N_max": 16}}
    if chi_ramp:
        rungs, c = [], 200
        while c < chi:
            rungs.append(c); c *= 2
        rungs.append(chi)
        dp["chi_list"] = {3 * i: r for i, r in enumerate(rungs)}
    info = dmrg_mod.run(psi, model, dp)
    E0 = float(info["E"])
    psi.canonical_form()
    ents = psi.entanglement_entropy()
    fit = fit_cc_obc(ents, L)
    res = {"model": "SO5_spinor_BBQ", "L": L, "theta": theta, "chi_max": chi,
           "E0": E0, "E0_per_bond": E0 / (L - 1), "S_Lhalf": float(ents[L // 2 - 1]),
           "S_max": float(max(ents)), "c_fit_smoothed": fit.get("c_fit_smoothed"),
           "R2_smoothed": fit.get("R2_smoothed"), "S_vs_l": fit["S_vs_l"],
           "total_charge": [int(x) for x in psi.get_total_charge(True)],
           "max_chi_used": int(max(psi.chi)), "runtime_sec": time.time() - t0}
    if measure_bonds:
        be = bond_energies(psi, theta)
        res["bond_energies"] = be
        # dimerization order parameter: bulk |e_i - e_{i+1}|, averaged over central third
        n = len(be)
        lo, hi = n // 3, 2 * n // 3
        dimer = [abs(be[i] - be[i + 1]) for i in range(lo, hi - 1)]
        res["dimerization"] = float(np.mean(dimer)) if dimer else None
        res["bond_e_central"] = be[n // 2 - 1: n // 2 + 2]
    if want_es:
        res["es_mid_by_charge"] = entanglement_spectrum_by_charge(psi, L // 2)
    if want_gap:
        # first excited state in the same (neutral) charge sector via orthogonalization
        psi2 = MPS.from_product_state(sites, _neutral_init(L), bc="finite")
        dp2 = dict(dp); dp2["orthogonal_to"] = [psi]
        info2 = dmrg_mod.run(psi2, model, dp2)
        res["E1"] = float(info2["E"])
        res["gap"] = float(info2["E"]) - E0
    return res, psi, model


def validate():
    print("[spinor] building SO(5) spinor rep ...", flush=True)
    h, B = _bond(0.0)
    ev = np.sort(np.linalg.eigvalsh(B).real)
    vals, cts = np.unique(np.round(ev, 6), return_counts=True)
    print(f"[spinor] B (bilinear) spectrum val x mult: {list(zip(vals.tolist(), cts.tolist()))}", flush=True)
    nt, nv = _charge_violations(h)
    print(f"[spinor] charge terms={nt} violations={nv}", flush=True)
    assert nv == 0
    print("[spinor] tiny DMRG L=12 chi=64 theta=0 ...", flush=True)
    res, psi, model = run(12, 0.0, 64, max_sweeps=20, want_gap=True)
    print(f"[spinor] E0={res['E0']:.6f} E0/bond={res['E0_per_bond']:.6f} "
          f"S(L/2)={res['S_Lhalf']:.4f} q={res['total_charge']} "
          f"dimerization={res['dimerization']:.4f} gap={res.get('gap'):.4f}", flush=True)
    print(f"[spinor] central bond energies: {[round(x,4) for x in res['bond_e_central']]}", flush=True)
    json.dump({k: v for k, v in res.items() if k not in ("S_vs_l", "bond_energies")},
              open(OUT / "validate_spinor.json", "w"), indent=2)
    print("[spinor] validation OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "production"], default="validate")
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--chi", type=int, default=400)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--max-sweeps", type=int, default=40)
    ap.add_argument("--chi-ramp", action="store_true")
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.mode == "validate":
        validate(); return
    assert args.L > 0
    tag = args.tag or f"spinor_L{args.L}_chi{args.chi}_th{args.theta:.4f}"
    res, psi, model = run(args.L, args.theta, args.chi, max_sweeps=args.max_sweeps,
                          chi_ramp=args.chi_ramp, want_gap=args.gap)
    print(f"[spinor] L={args.L} theta={args.theta:.4f} chi={args.chi} E0={res['E0']:.6f} "
          f"S(L/2)={res['S_Lhalf']:.4f} c_sm={res['c_fit_smoothed']} "
          f"dimer={res['dimerization']} gap={res.get('gap')} ({res['runtime_sec']:.0f}s)", flush=True)
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=2)
    print(f"[spinor] wrote {OUT / (tag + '.json')}", flush=True)


if __name__ == "__main__":
    main()
