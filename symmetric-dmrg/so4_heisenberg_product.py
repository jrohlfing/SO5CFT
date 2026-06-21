"""Exact, fully-converged n=4 (Spin(4)_1) control via the decoupling.

At the Reshetikhin point theta=0 the SO(4) vector chain is EXACTLY two decoupled
spin-1/2 Heisenberg chains (B = 2[S^A.S^A + S^B.S^B]). The ground state is a
product, so its half-cut Schmidt data factorizes:

    lambda_{ij} = lambda^A_i * lambda^B_j,
    eps_{ij}    = eps^A_i + eps^B_j,
    charge      = (q1,q2) = (m^A_i + m^B_j,  m^A_i - m^B_j)

where m = 2*S_z (integer Cartan charge). So a SINGLE high-chi Heisenberg chain
(2-dim site, cheap, fully convergeable) yields the EXACT SO(4) entanglement
tower -- avoiding the chi = chi_single^2 blow-up that limits the combined-site
DMRG. c = 2 (1 per chain) is then exact, and S(L/2) = 2 * S_single.

This is the rigorous primary n=4 result; the combined-site SO(4) DMRG
(so4_bbq_dmrg_sym.py) is the same-pipeline cross-check on the leading tower.

Writes so4_L{L}_prod.json in the es_mid_by_charge format consumed by
so_matched_tower.py (nominal chi_max=999999 so it is selected as the best n=4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

THREADS = int(os.environ.get("DMRG_THREADS_PER_WORKER", "16"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import numpy as np
from scipy import stats

import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg as dmrg_mod

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(parents=True, exist_ok=True)


def run_heisenberg(L, chi, max_sweeps=60):
    """Spin-1/2 Heisenberg AFM chain, OBC, Sz conserved. Returns (psi, model)."""
    model = SpinChain({"L": L, "S": 0.5, "Jx": 1.0, "Jy": 1.0, "Jz": 1.0,
                       "bc_MPS": "finite", "conserve": "Sz"})
    sites = model.lat.mps_sites()
    init = ["up", "down"] * (L // 2)
    psi = MPS.from_product_state(sites, init, bc="finite")
    chi_list = {}
    c, sw = 200, 0
    while c < chi:
        chi_list[sw] = c; c *= 2; sw += 3
    chi_list[sw] = chi
    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "mixer": True, "chi_list": chi_list,
        "max_sweeps": max_sweeps, "min_sweeps": 8, "max_E_err": 1e-11,
        "combine": True,
    }
    info = dmrg_mod.run(psi, model, dmrg_params)
    psi.canonical_form()
    return psi, float(info["E"]), int(max(psi.chi))


def single_chain_es_by_2sz(psi, bond):
    """Half-cut ES of the single chain as list of (m=2*Sz_int, eps). TeNPy charge
    is 2*Sz (integer) when conserve='Sz' with S=1/2 (qmod gives 2*Sz)."""
    spec = psi.entanglement_spectrum(by_charge=True)[bond]
    out = []
    for charge, ent in spec:
        m = int(charge[0])            # = 2*Sz block label
        for e in np.asarray(ent, dtype=float):
            out.append((m, float(e)))
    return out


def s_half_single(psi, L):
    return float(psi.entanglement_entropy()[L // 2 - 1])


def cc_c_single(psi, L):
    """Even-odd smoothed Calabrese-Cardy fit (the Heisenberg chain has a strong
    even-odd oscillation in S(l); average consecutive l before fitting)."""
    ents = [float(x) for x in psi.entanglement_entropy()]
    sm = [(ents[i] + ents[i + 1]) / 2 for i in range(len(ents) - 1)]
    lo, hi = max(2, L // 4), min(L - 2, 3 * L // 4)
    xs, ys = [], []
    for i, l in enumerate(range(1, L - 1)):
        if lo <= l <= hi:
            xs.append(math.log((2 * L / math.pi) * math.sin(math.pi * l / L)))
            ys.append(sm[i])
    res = stats.linregress(np.array(xs), np.array(ys))
    return float(6 * res.slope), float(res.rvalue ** 2)


def build_product_es(single, keep=4000):
    """Product of two identical single-chain spectra -> SO(4) ES by (q1,q2).
    Truncate to the lowest `keep` combined levels to bound output size."""
    # sort single levels, keep enough to cover the low tower
    single = sorted(single, key=lambda x: x[1])
    # cap single list to avoid O(N^2) blow-up; the low tower needs only the
    # smallest eps; keep up to 400 per chain (eps grows fast).
    sA = single[:400]
    combos = []
    for (mA, eA) in sA:
        for (mB, eB) in sA:
            # SO(4) Cartan charges: q1=(mA+mB)/2, q2=(mA-mB)/2 with m=2*Sz (odd).
            # mA,mB both odd -> sums even -> integer SO(4) charges. (0,0) requires
            # mA=mB=0, impossible (m odd) -> no zero weight: the even-parity signature.
            combos.append((eA + eB, (mA + mB) // 2, (mA - mB) // 2))
    combos.sort()
    combos = combos[:keep]
    # group into (q1,q2) sectors
    sectors = {}
    for (e, q1, q2) in combos:
        sectors.setdefault((q1, q2), []).append(e)
    es = []
    for (q1, q2), eps in sectors.items():
        eps = sorted(eps)
        es.append({"q1": q1, "q2": q2, "eps": eps,
                   "lambda": [math.exp(-0.5 * e) for e in eps]})
    es.sort(key=lambda d: min(d["eps"]))
    return es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--chi", type=int, default=1600)
    args = ap.parse_args()
    L = args.L

    psi, E_single, chi_used = run_heisenberg(L, args.chi)
    S_single = s_half_single(psi, L)
    c_single, r2 = cc_c_single(psi, L)
    single = single_chain_es_by_2sz(psi, L // 2)
    es = build_product_es(single)

    res = {
        "model": "SO4_via_2xHeisenberg_product", "N": 4, "theta": 0.0,
        "L": L, "chi_max": 999999, "chi_single_used": chi_used,
        "E_single_chain": E_single, "E0_so4_equiv": 4.0 * E_single,
        "S_single": S_single, "S_Lhalf": 2.0 * S_single,
        "c_single_chain": c_single, "c_fit_smoothed": 2.0 * c_single,
        "R2_single": r2,
        "es_mid_by_charge": es,
        "note": ("exact SO(4) half-cut tower from product of two converged "
                 "spin-1/2 Heisenberg chains (decoupling exact at theta=0)."),
    }
    out = RES / f"so4_L{L}_prod.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"[so4prod] L={L} chi_single={chi_used} S_single={S_single:.5f} "
          f"S(L/2)=2*={2*S_single:.5f} c_single={c_single:.4f} (x2={2*c_single:.4f}) "
          f"R2={r2:.5f}")
    # leading tower
    gmin = min(min(s["eps"]) for s in es)
    print("  leading sectors (q1,q2): n  eps-min")
    for s in es[:8]:
        print(f"    ({s['q1']:+d},{s['q2']:+d})  n={len(s['eps']):4d}  "
              f"eps-min={round(min(s['eps'])-gmin,4)}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
