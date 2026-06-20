"""
SO(5)_1 lattice realization test (Option B) driver.

Mirrors analysis/run_all.py. Feeds a NEW model (five critical Majorana chains)
to the UNCHANGED instrument (analysis/c_eff.py). See MODEL-DERIVATION.md.

Stages:
  Step 0  reproduce the three positive controls (XX, TFIM-crit, Potts3) through
          the unchanged instrument -- validation gate.
  Step 2  free-point c_eff: n=1,2 self-checks (-> 0.5, 1.0) then n=5 (-> 2.5)
          and n=4 control (-> 2.0).  [T1]
  Step 2b finite-size conformal spectrum: single-sigma (1/8), SO(5) spinor
          (5/8), vector (1); ordering check.  [T2]
  Step 3  unpaired-Majorana parity signature: lowest |E| vs N, n=5 (odd) vs
          n=4 (even) on the same axes.  [T3]
  Step 4  interacting Gross-Neveu scan: 5-leg Ising ladder DMRG, c_eff(g),
          run in PARALLEL across g values.  [T4 / marginality]

Outputs under results/so5/.
"""

from __future__ import annotations

import os

# Cap BLAS threads in the PARENT before numpy import; the parallel DMRG workers
# each get their own cap. The free-fermion linear algebra is tiny.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import json
import sys
import time
import multiprocessing as mp

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, CT_DIR)

from analysis.c_eff import (fit_central_charge, free_fermion_block_entropy,
                            geometric_L_grid, mps_block_entropy)
from models import so5_majorana as so5
from models.xx_chain import build_correlation_matrix

RESULTS_DIR = os.path.join(CT_DIR, "results", "so5")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# ---- parameters -------------------------------------------------------------
FREE_N = 512                      # sites for free-point entanglement (T1)
SPECTRUM_NS = [64, 128, 256, 512]  # ring sizes for finite-size spectrum (T2)
PARITY_NS = [8, 12, 16, 24, 32, 48, 64]   # chain lengths for parity (T3)
# interacting ladder (T4): 5-leg, snake MPS, parallel over g. The exact
# c=5/2 is fixed by the covariance route (Step 2); this scan needs only the
# marginal-flow TREND, so modest L/chi suffice. g=0 cross-check at chi=300
# gave c_eff=2.58 (r2=0.9998); chi=200 here trades a little accuracy for speed.
LADDER_N = 12                     # rungs (60 spin sites for n=5)
LADDER_CHI = 128
LADDER_SWEEPS = 14
LADDER_GS = [-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6]


# ============================================================================
# Step 0: reproduce positive controls
# ============================================================================

def _control_worker(args):
    """Run one positive control in a fresh interpreter; return (key, c_eff, r2)."""
    key, ct_dir = args
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    sys.path.insert(0, ct_dir)
    import numpy as _np
    from analysis.c_eff import (fit_central_charge as _fit, geometric_L_grid as _grid,
                                mps_block_entropy as _mbe)
    if key == "tfim_critical":
        from models import tfim_tenpy
        N, chi = 256, 64
        psi, E = tfim_tenpy.build_ground_state(L=N, g=1.0, chi=chi, verbose=False)
        Lv = _grid(N, n_points=14)
        S = _mbe(psi, [int(x) for x in Lv])
        fit = _fit(Lv, S, N=N, convention="quantum", window=(0.10, 0.50))
        return (key, fit.c_eff, fit.r2, 0.5)
    elif key == "potts3_critical":
        from models import potts3_tenpy
        N, chi = 128, 64
        psi, E = potts3_tenpy.build_ground_state(L=N, chi=chi, J=1.0, h=1.0,
                                                 verbose=False)
        Lv = _grid(N, n_points=14)
        S = _mbe(psi, [int(x) for x in Lv])
        fit = _fit(Lv, S, N=N, convention="quantum", window=(0.10, 0.50))
        return (key, fit.c_eff, fit.r2, 0.8)
    raise ValueError(key)


def step0_controls():
    print("[SO5] Step 0: reproduce positive controls", flush=True)
    out = {}
    # XX chain (free fermion, in-process, fast)
    C = build_correlation_matrix(512)
    Lxx = geometric_L_grid(512, n_points=14)
    Sxx = free_fermion_block_entropy(C, Lxx)
    fxx = fit_central_charge(Lxx, Sxx, N=512, convention="quantum",
                             window=(0.10, 0.50))
    out["xx_chain"] = {"c_eff": fxx.c_eff, "r2": fxx.r2, "expected": 1.0}
    print(f"  xx_chain      c_eff={fxx.c_eff:.4f} (exp 1.0)  r2={fxx.r2:.4f}",
          flush=True)
    # TFIM + Potts3 via parallel DMRG workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        for key, c, r2, exp in pool.imap_unordered(
                _control_worker,
                [("tfim_critical", CT_DIR), ("potts3_critical", CT_DIR)]):
            out[key] = {"c_eff": c, "r2": r2, "expected": exp}
            print(f"  {key:14s}c_eff={c:.4f} (exp {exp})  r2={r2:.4f}",
                  flush=True)
    # gate
    ok = (abs(out["xx_chain"]["c_eff"] - 1.0) < 0.1 and
          abs(out["tfim_critical"]["c_eff"] - 0.5) < 0.06 and
          abs(out["potts3_critical"]["c_eff"] - 0.8) < 0.08)
    out["reproduced"] = bool(ok)
    print(f"  -> controls reproduced: {ok}", flush=True)
    return out


# ============================================================================
# Step 2: free-point c_eff (T1)
# ============================================================================

def step2_free_ceff():
    print("[SO5] Step 2: free-point c_eff (T1)", flush=True)
    Lv = geometric_L_grid(FREE_N, n_points=14)
    out = {}
    for n in [1, 2, 4, 5]:
        S = so5.n_flavor_block_entropy(FREE_N, n, [int(x) for x in Lv])
        fit = fit_central_charge(Lv, S, N=FREE_N, convention="quantum",
                                 window=(0.10, 0.50))
        out[f"n{n}"] = {"n_flavor": n, "c_eff": fit.c_eff, "r2": fit.r2,
                        "expected": n * 0.5, "L_values": Lv.tolist(),
                        "S": S.tolist()}
        role = {1: "self-check", 2: "self-check", 4: "control Spin(4)_1",
                5: "SO(5)_1"}[n]
        print(f"  n={n} ({role:18s}) c_eff={fit.c_eff:.4f} "
              f"(exp {n*0.5})  r2={fit.r2:.4f}", flush=True)
    return out


# ============================================================================
# Step 2b: finite-size conformal spectrum (T2)
# ============================================================================

def step2b_spectrum():
    print("[SO5] Step 2b: finite-size conformal spectrum (T2)", flush=True)
    rows = [so5.finite_size_dimensions(N) for N in SPECTRUM_NS]
    for r in rows:
        print(f"  N={r['N']:4d}  single-sigma(Ising^5, not SO5)="
              f"{r['x_sigma_single']:.4f} (1/8)   "
              f"SO(5)-spinor={r['x_spinor_SO5']:.4f} (5/8)   "
              f"vector={r['x_vector']:.4f} (1)", flush=True)
    last = rows[-1]
    ordered = last["x_spinor_SO5"] < last["x_vector"]
    print(f"  -> spinor(5/8) < vector(1): {ordered}", flush=True)
    return {"rows": rows, "spinor_below_vector": bool(ordered)}


# ============================================================================
# Step 3: parity signature (T3)
# ============================================================================

def step3_parity():
    print("[SO5] Step 3: unpaired-Majorana parity signature (T3)", flush=True)
    out = {}
    for n in [5, 4]:
        es = [so5.edge_mode_splitting(n, N) for N in PARITY_NS]
        out[f"n{n}"] = {"n_flavor": n, "N": list(PARITY_NS),
                        "min_abs_E": [float(e) for e in es]}
        tag = "odd  -> expect protected zero mode" if n % 2 else \
              "even -> expect NO zero mode"
        print(f"  n={n} ({tag})", flush=True)
        for N, e in zip(PARITY_NS, es):
            print(f"      N={N:3d}  min|E|={e:.3e}", flush=True)
    return out


# ============================================================================
# Step 4: interacting Gross-Neveu scan (T4) -- parallel over g
# ============================================================================

def _ladder_worker(args):
    g, n, L, chi, ct_dir = args
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    sys.path.insert(0, ct_dir)
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as _np
    from analysis.c_eff import fit_central_charge as _fit
    from models import so5_majorana as _so5
    r = _so5.run_ladder_dmrg(n, L, g, chi, max_sweeps=LADDER_SWEEPS)
    SvN = _np.array(r["SvN"])
    # inter-rung bonds (cut all n legs): MPS index x*n+y, boundary after (x+1)*n-1
    idx = [(x + 1) * n - 1 for x in range(L - 1)]
    S_rung = SvN[idx]
    Lrung = _np.arange(1, L)
    fit = _fit(Lrung, S_rung, N=L, convention="quantum", window=(0.20, 0.50))
    r["c_eff"] = float(fit.c_eff)
    r["r2"] = float(fit.r2)
    r["accepted"] = bool(fit.accepted)
    r["S_rung"] = S_rung.tolist()
    return r


def step4_interacting():
    print("[SO5] Step 4: interacting Gross-Neveu scan (parallel over g)",
          flush=True)
    n = 5
    jobs = [(g, n, LADDER_N, LADDER_CHI, CT_DIR) for g in LADDER_GS]
    n_procs = min(len(jobs), 7)
    print(f"  launching {len(jobs)} DMRG workers ({n_procs} concurrent), "
          f"n={n} L={LADDER_N} chi={LADDER_CHI}", flush=True)
    ctx = mp.get_context("spawn")
    results = []
    t0 = time.time()
    with ctx.Pool(processes=n_procs) as pool:
        for r in pool.imap_unordered(_ladder_worker, jobs):
            print(f"  g={r['g']:+.2f}: c_eff={r['c_eff']:.3f} r2={r['r2']:.4f} "
                  f"S_mid={r['S_mid']:.3f} xi={r['corr_length']:.2f} "
                  f"maxchi={r['max_chi']} t={r['walltime_s']:.0f}s", flush=True)
            results.append(r)
    results.sort(key=lambda d: d["g"])
    print(f"  scan complete in {time.time()-t0:.0f}s", flush=True)
    return {"n_flavor": n, "L": LADDER_N, "chi": LADDER_CHI, "scan": results}


# ============================================================================
# Figures
# ============================================================================

def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # T1: free-point entanglement scaling
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    from analysis.c_eff import chord_length
    for n in [5, 4, 2, 1]:
        d = results["step2"][f"n{n}"]
        Lv = np.array(d["L_values"]); S = np.array(d["S"])
        x = np.log(chord_length(Lv, FREE_N))
        ax[0].plot(x, S, "o-", ms=3,
                   label=f"n={n}: c_eff={d['c_eff']:.3f} (exp {n*0.5})")
    ax[0].set_xlabel("ln chord(L;N)"); ax[0].set_ylabel("S(L)")
    ax[0].set_title("T1: free-point entanglement scaling")
    ax[0].legend(fontsize=8)
    # T3: parity
    for n, mk in [(5, "o-"), (4, "s--")]:
        d = results["step3"][f"n{n}"]
        ax[1].semilogy(d["N"], np.maximum(d["min_abs_E"], 1e-18), mk,
                       label=f"n={n} ({'odd' if n%2 else 'even'})")
    ax[1].set_xlabel("chain length N"); ax[1].set_ylabel("lowest |E| (edge)")
    ax[1].set_title("T3: unpaired-Majorana parity signature")
    ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "so5_T1_T3.png"), dpi=130)
    plt.close(fig)

    # T4: c_eff(g)
    if "step4" in results and results["step4"]["scan"]:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        scan = results["step4"]["scan"]
        gs = [r["g"] for r in scan]
        ce = [r["c_eff"] for r in scan]
        sm = [r["S_mid"] for r in scan]
        ax[0].plot(gs, ce, "o-")
        ax[0].axhline(2.5, ls=":", c="r", label="SO(5)_1 c=5/2")
        ax[0].set_xlabel("Gross-Neveu g"); ax[0].set_ylabel("c_eff")
        ax[0].set_title(f"T4: c_eff(g)  (5-leg ladder, L={results['step4']['L']})")
        ax[0].legend()
        ax[1].plot(gs, sm, "s-")
        ax[1].set_xlabel("Gross-Neveu g"); ax[1].set_ylabel("mid-chain S")
        ax[1].set_title("T4: entanglement vs g (gapping diagnostic)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "so5_T4_interacting.png"), dpi=130)
        plt.close(fig)


# ============================================================================
# main
# ============================================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    # step4-only mode: reuse already-saved free-fermion results, re-run only the
    # interacting scan (avoids re-running the Step-0 control DMRGs).
    if len(sys.argv) > 1 and sys.argv[1] == "step4":
        with open(os.path.join(RESULTS_DIR, "results_so5.json")) as f:
            results = json.load(f)
        results["step4"] = step4_interacting()
        with open(os.path.join(RESULTS_DIR, "results_so5.json"), "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
        make_figures(results)
        print("[SO5] step4-only: updated results_so5.json + figures", flush=True)
        return

    results = {}
    results["step0"] = step0_controls()
    if not results["step0"]["reproduced"]:
        print("[SO5] FATAL: positive controls did not reproduce; "
              "instrument unvalidated. Stopping.", flush=True)
        with open(os.path.join(RESULTS_DIR, "results_so5.json"), "w") as f:
            json.dump(results, f, indent=2)
        return
    results["step2"] = step2_free_ceff()
    results["step2b"] = step2b_spectrum()
    results["step3"] = step3_parity()
    results["step4"] = step4_interacting()

    with open(os.path.join(RESULTS_DIR, "results_so5.json"), "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"[SO5] wrote {os.path.join(RESULTS_DIR, 'results_so5.json')}",
          flush=True)

    make_figures(results)
    print("[SO5] wrote figures", flush=True)

    # gap-ratio table (T2)
    with open(os.path.join(RESULTS_DIR, "gap_ratio_table.csv"), "w") as f:
        f.write("N,x_sigma_single_Ising5,x_SO5_spinor,x_vector,"
                "ratio_spinor_over_vector\n")
        for r in results["step2b"]["rows"]:
            f.write(f"{r['N']},{r['x_sigma_single']:.5f},"
                    f"{r['x_spinor_SO5']:.5f},{r['x_vector']:.5f},"
                    f"{r['ratio_spinor_over_vector']:.5f}\n")
    print("[SO5] wrote gap_ratio_table.csv", flush=True)


if __name__ == "__main__":
    main()
