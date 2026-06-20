"""
T3-phase: topological-phase robustness of the unpaired-Majorana parity signature.

Upgrades T3 (RESULTS-SO5-LATTICE.md) from a single point to a PHASE. We sweep a
2D parameter region (chemical potential mu, inter-flavor coupling strength w) and
show that the n=5 (odd) protected Majorana zero mode survives at machine
precision across the gapped topological Kitaev phase and lifts only at the bulk
gap closing (mu = 2t), while the n=4 (even, Spin(4)_1) control has no protected
zero mode anywhere (for w>0) under the identical pipeline.

SCOPE (read carefully, governs the write-up):
  This is NOT a claim that the SO(5)_1 CRITICAL theory is robust at finite
  coupling. MODEL-DERIVATION.md sec 4 proves the Gross-Neveu coupling is marginal
  and no finite g gives a new c=5/2 critical theory; that no-go stands and is not
  retested. The CFT is realized at a POINT (the free critical point). The
  TOPOLOGICAL PARITY SIGNATURE is realized across a PHASE (the gapped topological
  region). The gap is what protects the edge mode; that is why the signature -
  not the CFT - is robust at finite coupling.

Reuses the unchanged core BdG machinery in models/so5_majorana.py
(multi_flavor_topological_matrix + single_particle_min_abs_energy, via
edge_mode_splitting). c_eff.py is not touched; no DMRG here -- pure free-fermion
diagonalisation.

Parity-preserving constraint: the inter-flavor coupling w enters as
lam * A_{ab} (i/2) gamma_a gamma_b with A real antisymmetric (seed-fixed,
generic). Every term in the BdG matrix M is a Majorana bilinear, so M is real
antisymmetric and H = (i/4) gamma^T M gamma conserves total fermion parity
exactly. The driver asserts M = -M^T (no parity-odd / single-Majorana term).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, CT_DIR)

from models import so5_majorana as so5

RESULTS_DIR = os.path.join(CT_DIR, "results", "so5", "phase")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# ---- fixed parameters -------------------------------------------------------
N = 64                       # prior T3 floor for n=5 (~3e-18 at mu=0.5)
T_HOP = 1.0
DELTA = 1.0                  # self-dual kinetic line
SEED = 0                     # same generic antisymmetric coupling as Option-B T3
PLATEAU_THRESH = 1e-10       # "protected" = min|E| below this

MU_GRID = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 4)    # 31 pts, crosses 2t
W_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 4)     # 11 pts
N_VALUES_BOUNDARY = [64, 128, 256]   # finite-size check near the boundary
MU_BOUNDARY_CHECK = [1.0, 1.5, 1.8, 1.9, 1.95, 2.0, 2.1, 2.5]


def confirm_parity_even() -> dict:
    """Assert the BdG matrix is real antisymmetric (parity-even Majorana
    bilinear) for representative (mu, w); return the check record."""
    worst = 0.0
    for mu in (0.0, 1.0, 2.0):
        for w in (0.0, 0.5, 1.0):
            M = so5.multi_flavor_topological_matrix(5, N, t=T_HOP, Delta=DELTA,
                                                    mu=mu, lam=w, seed=SEED)
            worst = max(worst, float(np.abs(M + M.T).max()),
                        0.0 if np.isrealobj(M) else 1.0)
    return {"max_abs_M_plus_MT": worst, "parity_even": bool(worst < 1e-12)}


def sweep(n_flavor: int) -> np.ndarray:
    """2D grid grid[i,j] = min|E| at (MU_GRID[i], W_GRID[j])."""
    grid = np.zeros((len(MU_GRID), len(W_GRID)), dtype=np.float64)
    for i, mu in enumerate(MU_GRID):
        for j, w in enumerate(W_GRID):
            grid[i, j] = so5.edge_mode_splitting(
                n_flavor, N, t=T_HOP, Delta=DELTA, mu=float(mu),
                lam=float(w), seed=SEED)
    return grid


def boundary_finite_size() -> dict:
    """min|E| vs mu near the boundary at several N (n=5, fixed w=0.5) to show
    the near-boundary rise is finite-size (shrinks with N as ξ→∞ at mu=2t)."""
    out = {}
    for Nv in N_VALUES_BOUNDARY:
        out[str(Nv)] = [
            float(so5.edge_mode_splitting(5, Nv, t=T_HOP, Delta=DELTA,
                                          mu=mu, lam=0.5, seed=SEED))
            for mu in MU_BOUNDARY_CHECK]
    return out


def robustness_margin(grid5: np.ndarray) -> dict:
    """For n=5, characterise the region where min|E| < PLATEAU_THRESH."""
    topo = MU_GRID < 2.0
    # per-w: largest mu (within topo) below which the whole column stays < thresh
    per_w = {}
    for j, w in enumerate(W_GRID):
        col = grid5[:, j]
        mu_below = MU_GRID[(col < PLATEAU_THRESH)]
        # contiguous plateau from mu=0: highest mu reached before first breach
        plateau_max = 0.0
        for i, mu in enumerate(MU_GRID):
            if col[i] < PLATEAU_THRESH:
                plateau_max = float(mu)
            else:
                break
        per_w[f"{w:.1f}"] = {"plateau_mu_max": plateau_max,
                             "n_pts_below_thresh": int((col < PLATEAU_THRESH).sum())}
    # max over the topological region for all w (the robustness number)
    max_in_topo = float(grid5[topo, :].max())
    # deepest robust sub-region (mu where ALL w stay below thresh)
    all_w_below = np.all(grid5 < PLATEAU_THRESH, axis=1)
    deep_mu_max = 0.0
    for i, mu in enumerate(MU_GRID):
        if all_w_below[i]:
            deep_mu_max = float(mu)
        else:
            break
    return {"per_w_plateau_mu_max": per_w,
            "max_minE_over_topological_region_allw": max_in_topo,
            "deep_plateau_mu_max_allw": deep_mu_max,
            "plateau_threshold": PLATEAU_THRESH}


def make_figures(grid5, grid4, boundary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = 1e-18
    ext = [W_GRID[0] - 0.05, W_GRID[-1] + 0.05,
           MU_GRID[0] - 0.05, MU_GRID[-1] + 0.05]

    # D1/D3: heat maps log10(min|E|) for n=5 and n=4
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for k, (g, n) in enumerate([(grid5, 5), (grid4, 4)]):
        im = ax[k].imshow(np.log10(np.maximum(g, floor)), origin="lower",
                          aspect="auto", extent=ext, cmap="viridis",
                          vmin=-18, vmax=0)
        ax[k].axhline(2.0, color="r", ls="--", lw=1.2,
                      label="bulk gap closing  mu=2t")
        ax[k].set_xlabel("inter-flavor coupling  w/t")
        ax[k].set_title(f"n={n} ({'odd, SO(5)_1' if n==5 else 'even, Spin(4)_1'})"
                        f": log10(min|E|)")
        ax[k].legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax[k], fraction=0.046, pad=0.04)
    ax[0].set_ylabel("chemical potential  mu/t")
    fig.suptitle("T3-phase: edge-mode splitting over (mu, w). "
                 "n=5 stays at floor in the topological phase; n=4 does not.")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "so5_phase_heatmaps.png"), dpi=130)
    plt.close(fig)

    # D2: min|E| vs mu line cuts at w = 0, 0.5, 1.0 (n=5 and n=4)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    wcuts = [0.0, 0.5, 1.0]
    for g, n, a in [(grid5, 5, ax[0]), (grid4, 4, ax[1])]:
        for w in wcuts:
            j = int(np.argmin(np.abs(W_GRID - w)))
            a.semilogy(MU_GRID, np.maximum(g[:, j], floor), "o-", ms=3,
                       label=f"w={w:.1f}")
        a.axvline(2.0, color="r", ls="--", lw=1.2, label="mu=2t")
        a.axhline(PLATEAU_THRESH, color="gray", ls=":", lw=1,
                  label=f"{PLATEAU_THRESH:g}")
        a.set_xlabel("mu/t"); a.set_title(f"n={n}: min|E| vs mu")
        a.legend(fontsize=8)
    ax[0].set_ylabel("lowest |E| (edge splitting)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "so5_phase_linecuts.png"), dpi=130)
    plt.close(fig)

    # boundary finite-size check (n=5, w=0.5)
    fig, a = plt.subplots(figsize=(6.5, 4.6))
    for Nv in N_VALUES_BOUNDARY:
        a.semilogy(MU_BOUNDARY_CHECK, np.maximum(boundary[str(Nv)], floor),
                   "o-", ms=4, label=f"N={Nv}")
    a.axvline(2.0, color="r", ls="--", lw=1.2, label="mu=2t")
    a.axhline(PLATEAU_THRESH, color="gray", ls=":", lw=1)
    a.set_xlabel("mu/t"); a.set_ylabel("lowest |E| (n=5, w=0.5)")
    a.set_title("Near-boundary rise is finite-size (shrinks with N)")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "so5_phase_boundary_finiteN.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("[T3-phase] confirming parity-even coupling", flush=True)
    parity = confirm_parity_even()
    print(f"  max|M+M^T|={parity['max_abs_M_plus_MT']:.2e}  "
          f"parity_even={parity['parity_even']}", flush=True)

    print(f"[T3-phase] 2D sweep n=5  ({len(MU_GRID)}x{len(W_GRID)}), N={N}",
          flush=True)
    grid5 = sweep(5)
    print(f"[T3-phase] 2D sweep n=4  control", flush=True)
    grid4 = sweep(4)

    print("[T3-phase] boundary finite-size check", flush=True)
    boundary = boundary_finite_size()

    margin = robustness_margin(grid5)
    print(f"  max min|E| over topological region (all w), n=5: "
          f"{margin['max_minE_over_topological_region_allw']:.3e}", flush=True)
    print(f"  deep plateau (all w below {PLATEAU_THRESH:g}) up to mu="
          f"{margin['deep_plateau_mu_max_allw']:.1f}", flush=True)

    # n=4 control: min over the topological region (should be finite for w>0)
    topo = MU_GRID < 2.0
    w_pos = W_GRID > 0.0
    n4_min_wpos = float(grid4[np.ix_(topo, w_pos)].min())
    n4_min_w0 = float(grid4[topo, 0].min())
    print(f"  n=4 control: min over topo region (w>0) = {n4_min_wpos:.3e} "
          f"(finite => no protected zero mode); w=0 column min = "
          f"{n4_min_w0:.3e} (decoupled limit)", flush=True)

    # ---- save raw grids -------------------------------------------------
    np.savez_compressed(os.path.join(RESULTS_DIR, "phase_grids.npz"),
                        mu_grid=MU_GRID, w_grid=W_GRID,
                        grid_n5=grid5, grid_n4=grid4, N=N)
    for n, g in [(5, grid5), (4, grid4)]:
        with open(os.path.join(RESULTS_DIR, f"minE_grid_n{n}.csv"), "w") as f:
            f.write("mu\\w," + ",".join(f"{w:.1f}" for w in W_GRID) + "\n")
            for i, mu in enumerate(MU_GRID):
                f.write(f"{mu:.1f}," +
                        ",".join(f"{g[i, j]:.6e}" for j in range(len(W_GRID)))
                        + "\n")

    results = {
        "N": N, "t": T_HOP, "Delta": DELTA, "seed": SEED,
        "mu_grid": MU_GRID.tolist(), "w_grid": W_GRID.tolist(),
        "parity_check": parity,
        "robustness_margin_n5": margin,
        "n4_control_min_topo_wpos": n4_min_wpos,
        "n4_control_min_topo_w0_decoupled": n4_min_w0,
        "boundary_finite_size": {"mu": MU_BOUNDARY_CHECK, "by_N": boundary},
    }
    with open(os.path.join(RESULTS_DIR, "results_phase.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[T3-phase] wrote results_phase.json + grids", flush=True)

    make_figures(grid5, grid4, boundary)
    print("[T3-phase] wrote figures", flush=True)


if __name__ == "__main__":
    main()
