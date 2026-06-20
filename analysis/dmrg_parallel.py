"""
Parallel DMRG worker pool for positive controls.

Each job is (model_name, params_dict). Workers spawn fresh Python interpreters
(safe under Windows mp). OMP_NUM_THREADS is capped in the parent before any
numpy/tenpy import.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import multiprocessing as mp
import sys
import time

import numpy as np


def _dmrg_worker(args):
    """Worker that builds a quantum ground state, computes block entropies,
    saves to per-job .npz, returns metadata."""
    job_id, model_name, params, L_values, ct_dir, tmpdir = args
    sys.path.insert(0, ct_dir)
    from analysis.c_eff import mps_block_entropy

    t0 = time.time()
    if model_name == "tfim_critical":
        from models import tfim_tenpy
        psi, E = tfim_tenpy.build_ground_state(
            L=params["L"], g=params["g"], chi=params["chi"],
            verbose=False, seed=params.get("seed", 0))
        meta = {"E_per_site": float(E / params["L"])}
    elif model_name == "tfim_gapped":
        from models import tfim_tenpy
        psi, E = tfim_tenpy.build_ground_state(
            L=params["L"], g=params["g"], chi=params["chi"],
            verbose=False, seed=params.get("seed", 0))
        meta = {"E_per_site": float(E / params["L"])}
    elif model_name == "potts3_critical":
        from models import potts3_tenpy
        psi, E = potts3_tenpy.build_ground_state(
            L=params["L"], chi=params["chi"], J=params.get("J", 1.0),
            h=params.get("h", 1.0), verbose=False, seed=params.get("seed", 0))
        meta = {"E_per_site": float(E / params["L"])}
    else:
        raise ValueError(model_name)

    S = mps_block_entropy(psi, [int(L) for L in L_values])
    dt = time.time() - t0
    fname = os.path.join(tmpdir, f"S_{job_id}_{model_name}.npz")
    np.savez_compressed(fname,
                         S=S, L_values=np.array(L_values),
                         dt=dt, **meta,
                         L_chain=params["L"], chi=params["chi"],
                         model=model_name)
    return (job_id, model_name, params, fname, dt, meta)


def run_dmrg_jobs(jobs: list[dict], ct_dir: str, tmpdir: str,
                   n_procs: int | None = None) -> list[dict]:
    """
    jobs: list of dicts with keys {model, params, L_values}.
    Each becomes one worker process. Returns list of result dicts.
    """
    if n_procs is None:
        n_procs = min(len(jobs), 6)
    os.makedirs(tmpdir, exist_ok=True)

    job_args = []
    for i, j in enumerate(jobs):
        job_args.append((i, j["model"], j["params"], j["L_values"],
                          ct_dir, tmpdir))

    print(f"[CC-DMRG] launching {len(jobs)} workers ({n_procs} concurrent)",
          flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        outs = []
        for res in pool.imap_unordered(_dmrg_worker, job_args):
            job_id, model, params, fname, dt, meta = res
            print(f"[CC-DMRG] done id={job_id} {model} L={params.get('L')} "
                  f"chi={params.get('chi')} in {dt:.1f}s  meta={meta}", flush=True)
            outs.append({"job_id": job_id, "model": model, "params": params,
                          "fname": fname, "dt": dt, "meta": meta})
    print(f"[CC-DMRG] all complete in {time.time()-t0:.1f}s", flush=True)
    return outs
