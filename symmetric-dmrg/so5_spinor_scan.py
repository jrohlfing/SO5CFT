"""Task A scan: sweep the BBQ angle theta in the SO(5) SPINOR-rep chain to locate
any critical point (S(L/2) peak, c_fit ~ 5/2). Gapped/dimerized phases show small,
saturating S(L/2) and large dimerization; a critical point shows growing S(L/2),
c_fit meaningful, dimerization dropping. Writes spinor_scan_L{L}.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from so5_spinor_dmrg import run  # noqa: E402

RES = THIS_DIR / "results"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--chi", type=int, default=300)
    ap.add_argument("--npts", type=int, default=21)
    ap.add_argument("--max-sweeps", type=int, default=24)
    ap.add_argument("--lo", type=float, default=-math.pi / 2)
    ap.add_argument("--hi", type=float, default=math.pi / 2)
    args = ap.parse_args()

    thetas = np.linspace(args.lo, args.hi, args.npts)
    out = []
    log = RES / f"spinor_scan_L{args.L}.log"
    for th in thetas:
        t0 = time.time()
        res, psi, model = run(args.L, float(th), args.chi, max_sweeps=args.max_sweeps,
                              chi_ramp=True, measure_bonds=False)
        rec = {"theta": float(th), "theta_deg": float(math.degrees(th)),
               "S_Lhalf": res["S_Lhalf"], "S_max": res["S_max"],
               "c_fit_smoothed": res["c_fit_smoothed"], "R2_smoothed": res["R2_smoothed"],
               "E0_per_bond": res["E0_per_bond"], "max_chi_used": res["max_chi_used"]}
        out.append(rec)
        line = (f"[{time.strftime('%H:%M:%S')}] th={math.degrees(th):+7.2f}deg "
                f"S(L/2)={res['S_Lhalf']:.4f} S_max={res['S_max']:.4f} "
                f"c_sm={res['c_fit_smoothed']} ({time.time()-t0:.0f}s)")
        print(line, flush=True)
        with open(log, "a", buffering=1) as fh:
            fh.write(line + "\n")
        json.dump(out, open(RES / f"spinor_scan_L{args.L}.json", "w"), indent=2)
    # report the peak
    peak = max(out, key=lambda r: r["S_Lhalf"])
    print(f"\nPEAK S(L/2): theta={peak['theta_deg']:.2f}deg S={peak['S_Lhalf']:.4f} "
          f"c_sm={peak['c_fit_smoothed']}", flush=True)


if __name__ == "__main__":
    main()
