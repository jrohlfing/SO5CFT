"""Converge the spinor-rep chain at its critical point theta = arctan(1/3) and
determine the universality class: c = 5/2 (SO(5)_1, spinor at Delta=5/8) vs c = 3
(SU(4)_1, fundamental at Delta=3/4).

Why theta = arctan(1/3): the BBQ couplings on the three fusion channels are
  h(singlet) = -5/2 cos t + 25/4 sin t,  h(vector_5) = -1/2 cos t + 1/4 sin t,
  h(adjoint_10) = +1/2 cos t + 1/4 sin t.
At tan t = 1/3 we get h(singlet) = h(vector) (both channels of the SU(4)
6 = 1+5 antisymmetric degenerate), so the model acquires full SU(4) symmetry --
the Uimin-Lai-Sutherland SU(4) point, expected SU(4)_1 (c=3). This is the natural
critical point reached when tuning the biquadratic out of the dimerized phase.

Reads c from consecutive-L slopes c = 6 dS/d ln(2L/pi) and the leading ES
multiplet (the 4 spinor weights (+-1,+-1) [= (+-1/2,+-1/2)]).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from so5_spinor_dmrg import run  # noqa: E402

RES = THIS_DIR / "results"
THETA = math.atan(1.0 / 3.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--Ls", default="32,48,64,96")
    ap.add_argument("--chi", type=int, default=800)
    args = ap.parse_args()
    Ls = [int(x) for x in args.Ls.split(",")]
    th = args.theta
    log = RES / "spinor_critical.log"

    def lg(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(log, "a", buffering=1) as fh:
            fh.write(line + "\n")

    lg(f"critical convergence at theta={math.degrees(th):.4f}deg (tan={math.tan(th):.5f}) chi={args.chi}")
    best = {}
    for L in Ls:
        t0 = time.time()
        res, psi, model = run(L, th, args.chi, max_sweeps=40, chi_ramp=True,
                              measure_bonds=False, want_es=True)
        best[L] = res
        lg(f"L={L:3d} S(L/2)={res['S_Lhalf']:.5f} c_sm={res['c_fit_smoothed']:.4f} "
           f"chi_used={res['max_chi_used']} ({time.time()-t0:.0f}s)")
        json.dump(res, open(RES / f"spinor_crit_L{L}.json", "w"), indent=2)

    # consecutive-L slopes
    def x(L):
        return math.log(2 * L / math.pi)
    lg("--- central charge from consecutive-L slopes c = 6 dS/d ln(2L/pi) ---")
    slopes = []
    for i in range(len(Ls) - 1):
        a, b = Ls[i], Ls[i + 1]
        c = 6 * (best[b]["S_Lhalf"] - best[a]["S_Lhalf"]) / (x(b) - x(a))
        slopes.append((a, b, c))
        lg(f"  L {a:3d}->{b:3d}: c = {c:.3f}")
    # leading ES multiplet at the largest L
    Lmax = Ls[-1]
    es = best[Lmax]["es_mid_by_charge"]
    gmin = min(min(s["eps"]) for s in es if s["eps"])
    lead = [(s["q1"], s["q2"]) for s in es if s["eps"] and min(s["eps"]) - gmin < 0.25]
    lg(f"--- leading ES shell at L={Lmax} (charges x2): {sorted(set(lead))} "
       f"(deg {len(lead)}) ---")
    interp = ("SU(4)_1 c=3 (fundamental Delta=3/4)" if slopes and abs(slopes[-1][2]-3) < abs(slopes[-1][2]-2.5)
              else "SO(5)_1 c=5/2 (spinor Delta=5/8)")
    lg(f"  nearest universality by last slope: {interp}")
    out = {"theta": th, "theta_deg": math.degrees(th), "Ls": Ls,
           "S_Lhalf": {str(L): best[L]["S_Lhalf"] for L in Ls},
           "c_fit_smoothed": {str(L): best[L]["c_fit_smoothed"] for L in Ls},
           "slopes": [{"a": a, "b": b, "c": c} for a, b, c in slopes],
           "leading_shell_charges_x2": sorted(set(lead)), "leading_deg": len(lead),
           "interp": interp}
    json.dump(out, open(RES / "spinor_critical_summary.json", "w"), indent=2)
    lg(f"wrote {RES/'spinor_critical_summary.json'}")


if __name__ == "__main__":
    main()
