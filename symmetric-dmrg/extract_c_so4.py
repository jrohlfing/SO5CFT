"""Central-charge sanity for the n=4 control: confirm the Reshetikhin theta=0
SO(4) chain sits at SO(4)_1 = SU(2)_1 x SU(2)_1, c = 2. Uses the best (largest)
chi per L; reports per-L smoothed c_fit, consecutive-L slopes c = 6 dS/d ln(2L/pi),
and a global log-corrected fit. Target band 2.0 +/- 0.2 (Heisenberg log
corrections push finite-L estimates below 2)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"


def x_of(L):
    return math.log(2 * L / math.pi)


def main():
    by_L = defaultdict(dict)
    for f in RES.glob("so4_L*_chi*.json"):
        d = json.load(open(f))
        by_L[d["L"]][d["chi_max"]] = d
    Ls = sorted(by_L)
    print("=== n=4 SO(4) Reshetikhin (theta=0) -- central charge check ===")
    print("  best (largest-chi) point per L:")
    best = {}
    for L in Ls:
        chi = max(by_L[L])
        d = by_L[L][chi]
        best[L] = d
        conv = ""
        if len(by_L[L]) >= 2:
            chis = sorted(by_L[L])
            dS = abs(by_L[L][chis[-1]]["S_Lhalf"] - by_L[L][chis[-2]]["S_Lhalf"])
            conv = f"  top2 dS={dS:.5f} {'CONVERGED' if dS < 0.005 else 'moving'}"
        print(f"   L={L:4d} chi={chi:5d}  S(L/2)={d['S_Lhalf']:.5f}  "
              f"c_sm={d.get('c_fit_smoothed'):.4f}  R2_sm={d.get('R2_smoothed'):.5f}{conv}")

    print("\n  consecutive-L slopes c = 6 dS/d ln(2L/pi):")
    for i in range(len(Ls) - 1):
        a, b = Ls[i], Ls[i + 1]
        c = 6 * (best[b]["S_Lhalf"] - best[a]["S_Lhalf"]) / (x_of(b) - x_of(a))
        flag = "  <-- ~2 (SO(4)_1)" if 1.8 <= c <= 2.2 else ""
        print(f"   L {a:3d}->{b:3d}: c = {c:.3f}{flag}")

    if len(Ls) >= 3:
        S = np.array([best[L]["S_Lhalf"] for L in Ls])
        x = np.array([x_of(L) for L in Ls])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, S, rcond=None)
        print(f"\n  naive global CC fit: c = {6*coef[0]:.3f}")
        if len(Ls) >= 4:
            A3 = np.vstack([x, np.ones_like(x), 1.0 / x]).T
            c3, *_ = np.linalg.lstsq(A3, S, rcond=None)
            print(f"  log-corrected global fit: c = {6*c3[0]:.3f}")

    out = {str(L): {"chi": max(by_L[L]), "S_Lhalf": best[L]["S_Lhalf"],
                    "c_fit_smoothed": best[L].get("c_fit_smoothed")} for L in Ls}
    with open(RES / "so4_c_check.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RES/'so4_c_check.json'}")


if __name__ == "__main__":
    main()
