"""Run the n=4 (Spin(4)_1) matched ladder: L in {32,48,64,96} at the Reshetikhin
point theta=0, chi-ramped, gating S(L/2) convergence (|dS| < 0.005 between top two
chi). Serial, incremental JSON per (L,chi). Light job (two decoupled Heisenberg
chains) so modest chi converges; we still verify the gate honestly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SCRIPT = THIS_DIR / "so4_bbq_dmrg_sym.py"
RES = THIS_DIR / "results"
LOG = RES / "so4_ladder.log"
GATE = 0.005
THREADS = int(os.environ.get("STAEDERT_THREADS_PER_WORKER", "16"))

# L -> list of chi to try (ascending); stop when S(L/2) gate met vs previous chi.
LADDER = {
    32: [400, 800],
    48: [400, 800],
    64: [600, 1000],
    96: [800, 1200, 1600],
}


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", buffering=1) as fh:
        fh.write(line + "\n")


def s_half(tag):
    f = RES / f"{tag}.json"
    return json.load(open(f))["S_Lhalf"] if f.exists() else None


def run_point(L, chi):
    tag = f"so4_L{L}_chi{chi}"
    if (RES / f"{tag}.json").exists():
        log(f"SKIP {tag} (exists S={s_half(tag):.5f})")
        return s_half(tag)
    env = dict(os.environ, STAEDERT_THREADS_PER_WORKER=str(THREADS))
    t0 = time.time()
    log(f"START {tag} ({THREADS} threads, chi-ramp)")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "production", "--L", str(L),
         "--chi", str(chi), "--max-sweeps", "40", "--chi-ramp", "--tag", tag],
        cwd=str(THIS_DIR), env=env, capture_output=True, text=True,
    )
    dt = time.time() - t0
    if proc.returncode != 0:
        log(f"FAIL {tag} rc={proc.returncode} ({dt:.0f}s) :: {proc.stderr[-800:]}")
        return None
    S = s_half(tag)
    log(f"DONE {tag} S(L/2)={S:.5f} ({dt:.0f}s)")
    return S


def main():
    RES.mkdir(parents=True, exist_ok=True)
    converged = {}
    for L in sorted(LADDER):
        prev_chi, prev_S = None, None
        for chi in LADDER[L]:
            S = run_point(L, chi)
            if S is None:
                log(f"ABORT L={L} (run failed)")
                break
            if prev_S is not None:
                d = abs(S - prev_S)
                status = "CONVERGED" if d < GATE else "STILL MOVING"
                log(f"L={L}: dS({prev_chi}->{chi})={d:.5f} (gate {GATE}) -> {status}")
                if d < GATE:
                    converged[L] = {"chi": chi, "S_Lhalf": S, "converged": True}
                    break
            prev_chi, prev_S = chi, S
        else:
            if L not in converged and prev_S is not None:
                converged[L] = {"chi": prev_chi, "S_Lhalf": prev_S, "converged": False}
        with open(RES / "so4_converged.json", "w") as fh:
            json.dump(converged, fh, indent=2)
    log(f"LADDER DONE: {converged}")


if __name__ == "__main__":
    main()
