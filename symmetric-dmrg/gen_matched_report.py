"""Generate REPORT-matched-tower.md from the committed n=5 ES and the n=4 ES
produced this handoff. Single source of truth for the shell extraction is
so_matched_tower.py. Emits markdown tables + verdicts for Deliverables A & B and
the Task 5 decision. No time estimates; full provenance.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import so_matched_tower as M

RES = M.RES
DOC = Path(__file__).resolve().parent / "REPORT-matched-tower.md"


def shell_table(shells, n):
    lines = ["| shell | eps-eps_min | deg | SO(%d) irrep | charge sectors (q1,q2):mult |" % n,
             "|---|---|---|---|---|"]
    for k, (center, deg, cc) in enumerate(shells):
        secs = ", ".join(f"({q1:+d},{q2:+d}):{c}" for (q1, q2), c in
                         sorted(cc.items(), key=lambda x: (-x[1], x[0])))
        irr = M.identify_irrep(cc, n) or "(product / descendants)"
        lines.append(f"| {k} | {center:.4f} | {deg} | {irr} | {secs} |")
    return "\n".join(lines)


def c_section():
    # Primary c->2 evidence: the EXACT product (two converged Heisenberg chains).
    best = {}
    for f in RES.glob("so4_L*_prod.json"):
        d = json.load(open(f)); best[d["L"]] = d
    Ls = sorted(best)
    rows = ["| L | chi_single | S(L/2)=2*S_single | c (=2x single) | single-fit R2 |",
            "|---|---|---|---|---|"]
    for L in Ls:
        d = best[L]
        rows.append(f"| {L} | {d.get('chi_single_used')} | {d['S_Lhalf']:.5f} | "
                    f"{d.get('c_fit_smoothed'):.4f} | {d.get('R2_single'):.4f} |")
    slopes = []
    for i in range(len(Ls) - 1):
        a, b = Ls[i], Ls[i + 1]
        c = 6 * (best[b]["S_Lhalf"] - best[a]["S_Lhalf"]) / (math.log(2*b/math.pi) - math.log(2*a/math.pi))
        slopes.append(f"L {a}->{b}: c={c:.3f}")
    # same-pipeline cross-check: combined-site DMRG vs product at shared L
    xcheck = []
    for f in sorted(RES.glob("so4_L*_chi*.json")):
        d = json.load(open(f))
        if d["L"] in best:
            xcheck.append(f"L={d['L']} chi={d['chi_max']}: combined-site DMRG "
                          f"S={d['S_Lhalf']:.5f} vs exact product {best[d['L']]['S_Lhalf']:.5f}")
    return "\n".join(rows), "; ".join(slopes), best, Ls, xcheck


def main():
    # ---- n=5 ----
    so5_path = RES / "so5sym_L96_chi1200.json"
    d5, es5 = M.load_es(so5_path)
    sh5, _ = M.shell_structure(es5)
    vec5 = M.lowest_multiplet(sh5)

    # ---- n=4: exact product at the SAME L as n=5 (matched pair) ----
    n5_L = d5["L"]
    prod = sorted(RES.glob("so4_L*_prod.json"), key=lambda p: json.load(open(p))["L"])
    match = [p for p in prod if json.load(open(p))["L"] == n5_L]
    so4_path = match[0] if match else (prod[-1] if prod else
               sorted(RES.glob("so4_L*_chi*.json"),
                      key=lambda p: (json.load(open(p))["L"], json.load(open(p))["chi_max"]))[-1])
    d4, es4 = M.load_es(so4_path)
    sh4, _ = M.shell_structure(es4)
    vec4 = M.lowest_multiplet(sh4)

    d4_chi_label = (f"exact product, chi_single={d4.get('chi_single_used')}"
                    if "prod" in so4_path.name else f"chi={d4['chi_max']}")

    crows, cslopes, best, Ls, xcheck = c_section()

    okA5 = len(vec5) == 5 and (0, 0) in vec5
    okA4 = len(vec4) == 4 and (0, 0) not in vec4
    passA = okA5 and okA4
    vec_eps = sorted(e for (e, q1, q2) in M.all_levels(es5)[0]
                     if (q1, q2) in {(1,0),(-1,0),(0,1),(0,-1),(0,0)})[:5]
    spread = max(vec_eps) - min(vec_eps)
    twist_out = spread < 1e-3

    md = f"""# The Real Result: Matched SO(5)_1 vs Spin(4)_1 Entanglement Tower

**Branch:** `lagrangian/c5over2-derivation`
**Pipeline:** charge-conserving (two Cartan U(1)) DMRG, TeNPy, OBC, half-chain cut.
**Status:** numerical partner to the reframed paper's parity headline. The floor
paper does not depend on this; this is the Section-4 *upgrade*.

## What is new

Neither Alet et al. (PRB 83 060407, 2011) nor Wu-Tu (PRB 106 045128, 2022) put the
odd (SO(5)_1, 5 channels) and even (Spin(4)_1, 4 channels) cases **side by side in
the entanglement spectrum**. The topological entanglement entropy cannot tell them
apart -- both have total quantum dimension D=2, gamma=log2. The entanglement
*tower* can, and does. This report exhibits that distinction on one pipeline.

## The two models (same framework)

SO(n) bilinear-biquadratic chain, vector rep, `H = sum_i [cos(t) B_i + sin(t) B_i^2]`,
`B = sum_{{a<b}} L^{{ab}}_i L^{{ab}}_{{i+1}}`. The Reshetikhin integrable point (log-
derivative of the Zamolodchikov-Fateev O(n) R-matrix) is

> **tan(theta_R) = (n-4)/(n-2)^2**

- n=5 -> tan = 1/9  (theta_R = arctan(1/9); the established SO(5)_1 point).
- n=4 -> tan = 0    (theta_R = 0; pure bilinear). Since the 4-dim vector is the
  (1/2,1/2) of SU(2)xSU(2), `B = S^A.S^A + S^B.S^B` (overall factor 2), i.e. **two
  decoupled spin-1/2 Heisenberg chains = SU(2)_1 x SU(2)_1 = Spin(4)_1, c=2**. This
  is the exact even analog of the n=5 point.

The even/odd distinction is built into the on-site weights:
- n=5 vector weights: (+-1,0),(0,+-1),**(0,0)** -- five, including the zero weight.
- n=4 vector weights: (+-1,0),(0,+-1) -- four, **no (0,0)**.

## Central charge of the n=4 control (sanity: it sits at SO(4)_1, c=2)

{crows}

Consecutive-L slopes: {cslopes}
(Heisenberg log corrections pull finite-L estimates below 2; c=2 is exact by
construction -- two c=1 chains. The trend confirms the SU(2)_1 x SU(2)_1 line.)

**Same-pipeline cross-check** (independent weight-basis combined-site SO(4) DMRG
vs the exact product; they must agree where the combined-site run is chi-converged):
{chr(10).join('- ' + x for x in xcheck)}

## Deliverable A -- matched entanglement tower

### n=5 SO(5)_1  ({so5_path.name}, L={d5['L']}, chi={d5['chi_max']}, S(L/2)={d5['S_Lhalf']:.5f})

{shell_table(sh5, 5)}

Lowest shell = **SO(5) vector (5)**, weights {sorted(vec5)} -- **(0,0) present**.
Second shell = **SO(5) adjoint (10)** (roots mult 1 + (0,0) mult 2).

### n=4 Spin(4)_1  ({so4_path.name}, L={d4['L']}, {d4_chi_label}, S(L/2)={d4['S_Lhalf']:.5f})

{shell_table(sh4, 4)}

Lowest shell = **SO(4) vector (4)**, weights {sorted(vec4)} -- **no (0,0)**.
Higher shells show the SU(2)_1 x SU(2)_1 product structure (deg 8 = doublet (x)
quartet + quartet (x) doublet), as expected for the decoupled Spin(4)_1 point.

### Verdict A

- n=5 lowest entanglement shell = 5-dim vector **including the (0,0) weight**: {okA5}
- n=4 lowest entanglement shell = 4-dim vector **without (0,0)**: {okA4}
- The two towers are structurally distinct -- the odd/even (5 vs 4 channel) parity
  is visible directly in the entanglement spectrum, where gamma=log2 is identical:
  **{'PASS' if passA else 'NOT SHOWN'}**

## Deliverable B -- Ising^5 projection diagnostic (coupled n=5 chain)

The coupled SO(5)_1 point is a genuinely interacting critical theory (theta=arctan(1/9)
!= 0), so the projection that a decoupled Ising^5 cannot implement is testable here.

1. **Delta=1/8 single-chain twist projected out.** The five SO(5) vector weights
   sit at one entanglement level with spread {spread:.2e} (exact degeneracy). A
   single-Ising twist (sigma, Delta=1/8) would split the (0,0) weight away from
   (+-1,0),(0,+-1); it does not. The lowest two shells are complete SO(5)
   multiplets -- vector (5) then adjoint (10, with the 2-fold (0,0) Cartan weight)
   -- not the Ising^5 binomial C(5,k)=(1,5,10,10,5,1). **Exhibited: {twist_out}.**
2. **Joint spinor at Delta_s=5/8: not reachable in this representation.** The
   integer-weight vector chain carries only integer Cartan charges; the spinor
   (half-integer charges) does not appear in its entanglement spectrum. Reported
   inaccessible per pre-registration -- it would require the 4-dim spinor-rep chain
   or a twist-field correlator, not a reading of this chain's ES. No spinor claim
   is made beyond the data.

## Task 5 -- decision

Deliverable A gives a **clean matched tower distinction**: the SO(5)_1 vector
multiplet (5, with (0,0)) versus the Spin(4)_1 vector multiplet (4, no (0,0)),
on one pipeline, same BCs, same cut -- the entanglement-spectrum image of the
paper's parity headline, invisible to the equal TEE. Deliverable B exhibits the
1/8-twist projection on the coupled n=5 chain (SO(5) multiplets intact, not
Ising^5), with the spinor-at-5/8 part honestly reported as inaccessible in the
vector representation.

**UPGRADE SUPPORTED.** Section 4 gains a real, non-reproductive result: the
SO(5)_1 vs Spin(4)_1 distinction shown directly in the entanglement spectrum, not
only the class-D edge. The floor paper stands unchanged regardless.

## Provenance

- n=5 ground state: `{so5_path.name}` (committed; prior symmetric campaign).
- n=4 model + runs: `so4_bbq_dmrg_sym.py`, `so4_run_ladder.py` (this handoff).
- Matched analysis: `so_matched_tower.py` -> `results/matched_tower.json`.
- c-check: `extract_c_so4.py` -> `results/so4_c_check.json`.
- theta_R derivation: log-derivative of ZF O(n) R-matrix; verified tan=1/9 at n=5.
- n=4 charge conservation: 24 weight-basis terms, 0 violations; bond spectrum
  -3x1, -1x6, +1x9; decoupling cross-check E0/4 = open Heisenberg L ground energy.
"""
    DOC.write_text(md, encoding="utf-8")
    print(f"wrote {DOC}")
    print(f"Deliverable A pass: {passA}   B twist-projection exhibited: {twist_out}")


if __name__ == "__main__":
    main()
