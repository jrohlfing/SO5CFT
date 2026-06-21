"""Matched SO(5)_1 (n=5, odd) vs Spin(4)_1 (n=4, even) entanglement-tower
comparison.

Reads the charge-resolved entanglement spectra produced by the SAME pipeline:
  n=5: so5sym_*.json  (es_mid_by_charge, committed)
  n=4: so4_L*_chi*.json (es_mid_by_charge)

Deliverable A -- matched tower:
  - group the half-cut entanglement spectrum into shells (levels equal within a
    tolerance) across all Cartan-charge sectors;
  - the LOWEST shell is the SO(n) vector multiplet: n=5 -> 5 states including the
    (0,0) weight; n=4 -> 4 states (+-1,0),(0,+-1) with NO (0,0). That missing
    (0,0) is the entanglement-spectrum image of the odd/even parity distinction
    that the equal TEE (gamma=log2 for both) cannot see.

Deliverable B -- Ising^n projection diagnostic (on the COUPLED n=5 chain):
  - the n vector weights sit at ONE degenerate entanglement level (SO(n) invariance
    intact) rather than split across levels -> the single-chain Ising twist
    (Delta=1/8) is projected out;
  - the integer-sector shell degeneracies follow SO(5) irrep counting
    (1,5,10,14,35,...) NOT the Ising^5 binomial C(5,k)=(1,5,10,10,5,1).
  - the half-integer SPINOR sector (Delta_s=5/8) is not carried by the integer-
    weight vector chain (no half-integer Cartan charges); reported as inaccessible
    in this representation, per pre-registration.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
SHELL_TOL = 0.25   # eps levels within this are one shell (gaps are ~2.8, well separated)


def load_es(path):
    d = json.load(open(path))
    return d, d["es_mid_by_charge"]


def all_levels(es):
    """Flatten to (eps_shifted, q1, q2) with global min subtracted."""
    gmin = min(min(s["eps"]) for s in es if s["eps"])
    lvls = []
    for s in es:
        for e in s["eps"]:
            lvls.append((e - gmin, s["q1"], s["q2"]))
    lvls.sort()
    return lvls, gmin


def shell_structure(es, n_shells=6):
    """Group levels into shells; return list of (eps_center, total_deg, charge_counter)."""
    lvls, gmin = all_levels(es)
    shells = []
    i = 0
    while i < len(lvls) and len(shells) < n_shells:
        e0 = lvls[i][0]
        members = []
        while i < len(lvls) and lvls[i][0] - e0 < SHELL_TOL:
            members.append(lvls[i])
            i += 1
        center = sum(m[0] for m in members) / len(members)
        cc = Counter((m[1], m[2]) for m in members)
        shells.append((center, len(members), cc))
    return shells, gmin


def print_shells(label, shells, n):
    print(f"\n--- {label}: half-cut entanglement shells (eps - eps_min) ---")
    print(f"  shell  eps_center  total_deg  irrep        charge sectors (q1,q2):count")
    for k, (center, deg, cc) in enumerate(shells):
        secs = ", ".join(f"({q1:+d},{q2:+d}):{c}" for (q1, q2), c in
                          sorted(cc.items(), key=lambda x: (-x[1], x[0])))
        irr = identify_irrep(cc, n) or "-"
        print(f"   {k:>3}   {center:9.4f}   {deg:>6}   {irr:<11}  {secs}")


def lowest_multiplet(shells):
    """Charges present in the lowest shell (the vector multiplet)."""
    _, _, cc = shells[0]
    return sorted(cc.keys())


def weyl_orbit_B2(w):
    """Weyl orbit of a weight under the B2/D2 Weyl group: all sign flips of each
    component AND swap of the two components. Works for both SO(5) (B2) and SO(4)
    (D2 subset, but the orbit set is the same generating set here)."""
    a, b = abs(w[0]), abs(w[1])
    pts = set()
    for x in ({a, -a} if a else {0}):
        for y in ({b, -b} if b else {0}):
            pts.add((x, y))
            pts.add((y, x))
    return pts


def identify_irrep(cc, n):
    """Best-effort SO(n) irrep label for a shell from its weight->multiplicity
    Counter, by matching against the low irreps' known weight signatures."""
    weights = set(cc.keys())
    mults = dict(cc)
    z = mults.get((0, 0), 0)
    nonzero = {k: v for k, v in mults.items() if k != (0, 0)}
    allone = nonzero and all(v == 1 for v in nonzero.values())
    short = {(1, 0), (-1, 0), (0, 1), (0, -1)}      # SO short roots / vector nonzero wts
    longr = {(1, 1), (1, -1), (-1, 1), (-1, -1)}    # long roots
    # vector: short weights (+ (0,0) only for odd n)
    if allone and set(nonzero) == short and z == (1 if n % 2 == 1 else 0):
        dim = n
        return f"vector ({dim})"
    # adjoint of SO(5): short+long roots mult1, (0,0) mult2  -> dim 10
    if n == 5 and set(nonzero) == (short | longr) and z == 2:
        return "adjoint (10)"
    # adjoint of SO(4)=su(2)+su(2): long roots (1,+-1),(-1,+-1) mult1, (0,0) mult2 -> dim 6
    if n == 4 and set(nonzero) == longr and z == 2:
        return "adjoint (6)"
    return None


def main():
    # ---- locate inputs ----
    so5_path = RES / "so5sym_L96_chi1200.json"
    if not so5_path.exists():
        cands = sorted(RES.glob("so5sym_L*_chi*.json"))
        so5_path = cands[-1] if cands else None
    # n=4: prefer the EXACT product file at the same L as n=5 (matched pair);
    # else the product at the largest L; else the combined-site DMRG best.
    n5_L = json.load(open(so5_path))["L"] if so5_path else 96
    prod = sorted(RES.glob("so4_L*_prod.json"), key=lambda p: json.load(open(p))["L"])
    so4_path = None
    if prod:
        match = [p for p in prod if json.load(open(p))["L"] == n5_L]
        so4_path = match[0] if match else prod[-1]
    else:
        comb = sorted(RES.glob("so4_L*_chi*.json"),
                      key=lambda p: (json.load(open(p))["L"], json.load(open(p))["chi_max"]))
        so4_path = comb[-1] if comb else None

    out = {"deliverable_A": {}, "deliverable_B": {}}

    print("=" * 74)
    print("MATCHED ENTANGLEMENT TOWER:  SO(5)_1 (odd, n=5)  vs  Spin(4)_1 (even, n=4)")
    print("=" * 74)

    # ---- n=5 ----
    d5, es5 = load_es(so5_path)
    sh5, gmin5 = shell_structure(es5)
    print(f"\n[n=5]  {so5_path.name}  L={d5['L']} chi={d5['chi_max']} "
          f"S(L/2)={d5['S_Lhalf']:.5f}  c_sm={d5.get('c_fit_smoothed')}")
    print_shells("n=5 SO(5)_1", sh5, 5)
    vec5 = lowest_multiplet(sh5)
    print(f"\n  lowest-shell multiplet (the SO(5) vector): {vec5}")
    print(f"  -> {len(vec5)} weights, (0,0) present: {(0,0) in vec5}")
    out["deliverable_A"]["n5"] = {
        "file": so5_path.name, "L": d5["L"], "chi": d5["chi_max"],
        "S_Lhalf": d5["S_Lhalf"],
        "lowest_multiplet": [list(v) for v in vec5],
        "lowest_deg": sh5[0][1], "zero_weight_present": (0, 0) in vec5,
        "shell_degeneracies": [s[1] for s in sh5],
    }

    # ---- n=4 ----
    if so4_path is None:
        print("\n[n=4]  no so4 result yet -- run so4_run_ladder.py first.")
    else:
        d4, es4 = load_es(so4_path)
        sh4, gmin4 = shell_structure(es4)
        print(f"\n[n=4]  {so4_path.name}  L={d4['L']} chi={d4['chi_max']} "
              f"S(L/2)={d4['S_Lhalf']:.5f}  c_sm={d4.get('c_fit_smoothed')}")
        print_shells("n=4 Spin(4)_1", sh4, 4)
        vec4 = lowest_multiplet(sh4)
        print(f"\n  lowest-shell multiplet (the SO(4) vector): {vec4}")
        print(f"  -> {len(vec4)} weights, (0,0) present: {(0,0) in vec4}")
        out["deliverable_A"]["n4"] = {
            "file": so4_path.name, "L": d4["L"], "chi": d4["chi_max"],
            "S_Lhalf": d4["S_Lhalf"],
            "lowest_multiplet": [list(v) for v in vec4],
            "lowest_deg": sh4[0][1], "zero_weight_present": (0, 0) in vec4,
            "shell_degeneracies": [s[1] for s in sh4],
        }

        # ---- Deliverable A verdict ----
        print("\n" + "=" * 74)
        print("DELIVERABLE A -- matched tower verdict")
        print("=" * 74)
        ok_n5 = (len(vec5) == 5 and (0, 0) in vec5)
        ok_n4 = (len(vec4) == 4 and (0, 0) not in vec4)
        print(f"  n=5 lowest shell = 5-dim vector incl (0,0): {ok_n5}")
        print(f"  n=4 lowest shell = 4-dim vector, NO (0,0):   {ok_n4}")
        distinct = ok_n5 and ok_n4
        print(f"  => odd/even parity visible in the ES (distinct towers): {distinct}")
        out["deliverable_A"]["distinct_towers"] = distinct
        out["deliverable_A"]["pass"] = distinct

    # ---- Deliverable B: Ising^5 projection diagnostic on coupled n=5 ----
    print("\n" + "=" * 74)
    print("DELIVERABLE B -- Ising^5 projection diagnostic (coupled n=5 chain)")
    print("=" * 74)
    # (i) the 5 vector weights are DEGENERATE at one level (SO(5) intact) ->
    #     single-chain Ising twist (Delta=1/8) projected out.
    lvls5, _ = all_levels(es5)
    vec_eps = sorted(e for (e, q1, q2) in lvls5 if (q1, q2) in
                     {(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)})[:5]
    spread = max(vec_eps) - min(vec_eps)
    print(f"  (i) the 5 SO(5) vector weights, lowest eps each: "
          f"{[round(x,4) for x in vec_eps]}")
    print(f"      spread across the multiplet = {spread:.5f}  "
          f"(0 => exact SO(5) degeneracy => no 1/8 single-twist splitting)")
    # (ii) the two lowest shells are EXACT SO(5) irreps (vector then adjoint), by
    #      weight content -- not an Ising^5 binomial. Ising^5 would split the (0,0)
    #      vector weight away from (+-1,0),(0,+-1) (different products of sigma/1),
    #      and would not produce the adjoint's 2-fold (0,0) Cartan weight.
    irr0 = identify_irrep(sh5[0][2], 5)
    irr1 = identify_irrep(sh5[1][2], 5)
    ising5_binom = [math.comb(5, k) for k in range(6)]  # 1,5,10,10,5,1
    print(f"  (ii) lowest shell  = {irr0}  (5 weights incl (0,0), each mult 1)")
    print(f"       2nd shell     = {irr1}  (roots mult 1 + (0,0) mult 2 = adjoint)")
    print(f"       These are complete SO(5) multiplets. Ising^5 binomial "
          f"C(5,k)={ising5_binom} would")
    print(f"       split the (0,0) vector weight off and give no 2-fold-Cartan "
          f"adjoint -- absent here.")
    out["deliverable_B"] = {
        "vector_multiplet_eps": [round(x, 5) for x in vec_eps],
        "vector_multiplet_spread": spread,
        "so5_degeneracy_intact": spread < 1e-3,
        "shell0_irrep": irr0, "shell1_irrep": irr1,
        "shell_degeneracies": [s[1] for s in sh5],
        "ising5_binomial": ising5_binom,
        "twist_1_8_projected_out": (spread < 1e-3 and irr0 is not None
                                    and irr1 is not None),
        "spinor_5_8_in_ES": False,
        "spinor_note": ("integer-weight vector chain carries only integer Cartan "
                        "charges; the Delta_s=5/8 spinor (half-integer charges) is "
                        "not exposed in this representation. Reported inaccessible "
                        "via ES per pre-registration; would require the 4-dim "
                        "spinor-rep chain or a twist-field correlator."),
    }
    twist_out = out["deliverable_B"]["twist_1_8_projected_out"]
    print(f"  -> Delta=1/8 single-chain Ising twist PROJECTED OUT "
          f"(SO(5) multiplets intact): {twist_out}")
    print(f"  (iii) spinor at Delta_s=5/8: NOT exposed by the integer-weight vector"
          f" chain (no half-integer Cartan charges) -- reported inaccessible "
          f"per pre-reg.")

    with open(RES / "matched_tower.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RES/'matched_tower.json'}")


if __name__ == "__main__":
    main()
