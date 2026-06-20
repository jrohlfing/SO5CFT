# The Real Result: Matched SO(5)_1 vs Spin(4)_1 Entanglement Tower

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
`B = sum_{a<b} L^{ab}_i L^{ab}_{i+1}`. The Reshetikhin integrable point (log-
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

| L | chi_single | S(L/2)=2*S_single | c (=2x single) | single-fit R2 |
|---|---|---|---|---|
| 32 | 432 | 1.44299 | 1.6658 | 0.9480 |
| 48 | 683 | 1.58891 | 1.7267 | 0.9748 |
| 64 | 932 | 1.69055 | 1.7615 | 0.9853 |
| 96 | 1406 | 1.83164 | 1.8016 | 0.9932 |
| 128 | 1600 | 1.93047 | 1.8248 | 0.9961 |

Consecutive-L slopes: L 32->48: c=2.159; L 48->64: c=2.120; L 64->96: c=2.088; L 96->128: c=2.061
(Heisenberg log corrections pull finite-L estimates below 2; c=2 is exact by
construction -- two c=1 chains. The trend confirms the SU(2)_1 x SU(2)_1 line.)

**Same-pipeline cross-check** (independent weight-basis combined-site SO(4) DMRG
vs the exact product; they must agree where the combined-site run is chi-converged):
- L=32 chi=400: combined-site DMRG S=1.44297 vs exact product 1.44299
- L=32 chi=800: combined-site DMRG S=1.44299 vs exact product 1.44299
- L=48 chi=400: combined-site DMRG S=1.58870 vs exact product 1.58891
- L=48 chi=800: combined-site DMRG S=1.58890 vs exact product 1.58891

## Deliverable A -- matched entanglement tower

### n=5 SO(5)_1  (so5sym_L96_chi1200.json, L=96, chi=1200, S(L/2)=2.34538)

| shell | eps-eps_min | deg | SO(5) irrep | charge sectors (q1,q2):mult |
|---|---|---|---|---|
| 0 | 0.0000 | 5 | vector (5) | (-1,+0):1, (+0,-1):1, (+0,+0):1, (+0,+1):1, (+1,+0):1 |
| 1 | 2.8320 | 10 | adjoint (10) | (+0,+0):2, (-1,-1):1, (-1,+0):1, (-1,+1):1, (+0,-1):1, (+0,+1):1, (+1,-1):1, (+1,+0):1, (+1,+1):1 |
| 2 | 3.1429 | 5 | vector (5) | (-1,+0):1, (+0,-1):1, (+0,+0):1, (+0,+1):1, (+1,+0):1 |
| 3 | 5.1182 | 35 | (product / descendants) | (-1,+0):3, (+0,-1):3, (+0,+0):3, (+0,+1):3, (+1,+0):3, (-1,-1):2, (-1,+1):2, (+1,-1):2, (+1,+1):2, (-2,-1):1, (-2,+0):1, (-2,+1):1, (-1,-2):1, (-1,+2):1, (+0,-2):1, (+0,+2):1, (+1,-2):1, (+1,+2):1, (+2,-1):1, (+2,+0):1, (+2,+1):1 |
| 4 | 6.4158 | 5 | vector (5) | (-1,+0):1, (+0,-1):1, (+0,+0):1, (+0,+1):1, (+1,+0):1 |
| 5 | 6.6898 | 10 | adjoint (10) | (+0,+0):2, (-1,-1):1, (-1,+0):1, (-1,+1):1, (+0,-1):1, (+0,+1):1, (+1,-1):1, (+1,+0):1, (+1,+1):1 |

Lowest shell = **SO(5) vector (5)**, weights [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)] -- **(0,0) present**.
Second shell = **SO(5) adjoint (10)** (roots mult 1 + (0,0) mult 2).

### n=4 Spin(4)_1  (so4_L96_prod.json, L=96, exact product, chi_single=1406, S(L/2)=1.83164)

| shell | eps-eps_min | deg | SO(4) irrep | charge sectors (q1,q2):mult |
|---|---|---|---|---|
| 0 | 0.0000 | 4 | vector (4) | (-1,+0):1, (+0,-1):1, (+0,+1):1, (+1,+0):1 |
| 1 | 3.0053 | 8 | (product / descendants) | (-1,+0):2, (+0,-1):2, (+0,+1):2, (+1,+0):2 |
| 2 | 4.8535 | 16 | (product / descendants) | (-1,+0):2, (+0,-1):2, (+0,+1):2, (+1,+0):2, (-2,-1):1, (-2,+1):1, (-1,-2):1, (-1,+2):1, (+1,-2):1, (+1,+2):1, (+2,-1):1, (+2,+1):1 |
| 3 | 6.0106 | 4 | vector (4) | (-1,+0):1, (+0,-1):1, (+0,+1):1, (+1,+0):1 |
| 4 | 6.8769 | 8 | (product / descendants) | (-1,+0):2, (+0,-1):2, (+0,+1):2, (+1,+0):2 |
| 5 | 7.8588 | 16 | (product / descendants) | (-1,+0):2, (+0,-1):2, (+0,+1):2, (+1,+0):2, (-2,-1):1, (-2,+1):1, (-1,-2):1, (-1,+2):1, (+1,-2):1, (+1,+2):1, (+2,-1):1, (+2,+1):1 |

Lowest shell = **SO(4) vector (4)**, weights [(-1, 0), (0, -1), (0, 1), (1, 0)] -- **no (0,0)**.
Higher shells show the SU(2)_1 x SU(2)_1 product structure (deg 8 = doublet (x)
quartet + quartet (x) doublet), as expected for the decoupled Spin(4)_1 point.

### Verdict A

- n=5 lowest entanglement shell = 5-dim vector **including the (0,0) weight**: True
- n=4 lowest entanglement shell = 4-dim vector **without (0,0)**: True
- The two towers are structurally distinct -- the odd/even (5 vs 4 channel) parity
  is visible directly in the entanglement spectrum, where gamma=log2 is identical:
  **PASS**

## Deliverable B -- Ising^5 projection diagnostic (coupled n=5 chain)

The coupled SO(5)_1 point is a genuinely interacting critical theory (theta=arctan(1/9)
!= 0), so the projection that a decoupled Ising^5 cannot implement is testable here.

1. **Delta=1/8 single-chain twist projected out.** The five SO(5) vector weights
   sit at one entanglement level with spread 4.74e-05 (exact degeneracy). A
   single-Ising twist (sigma, Delta=1/8) would split the (0,0) weight away from
   (+-1,0),(0,+-1); it does not. The lowest two shells are complete SO(5)
   multiplets -- vector (5) then adjoint (10, with the 2-fold (0,0) Cartan weight)
   -- not the Ising^5 binomial C(5,k)=(1,5,10,10,5,1). **Exhibited: True.**
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

- n=5 ground state: `so5sym_L96_chi1200.json` (committed; prior symmetric campaign).
- n=4 model + runs: `so4_bbq_dmrg_sym.py`, `so4_run_ladder.py` (this handoff).
- Matched analysis: `so_matched_tower.py` -> `results/matched_tower.json`.
- c-check: `extract_c_so4.py` -> `results/so4_c_check.json`.
- theta_R derivation: log-derivative of ZF O(n) R-matrix; verified tan=1/9 at n=5.
- n=4 charge conservation: 24 weight-basis terms, 0 violations; bond spectrum
  -3x1, -1x6, +1x9; decoupling cross-check E0/4 = open Heisenberg L ground energy.
