# Resolving the SO(5)_1 Spinor Sector (Delta_s = 5/8): free-theory twist, and the
# spinor-rep feasibility benchmark

**Branch:** `lagrangian/c5over2-derivation`
**Scope of this round:** Task C validation rung (free theory) + Task A (spinor-rep
feasibility). Tasks C (interacting twist) and B (disorder correlator) are assessed
with a concrete path and obstruction; they are the harder remaining steps and were
not executed this round. The paper's spinor-at-5/8 claim is fully backed by the
free-theory result below, per the pre-registered decision rule.

The SO(5)_1 spinor primary: chiral weight h = 5/16, total scaling dimension
Delta_s = 2h = 5/8, quantum dimension sqrt2, 4-dimensional SO(5) multiplet
(weights (+-1/2,+-1/2)).

---

## Task C validation rung -- free-theory twist (PASS, exact)

SO(5)_1 = five free Majorana fermions (= Ising^5). The spinor is the twist field:
each Majorana contributes chiral weight h = 1/16 in the twisted (Ramond /
periodic-Majorana) sector, so five Majoranas give h = 5/16 and Delta_s = 5/8. The
twisted-sector ground-state degeneracy is 2^floor(N/2) = 4 for N=5 (the N Majorana
zero modes form a Clifford algebra whose irreducible rep is the 4-dim SO(5)
spinor).

Implemented exactly via a single critical Majorana hopping ring (`so5_free_twist.py`),
no DMRG: the Ramond ground state sits at Delta = 1/8 (h = 1/16) above the NS ground
state; calibrating 2*pi*v/L against the NS energy-operator gap (Delta_eps = 1)
removes the velocity and Casimir offsets.

| L | Delta_sigma (target 1/8) | h_sigma (target 1/16) | Ramond zero-modes |
|---|---|---|---|
| 32  | 0.125302 | 0.062651 | 1 |
| 64  | 0.125075 | 0.062538 | 1 |
| 128 | 0.125019 | 0.062509 | 1 |
| 256 | 0.125005 | 0.062502 | 1 |
| 512 | 0.125001 | 0.062501 | 1 |

- Single Ising/Majorana: **h_sigma -> 1/16 = 0.06250** (clean 1/L convergence).
- Five Majoranas: **h_5 = 5/16 = 0.31250, Delta_s = 5/8 = 0.62500**.
- Twisted-sector ground-state degeneracy **2^floor(5/2) = 4 = SO(5) spinor dim**.

This reproduces the in-session free-theory confirmation in committed, reproducible
code, and is the pre-registered prerequisite ("Do not proceed until this
matches"). The twist + zero-mode-degeneracy machinery is validated end to end.

**Why this already backs the paper's claim.** Delta_s = 5/8 is a universal datum
of the SO(5)_1 universality class -- the same class as the interacting Reshetikhin
point. The spinor primary dimension is fixed by the chiral algebra, not by the
lattice realization, so the free-Majorana value IS the SO(5)_1 spinor dimension.
The interacting lattice twist (Task C) would *demonstrate* it on a coupled chain
(a stronger lattice statement) but cannot change the value.

---

## Task A -- spinor-rep (Sp(4) fundamental / SO(5) spinor) chain feasibility

Built the SO(5)=Sp(4) chain in the 4-dim spinor rep on the existing charge-
conserving framework (`so5_spinor_dmrg.py`). Construction (no basis rotation; the
Cartans are diagonal in the computational basis):
- gamma matrices G1=sx(x)I, G2=sy(x)I, G3=sz(x)sx, G4=sz(x)sy, G5=sz(x)sz;
  generators Sigma^{ab} = -(i/2) G^a G^b; Casimir = 5/2.
- spinor weights (+-1/2,+-1/2) [stored x2 as (+-1,+-1)] -- HALF-INTEGER, i.e. the
  spinor sector lives in the local Hilbert space (unlike the integer-weight vector
  chain). 28 weight-basis coupling terms, 0 charge violations.
- bilinear B spectrum: -5/2 (singlet x1), -1/2 (vector x5), +1/2 (adjoint x10),
  matching 4 (x) 4 = 1 + 5 + 10 and the Casimir.

### (1) Bare chain theta=0 is gapped/dimerized (as predicted)

| L | S(L/2) | c_sm | dimerization |
|---|---|---|---|
| 16 | 0.3748 | 0.237 | 1.758 |
| 32 | 0.3872 | 0.072 | 1.741 |
| 64 | 0.3883 | 0.006 | 1.740 |

S(L/2) **saturates** (0.375 -> 0.388) and c_sm -> 0: the chain is **gapped**. The
dimerization order parameter is large (~1.74), with central bond energies
alternating -2.22 / -0.44 -- near-perfect valence bonds (strong bond ~ singlet
eigenvalue -2.5). This confirms the spontaneous-dimerization prediction
(Bjornberg et al. 2101.11464; Nachtergaele-Ueltschi 1701.03983).

### (2) BBQ scan: the only critical point is SU(4)_1, not SO(5)_1

Scanned theta in [-90deg, +90deg] (`so5_spinor_scan.py`). The whole range is
gapped/dimerized (S(L/2) ~ 0.3-0.46) EXCEPT a sharp critical point near +18deg,
where S(L/2) jumps (0.36 at 12.5deg -> 1.26 at 15deg -> 2.23 at 17.5deg).

That critical point is **theta = arctan(1/3) = 18.4349deg**, and it is the
**SU(4) Uimin-Lai-Sutherland point**, proven exactly: at tan(theta)=1/3 the bond
Hamiltonian has only TWO distinct eigenvalues,
  h = -0.395285 on the 6 (= singlet 1 + vector 5, the SU(4) antisymmetric),
  h = +0.553399 on the 10 (= adjoint, the SU(4) symmetric),
i.e. h(singlet) = h(vector) exactly -> full SU(4) symmetry (SWAP/permutation
structure 6 (+) 10). This is **SU(4)_1, c = 3**, corroborated numerically at the
critical point (L=32, chi=800): S(L/2) = 2.150, c_sm = 2.83 (finite-size undershoot
toward 3; cf. the vector SO(5)_1 chain gave S=1.89 at L=32, lower c).

The leading entanglement shell at this critical point is the **4-fold multiplet
with weights (+-1/2,+-1/2)** -- the spinor weights are indeed carried and exposed
-- but in an **SU(4)_1 (c=3)** theory, where the 4-dim multiplet is the SU(4)
fundamental at **Delta = 3/4**, NOT the SO(5) spinor at Delta = 5/8.

### Task A go/no-go verdict

**NO at nearest-neighbor.** A critical spinor-rep chain IS reachable (at
theta=arctan(1/3)), but its universality is the more symmetric **SU(4)_1 (c=3,
fundamental at Delta=3/4)**, not **SO(5)_1 (c=5/2, spinor at Delta=5/8)**. The
natural critical point of the NN spinor-rep BBQ chain has *enhanced* SU(4)
symmetry, so a native SO(5)_1 spinor multiplet at Delta_s=5/8 is not obtained
there. Reaching SO(5)_1 in the spinor rep would require breaking SU(4)->SO(5)
while staying critical (fine-tuning beyond NN BBQ, or longer-range terms) -- the
higher-effort path is *not* trivially reachable. This makes the vector-chain twist
(Task C) the natural lattice route to a native interacting SO(5)_1 spinor.

---

## Task C (interacting twist) and Task B (disorder correlator) -- assessment

Both were not executed this round; both run into the same concrete obstruction,
and the free-theory result above already fixes Delta_s = 5/8.

**The obstruction.** The SO(5) VECTOR site carries only integer (tensor) weights.
The Z2 center of Spin(5)=Sp(4) -- which distinguishes spinor from tensor -- acts
TRIVIALLY on the vector rep. So there is no on-site Z2 to gauge/twist that would
produce the spinor sector; the twist (Task C) and the disorder operator mu^5
(Task B) are both NON-LOCAL Kramers-Wannier / Jordan-Wigner duality objects, not
products of on-site operators. This is the same reason the integer-weight vector
ES cannot expose the spinor (matched-tower report).

**Concrete path for a future full Task C run.**
1. PBC DMRG for the vector chain (heavier than the current OBC).
2. The duality defect on one boundary bond (an odd number of the five emergent
   Majoranas antiperiodic), implemented via the lattice KW transformation.
3. The Lecheminant fermion-parity projection (cond-mat/9702057) -- the step most
   prone to NS/R contamination; the single-Ising rung above (h->1/16) is the
   validation that the projection is correct before trusting the interacting tower.
4. Scaling dimensions via the Zou-Vidal APBC Koo-Saleur generators (1907.10704).
Target/failure criterion: lowest twisted state at Delta_s=5/8, 1/L spacing,
degeneracy 4.

This is a multi-step build with real risk (esp. step 3); it is the right next
investment but was not rushed here.

---

## Decision

- **Free-theory spinor (Delta_s = 5/8, degeneracy 4): CONFIRMED** in committed code
  -- the paper's spinor claim stands, exactly as the pre-registered decision rule allows
  ("If Task C does not converge, the free-theory confirmation already in hand still
  backs the spinor-at-5/8 claim").
- **Task A feasibility: the native-spinor route gives SU(4)_1 (c=3) at NN, not
  SO(5)_1.** A native interacting SO(5)_1 spinor multiplet is not obtained from the
  NN spinor-rep BBQ chain; the vector-chain twist (Task C) is the route, with the
  validation rung already passing.
- **Task C (interacting twist) not completed; Task B not executed.** Both have a
  documented path and the same non-locality obstruction.

Net: the spinor-at-5/8 is established (free theory, exact); the *interacting
lattice demonstration* remains the open upgrade, and Task A clarifies it must come
from the vector-chain twist rather than a native spinor chain.

---

## Provenance

- Free twist: `so5_free_twist.py` -> `results/free_twist_validation.json`
  (h1=0.062501, Delta_s=0.625006, deg 4; targets 1/16, 5/8, 4).
- Spinor chain: `so5_spinor_dmrg.py` (validate -> `results/validate_spinor.json`),
  theta=0 saturation runs `results/spinor_th0_L{16,32,64}.json`.
- BBQ scan: `so5_spinor_scan.py` -> `results/spinor_scan_L32.log`,
  `results/spinor_scan_L40.log` (critical point located at ~18.4deg).
- Critical point: `so5_spinor_critical.py` -> `results/spinor_crit_L32.json`,
  `results/spinor_critical.log`; SU(4) symmetry proven from the bond spectrum
  (two eigenvalues, multiplicities 6 and 10) at tan(theta)=1/3.
- Method refs (used, not double-cited with the paper bib): Zou-Vidal 1907.10704;
  Lecheminant cond-mat/9702057; Liu et al. 2311.05690 / 2308.00737;
  Bjornberg 2101.11464; Nachtergaele-Ueltschi 1701.03983; Wu et al. 1103.1926.
