# SO(5)_1 Lattice Candidate — Model Derivation

This note is committed **before** any numerics so the candidate Hamiltonian is
on record independently of what the instrument returns. It states the lattice
model, its symmetry generators, the free-point central charge, the primary
content, and — critically — an honest account of what an interaction can and
cannot do. It also flags one framing that is, on the
rigorous CFT side, imprecise, and resolves it.

## 0. Target

SO(5)_1 WZW: central charge `c = 5/2`, three primaries

| primary | name | chiral weight h | non-chiral Δ = h + h̄ | quantum dim d |
|---------|------|-----------------|------------------------|---------------|
| 1 | identity | 0 | 0 | 1 |
| v | vector | 1/2 | 1 | 1 |
| s | spinor | 5/16 | 5/8 = 0.625 | √2 |

Fusion: `s × s = 1 + v`, `v × v = 1`, `v × s = s`. Total quantum dimension
`D = √(1 + 1 + 2) = 2`. (Consistent with Calc 20 in SOFT-MODES-WORKING.md.)

## 1. The lattice model: five critical Majorana (Kitaev) chains

The standard field-theory fact we build on:

> **N free massless Majorana fermions = SO(N)_1 WZW**, central charge `c = N/2`.
> The currents `J^{ab}(x) = i ψ_a(x) ψ_b(x)`, `1 ≤ a < b ≤ N`, generate the
> `so(N)_1` Kac–Moody algebra at level 1.

This is textbook (e.g. Di Francesco–Mathieu–Sénéchal §15.5; Witten's
non-abelian bosonization). It is the reason SO(N)_1 is the natural target for a
Majorana lattice regularization: we do not need to invent a new critical point,
we need a lattice model whose low-energy limit is N gapless Majoranas with the
SO(N) rotation acting on the flavor index manifest.

**Lattice Hamiltonian.** Take `n = 5` identical Kitaev (Majorana) chains
indexed by flavor `a = 1..n`, each of `N` sites:

```
H_0 = Σ_a Σ_{i=1}^{N-1} [ -t (c†_{a,i} c_{a,i+1} + h.c.) + Δ (c_{a,i} c_{a,i+1} + h.c.) ]
          - μ Σ_a Σ_{i=1}^{N} (c†_{a,i} c_{a,i} - 1/2)
```

In Majorana variables `c_{a,i} = (γ_{a,i}^A + i γ_{a,i}^B)/2` this is a
quadratic form `H_0 = (i/4) Σ γ_α (A_0)_{αβ} γ_β` with `A_0` real
antisymmetric. Each chain is the transverse-field Ising chain in disguise; at
the self-dual point `μ = 2t` (with `Δ = t`) a single chain is gapless with
`c = 1/2` and velocity `v = 2t` (lattice units). Five such chains give
`c = 5 × 1/2 = 5/2` at the free point. Equivalently, one may read the model as
five decoupled critical transverse-field Ising chains
`H_a = -Σ_i σ^x_{a,i} σ^x_{a,i+1} - Σ_i σ^z_{a,i}`, which is the spin
representation we use for the interacting DMRG (Section 4).

**SO(5) generators.** The 10 = (5·4/2) bilinears

```
J^{ab} = (i/2) Σ_i γ_{a,i} γ_{b,i},    1 ≤ a < b ≤ 5
```

(taking one Majorana species per flavor in the low-energy continuum limit)
rotate the five Majoranas into one another and commute with the kinetic term,
which is `Σ_a (i/2) γ_a ∂_x γ_a` — manifestly invariant under
`γ_a → R_{ab} γ_b`, `R ∈ SO(5)`. These ten generators close into so(5). This
is the symmetry the target rests on.

## 2. Honest statement of the subtlety (this is the crux)

One framing to avoid writes "five free Majoranas have c = 5/2 but are NOT
SO(5)_1 (they are five copies of Ising)." The precise statement is sharper and
worth getting exactly right, because a referee will:

- **At the level of the Majorana FERMIONS, the free point IS SO(5)_1.** The
  chiral algebra of 5 free Majoranas is exactly `so(5)_1`; the energy–momentum
  tensor, the currents `J^{ab}`, and the primaries `{1, v, s}` are all present.
- **At the level of the SPIN (Ising) operators, five decoupled Ising chains
  realize the larger theory `Ising^5`**, whose symmetry is only `(Z_2)^5 ⋊ S_5`
  and which has *more* primaries than SO(5)_1. SO(5)_1 is the SO(5)-invariant
  sub-theory obtained by a GSO-type projection (the diagonal SO(5)_1 modular
  invariant keeps only `1, v, s`).

Concretely the **spinor primary `s` is the JOINT twist field of all five
Majoranas simultaneously**, `s ~ σ_1 σ_2 σ_3 σ_4 σ_5`, with chiral weight
`h_s = 5 × (1/16) = 5/16` and `Δ_s = 5/8`. The single-chain disorder field
`σ_a` (Δ = 1/8) is an operator of `Ising^5` that is **not** an SO(5)_1 primary:
it is removed by the SO(5)-invariance projection because it transforms under a
single-flavor twist that is not in the SO(5)_1 spectrum. The vector primary `v`
is the energy operator `ε_a = i ψ_{L,a} ψ_{R,a}` on one chain (Δ_v = 1), and
the five `ε_a` form the SO(5) vector.

**What this means for the tests:**

- `c = 5/2` is necessary, not sufficient. `Ising^5` and SO(5)_1 share it.
- The decisive SO(5)_1 fingerprints are (T2) the *joint* twist at `Δ = 5/8`
  sitting **below** the vector at `Δ = 1`, and (T3) the single unpaired Majorana
  that `n = 5` (odd) carries and `n = 4` (even, Spin(4)_1) does not.
- We therefore measure the joint-twist gap (all five chains twisted together),
  not a single-chain σ, and we explicitly note in the results that the
  unprojected lattice `Ising^5` *also* contains lower operators (single σ at
  Δ = 1/8) that are not SO(5)_1 primaries. Reporting otherwise would be the
  exact error to avoid.

## 3. The free point and the entanglement instrument

For five critical Majorana chains the half-cut entanglement entropy obeys the
Calabrese–Cardy law `S(L) = (c/6) ln(chord) + const` with `c = 5/2`. We compute
`S(L)` from the **Majorana covariance matrix** (Peschel's method for a BdG
ground state) — a standard construction — and feed the resulting `S(L)` array
into the **unchanged** `analysis/c_eff.py` fitter. The estimator (slope → c,
R² ≥ 0.9 gate) is untouched; only the entropy of a new (Majorana) state is
supplied, exactly as `free_fermion_block_entropy` supplies it for the complex
XX chain.

We self-validate the covariance-entropy helper against already-validated
instrument numbers: `n = 1` critical Majorana chain must return `c_eff ≈ 0.5`
(matches the TFIM-critical control), `n = 2` must return `≈ 1.0` (matches the XX
control). Only after that do we trust `n = 5 → 2.5` and the `n = 4 → 2.0`
control.

## 4. The interaction (Gross–Neveu) — what it can and cannot do

The SO(5)-symmetric quartic interaction is the Gross–Neveu term

```
H_int = g Σ_i ( Σ_a i ψ_{L,a} ψ_{R,a} )^2_i
      = g Σ_i ( Σ_a ε_{a,i} )^2
      = g Σ_i [ Σ_a ε_{a,i}^2 + 2 Σ_{a<b} ε_{a,i} ε_{b,i} ]
      ∝ g Σ_i Σ_{a<b} ε_{a,i} ε_{b,i}  + const
```

i.e. an **energy–energy coupling between all pairs of flavors**. On the spin
lattice the energy density is `ε_{a,i} ~ σ^z_{a,i}` (the transverse-field term),
so the interacting model we hand to DMRG is a five-leg critical Ising ladder
with rung coupling `g Σ_i Σ_{a<b} σ^z_{a,i} σ^z_{b,i}`. This carries the
`(Z_2)^5 ⋊ S_5` lattice symmetry that enhances to SO(5) in the IR.

**Renormalization-group fact (derived on paper, before running):** the
operator `(Σ_a ε_a)^2` is built from the energy operator of dimension Δ_ε = 1,
so the Gross–Neveu coupling is a **current–current / marginal** perturbation
(dimension 2 in 1+1D). The SO(N) Gross–Neveu model is the canonical example:

- one sign of `g` is **marginally irrelevant** — the theory flows back to the
  free SO(5)_1 fixed point (`c` stays 5/2, logarithmic corrections only);
- the other sign is **marginally relevant** — it dynamically generates a mass,
  the system **gaps out**, and `c → 0` in the deep IR.

There is no third option in which `g ≠ 0` produces a *new* `c = 5/2` critical
theory distinct from the free point. **SO(5)_1 lives at the free point.** This
is the honest prior. The Step-4 scan therefore tests a falsifiable prediction:

- small `|g|`, irrelevant sign → `c_eff` stays ≈ 2.5 (free fixed point robust);
- relevant sign / large `|g|` → `c_eff` falls below 2.5 (system gaps).

If instead the interaction were to leave a *robust* `c = 5/2` with the
three-primary tower across a finite window, that would be the strong Option-B
result. The derivation predicts it will not; we report whichever occurs.

## 5. The parity signature (T3) — index argument

A free-Majorana BdG Hamiltonian is `H = (i/4) γ^T A γ` with `A` real
antisymmetric. The single-particle spectrum is `±` pairs (eigenvalues of `iA`).
A **real antisymmetric matrix of odd dimension is singular** — it has a
guaranteed zero eigenvalue. More physically: put the five chains in the
**topological** Kitaev phase (`|μ| < 2t`). Each open chain then carries one
exact Majorana zero mode at each end, so the left edge hosts `n` edge
Majoranas. Any fermion-parity-preserving perturbation coupling them is a real
antisymmetric `n × n` matrix; it pairs them up, and the number of surviving
zero modes is `n mod 2`:

- `n = 5` (odd) → exactly **one** unpaired Majorana zero mode survives at each
  edge; its splitting with the opposite edge is `~ e^{-N/ξ}` (exponentially
  small in chain length). This is the SO(5)_1 spinor edge mode (`d_s = √2`).
- `n = 4` (even) → the four edge Majoranas pair up completely; **no** protected
  zero mode (Spin(4)_1 = SU(2)×SU(2), `c = 2`, no unpaired mode).

We diagonalize the open-chain BdG matrix directly and plot the lowest `|E|`
versus `N` for `n = 5` and `n = 4` on the same axes. The control is mandatory:
the claim is the *difference* between odd and even under an identical pipeline.

## 6. Summary of what each test decides

| test | quantity | SO(5)_1 expectation | what a fail means |
|------|----------|---------------------|-------------------|
| T1 | c_eff (free, n=5) | 2.5 ± 0.15 | entropy helper or model wrong |
| T1' | c_eff (free, n=4) | 2.0 (control) | — |
| T2 | joint-twist Δ_s vs vector Δ_v | 5/8 < 1, ordered | tower not SO(5)_1 |
| T3 | lowest |E| vs N, n=5 vs n=4 | odd→0, even→finite | no unpaired-mode signature |
| T4 | c_eff(g) interacting | 2.5 (irrel.) or gap (rel.) | — (marginal, expected) |

A pass on T1+T2+T3 at the free point is the full Option-B realization:
"this candidate (five critical Majorana chains) flows to SO(5)_1 within
instrument tolerance, carries the spinor tower, and shows the odd-Majorana
parity signature absent in the even control." We will not state it more
strongly than the instrument's ~3–6% tolerance allows.
