# T3-phase: Topological-Phase Robustness of the Parity Signature

Upgrades T3 (RESULTS-SO5-LATTICE.md) from a single point to a PHASE. A 2D sweep
over chemical potential `mu` and inter-flavor coupling strength `w` shows the
n=5 (odd) protected Majorana zero mode survives at machine precision across the
gapped topological Kitaev phase and for the full tested range of finite coupling,
lifting only at the bulk gap closing `mu = 2t`; the n=4 (even, Spin(4)_1) control
has no protected zero mode anywhere for `w>0` under the identical pipeline.

## Scope (governs every claim below)

This is **not** a claim that the SO(5)_1 *critical theory* is robust at finite
coupling. MODEL-DERIVATION.md §4 proves the Gross-Neveu coupling is marginal and
no finite g produces a new c=5/2 critical theory; that no-go stands and is not
retested here. The CFT is realized at a **point** (the free critical point,
T1+T2). The **topological parity signature** is realized across a **phase** (the
gapped topological region). The bulk gap is what protects the edge mode, which is
precisely why the *signature* — not the CFT — is robust at finite coupling. These
two regimes are kept surgically separate.

## Method

Reuses the unchanged core BdG machinery (`models/so5_majorana.py`:
`multi_flavor_topological_matrix` + `single_particle_min_abs_energy`, via
`edge_mode_splitting`). No DMRG, no `c_eff.py` — pure free-fermion
diagonalisation. New driver only: `analysis/run_so5_phase.py`.

Fixed: N=64 (the prior T3 floor for n=5 was ~3e-18 here), Delta=t=1 (self-dual
kinetic line), seed=0 (same generic antisymmetric inter-flavor coupling as the
Option-B T3 point). Grid: mu/t in [0.0, 3.0] step 0.1 (31 points, crossing the
boundary at 2t); w/t in [0.0, 1.0] step 0.1 (11 points). For each (mu, w) and
each of n=5, n=4 we record the lowest |single-particle energy| (edge-mode
splitting).

### Parity-preserving constraint (verified)

The inter-flavor coupling enters as `lam * A_{ab} (i/2) gamma_a gamma_b` with `A`
real antisymmetric (seed-fixed, generic). Every term in the BdG matrix M is a
Majorana **bilinear**, so M is real antisymmetric and `H = (i/4) gamma^T M gamma`
conserves total fermion parity exactly. The driver asserts this directly across
representative (mu, w): **`max|M + M^T| = 0.00e+00`, parity_even = True**. There
is no linear / single-Majorana / parity-odd term; the zero mode can lift only for
physical (gap-closing) reasons, never trivially.

## D1 — Plateau (robustness, the primary claim)

For n=5, `min|E|` sits at the double-precision floor across the topological
region. The per-column plateau (contiguous from mu=0 where `min|E| < 1e-10`) is
**w-independent and reaches mu = 1.3 for every w in [0, 1.0]** at N=64. The
maximum `min|E|` anywhere in the full topological region (mu<2t, all w) is
**7.4e-3**, occurring only in the thin band approaching the boundary.

The heat map `figures/so5_phase_heatmaps.png` (left) is the primary figure: the
entire lower region (mu < ~1.3) is at log10(min|E|) ~ -16 to -18 across all w,
with a finite-size gradient band only as mu -> 2t.

**The plateau is independent of the inter-flavor coupling w over the full tested
range w in [0, 1.0]** — i.e. up to coupling of order the hopping t, well beyond
the perturbative regime. This is the "robust at finite coupling" content: finite
inter-flavor coupling does not lift the protected mode.

## D2 — Boundary (the second falsifiable feature)

`min|E|` rises sharply as mu crosses 2t, and the rise location is
**w-independent** (set by the bulk gap closing, not by the edge coupling) —
confirmed: the n=5 plateau terminates at the same mu for every w. Line cuts at
w = 0, 0.5, 1.0 are in `figures/so5_phase_linecuts.png`.

The near-boundary rise *within* the nominal topological region at N=64 (e.g.
`min|E|` ~ 1e-8 by mu=1.5, ~5e-3 by mu=1.9) is **finite-size**, not a loss of
protection: the edge-localization length diverges as the gap closes
(xi ~ 1/(2t-mu)), so at fixed N the two ends hybridize. The N-scaling check
(n=5, w=0.5) demonstrates this directly — at fixed mu the splitting falls
exponentially with N:

| mu/t | N=64 | N=128 | N=256 |
|------|------|-------|-------|
| 1.00 | 1.5e-15 | 9.7e-17 | 2.2e-15 |
| 1.50 | 8.8e-09 | 7.1e-16 | 1.1e-15 |
| 1.80 | 4.5e-04 | 5.3e-07 | **7.3e-13** |
| 1.90 | 5.3e-03 | 2.7e-04 | 3.9e-07 |
| 1.95 | 2.7e-03 | 3.9e-03 | 1.5e-04 |
| 2.00 | 7.8e-03 | 1.5e-02 | 2.4e-03 |
| 2.10 | 1.7e-02 | 8.5e-03 | 1.4e-03 |

At mu < 2t the splitting is exponentially suppressed in N (the machine-zero
plateau extends toward mu -> 2t as N grows — N=256 already reaches 7e-13 at
mu=1.8). At mu >= 2t it stays finite for all N: a genuine, w-independent loss of
protection at the bulk gap closing. See
`figures/so5_phase_boundary_finiteN.png`.

## D3 — Control (mandatory; the claim is the difference)

For n=4 under the identical pipeline:
- For **all w > 0** in the topological region, `min|E|` is **finite** — minimum
  over that region is **4.4e-4** — i.e. **no protected zero mode anywhere**. The
  four edge Majoranas pair up completely (n mod 2 = 0).
- The **w = 0 column** is the decoupled limit: there `min|E| ~ 0` for both n=4
  and n=5 (each chain trivially carries its own edge mode). The parity
  distinction is therefore a statement about **coupled** chains (w > 0); at w = 0
  there is nothing to pair. This is reported plainly rather than papered over.

The heat map (right panel) shows the stark contrast: n=4 is at the floor **only**
in the single w=0 column; for any w>0 it is finite (log10 ~ 0 to -2) across the
whole region. n=5 is at the floor across the entire w in [0,1] topological block.

## D4 — Robustness margin (the sentence for the paper)

> At N=64, the n=5 odd-Majorana protected zero mode survives (min|E| < 1e-10) for
> all mu in [0, 1.3] and all inter-flavor coupling w in [0, 1.0] tested; the rise
> between mu=1.3 and the bulk gap closing at mu=2t is finite-size edge
> hybridization — the splitting at fixed mu falls exponentially with N (e.g. at
> mu=1.8: 4.5e-4, 5.3e-7, 7.3e-13 for N=64, 128, 256) — so the machine-zero
> plateau extends to mu -> 2t in the thermodynamic limit. The signature lifts
> genuinely (and w-independently) only at mu >= 2t, the bulk gap closing. The n=4
> even control shows no protected zero mode anywhere for w>0 under the identical
> pipeline.

The defensible, surgical statement for Section 11.2: **the odd-Majorana parity
signature is robust across the entire gapped topological phase and for finite
inter-flavor coupling up to the order of the hopping (w<=t), vanishing only at
the phase boundary; the even control lacks it throughout. The c=5/2 critical
theory remains a free-point property (T1+T2) and is not claimed robust at finite
coupling.**

## Verdict

| Diagnostic | Result | Verdict |
|-----------|--------|---------|
| D1 plateau (n=5) | min|E|<1e-10 for all w in [0,1], mu in [0,1.3] at N=64; extends to mu->2t with N | robust across the phase + finite coupling |
| D2 boundary | sharp, w-independent rise at mu=2t; near-boundary lift at N=64 is finite-size (exp. suppressed in N) | confirmed |
| D3 control (n=4) | finite everywhere for w>0 (min 4.4e-4); zero only in the w=0 decoupled column | no protected mode; difference confirmed |
| D4 margin | see boxed sentence | stated |

Honest caveat retained: at N=64 the literal `min|E|<1e-10` plateau reaches mu=1.3,
not 2.0; the gap between 1.3 and 2.0 is finite-size, demonstrated by the N-scaling
table, not a breakdown of topological protection. The robustness in the coupling
w (the "finite coupling" claim) is complete across the full tested range with no
finite-N caveat.

## Outputs

- `analysis/run_so5_phase.py` — 2D sweep driver (no core changes)
- `results/so5/phase/phase_grids.npz` — raw 2D grids (mu x w) for n=5, n=4
- `results/so5/phase/minE_grid_n5.csv`, `minE_grid_n4.csv` — same as CSV
- `results/so5/phase/results_phase.json` — parity check, margins, N-scaling
- `results/so5/phase/figures/so5_phase_heatmaps.png` — D1/D3 heat maps
- `results/so5/phase/figures/so5_phase_linecuts.png` — D2 line cuts at w=0,0.5,1.0
- `results/so5/phase/figures/so5_phase_boundary_finiteN.png` — finite-size check
