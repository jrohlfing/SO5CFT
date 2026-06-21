# SO(5)$_1$ boundary CFT: lattice realization and diagnostics

Reproduction code and numerical results for the coupled-Majorana-chain
realization of the SO(5)$_1$ Wess-Zumino-Witten model, supporting the
lattice section of the manuscript *A Parity Signature Distinguishing SO(5)$_1$
from Spin(4)$_1$ Boundaries* (J. Rohlfing).

This repository contains only the lattice computations referenced in the
data-availability statement of that paper.

## What is computed

Five coupled critical Kitaev (Majorana) chains with a generic SO(5)-direction
inter-flavor coupling. Four diagnostics:

1. **Central charge.** Half-chain entanglement entropy fitted to the
   Calabrese-Cardy form returns $c_{\mathrm{eff}}=2.513$ for $n=5$ (target
   $5/2$), with $n=1,2,4$ recovering $0.50,\,1.01,\,2.01$ as controls.
2. **Operator content.** Finite-size scaling dimensions: the single-chain twist
   at $1/8$ (an Ising$^5$ operator) is removed by the SO(5) projection; the
   joint five-chain spinor sits at $\Delta_s=5/8$ below the vector at $1$ — the
   diagnostic separating SO(5)$_1$ from Ising$^5$. Stable across $N=64$--$512$.
3. **Parity signature.** The odd channel count ($n=5$) leaves one protected
   Majorana zero mode per edge (splitting falls to the double-precision floor);
   the even control ($n=4$) leaves none. Altland-Zirnbauer class D, $\mathbb{Z}_2$
   index $n \bmod 2$. Robust across the $(\mu,w)$ phase plane.
4. **Marginality.** A DMRG scan of the SO(5)-symmetric Gross-Neveu coupling
   shows $c$ maximal at the free point and falling both sides; the interaction
   is marginal and the SO(5)$_1$ realization is the free Majorana fixed point.

## Symmetric-DMRG cross-check (`symmetric-dmrg/`)

An independent, *interacting*-lattice confirmation via SO(5)-symmetric DMRG on the
bilinear-biquadratic (BBQ) chain at the Reshetikhin point, charge-conserving in the
two Cartan $U(1)$s, complementing the free-Majorana realization above:

- **Matched entanglement tower.** The odd/even ($n=5$ vs $n=4$) parity shown
  directly in the half-cut *entanglement spectrum*, which the equal topological
  entanglement entropy ($\gamma=\log 2$ for both) cannot see: the SO(5)$_1$ leading
  shell is the 5-dim vector multiplet *including the $(0,0)$ weight*, then the
  adjoint (10) with a 2-fold $(0,0)$ Cartan; the Spin(4)$_1$ control's leading
  shell is the 4-dim vector *without* $(0,0)$. The n=4 control is the exact product
  of two Heisenberg chains (Spin(4)$_1=$SU(2)$_1\times$SU(2)$_1$, $c=2$); the
  Reshetikhin point is derived as $\tan\theta_R=(n-4)/(n-2)^2$ (so $1/9$ at $n=5$,
  $0$ at $n=4$).
- **Spinor sector.** An exact free-Majorana twist computation reconfirms the
  spinor at $\Delta_s=5/8$ (chiral $h=1/16$ per Majorana, twisted-sector
  ground-state degeneracy $2^{\lfloor 5/2\rfloor}=4$). A feasibility benchmark of
  the *native* 4-dim spinor-rep (Sp(4)-fundamental) chain finds it dimerized at the
  Heisenberg point, with its only nearest-neighbor critical point at
  $\theta=\arctan(1/3)$ being the enhanced-symmetry SU(4)$_1$ ($c=3$, fundamental
  at $\Delta=3/4$) rather than SO(5)$_1$ — so the SO(5)$_1$ spinor is best obtained
  from the twist/free-Majorana route, not a native spinor chain.

See `symmetric-dmrg/REPORT-matched-tower.md` and
`symmetric-dmrg/REPORT-spinor-sector.md` for the full tables and verdicts.

## Layout

```
models/        Hamiltonians: so5_majorana.py (+ TFIM, Potts-3, XX, classical controls)
analysis/      run_so5.py, run_so5_phase.py (drivers); c_eff.py, dmrg_parallel.py
results/so5/   results_so5.json, gap_ratio_table.csv, phase/ (minE grids, results_phase.json)
figures/       fig1.png, fig2.png, fig3.png, make_figs.py (regenerates all three from results/)
symmetric-dmrg/  SO(5)-symmetric BBQ DMRG: matched SO(5)/Spin(4) tower + spinor sector
                 (code, results/*.json, REPORT-matched-tower.md, REPORT-spinor-sector.md)
MODEL-DERIVATION.md     exact Hamiltonian, SO(5) generators, pre-registered targets
RESULTS-SO5-LATTICE.md  the four tests with verdicts
RESULTS-SO5-PHASE.md    the (mu,w) phase scan
```

## Reproduce

```
pip install -r requirements.txt
python -m analysis.run_so5          # tests 1-3, writes results/so5/results_so5.json
python -m analysis.run_so5_phase    # phase scan, writes results/so5/phase/
python figures/make_figs.py         # regenerates fig1.png, fig2.png, fig3.png from results/
```

The committed `results/` JSON and CSV files are the outputs used in the paper;
`make_figs.py` reads them directly, so the figures can be regenerated without
rerunning the (slower) DMRG and exact-diagonalization drivers.

## Requirements

numpy, scipy, matplotlib, torch (free-fermion covariance), physics-tenpy (DMRG).
Computations were run with TeNPy on a single workstation; the free-fermion
parts are inexpensive, the Gross-Neveu DMRG scan is the costliest step.

## License

MIT (see LICENSE).
