# Symmetric-DMRG cross-check: matched SO(5)$_1$/Spin(4)$_1$ tower + spinor sector

Independent, interacting-lattice confirmation of the SO(5)$_1$ diagnostics, via
**SO(5)-symmetric DMRG** on the bilinear-biquadratic (BBQ) chain at the Reshetikhin
point, charge-conserving in the two Cartan $U(1)$ charges (TeNPy). Complements the
free-Majorana realization in the repository root.

## Contents

Two reports carry the results and verdicts:

- **`REPORT-matched-tower.md`** — the odd/even ($n=5$ SO(5)$_1$ vs $n=4$ Spin(4)$_1$)
  parity distinction in the half-cut entanglement spectrum.
- **`REPORT-spinor-sector.md`** — the spinor at $\Delta_s=5/8$ (exact free-Majorana
  twist) and the native spinor-rep feasibility benchmark (its NN critical point is
  SU(4)$_1$, not SO(5)$_1$).

### Code

| file | role |
|---|---|
| `so5_bbq_dmrg_sym.py` | $n=5$ SO(5) vector BBQ chain, weight-basis charge-conserving DMRG (Reshetikhin $\theta=\arctan(1/9)$) |
| `so4_bbq_dmrg_sym.py` | $n=4$ Spin(4) control on the same framework ($\theta=0$) |
| `so4_heisenberg_product.py` | exact $n=4$ tower as the product of two converged Heisenberg chains (avoids the $\chi=\chi_\text{single}^2$ blow-up) |
| `so4_run_ladder.py` | combined-site $n=4$ $\chi$-ladder (same-pipeline cross-check) |
| `so_matched_tower.py` | matched-tower analysis (shell/irrep identification) |
| `extract_c_so4.py`, `gen_matched_report.py` | $n=4$ central-charge check; report generator |
| `so5_free_twist.py` | exact free-Majorana twist: $h_\sigma\to1/16$, $\Delta_s\to5/8$, twisted-sector degeneracy 4 |
| `so5_spinor_dmrg.py`, `so5_spinor_scan.py`, `so5_spinor_critical.py` | Sp(4)-fundamental (SO(5) spinor) chain: build, BBQ $\theta$-scan, critical-point convergence |

### Results (`results/`)

`so5sym_L96_chi1200.json` ($n=5$ ES), `so4_L*_prod.json` / `so4_L*_chi*.json`
($n=4$), `matched_tower.json` (matched comparison); `free_twist_validation.json`,
`validate_spinor.json`, `spinor_th0_L*.json`, `spinor_scan_L*.json`,
`spinor_crit_L32.json` (spinor sector).

## Reproduce

```
pip install -r ../requirements.txt          # numpy, scipy, physics-tenpy
# n=5 vector ground state + ES (Reshetikhin point):
python so5_bbq_dmrg_sym.py --mode production --L 96 --chi 1200 --chi-ramp --tag so5sym_L96_chi1200
# n=4 Spin(4) control (exact product of two Heisenberg chains):
python so4_heisenberg_product.py --L 96 --chi 2000
# matched tower + report:
python so_matched_tower.py && python gen_matched_report.py
# spinor sector:
python so5_free_twist.py                     # exact, instant: 1/16, 5/8, deg 4
python so5_spinor_dmrg.py --mode validate    # spinor-rep build + theta=0 dimerization
python so5_spinor_critical.py --Ls 32        # SU(4)_1 critical point at arctan(1/3)
```

Thread count is set by `DMRG_THREADS_PER_WORKER` (default 16/30). The
free-twist step is exact and instant; the DMRG steps are the costly ones
(the critical-point convergence is the slowest).
