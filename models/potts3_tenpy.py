"""
3-state quantum Potts chain via TenPy DMRG.

H = -J sum_i (Z_i Z*_{i+1} + Z*_i Z_{i+1}) - h sum_i (X_i + X*_i)

where Z, X are Z_3 clock generators (3x3 unitary matrices). At the self-dual
critical point J = h, the model is described by the parafermion CFT with
central charge c = 4/5 (the 3-state Potts CFT).

TenPy implementation: build via SpinSite with custom on-site ops, or via
tenpy.networks.site for a generic d=3 site. We use a custom CouplingModel.
"""

from __future__ import annotations

import time

import numpy as np
import tenpy
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.algorithms.dmrg import TwoSiteDMRGEngine


# Z_3 clock matrices
omega = np.exp(2j * np.pi / 3)
Z3 = np.diag([1.0, omega, omega.conjugate()])
X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
Zd = Z3.conjugate().T  # Z^dagger
Xd = X3.conjugate().T


def make_potts_site() -> Site:
    """Generic d=3 site with Z and X clock operators."""
    leg = tenpy.linalg.charges.LegCharge.from_trivial(3)
    site = Site(leg, ["0", "1", "2"])
    site.add_op("Z", Z3, hc="Zd")
    site.add_op("Zd", Zd, hc="Z")
    site.add_op("X", X3, hc="Xd")
    site.add_op("Xd", Xd, hc="X")
    return site


class PottsChain(CouplingMPOModel, NearestNeighborModel):
    def init_sites(self, model_params):
        return make_potts_site()

    def init_terms(self, model_params):
        J = model_params.get("J", 1.0)
        h = model_params.get("h", 1.0)
        # -J (Z_i Zd_{i+1} + Zd_i Z_{i+1})
        self.add_coupling(-J, 0, "Z", 0, "Zd", 1, plus_hc=False)
        self.add_coupling(-J, 0, "Zd", 0, "Z", 1, plus_hc=False)
        # -h (X + Xd)
        self.add_onsite(-h, 0, "X")
        self.add_onsite(-h, 0, "Xd")


def build_ground_state(L: int, chi: int, J: float = 1.0, h: float = 1.0,
                       seed: int = 0, verbose: bool = True):
    """
    DMRG ground state of the 3-state Potts chain.
    Critical at J = h.
    """
    model_params = {
        "L": L,
        "J": J,
        "h": h,
        "bc_MPS": "finite",
    }
    M = PottsChain(model_params)
    # initial state: random product
    rng = np.random.default_rng(seed)
    init = rng.integers(0, 3, size=L).tolist()
    psi = MPS.from_lat_product_state(M.lat, [[s] for s in init])
    dmrg_params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "max_E_err": 1e-7,
        "max_S_err": 1e-5,
        "mixer": True,
        "max_sweeps": 30,
        "min_sweeps": 6,
    }
    if verbose:
        print(f"[Potts-DMRG] L={L} J={J} h={h} chi={chi}: starting", flush=True)
    t0 = time.time()
    eng = TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, _ = eng.run()
    if verbose:
        print(f"[Potts-DMRG] L={L} J={J} h={h}: E/site={E/L:.6f}  "
              f"in {time.time()-t0:.1f}s", flush=True)
    return psi, E
