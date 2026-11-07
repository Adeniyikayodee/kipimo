import numpy as np

from kipimo.belief.grid import BeliefGrid
from kipimo.config import BeliefCfg
from kipimo.types import Parcel, ParcelBelief

Z = 3.92  # width of a 95 percent interval in standard deviations


def prior_width(cfg: BeliefCfg) -> float:
    return cfg.ref_width


def width_from_n(n: np.ndarray, cfg: BeliefCfg) -> float:
    """Parcel CI width given per cell observation evidence. Depends on evidence only."""
    w = n / (n + cfg.h)
    we = Z * cfg.sigma_extrap / np.sqrt(1.0 + w.sum() / cfg.corr_cells)
    wm = np.minimum(Z * 0.5 * cfg.rho / np.sqrt(n + 1.0), we)
    return float((w * wm + (1 - w) * we).mean())


def parcel_stats(grid: BeliefGrid, cells: np.ndarray, om=None) -> tuple[float, float]:
    """Loss mean and CI width for one parcel.

    Observed cells use their measurement. Unobserved cells use the satellite prior,
    bias corrected by the parcel's observed residual, with an extrapolation width that
    shrinks as more independent patches of the parcel are seen.
    """
    n = grid.n[cells]
    w = n / (n + grid.cfg.h)
    p = grid.prior[cells]
    x = (grid.obs_mean() if om is None else om)[cells]  # om: the whole map's, computed once
    sw = w.sum()
    delta = float((w * (x - p)).sum() / sw) if sw > 0 else 0.0
    mean = float((w * x + (1 - w) * (p + delta)).mean())
    return mean, width_from_n(n, grid.cfg)


def aggregate(belief, parcels: tuple[Parcel, ...]) -> dict[str, ParcelBelief]:
    w0 = prior_width(belief.cfg)
    means, widths = belief.parcel_moments(parcels)
    out = {}
    for p, mean, width in zip(parcels, means.tolist(), widths.tolist(), strict=True):
        lo, hi = max(0.0, mean - width / 2), min(1.0, mean + width / 2)
        conf = min(max(1 - width / w0, 0.0), 1.0)
        out[p.id] = ParcelBelief(p.id, mean, (lo, hi), conf, belief.observed(p.cells), len(p.cells))
    return out
