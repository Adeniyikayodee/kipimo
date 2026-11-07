import numpy as np

from kipimo.config import BeliefCfg
from kipimo.types import Observation


class BeliefGrid:
    """Per cell Beta posterior over crop loss. Prior pseudo counts come from the satellite estimate.

    alpha = k0 * prior + sum(k * q * x)
    beta  = k0 * (1 - prior) + sum(k * q * (1 - x))
    """

    def __init__(self, prior: np.ndarray, cfg: BeliefCfg):
        self.cfg = cfg
        self.prior = prior.reshape(-1).astype(np.float64)
        self.n = np.zeros_like(self.prior)  # observation pseudo counts
        self.sx = np.zeros_like(self.prior)  # pseudo count weighted loss sum

    def update(self, obs: Observation) -> None:
        w = self.cfg.k * obs.quality
        self.n[obs.cell_ids] += w
        self.sx[obs.cell_ids] += w * obs.crop_loss.astype(np.float64)

    @property
    def alpha(self) -> np.ndarray:
        return self.cfg.k0 * self.prior + self.sx

    @property
    def beta(self) -> np.ndarray:
        return self.cfg.k0 * (1 - self.prior) + self.n - self.sx

    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def obs_mean(self) -> np.ndarray:
        return np.divide(self.sx, self.n, out=self.prior.copy(), where=self.n > 0)

    def weight(self) -> np.ndarray:
        """How much a cell's estimate rests on direct observation, 0..1."""
        return self.n / (self.n + self.cfg.h)

    def parcel_moments(self, parcels) -> tuple[np.ndarray, np.ndarray]:
        """Loss mean and 95% interval width per parcel."""
        from kipimo.belief.parcels import parcel_stats

        om = self.obs_mean()
        m, w = zip(*(parcel_stats(self, p.cells, om) for p in parcels), strict=True)
        return np.array(m), np.array(w)

    def doubt(self, pid: str, factor: float = 3.0) -> bool:
        return False  # the grid prior has no parcel term to widen

    def observed(self, cells: np.ndarray) -> int:
        return int((self.n[cells] > 0).sum())

    def evidence(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.n[cells], self.sx[cells]

    def copy(self) -> "BeliefGrid":
        g = BeliefGrid.__new__(BeliefGrid)
        g.cfg, g.prior, g.n, g.sx = self.cfg, self.prior, self.n.copy(), self.sx.copy()
        return g
