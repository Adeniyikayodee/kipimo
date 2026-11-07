"""Gaussian belief over crop loss with scene, parcel and local random effects.

    loss_i = prior_i + phi_i . w + r_i
    w   ~ N(0, diag(s^2))     phi = (1, radar prior, crop response, parcel one-hot[, embeddings])
    r_i ~ sum of block terms (variance, cells per block): radar pixels, local, regional patches
    measurements carry a bias shared by a parcel's cells, bias_frac of their noise sd

Equivalently a GP whose kernel is the sum of those terms. Scene terms carry evidence across
fields: a few looks recalibrate the radar for every parcel. Correlated residuals are counted once
per block, so repeated nearby cells do not inflate confidence. State is D x D with D = 3 + parcels.
"""

import numpy as np

from kipimo.config import BeliefCfg
from kipimo.types import Observation, Parcel
from kipimo.world.generate import World, response_map

N_SCENE = 3
Z = 3.92  # width of a 95 percent interval in sds


def sigma_meas(q: float) -> float:
    return 0.03 + 0.15 * (1 - q)


class GPBelief:
    def __init__(self, world: World, cfg: BeliefCfg):
        g = cfg.gp
        self.cfg = cfg
        self.blocks = np.array(g.blocks)
        self.r2 = self.blocks[:, 0].sum()  # per cell residual variance
        self.block2 = self.blocks.prod(1).sum()  # residual variance of one block's worth
        self.sys2 = g.sigma_sys**2
        self.prior = world.prior.reshape(-1).astype(np.float64)
        self.resp = response_map(world)
        self.pid = world.parcel_map.reshape(-1).astype(np.int64)
        self.index = {p.id: i for i, p in enumerate(world.parcels)}
        P = len(world.parcels)
        self.ncell = np.array([len(p.cells) for p in world.parcels], float)
        self.S = np.stack([self.phi(p.cells).sum(0) for p in world.parcels])
        self.prior_sum = np.array([self.prior[p.cells].sum() for p in world.parcels])
        s2 = np.r_[g.s_bias**2, g.s_prior**2, g.s_resp**2, np.full(P, g.s_parcel**2)]
        self.lam = np.diag(1 / s2)
        self.eta = np.zeros(len(s2))
        self.prec = np.zeros(len(self.prior))  # sum of 1 / sigma_meas^2 per cell
        self.ysum = np.zeros(len(self.prior))  # precision weighted measurement sum per cell
        self.sb = np.zeros(len(self.prior))  # sum of 1 / sigma_meas per cell, for the bias sd
        self.C = np.zeros((P, len(s2)))  # observed cells: sum of shrink * phi
        self.K = np.zeros(P)  # observed cells: sum of shrink * (ybar - prior)
        self.V = np.zeros(P)  # observed cells: sum of posterior residual variance
        self.B = np.zeros(P)  # observed cells: sum of shrink * measurement bias sd
        self.unseen = self.ncell.copy()
        self.doubted: set[str] = set()
        self._sigma = None

    def phi(self, ids: np.ndarray) -> np.ndarray:
        F = np.zeros((len(ids), N_SCENE + len(self.ncell)))
        F[:, 0], F[:, 1], F[:, 2] = 1.0, self.prior[ids], self.resp[ids]
        F[np.arange(len(ids)), N_SCENE + self.pid[ids]] = 1.0
        return F

    def terms(self, prec: np.ndarray):
        """Per cell (seen, residual variance, shrink, evidence on w) from look precision."""
        seen = prec > 0
        v = self.sys2 + 1 / np.where(seen, prec, np.inf)
        s = np.where(seen, self.r2 / (self.r2 + v), 0.0)
        g = np.where(seen, 1 / (self.block2 + v), 0.0)  # once per correlated block
        return seen, s * v, s, g

    def bias(self, s: np.ndarray, sb: np.ndarray, prec: np.ndarray) -> np.ndarray:
        return self.cfg.gp.bias_frac * s * sb / np.where(prec > 0, prec, 1)

    def _cell(self, ids: np.ndarray):
        prec = self.prec[ids]
        seen, pv, s, g = self.terms(prec)
        ybar = np.where(seen, self.ysum[ids] / np.where(seen, prec, 1), self.prior[ids])
        return seen, pv, s, ybar, g

    def update(self, obs: Observation) -> None:
        keep = self.pid[obs.cell_ids] >= 0
        if not keep.any():
            return
        ids, inv = np.unique(obs.cell_ids[keep], return_inverse=True)
        tau = 1 / sigma_meas(obs.quality) ** 2
        seen0, pv0, s0, y0, g0 = self._cell(ids)
        b0 = self.bias(s0, self.sb[ids], self.prec[ids])
        self.prec[ids] += tau * np.bincount(inv)
        self.ysum[ids] += tau * np.bincount(inv, obs.crop_loss[keep].astype(np.float64))
        self.sb[ids] += np.sqrt(tau) * np.bincount(inv)
        _, pv1, s1, y1, g1 = self._cell(ids)
        np.add.at(self.B, self.pid[ids], self.bias(s1, self.sb[ids], self.prec[ids]) - b0)
        F, p, k = self.phi(ids), self.prior[ids], self.pid[ids]
        self.lam += F.T @ ((g1 - g0)[:, None] * F)
        self.eta += F.T @ (g1 * (y1 - p) - g0 * (y0 - p))
        np.add.at(self.C, k, (s1 - s0)[:, None] * F)
        np.add.at(self.K, k, s1 * (y1 - p) - s0 * (y0 - p))
        np.add.at(self.V, k, pv1 - pv0)
        np.add.at(self.unseen, k, -(~seen0).astype(float))
        self._sigma = None

    @property
    def sigma(self) -> np.ndarray:
        if self._sigma is None:
            self._sigma = np.linalg.inv(self.lam)
        return self._sigma

    @property
    def mu(self) -> np.ndarray:
        return self.sigma @ self.eta

    def unseen_var(self, i: np.ndarray, u: np.ndarray) -> np.ndarray:
        v, c = self.blocks.T
        return u * (v * np.minimum(c, u[:, None])).sum(1) / self.ncell[i] ** 2

    def parcel_moments(self, parcels: tuple[Parcel, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Loss mean and 95% interval width per parcel."""
        i = np.array([self.index[p.id] for p in parcels])
        n = self.ncell[i]
        A = (self.S[i] - self.C[i]) / n[:, None]
        mean = (self.prior_sum[i] + A @ self.mu * n + self.K[i]) / n
        var = np.einsum("pd,de,pe->p", A, self.sigma, A) + self.unseen_var(i, self.unseen[i])
        return np.clip(mean, 0, 1), Z * np.sqrt(var + self.V[i] / n**2 + (self.B[i] / n) ** 2)

    def mean(self) -> np.ndarray:
        out = np.zeros_like(self.prior)
        ids = np.flatnonzero(self.pid >= 0)
        _, _, s, ybar, _ = self._cell(ids)
        fit = self.prior[ids] + self.phi(ids) @ self.mu
        out[ids] = (1 - s) * fit + s * ybar
        return np.clip(out, 0, 1)

    def doubt(self, pid: str, factor: float = 3.0) -> bool:
        """Widen one parcel's prior, e.g. when its farmer reports a loss the radar disagrees with.

        Only the prior variance grows, so the estimate never moves on a report alone; the planner
        sees more money at risk there and sends the drone to look.
        """
        if pid in self.doubted:
            return False
        k = N_SCENE + self.index[pid]
        self.lam[k, k] -= (1 - factor**-2) / self.cfg.gp.s_parcel**2
        self.doubted.add(pid)
        self._sigma = None
        return True

    def observed(self, cells: np.ndarray) -> int:
        return int((self.prec[cells] > 0).sum())

    def evidence(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.prec[cells], self.ysum[cells]

    def weight(self) -> np.ndarray:
        """How much a cell rests on direct observation, 0..1."""
        out = np.zeros_like(self.prior)
        ids = np.flatnonzero(self.prec > 0)
        out[ids] = self._cell(ids)[2]
        return out
