"""Wind at the drone: the forecast's hourly mean on a seeded bearing, plus gusts.

Gusts are an Ornstein-Uhlenbeck process per axis, the first order core of the Dryden turbulence
model: a fixed share of the mean wind, correlated over a few seconds.
"""

import math
from collections.abc import Callable

import numpy as np


class Wind:
    def __init__(self, seed: int, mean_mps: Callable[[float], float], gust: float, tau_s: float):
        self.rng = np.random.default_rng([seed, 13])
        self.bearing = float(self.rng.uniform(0, 2 * math.pi))  # direction the wind blows to
        self.mean_mps, self.share, self.tau = mean_mps, gust, tau_s
        self.g = np.zeros(2)

    def mean(self, t: float) -> tuple[float, float]:
        return self.vec(self.mean_mps(t))

    def vec(self, s: float) -> tuple[float, float]:
        return s * math.cos(self.bearing), s * math.sin(self.bearing)

    def step(self, t: float, dt: float) -> tuple[float, float]:
        a = math.exp(-dt / self.tau)
        sd = self.share * self.mean_mps(t) * math.sqrt(1 - a * a)
        self.g = a * self.g + self.rng.normal(0, sd, 2)
        mx, my = self.mean(t)
        return mx + float(self.g[0]), my + float(self.g[1])

    def from_deg(self) -> int:
        """Compass bearing the wind comes from, with y pointing south on the map."""
        return (
            round(math.degrees(math.atan2(-math.cos(self.bearing), math.sin(self.bearing)))) % 360
        )
