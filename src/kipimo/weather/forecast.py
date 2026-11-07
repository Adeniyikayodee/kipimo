"""Wind forecast ensemble and the wind that actually happens, seeded per mission.

Synthetic until WeatherNext data is wired in: same shape (members x hours), same use. A flight
window is a run of hours where at least `p` of the members stay under the drone's wind limit.
The actual wind is a held-out member, so plans face real forecast error.
"""

from dataclasses import dataclass

import numpy as np

from kipimo.config import WeatherCfg


@dataclass(frozen=True)
class Weather:
    members: np.ndarray  # (members, hours) wind m/s
    actual: np.ndarray  # (hours,)
    limit: float

    def hour(self, t_s: float) -> int:
        return min(int(t_s // 3600), self.actual.shape[0] - 1)

    def wind(self, t_s: float) -> float:
        return float(self.actual[self.hour(t_s)])

    def safe_share(self, h: int) -> float:
        return float((self.members[:, h] < self.limit).mean())

    def window(self, t_s: float, need_h: int = 1, p: float = 0.95) -> int | None:
        """First hour from now that starts `need_h` hours where at least `p` of members are calm."""
        ok = (self.members < self.limit).mean(0) >= p
        for h in range(self.hour(t_s), len(ok) - need_h + 1):
            if ok[h : h + need_h].all():
                return h
        return None


def make(seed: int, cfg: WeatherCfg) -> Weather | None:
    if not cfg.enabled:
        return None
    rng = np.random.default_rng([seed, 11])
    hours = np.arange(cfg.hours)
    storm = cfg.storm_peak * np.exp(-0.5 * ((hours - cfg.storm_hour) / cfg.storm_width) ** 2)
    base = cfg.calm + storm
    walks = np.cumsum(rng.normal(0, cfg.spread, (cfg.members + 1, cfg.hours)), 1) * 0.5
    all_ = np.clip(base + walks + rng.normal(0, cfg.spread, (cfg.members + 1, 1)), 0, None)
    return Weather(all_[:-1], all_[-1], cfg.limit)
