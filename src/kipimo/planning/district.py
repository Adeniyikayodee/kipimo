"""A district of villages and the drone hubs that serve them, placed like ambulance stations.

Hubs go where they cover the most insured money within one battery's reach: the maximal
covering location problem (Church and ReVelle 1974), solved greedily, which is within
(1 - 1/e) of optimal because coverage is submodular.
"""

import math
from dataclasses import dataclass

import numpy as np

from kipimo import config
from kipimo.payout import triggers
from kipimo.robot.kinematics import power_w
from kipimo.world.generate import generate, parcel_truth

SPAN_KM = 24.0
SITE_STEP_KM = 4.0
SURVEY_KM = 8.0  # a typical village survey, from the benchmark


@dataclass(frozen=True)
class Village:
    id: str
    seed: int
    x: float  # km
    y: float
    insured: float
    expected: float  # payout the radar prior implies, KES
    owed: float  # payout the truth implies, KES
    flooded: bool


def reach_km(cfg: config.RobotCfg, survey_km: float = SURVEY_KM) -> float:
    """One-way radius a hub can serve: charge left after a survey, split out and back."""
    per_m = 1.15 * power_w(cfg, cfg.speed_mps) / cfg.speed_mps / 3600
    usable_km = (cfg.battery_wh - cfg.reserve_wh) / per_m / 1000
    return max(0.0, (usable_km - survey_km) / 2)


def villages(seed: int, n: int = 20, flood_share: float = 0.6) -> list[Village]:
    rng = np.random.default_rng([seed, 31])
    out = []
    for i in range(n):
        vseed = seed * 100 + i
        flooded = rng.random() < flood_share
        stage = [1.6, 2.6] if flooded else [0.2, 0.4]
        w = generate(vseed, config.WorldCfg(flood_stage=stage))
        prior, truth = w.prior.reshape(-1), parcel_truth(w)
        exp = sum(triggers.score(p, float(prior[p.cells].mean())).amount for p in w.parcels)
        owed = sum(triggers.score(p, truth[p.id]).amount for p in w.parcels)
        x, y = rng.uniform(0, SPAN_KM, 2)
        si = sum(p.sum_insured for p in w.parcels)
        out.append(Village(f"V{i:02d}", vseed, round(x, 2), round(y, 2), si, exp, owed, flooded))
    return out


def mission_config(v: Village) -> config.Config:
    stage = [1.6, 2.6] if v.flooded else [0.2, 0.4]
    return config.load(None, seed=v.seed, world={"flood_stage": stage}, belief={"model": "gp"})


def sites() -> list[tuple[float, float]]:
    ticks = np.arange(SITE_STEP_KM / 2, SPAN_KM, SITE_STEP_KM)
    return [(float(x), float(y)) for x in ticks for y in ticks]


def place_hubs(vs: list[Village], k: int, radius: float) -> tuple[list[tuple[float, float]], float]:
    """Greedy maximal covering by sum insured. Returns hub sites and the share covered."""
    cands, covered, hubs = sites(), set(), []
    for _ in range(k):
        gain = [
            sum(v.insured for v in vs if v.id not in covered and math.dist((v.x, v.y), c) <= radius)
            for c in cands
        ]
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        hubs.append(cands[best])
        covered |= {v.id for v in vs if math.dist((v.x, v.y), cands[best]) <= radius}
    total = sum(v.insured for v in vs)
    return hubs, sum(v.insured for v in vs if v.id in covered) / total
