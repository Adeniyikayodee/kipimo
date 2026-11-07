"""Empirical Bayes fit of the GP residual model on calibration floods.

Per flood, regress truth minus radar prior on (bias, prior, crop response); the prior sd of each
coefficient is its root mean square across floods. Parcel means of what is left give the parcel
effect; a three scale variogram of the rest gives radar pixel noise, local and regional patchiness,
each as (variance, cells per independent block). Seeds are disjoint from bench, demo, training.
"""

import argparse

import numpy as np

from kipimo import config
from kipimo.world.generate import generate, response_map

SEEDS = range(3000, 3030)
LAGS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])  # cells
PIXEL = 1.5  # radar pixel noise decorrelation, cells


def residuals(seed: int, cfg: config.Config):
    w = generate(seed, cfg.world)
    on = w.parcel_map.reshape(-1) >= 0
    r = (w.loss - w.prior).reshape(-1)
    X = np.column_stack([np.ones(on.sum()), w.prior.reshape(-1)[on], response_map(w)[on]])
    beta, *_ = np.linalg.lstsq(X, r[on], rcond=None)
    e = np.full(r.shape, np.nan)
    e[on] = r[on] - X @ beta
    return beta, e.reshape(w.size, w.size)


def variogram(e: np.ndarray) -> np.ndarray:
    out = []
    for h in LAGS:
        d = np.concatenate([(e[:, h:] - e[:, :-h]).ravel(), (e[h:] - e[:-h]).ravel()])
        out.append(0.5 * np.nanmean(d**2))
    return np.array(out)


def fit(cfg: config.Config) -> dict:
    betas, parcel_means, gams = [], [], []
    for s in SEEDS:
        w = generate(s, cfg.world)
        b, e = residuals(s, cfg)
        betas.append(b)
        within = e.copy().reshape(-1)
        for p in w.parcels:
            m = within[p.cells].mean()
            parcel_means.append(m)
            within[p.cells] -= m
        gams.append(variogram(within.reshape(e.shape)))
    sd = np.sqrt(np.mean(np.square(betas), 0))
    gam = np.mean(gams, 0)
    best = None
    for l1 in np.linspace(3, 20, 18):  # local and regional patch scales, cells
        for l2 in np.linspace(l1 + 4, 80, 20):
            X = np.column_stack([1 - np.exp(-LAGS / PIXEL), _gauss(l1), _gauss(l2)])
            v, *_ = np.linalg.lstsq(X, gam, rcond=None)
            err = np.sum((X @ v - gam) ** 2)
            if (v > 0).all() and (best is None or err < best[0]):
                best = (err, (PIXEL, l1, l2), v)
    _, scales, v = best
    return {
        "s_bias": round(float(sd[0]), 4),
        "s_prior": round(float(sd[1]), 4),
        "s_resp": round(float(sd[2]), 4),
        "s_parcel": round(float(np.sqrt(np.mean(np.square(parcel_means)))), 4),
        "bias_frac": round(bias_frac(cfg), 3),
        "blocks": [
            [round(float(x), 5), round(float(2 * np.pi * L**2), 1)]
            for x, L in zip(v, scales, strict=True)
        ],
    }


def bias_frac(cfg: config.Config, looks: int = 8) -> float:
    """RMS parcel mean measurement bias over the per look noise sd, from simulated looks."""
    from kipimo.belief.gp import sigma_meas

    out = []
    for q in (0.77, 0.3):
        sm, b = sigma_meas(q), []
        for s in SEEDS:
            w, rng = generate(s, cfg.world), np.random.default_rng([s, 23])  # not the world stream
            loss = w.loss.reshape(-1)
            for p in w.parcels:
                x = loss[p.cells]
                y = np.mean([np.clip(x + rng.normal(0, sm, len(x)), 0, 1) for _ in range(looks)], 0)
                b.append(y.mean() - x.mean())
        out.append(np.sqrt(np.mean(np.square(b))) / sm)
    return float(max(out))


def _gauss(L: float) -> np.ndarray:
    return 1 - np.exp(-(LAGS**2) / (2 * L**2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    print(fit(config.load(ap.parse_args().scenario)))


if __name__ == "__main__":
    main()
