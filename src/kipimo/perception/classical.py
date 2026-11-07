"""Classical baseline: HSV plus local texture, mapped to (water, loss) by a least squares fit.

No learning beyond a linear calibration, so it doubles as the fallback when the model is missing.
"""

from functools import cache

import numpy as np

from kipimo.world.generate import generate
from kipimo.world.render import render_tile, tile_truth


def hsv(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(-1), img.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return np.stack([h / 6, d / (mx + 1e-6), mx], -1)


def features(img: np.ndarray) -> np.ndarray:
    f = hsv(img)
    v = img.mean(-1)
    pad = np.pad(v, 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, (5, 5))
    tex = win.std(axis=(-1, -2))
    h, s, val = f[..., 0], f[..., 1], f[..., 2]
    cols = [h, s, val, tex, h * s, s * s, h * h, tex * s, np.ones_like(h)]
    return np.stack(cols, -1)


@cache
def calibration(n_tiles: int = 160, seed: int = 1000) -> np.ndarray:
    """Fit on renders of worlds disjoint from the benchmark seeds."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in range(n_tiles):
        w = generate(seed + i // 20)
        span = w.size * w.cell_m
        x, y = rng.uniform(80, span - 80, 2)
        img, ids = render_tile(w, x, y, 34.0, float(rng.uniform(0.3, 1.0)), rng)
        keep = ids >= 0
        X.append(features(img)[keep])
        Y.append(tile_truth(w, ids)[:, keep].T)
    X, Y = np.concatenate(X), np.concatenate(Y)
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def predict(img: np.ndarray) -> np.ndarray:
    """Per pixel (water, loss), shape (2, H, W)."""
    out = features(img) @ calibration()
    return np.clip(out, 0, 1).transpose(2, 0, 1).astype(np.float32)
