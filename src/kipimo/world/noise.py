import numpy as np


def _upsample(g: np.ndarray, n: int) -> np.ndarray:
    m = g.shape[0] - 1
    u = np.linspace(0, m, n, endpoint=False)
    i = np.floor(u).astype(int)
    f = u - i
    f = f * f * (3 - 2 * f)
    a = g[i][:, i] * (1 - f)[None, :] + g[i][:, i + 1] * f[None, :]
    b = g[i + 1][:, i] * (1 - f)[None, :] + g[i + 1][:, i + 1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def fractal(rng: np.random.Generator, n: int, base: int = 4, octaves: int = 5) -> np.ndarray:
    """Smoothstep value noise, summed over octaves, normalised to [-1, 1]."""
    out = np.zeros((n, n))
    amp, cells, total = 1.0, base, 0.0
    for _ in range(octaves):
        out += amp * _upsample(rng.uniform(-1, 1, (cells + 1, cells + 1)), n)
        total += amp
        amp *= 0.5
        cells *= 2
    return out / total
