import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kipimo.config import WorldCfg
from kipimo.types import Crop, Parcel
from kipimo.world import terrain
from kipimo.world.noise import fractal

EVENT_DAY = 63
MIN_PARCEL_CELLS = 200
CROPS: tuple[Crop, ...] = ("maize", "rice", "sorghum")
SENSITIVITY = {"maize": 1.0, "rice": 0.55, "sorghum": 0.8}
VALUE_PER_HA = {"maize": 900.0, "rice": 1300.0, "sorghum": 700.0}
HOLDERS = (
    "Amina Okello",
    "Baraka Mwangi",
    "Chiku Njeri",
    "Daudi Otieno",
    "Esther Wanjiru",
    "Faraji Kamau",
    "Grace Achieng",
    "Hassan Mutua",
    "Imani Chebet",
    "Juma Odhiambo",
    "Kesi Nyambura",
    "Lulu Wekesa",
    "Makena Kiprop",
    "Neema Atieno",
    "Omari Kibet",
    "Pendo Auma",
    "Rehema Waweru",
    "Saida Onyango",
    "Tumaini Rotich",
    "Zawadi Muthoni",
)


@dataclass(frozen=True)
class World:
    size: int
    cell_m: float
    elevation: np.ndarray  # (n, n) metres above river datum
    water: np.ndarray  # (n, n) flooded fraction 0..1
    loss: np.ndarray  # (n, n) ground truth crop loss 0..1
    parcel_map: np.ndarray  # (n, n) int16, parcel index or -1
    prior: np.ndarray  # (n, n) coarse satellite loss estimate
    parcels: tuple[Parcel, ...]
    site: str = "synthetic"
    stage: float = 0.0  # river stage, m above datum; the flood surface is flat at it
    rect: tuple[float, float, float, float] | None = None  # footprint in the real terrain crop

    def parcel(self, pid: str) -> Parcel:
        return next(p for p in self.parcels if p.id == pid)


def crop_response(crop: str, planted_day: int) -> float:
    """Loss at full submersion: crop sensitivity times growth stage, worst near flowering.

    This is the agronomic damage curve an insurer holds for each policy.
    """
    age = EVENT_DAY - planted_day
    return SENSITIVITY[crop] * (0.6 + 0.4 * np.exp(-(((age - 55) / 20) ** 2)))


def response_map(world: "World") -> np.ndarray:
    """Flat per cell crop_response from the policy records, 0 off parcels."""
    out = np.zeros(world.size * world.size)
    for p in world.parcels:
        out[p.cells] = crop_response(p.crop, p.planted_day)
    return out


def _river(rng: np.random.Generator, n: int) -> np.ndarray:
    """Distance in cells from a meandering channel running left to right."""
    xs = np.arange(n)
    ph, f = rng.uniform(0, 2 * np.pi, 2), rng.uniform(1.0, 2.0, 2)
    cy = n * (0.5 + 0.12 * np.sin(2 * np.pi * f[0] * xs / n + ph[0]))
    cy += n * 0.05 * np.sin(2 * np.pi * f[1] * 2.3 * xs / n + ph[1])
    return np.abs(np.arange(n)[:, None] - cy[None, :])


def _parcels(rng: np.random.Generator, cfg: WorldCfg) -> np.ndarray:
    n = cfg.size
    k = int(rng.integers(cfg.n_parcels[0], cfg.n_parcels[1] + 1))
    margin = n * 0.08
    seeds = rng.uniform(margin, n - margin, (k, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(yy[..., None] - seeds[:, 0], xx[..., None] - seeds[:, 1])
    order = np.argsort(d, axis=-1)
    first = np.take_along_axis(d, order[..., :1], -1)[..., 0]
    second = np.take_along_axis(d, order[..., 1:2], -1)[..., 0]
    pm = order[..., 0].astype(np.int16)
    pm[(second - first) < 1.5] = -1  # hedgerows between parcels
    pm[
        (yy < margin * 0.5)
        | (yy >= n - margin * 0.5)
        | (xx < margin * 0.5)
        | (xx >= n - margin * 0.5)
    ] = -1
    pm[first > n * 0.22] = -1  # bush beyond the farmed area
    return pm


def _radar_prior(rng: np.random.Generator, water: np.ndarray, vuln: np.ndarray) -> np.ndarray:
    """Loss prior from a Sentinel-1 style radar flood map plus the insurer's crop records.

    Radar pixels are 16 m with speckle, the scene has a calibration bias, and the crop response
    model misses local plant health, so the prior is useful but not settlement grade.
    """
    n, b = water.shape[0], 4
    sar = water.reshape(n // b, b, n // b, b).mean(axis=(1, 3))
    sar = np.clip(sar * rng.uniform(0.8, 1.2) + rng.normal(0, 0.12, sar.shape), 0, 1)
    sar = np.kron(sar, np.ones((b, b)))
    m = 32
    model_err = np.kron(rng.normal(0, 0.04, (n // m, n // m)), np.ones((m, m)))
    return np.clip(vuln * sar**0.8 + model_err, 0.02, 0.98)


def generate(seed: int, cfg: WorldCfg | None = None) -> World:
    cfg = cfg or WorldCfg()
    rng = np.random.default_rng(seed)
    n = cfg.size
    if cfg.terrain == "tana":
        elevation, site, rect = terrain.window(np.random.default_rng([seed, 17]), n, cfg.cell_m)
    else:
        site, rect = "synthetic", None
        elevation = 0.025 * _river(rng, n) + 1.2 * fractal(rng, n, base=4) + 0.4
    stage = rng.uniform(*cfg.flood_stage)
    water = np.clip((stage - elevation) / 0.6, 0, 1)

    pm = _parcels(rng, cfg)
    if site != "synthetic":
        pm[elevation < 0.3] = -1  # the channel and oxbows hold water before any flood
    k = int(pm.max()) + 1
    parcels = []
    health_noise = fractal(rng, n, base=16, octaves=3)
    loss = np.zeros((n, n))
    vuln = np.zeros(n * n)
    names = rng.permutation(len(HOLDERS))
    flat_pm = pm.reshape(-1)
    for i in range(k):
        cells = np.flatnonzero(flat_pm == i).astype(np.int32)
        if len(cells) < MIN_PARCEL_CELLS:  # slivers left by hedgerows and bush are not farms
            flat_pm[cells] = -1
            continue
        j = len(parcels)
        flat_pm[cells] = j
        crop = CROPS[int(rng.integers(len(CROPS)))]
        planted = int(rng.integers(0, 45))
        ha = len(cells) * cfg.cell_m**2 / 1e4
        si = round(VALUE_PER_HA[crop] * ha * rng.uniform(0.8, 1.2), -1)
        vuln[cells] = crop_response(crop, planted)
        flat = loss.reshape(-1)
        w = water.reshape(-1)[cells]
        flat[cells] = vuln[cells] * w**0.8 + 0.06 * health_noise.reshape(-1)[cells]
        parcels.append(
            Parcel(f"P{j:02d}", cells, HOLDERS[names[j % len(HOLDERS)]], si, crop, planted)
        )
    loss = np.clip(loss, 0, 1)
    loss[pm < 0] = 0

    prior = _radar_prior(rng, water, vuln.reshape(n, n))

    return World(
        n, cfg.cell_m, elevation, water, loss, pm, prior, tuple(parcels), site, float(stage), rect
    )


def parcel_truth(world: World) -> dict[str, float]:
    flat = world.loss.reshape(-1)
    return {p.id: float(flat[p.cells].mean()) for p in world.parcels}


def save(world: World, path: Path) -> None:
    np.savez_compressed(
        path,
        elevation=world.elevation,
        water=world.water,
        loss=world.loss,
        parcel_map=world.parcel_map,
        prior=world.prior,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--png", type=Path)
    a = ap.parse_args()
    w = generate(a.seed)
    save(w, a.out)
    if a.png:
        from kipimo.world.render import truth_png

        truth_png(w, a.png)
    for pid, v in parcel_truth(w).items():
        p = w.parcel(pid)
        print(f"{pid} {p.crop:8s} si={p.sum_insured:8.0f} cells={len(p.cells):5d} loss={v:.3f}")


if __name__ == "__main__":
    main()
