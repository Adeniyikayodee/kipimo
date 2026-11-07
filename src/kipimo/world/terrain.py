"""Real floodplain relief: Copernicus GLO-30 over the lower Tana River near Garsen, Kenya.

Each seed takes a window that a mid flood stage partly drowns, levels it to the river (its 2nd
percentile) and resamples the 30 m posts onto the world grid with a little sub-post relief.
GLO-30 is a surface model, so riparian trees stand up as high ground.
Copernicus DEM: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018,
provided under COPERNICUS by the European Union and ESA; all rights reserved.

Imagery is Sentinel-2 cloudless 2016 by EOX IT Services GmbH (contains modified Copernicus
Sentinel data 2016), CC BY 4.0, stitched from zoom 14 tiles onto the same crop.

Rebuild both packs from the raw tiles in data/dem and data/s2:
    uv run --group bench python -m kipimo.world.terrain
"""

import functools

import numpy as np

from kipimo.config import CONFIG_DIR, ROOT
from kipimo.world.noise import fractal

TILE = "Copernicus_DSM_COG_10_S03_00_E040_00_DEM.tif"
PACK = CONFIG_DIR / "terrain" / "tana_garsen.npz"
IMAGERY = CONFIG_DIR / "terrain" / "tana_garsen_s2.jpg"
S2_ZOOM, S2_X, S2_Y = 14, (10017, 10021), (8293, 8298)  # web mercator tiles covering CROP
CORNER = (-2.0, 40.0)  # tile's north west corner, lat lon
CROP = (850, 1250, 380, 680)  # rows and cols of the tile kept
POST_M, POST_DEG = 30.9, 1 / 3600
STAGE_M, SHARE = 2.1, (0.25, 0.65)  # windows a 2.1 m stage floods by this share


def build() -> None:
    from PIL import Image

    r0, r1, c0, c1 = CROP
    dem = np.array(Image.open(ROOT / "data" / "dem" / TILE))[r0:r1, c0:c1]
    np.savez_compressed(PACK, dem_cm=np.round(dem * 100).astype(np.int16))
    build_imagery()


def build_imagery(per_post: int = 3) -> None:
    """Resample the mercator mosaic onto the crop's lat lon grid, 3 pixels per 30 m post."""
    from PIL import Image

    (x0, x1), (y0, y1), z = S2_X, S2_Y, S2_ZOOM
    mosaic = Image.new("RGB", (256 * (x1 - x0 + 1), 256 * (y1 - y0 + 1)))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = Image.open(ROOT / "data" / "s2" / f"{z}_{x}_{y}.jpg")
            mosaic.paste(tile, (256 * (x - x0), 256 * (y - y0)))
    r0, r1, c0, c1 = CROP
    k = np.arange(0.5, (r1 - r0) * per_post) / per_post
    lat = np.radians(CORNER[0] - (r0 + k) * POST_DEG)
    lon = CORNER[1] + (c0 + np.arange(0.5, (c1 - c0) * per_post) / per_post) * POST_DEG
    world = 256 * 2**z
    px = (lon + 180) / 360 * world - 256 * x0
    py = (1 - np.arcsinh(np.tan(lat)) / np.pi) / 2 * world - 256 * y0
    src = np.asarray(mosaic)
    out = src[np.round(py).astype(int)[:, None], np.round(px).astype(int)[None, :]]
    Image.fromarray(out).save(IMAGERY, quality=88, optimize=True)


@functools.cache
def _dem() -> np.ndarray:
    return np.load(PACK)["dem_cm"] / 100.0


@functools.cache
def _windows(k: int) -> tuple[tuple[int, int], ...]:
    dem, out = _dem(), []
    for r in range(0, dem.shape[0] - k, 4):
        for c in range(0, dem.shape[1] - k, 4):
            w = dem[r : r + k, c : c + k]
            if SHARE[0] <= (w - np.percentile(w, 2) < STAGE_M).mean() <= SHARE[1]:
                out.append((r, c))
    return tuple(out)


def window(rng: np.random.Generator, n: int, cell_m: float) -> tuple[np.ndarray, str, tuple]:
    """Elevation above the river in metres on an n x n grid, where it is, and its footprint as
    (left, top, right, bottom) fractions of the crop, to cut the same window from the imagery."""
    dem, step = _dem(), cell_m / POST_M
    wins = _windows(int(np.ceil(n * step)) + 1)
    r, c = wins[int(rng.integers(len(wins)))]
    u = np.arange(n) * step + step / 2
    i, f = np.floor(u).astype(int), u - np.floor(u)
    g = dem[r : r + i[-1] + 2, c : c + i[-1] + 2]
    rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
    z = rows[:, i] * (1 - f) + rows[:, i + 1] * f
    z = z - np.percentile(z, 2) + 0.15 * fractal(rng, n, base=32, octaves=2)
    lat = CORNER[0] - (CROP[0] + r + u[-1] / 2) * POST_DEG
    lon = CORNER[1] + (CROP[2] + c + u[-1] / 2) * POST_DEG
    h, w = dem.shape
    rect = (c / w, r / h, (c + n * step) / w, (r + n * step) / h)
    return z, f"Tana River near Garsen, {-lat:.3f} S {lon:.3f} E", rect


if __name__ == "__main__":
    build()
