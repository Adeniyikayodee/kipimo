import struct
import zlib
from pathlib import Path

import numpy as np

from kipimo.world.generate import World


def write_png(rgb: np.ndarray, path: Path) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def truth_rgb(world: World) -> np.ndarray:
    loss = world.loss
    rgb = np.empty((*loss.shape, 3))
    rgb[..., 0] = 60 + 190 * loss
    rgb[..., 1] = 150 - 90 * loss
    rgb[..., 2] = 50
    bare = world.parcel_map < 0
    rgb[bare] = (110, 100, 80)
    water = world.water[..., None]
    rgb = rgb * (1 - 0.6 * water) + np.array([40, 90, 200]) * 0.6 * water
    return rgb.clip(0, 255).astype(np.uint8)


def truth_png(world: World, path: Path, scale: int = 2) -> None:
    write_png(np.kron(truth_rgb(world), np.ones((scale, scale, 1), dtype=np.uint8)), path)


TILE_PX = 64
HEALTHY = np.array([0.24, 0.55, 0.20])
DAMAGED = np.array([0.60, 0.48, 0.25])
WATER = np.array([0.36, 0.40, 0.42])
SOIL = np.array([0.45, 0.38, 0.30])


def tile_cells(world: World, x: float, y: float, half_m: float, px: int = TILE_PX) -> np.ndarray:
    """Flat cell index under each pixel of a nadir camera tile, -1 outside the world."""
    n, c = world.size, world.cell_m
    u = (np.arange(px) + 0.5) / px * 2 * half_m - half_m
    cols = np.floor((x + u) / c).astype(int)
    rows = np.floor((y + u) / c).astype(int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
    return np.where(ok, rr * n + cc, -1).astype(np.int32)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return img
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, pad)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, out)


def render_tile(
    world: World, x: float, y: float, half_m: float, q: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic RGB camera tile in 0..1 and the cell index under each pixel.

    Crops shade from green to straw with loss and carry row texture; flood water is a flat
    grey sheet; lighting, blur and sensor noise all vary, and worsen as quality q drops.
    """
    ids = tile_cells(world, x, y, half_m)
    px = ids.shape[0]
    flat = np.maximum(ids, 0)
    loss = world.loss.reshape(-1)[flat][..., None]
    water = world.water.reshape(-1)[flat][..., None]
    farmed = (world.parcel_map.reshape(-1)[flat] >= 0)[..., None]
    rows = np.sin(np.linspace(0, px / 2.5, px))[None, :, None] * 0.04
    crop = HEALTHY * (1 - loss) + DAMAGED * loss + rows + rng.normal(0, 0.03, (px, px, 1))
    ground = np.where(farmed, crop, SOIL + rng.normal(0, 0.04, (px, px, 1)))
    w = np.clip(water * 1.15, 0, 1)
    img = ground * (1 - w) + (WATER + rng.normal(0, 0.008, (px, px, 1))) * w
    img = img * rng.uniform(0.85, 1.15) + rng.normal(0, 0.02, 3)
    img = _blur(img, 2.5 * (1 - q))
    img = img + rng.normal(0, 0.01 + 0.04 * (1 - q), img.shape)
    img[ids < 0] = 0
    return np.clip(img, 0, 1).astype(np.float32), ids


def tile_truth(world: World, ids: np.ndarray) -> np.ndarray:
    """Per pixel (water, loss) targets, shape (2, H, W)."""
    flat = np.maximum(ids, 0)
    return np.stack([world.water.reshape(-1)[flat], world.loss.reshape(-1)[flat]]).astype(
        np.float32
    )
