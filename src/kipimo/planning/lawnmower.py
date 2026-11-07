import numpy as np

from kipimo.world.generate import World


def farm_bbox(world: World) -> tuple[float, float, float, float]:
    rows, cols = np.nonzero(world.parcel_map >= 0)
    c = world.cell_m
    return cols.min() * c, rows.min() * c, (cols.max() + 1) * c, (rows.max() + 1) * c


def plan(world: World, half_m: float, overlap: float = 0.1) -> list[tuple[float, float]]:
    """Boustrophedon lanes along x, spaced by the camera swath."""
    if overlap >= 1:
        raise ValueError(f"overlap {overlap} leaves no lane spacing")
    x0, y0, x1, y1 = farm_bbox(world)
    step = 2 * half_m * (1 - overlap)
    ys = np.arange(y0 + half_m, y1 - half_m + step, step)
    ys[-1] = min(ys[-1], y1 - half_m)
    wps = []
    for i, y in enumerate(ys):
        a, b = (x0 + half_m, x1 - half_m) if i % 2 == 0 else (x1 - half_m, x0 + half_m)
        wps += [(float(a), float(y)), (float(b), float(y))]
    return wps
