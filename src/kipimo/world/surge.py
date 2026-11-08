"""A river breach mid-mission: the lowest fields that are still dry flood, and a coarse radar
pass reports it.

The world changes for real, so the drone's camera sees the new water and damage when it gets
there. The radar pass is the first, blurry view of it, and it is what makes the planner look.
"""

import numpy as np

from kipimo.types import Observation, Pose
from kipimo.world.generate import World, response_map

DRY = 0.15  # a parcel with less true loss than this still counts as dry
FIELDS = 3  # how many fields one breach reaches
RADAR_QUALITY = 0.2  # a coarse satellite radar pass, far below the drone camera


def breach(world: World, skip: set[str] = frozenset()) -> list[str]:
    """Flood the lowest dry parcels in place. Returns their ids, lowest first."""
    elev, loss, water = world.elevation.flat, world.loss.flat, world.water.flat  # writes through
    dry = [p for p in world.parcels if p.id not in skip and loss[p.cells].mean() < DRY]
    hit = sorted(dry, key=lambda p: float(np.median(elev[p.cells])))[:FIELDS]
    resp = response_map(world)
    for p in hit:
        water[p.cells] = np.maximum(water[p.cells], 0.9)
        loss[p.cells] = np.maximum(loss[p.cells], 0.9 * resp[p.cells])
    return [p.id for p in hit]


def radar_pass(world: World, pids: list[str], t: float, rng: np.random.Generator) -> Observation:
    """What a satellite radar sees over the breached fields: right on average, very noisy."""
    cells = np.concatenate([world.parcel(pid).cells for pid in pids]).astype(np.int32)
    noise = rng.normal(0, 0.15, (2, len(cells)))
    return Observation(
        t=t,
        pose=Pose(-1.0, -1.0, 0.0),
        cell_ids=cells,
        water_frac=np.clip(world.water.flat[cells] + noise[0], 0, 1).astype(np.float32),
        crop_loss=np.clip(world.loss.flat[cells] + noise[1], 0, 1).astype(np.float32),
        quality=RADAR_QUALITY,
    )
