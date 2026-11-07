from collections.abc import Callable
from typing import Protocol

import numpy as np

from kipimo.robot.kinematics import Robot
from kipimo.types import Observation
from kipimo.world.generate import World


class SensorSource(Protocol):
    degraded: bool

    def observe(self, robot: Robot, t: float) -> Observation: ...


def footprint_cells(world: World, x: float, y: float, half_m: float) -> np.ndarray:
    n, c = world.size, world.cell_m
    r0, r1 = (int(min(max(v, 0), n)) for v in ((y - half_m) // c, (y + half_m) // c + 1))
    c0, c1 = (int(min(max(v, 0), n)) for v in ((x - half_m) // c, (x + half_m) // c + 1))
    return (np.arange(r0, r1)[:, None] * n + np.arange(c0, c1)).ravel().astype(np.int32)


def quality(altitude_m: float, speed_mps: float) -> float:
    q_alt = min(max(1.0 - (altitude_m - 20.0) / 120.0, 0.2), 1.0)
    q_blur = min(max(1.0 - 0.04 * max(0.0, speed_mps - 6.0), 0.4), 1.0)
    return float(q_alt * q_blur)


DEGRADED = 0.3  # camera quality at the default flight settings when dirty or fogged


def camera_quality(robot: Robot, degraded: bool) -> float:
    """Quality now. A degraded camera scales with how the drone flies: lower and slower helps."""
    q = quality(robot.altitude_m, robot.speed_mps)
    if not degraded:
        return q
    return DEGRADED * (q / quality(robot.cfg.altitude_m, robot.cfg.speed_mps))


class SimSensor:
    def __init__(self, world: World, rng: np.random.Generator):
        self.world = world
        self.rng = rng
        self.degraded = False

    def observe(self, robot: Robot, t: float) -> Observation:
        p = robot.pose
        ids = footprint_cells(self.world, p.x, p.y, robot.footprint_half_m())
        q = camera_quality(robot, self.degraded)
        sigma = 0.03 + 0.15 * (1 - q)
        water = self.world.water.reshape(-1)[ids]
        loss = self.world.loss.reshape(-1)[ids]
        noise = self.rng.normal(0, sigma, (2, len(ids)))
        return Observation(
            t=t,
            pose=p,
            cell_ids=ids,
            water_frac=np.clip(water + noise[0], 0, 1).astype(np.float32),
            crop_loss=np.clip(loss + noise[1], 0, 1).astype(np.float32),
            quality=q,
        )


class PerceptionSensor:
    """Renders a camera tile at the pose and reads it with a perception backend.

    The backend maps an HxWx3 image to per pixel (water, loss); pixels are averaged per cell.
    Crops under water cannot be seen, so the submerged share of a cell takes its loss from the
    policy's crop damage curve applied to the measured water, and the dry share from the camera.
    """

    def __init__(self, world: World, rng: np.random.Generator, backend: Callable, name: str):
        from kipimo.world.generate import response_map

        self.response = response_map(world)
        self.world = world
        self.rng = rng
        self.backend = backend
        self.name = name
        self.degraded = False

    def observe(self, robot: Robot, t: float) -> Observation:
        from kipimo.world.render import render_tile

        p = robot.pose
        q = camera_quality(robot, self.degraded)
        img, ids = render_tile(self.world, p.x, p.y, robot.footprint_half_m(), q, self.rng)
        pred = self.backend(img).reshape(2, -1)
        flat = ids.reshape(-1)
        ok = flat >= 0
        cells, inv = np.unique(flat[ok], return_inverse=True)
        cnt = np.bincount(inv)
        water = np.bincount(inv, pred[0, ok]) / cnt
        loss = np.bincount(inv, pred[1, ok]) / cnt
        keep = cnt >= 3
        cells, water, loss = cells[keep], np.clip(water[keep], 0, 1), loss[keep]
        submerged = self.response[cells] * water**0.8
        loss = np.clip(water * submerged + (1 - water) * loss, 0, 1)
        return Observation(
            t=t,
            pose=p,
            cell_ids=cells.astype(np.int32),
            water_frac=water.astype(np.float32),
            crop_loss=loss.astype(np.float32),
            quality=q,
        )
