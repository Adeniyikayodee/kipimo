"""A multirotor as a point mass: thrust tilts to accelerate, the wind carries it, rotors draw power.

Power is the rotary wing model of Zeng, Xu and Zhang (2019): blade profile, induced and parasite
terms, scaled so the airframe hovers on `hover_w`. Altitude is held, so climbs are not costed.
"""

import math
from dataclasses import dataclass, field

from kipimo.config import RobotCfg
from kipimo.types import Pose

ALTITUDE_M = (20.0, 60.0)
SPEED_MPS = (3.0, 10.0)
G = 9.81
SUBSTEPS = 10
ACCEPT_M = 1.0
# Zeng et al. 2019, Table I: profile and induced hover power W, tip and induced speed m/s, drag
P0, PI, U_TIP, V0, DRAG = 79.86, 88.63, 120.0, 4.03, 0.5 * 0.6 * 1.225 * 0.05 * 0.503

Vec = tuple[float, float]


def power_w(cfg: RobotCfg, v: float) -> float:
    """Rotor power at airspeed v."""
    induced = math.sqrt(math.sqrt(1 + v**4 / (4 * V0**4)) - v**2 / (2 * V0**2))
    return (P0 * (1 + 3 * v**2 / U_TIP**2) + PI * induced + DRAG * v**3) * cfg.hover_w / (P0 + PI)


def cruise(cfg: RobotCfg, speed: float, u: Vec, wind: Vec) -> tuple[float, float]:
    """Ground and air speed holding `speed` along unit u. Past max airspeed, the wind triangle."""
    air = math.hypot(speed * u[0] - wind[0], speed * u[1] - wind[1])
    if air <= cfg.max_airspeed:
        return speed, air
    wu = wind[0] * u[0] + wind[1] * u[1]
    disc = cfg.max_airspeed**2 - wind[0] ** 2 - wind[1] ** 2 + wu**2
    return (max(0.0, wu + math.sqrt(disc)) if disc >= 0 else 0.0), cfg.max_airspeed


def _clip(x: float, y: float, lim: float) -> Vec:
    m = math.hypot(x, y)
    return (x, y) if m <= lim else (x * lim / m, y * lim / m)


@dataclass
class Robot:
    cfg: RobotCfg
    pose: Pose
    battery_wh: float = field(init=False)
    distance_m: float = 0.0
    altitude_m: float = field(init=False)
    speed_mps: float = field(init=False)
    air: Vec = (0.0, 0.0)
    wind: Vec = (0.0, 0.0)
    power: float = 0.0

    def __post_init__(self):
        self.battery_wh = self.cfg.battery_wh
        self.altitude_m = self.cfg.altitude_m
        self.speed_mps = self.cfg.speed_mps

    def step(self, tx: float, ty: float, dt: float, wind: Vec = (0.0, 0.0)) -> bool:
        """Track (tx, ty) at cruise ground speed, braking onto it. True once within ACCEPT_M."""
        a_max = G * math.tan(math.radians(self.cfg.tilt_deg))
        h, (x, y), (ax, ay) = dt / SUBSTEPS, (self.pose.x, self.pose.y), self.air
        arrived, wh = False, 0.0
        for _ in range(SUBSTEPS):
            dx, dy = tx - x, ty - y
            d = math.hypot(dx, dy)
            arrived |= d < ACCEPT_M
            v = 0.0 if arrived else min(self.speed_mps, math.sqrt(a_max * d))
            gx, gy = (dx / d * v, dy / d * v) if d > 1e-9 else (0.0, 0.0)
            cx, cy = _clip(gx - wind[0], gy - wind[1], self.cfg.max_airspeed)
            sx, sy = _clip(cx - ax, cy - ay, a_max * h)
            ax, ay = ax + sx, ay + sy
            mx, my = (ax + wind[0]) * h, (ay + wind[1]) * h
            x, y = x + mx, y + my
            self.distance_m += math.hypot(mx, my)
            wh += power_w(self.cfg, math.hypot(ax, ay)) * h / 3600
        yaw = math.atan2(ay, ax) if math.hypot(ax, ay) > 0.5 else self.pose.yaw
        self.pose, self.air, self.wind = Pose(x, y, yaw), (ax, ay), wind
        self.power = wh * 3600 / dt
        self.battery_wh = max(0.0, self.battery_wh - wh)
        return arrived

    def leg_wh(self, a: Vec, b: Vec, wind: Vec, settle_s: float = 0.0) -> float:
        d = math.dist(a, b)
        if d < 1e-9:
            return power_w(self.cfg, math.hypot(*wind)) * settle_s / 3600
        g, air = cruise(self.cfg, self.speed_mps, ((b[0] - a[0]) / d, (b[1] - a[1]) / d), wind)
        return power_w(self.cfg, air) * (d / max(g, 0.1) + settle_s) / 3600

    def wh_per_m(self, wind_mps: float = 0.0) -> float:
        """Energy per metre flown straight into a headwind, the worst heading."""
        g, air = cruise(self.cfg, self.speed_mps, (1.0, 0.0), (-wind_mps, 0.0))
        return power_w(self.cfg, air) / max(g, 0.1) / 3600

    def set_flight(self, altitude_m: float, speed_mps: float) -> tuple[float, float]:
        """Clamp to the airframe's limits and apply. Returns what was set."""
        self.altitude_m = min(max(altitude_m, ALTITUDE_M[0]), ALTITUDE_M[1])
        self.speed_mps = min(max(speed_mps, SPEED_MPS[0]), SPEED_MPS[1])
        return self.altitude_m, self.speed_mps

    def footprint_half_m(self) -> float:
        return self.altitude_m * math.tan(math.radians(self.cfg.fov_deg) / 2)
