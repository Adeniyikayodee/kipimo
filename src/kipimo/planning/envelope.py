import math
from dataclasses import dataclass

from kipimo.config import EnvelopeCfg
from kipimo.planning.lawnmower import farm_bbox
from kipimo.world.generate import World

Pt = tuple[float, float]


def seg_dist(p: Pt, a: Pt, b: Pt) -> tuple[float, Pt]:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    q = (ax + t * dx, ay + t * dy)
    return math.dist(p, q), q


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str


class Envelope:
    """Hard limits the planner and the agent cannot override."""

    def __init__(self, world: World, cfg: EnvelopeCfg, home: Pt | None = None):
        self.cfg = cfg
        x0, y0, x1, y1 = farm_bbox(world)
        if home:  # the pad is always inside the fence
            x0, y0, x1, y1 = min(x0, home[0]), min(y0, home[1]), max(x1, home[0]), max(y1, home[1])
        m, span = cfg.geofence_margin_m, world.size * world.cell_m
        self.fence = (max(0.0, x0 - m), max(0.0, y0 - m), min(span, x1 + m), min(span, y1 + m))
        self.zones = [(z.x, z.y, z.r) for z in cfg.no_fly]

    def inside_fence(self, p: Pt) -> bool:
        x0, y0, x1, y1 = self.fence
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def segment_ok(self, a: Pt, b: Pt) -> bool:
        m = self.cfg.zone_margin_m
        return all(seg_dist((x, y), a, b)[0] > r + m for x, y, r in self.zones)

    def route(self, a: Pt, b: Pt, depth: int = 0) -> list[Pt]:
        """Waypoints from a to b that skirt no fly zones. Excludes a, includes b."""
        m = self.cfg.zone_margin_m
        for x, y, r in self.zones:
            d, q = seg_dist((x, y), a, b)
            if d <= r + m and depth < 4:
                ux, uy = q[0] - x, q[1] - y
                if math.hypot(ux, uy) < 1e-6:
                    ux, uy = -(b[1] - a[1]), b[0] - a[0]
                k = (r + 2 * m) / math.hypot(ux, uy)
                via = (x + ux * k, y + uy * k)
                return self.route(a, via, depth + 1) + self.route(via, b, depth + 1)
        return [b]

    def check_point(self, p: Pt) -> Verdict:
        if not self.inside_fence(p):
            return Verdict(False, "outside_geofence")
        if any(math.dist(p, (x, y)) <= r for x, y, r in self.zones):
            return Verdict(False, "inside_no_fly_zone")
        return Verdict(True, "ok")

    def check_battery(self, battery_wh: float, need_wh: float, reserve_wh: float) -> Verdict:
        if battery_wh - need_wh < reserve_wh:
            return Verdict(
                False, f"battery_reserve: need {need_wh:.1f} Wh of {battery_wh - reserve_wh:.1f}"
            )
        return Verdict(True, "ok")
