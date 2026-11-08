import math
from dataclasses import dataclass

import numpy as np

from kipimo.belief.parcels import Z, parcel_stats, prior_width, width_from_n
from kipimo.payout import triggers
from kipimo.planning.envelope import Envelope
from kipimo.planning.lawnmower import farm_bbox
from kipimo.types import Objective, Parcel
from kipimo.world.generate import World

Pt = tuple[float, float]


@dataclass(frozen=True)
class Lane:
    parcel_id: str
    a: Pt
    b: Pt
    local: np.ndarray  # indices into parcel.cells under the swath

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)


def build_lanes(world: World, half_m: float, spacing: float, envelope: Envelope) -> list[Lane]:
    """Per parcel lane segments on a shared farm wide row grid, so neighbouring lanes chain."""
    n, c = world.size, world.cell_m
    _, fy0, _, fy1 = farm_bbox(world)
    grid = np.arange(fy0 + half_m, fy1, spacing)
    lanes = []
    for p in world.parcels:
        rows, cols = np.divmod(p.cells, n)
        yc, xc = (rows + 0.5) * c, (cols + 0.5) * c
        ys = [y for y in grid if rows.min() * c <= y < (rows.max() + 1) * c]
        if not ys:
            ys = [float(yc.mean())]
        for y in ys:
            band = cols[rows == rows[np.abs(rows - int(y // c)).argmin()]]  # nearest row with cells
            a = ((band.min() + 0.5) * c, float(y))
            b = ((band.max() + 0.5) * c, float(y))
            if not envelope.segment_ok(a, b):
                continue
            under = (np.abs(yc - y) <= half_m) & (xc >= a[0] - half_m) & (xc <= b[0] + half_m)
            lanes.append(Lane(p.id, a, b, np.flatnonzero(under)))
    return lanes


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def money_at_risk(parcel: Parcel, mean: float, width: float, w0: float, target: float) -> float:
    """Expected payout error plus payable money still blocked by the confidence gate."""
    conf = 1 - width / w0
    sd = max(width / Z, 1e-6)
    risk, prev = 0.0, 0.0
    for b in sorted(triggers.boundaries(parcel.crop)):
        pays = triggers.score(parcel, b).tier
        risk += (pays - prev) * _phi(-abs(mean - b) / sd)
        prev = pays
    blocked = triggers.score(parcel, mean).tier * max(0.0, target - conf) / target
    return parcel.sum_insured * (risk + blocked)


def resolved(parcel: Parcel, mean: float, width: float, w0: float, target: float) -> bool:
    """Settled for payout: passes the gate, or the whole interval sits below every trigger."""
    return 1 - width / w0 >= target or mean + width / 2 < min(triggers.boundaries(parcel.crop))


class GridForecast:
    """Planning view of the grid belief: a lane only informs its own parcel."""

    def __init__(self, planner: "InfoGainPlanner", belief, objective: Objective, q: float):
        self.parcels, self.cfg = planner.parcels, belief.cfg
        self.w0, self.target = prior_width(belief.cfg), objective.target_confidence
        self.dn = planner.evidence_per_pass(q)
        self.state = {}
        for pid in objective.parcel_ids:
            p = self.parcels[pid]
            m, w = parcel_stats(belief, p.cells)
            n = belief.n[p.cells].copy()
            self.state[pid] = [m, n, money_at_risk(p, m, w, self.w0, self.target)]

    def settled(self) -> bool:
        return all(
            1 - width_from_n(st[1], self.cfg) / self.w0 >= self.target for st in self.state.values()
        )

    def open(self, lane: Lane, until_target: bool) -> bool:
        st = self.state.get(lane.parcel_id)
        if st is None or st[2] <= 0:
            return False
        p, w = self.parcels[lane.parcel_id], width_from_n(st[1], self.cfg)
        return not (until_target and resolved(p, st[0], w, self.w0, self.target))

    def _after(self, lane: Lane):
        st = self.state[lane.parcel_id]
        n = st[1].copy()
        n[lane.local] += self.dn
        p = self.parcels[lane.parcel_id]
        return n, money_at_risk(p, st[0], width_from_n(n, self.cfg), self.w0, self.target)

    def gain(self, lane: Lane) -> float:
        return self.state[lane.parcel_id][2] - self._after(lane)[1]

    def commit(self, lane: Lane) -> None:
        st = self.state[lane.parcel_id]
        st[1], st[2] = self._after(lane)


class GPForecast:
    """Planning view of the GP belief: a lane also informs every parcel through the scene terms.

    A lane touches 4 of the D weights (3 scene, 1 parcel), so its effect on every parcel variance
    is a rank 4 Woodbury update of the weight covariance.
    """

    def __init__(self, planner: "InfoGainPlanner", belief, objective: Objective, q: float):
        from kipimo.belief.gp import sigma_meas

        b = self.b = belief
        self.parcels = planner.parcels
        self.w0, self.target = prior_width(b.cfg), objective.target_confidence
        looks = 2 * planner.half_m / (planner.speed * planner.cfg.sim.dt)
        self.add, self.add_sb = looks / sigma_meas(q) ** 2, looks / sigma_meas(q)
        self.rows = {pid: k for k, pid in enumerate(objective.parcel_ids)}
        self.ps = [self.parcels[pid] for pid in objective.parcel_ids]
        self.i = np.array([b.index[pid] for pid in objective.parcel_ids])
        self.n = b.ncell[self.i]
        self.lam, self.sigma = b.lam.copy(), b.sigma.copy()
        self.C, self.V, self.B = b.C[self.i].copy(), b.V[self.i].copy(), b.B[self.i].copy()
        self.unseen = b.unseen[self.i].copy()
        self.prec, self.sb = b.prec.copy(), b.sb.copy()
        self.mean = b.parcel_moments(tuple(self.ps))[0]
        self.width = b.parcel_moments(tuple(self.ps))[1]
        self.risk = self._risk(self.width)

    def _risk(self, widths: np.ndarray) -> np.ndarray:
        return np.array(
            [
                money_at_risk(p, m, w, self.w0, self.target)
                for p, m, w in zip(self.ps, self.mean.tolist(), widths.tolist(), strict=True)
            ]
        )

    def settled(self) -> bool:
        return bool((1 - self.width / self.w0 >= self.target).all())

    def open(self, lane: Lane, until_target: bool) -> bool:
        k = self.rows.get(lane.parcel_id)
        if k is None or self.risk.sum() <= 0:
            return False
        p = self.ps[k]
        return not (until_target and resolved(p, self.mean[k], self.width[k], self.w0, self.target))

    def _effect(self, lane: Lane):
        b, k = self.b, self.rows[lane.parcel_id]
        cells = self.ps[k].cells[lane.local]
        p0, sb0 = self.prec[cells], self.sb[cells]
        p1, sb1 = p0 + self.add, sb0 + self.add_sb
        seen0, pv0, s0, g0 = b.terms(p0)
        _, pv1, s1, g1 = b.terms(p1)
        idx = np.array([0, 1, 2, 3 + self.i[k]])
        f = np.column_stack(
            [np.ones(len(cells)), b.prior[cells], b.resp[cells], np.ones(len(cells))]
        )
        dlam = f.T @ ((g1 - g0)[:, None] * f)
        sx = self.sigma[:, idx]
        x = np.linalg.solve(np.eye(4) + dlam @ sx[idx], dlam)  # sigma' = sigma - sx x sx^T
        A = (b.S[self.i] - self.C) / self.n[:, None]
        A[k, idx] -= (s1 - s0) @ f / self.n[k]
        B = A @ sx
        var = np.einsum("pd,de,pe->p", A, self.sigma, A) - np.einsum("pa,ab,pb->p", B, x, B)
        unseen, V, Bs = self.unseen.copy(), self.V.copy(), self.B.copy()
        unseen[k] -= (~seen0).sum()
        V[k] += (pv1 - pv0).sum()
        Bs[k] += (b.bias(s1, sb1, p1) - b.bias(s0, sb0, p0)).sum()
        var += b.unseen_var(self.i, unseen) + V / self.n**2 + (Bs / self.n) ** 2
        return cells, idx, dlam, s1 - s0, f, unseen, V, Bs, Z * np.sqrt(var)

    def gain(self, lane: Lane) -> float:
        return float(self.risk.sum() - self._risk(self._effect(lane)[-1]).sum())

    def commit(self, lane: Lane) -> None:
        cells, idx, dlam, ds, f, self.unseen, self.V, self.B, self.width = self._effect(lane)
        k = self.rows[lane.parcel_id]
        self.lam[np.ix_(idx, idx)] += dlam
        self.sigma = np.linalg.inv(self.lam)
        self.C[k, idx] += ds @ f
        self.prec[cells] += self.add
        self.sb[cells] += self.add_sb
        self.risk = self._risk(self.width)


class InfoGainPlanner:
    def __init__(self, world: World, envelope: Envelope, half_m: float, ep_cfg):
        self.world = world
        self.cfg = ep_cfg
        self.half_m = half_m
        self.envelope, self.speed = envelope, ep_cfg.robot.speed_mps
        self.lanes = build_lanes(world, half_m, 2 * half_m * 0.9, envelope)
        self.parcels = {p.id: p for p in world.parcels}

    def retune(self, half_m: float, speed: float) -> None:
        """Lanes and evidence follow the camera footprint and speed actually flown."""
        self.half_m, self.speed = half_m, speed
        self.lanes = build_lanes(self.world, half_m, 2 * half_m * 0.9, self.envelope)

    def evidence_per_pass(self, q: float) -> float:
        return self.cfg.belief.k * q * 2 * self.half_m / (self.speed * self.cfg.sim.dt)

    def forecast(self, belief, objective: Objective, q: float):
        cls = GPForecast if hasattr(belief, "terms") else GridForecast
        return cls(self, belief, objective, q)

    def risk_table(self, belief, objective: Objective) -> dict[str, float]:
        w0 = prior_width(belief.cfg)
        ps = tuple(self.parcels[pid] for pid in objective.parcel_ids)
        means, widths = belief.parcel_moments(ps)
        return {
            p.id: money_at_risk(p, m, w, w0, objective.target_confidence)
            for p, m, w in zip(ps, means.tolist(), widths.tolist(), strict=True)
        }

    def plan_batch(self, belief, pose: Pt, objective: Objective, q: float, until_target: bool):
        """Choose a batch of lanes on predicted belief, then order them into a tour from pose.

        Each candidate is charged its length plus its distance to the nearest already chosen lane,
        so the batch stays spatially compact instead of zig-zagging across the farm.
        """
        fc = self.forecast(belief, objective, q)
        anchors = [pose]
        chosen: list[tuple[Lane, float]] = []
        uses: dict[int, int] = {}
        floor = 0.0 if until_target else self.cfg.planner.min_gain_per_m
        while not (until_target and fc.settled()):
            best = None
            for li, lane in enumerate(self.lanes):
                if uses.get(li, 0) >= 2 or not fc.open(lane, until_target):
                    continue
                gain = fc.gain(lane)
                hop = min(min(math.dist(lane.a, a), math.dist(lane.b, a)) for a in anchors)
                rate = gain / (lane.length + hop + self.cfg.planner.fixed_cost_m)
                if best is None or rate > best[0]:
                    best = (rate, gain, li)
            if best is None or best[1] <= 0 or best[0] < floor:
                break
            _, gain, li = best
            lane = self.lanes[li]
            fc.commit(lane)
            uses[li] = uses.get(li, 0) + 1
            anchors += [lane.a, lane.b]
            chosen.append((lane, gain))
        return _tour(chosen, pose)


def _tour(chosen: list[tuple["Lane", float]], pose: Pt) -> list[tuple["Lane", Pt, Pt, float]]:
    """Boustrophedon order over the chosen lanes: row by row from the nearer edge, alternating."""
    if not chosen:
        return []
    rows: dict[float, list[tuple[Lane, float]]] = {}
    for lane, gain in chosen:
        rows.setdefault(round(lane.a[1], 3), []).append((lane, gain))
    ys = sorted(rows)
    if abs(pose[1] - ys[-1]) < abs(pose[1] - ys[0]):
        ys.reverse()
    first = rows[ys[0]]
    lo = min(min(ln.a[0], ln.b[0]) for ln, _ in first)
    hi = max(max(ln.a[0], ln.b[0]) for ln, _ in first)
    rightward = abs(pose[0] - lo) <= abs(pose[0] - hi)
    out = []
    for y in ys:
        row = sorted(rows[y], key=lambda t: min(t[0].a[0], t[0].b[0]), reverse=not rightward)
        for lane, gain in row:
            a, b = sorted((lane.a, lane.b), reverse=not rightward)
            out.append((lane, a, b, gain))
        rightward = not rightward
    return out


def survey(
    ep,
    planner: InfoGainPlanner,
    envelope: Envelope,
    objective: Objective,
    until_target: bool = False,
    on_degrade=None,
    stop_on_camera: bool = False,
) -> str:
    """Closed loop: plan a batch of lanes, fly it, update belief, replan. Returns the stop reason.

    until_target: keep flying until every objective parcel reaches target confidence,
    ignoring the diminishing returns floor. Used to settle payable parcels.
    stop_on_camera: return "camera_changed" when the camera degrades or recovers, so an agent
    decides how to respond instead of the rules.
    """
    from kipimo.robot.sensors import camera_quality

    r = ep.robot
    start = r.distance_m
    for _ in range(24):
        degraded = ep.sensor.degraded
        q = camera_quality(r, degraded)
        here = (r.pose.x, r.pose.y)
        batch = planner.plan_batch(ep.belief, here, objective, q, until_target)
        if not batch:
            return "objective_met" if until_target else "diminishing_returns"
        ep.bus.emit(
            "PLAN_BATCH",
            lanes=len(batch),
            parcels=sorted({ln.parcel_id for ln, *_ in batch}),
            path=[[a, b] for _, a, b, _ in batch],
        )
        version = ep.version
        for lane, a, b, gain in batch:
            if ep.sensor.degraded != degraded:
                if on_degrade:
                    on_degrade(ep.sensor.degraded)
                q = round(camera_quality(r, ep.sensor.degraded), 3)  # what the next plan sees
                ep.bus.emit("DEGRADED_SENSOR_REPLAN", degraded=ep.sensor.degraded, quality=q)
                if stop_on_camera:
                    return "camera_changed"
                break
            here = (r.pose.x, r.pose.y)
            path = envelope.route(here, a) + envelope.route(a, b)
            legs = [here, *path]
            leg_m = sum(math.dist(u, v) for u, v in zip(legs, legs[1:], strict=False))
            if r.distance_m - start + leg_m > objective.max_distance_m:
                return "distance_budget"
            need = sum(ep.energy_wh(u, v) for u, v in zip(legs, legs[1:], strict=False))
            verdict = envelope.check_battery(
                r.battery_wh, need + ep.energy_wh(b, ep.home), r.cfg.reserve_wh
            )
            if not verdict.ok:
                ep.bus.emit("RETURN_TO_HOME", reason=verdict.reason, battery_wh=r.battery_wh)
                return "battery_reserve"
            ep.bus.emit("PLAN_LANE", parcel=lane.parcel_id, a=a, b=b, gain=gain)
            flown = all(
                ep.fly_to(x, y, stop_on_camera=stop_on_camera, stop_on_change=True) for x, y in path
            )
            if not flown and ep.version != version:  # the world changed mid-lane: plan again
                ep.bus.emit("SURGE_REPLAN", parcels=sorted(ep.surged))
                break
            if not flown:
                if stop_on_camera and ep.sensor.degraded != degraded:
                    q = round(camera_quality(r, ep.sensor.degraded), 3)
                    ep.bus.emit("DEGRADED_SENSOR_REPLAN", degraded=ep.sensor.degraded, quality=q)
                    return "camera_changed"
                return "leg_aborted"
    return "replan_limit"
