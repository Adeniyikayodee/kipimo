import math
from pathlib import Path

import numpy as np

from kipimo.belief.gp import GPBelief
from kipimo.belief.grid import BeliefGrid
from kipimo.belief.parcels import aggregate
from kipimo.bus import Bus
from kipimo.config import Config
from kipimo.payout import triggers
from kipimo.planning.envelope import Envelope
from kipimo.robot.kinematics import Robot
from kipimo.robot.sensors import PerceptionSensor, SensorSource, SimSensor
from kipimo.types import Observation, Pose
from kipimo.weather import forecast
from kipimo.weather.wind import Wind
from kipimo.world.generate import World, generate, parcel_truth


def make_sensor(kind: str, world: World, rng: np.random.Generator) -> SensorSource:
    if kind == "sim":
        return SimSensor(world, rng)
    if kind == "classical":
        from kipimo.perception import classical

        return PerceptionSensor(world, rng, classical.predict, "classical")
    if kind == "onnx":
        from kipimo.perception.infer import OnnxPerception

        return PerceptionSensor(world, rng, OnnxPerception(), "onnx")
    raise ValueError(f"unknown sensor {kind}")


class Episode:
    def __init__(self, cfg: Config, bus: Bus | None = None, out: Path | None = None):
        self.cfg = cfg
        self.bus = bus or Bus(out)
        self.rng = np.random.default_rng(cfg.seed)
        self.world: World = generate(cfg.seed, cfg.world)
        home = 2.0 * cfg.world.cell_m
        self.home = (home, home)
        self.robot = Robot(cfg.robot, Pose(home, home, 0.0))
        self.env = Envelope(self.world, cfg.envelope, self.home)
        self.down = False  # forced landing: the airframe never moves again
        self.live_time = False  # the dashboard sets it: sim time runs while an agent thinks
        self.sensor: SensorSource = make_sensor(cfg.sim.sensor, self.world, self.rng.spawn(1)[0])
        self.belief = (
            GPBelief(self.world, cfg.belief)
            if cfg.belief.model == "gp"
            else BeliefGrid(self.world.prior, cfg.belief)
        )
        self.observers: list = [self.belief.update]
        self.tickers: list = []
        self.weather = forecast.make(cfg.seed, cfg.weather)
        w = cfg.weather
        mean = self.weather.wind if self.weather else (lambda t: w.breeze)
        self.wind = Wind(cfg.seed, mean, w.gust, w.gust_s)

    def observe(self) -> Observation:
        obs = self.sensor.observe(self.robot, self.bus.t)
        for fn in self.observers:
            fn(obs)
        r = self.robot
        self.bus.emit(
            "OBSERVATION",
            pose=r.pose,
            alt=r.altitude_m,
            battery_wh=r.battery_wh,
            distance_m=r.distance_m,
            wind=r.wind,
            power_w=r.power,
            n_cells=len(obs.cell_ids),
            quality=obs.quality,
            loss_mean=float(obs.crop_loss.mean()) if obs.crop_loss.size else None,
        )
        return obs

    def no_go(self) -> str:
        w, p = self.weather, self.robot.pose
        if w and w.wind(self.bus.t) > w.limit:
            return "wind"
        return "" if self.env.inside_fence((p.x, p.y)) else "geofence"

    def fly_to(self, x: float, y: float, abort: bool = True) -> bool:
        """Fly to a waypoint, observing every tick. False if it stopped short.

        abort: stop on wind over the limit or a geofence breach. Homeward legs pass False.
        """
        dt = self.cfg.sim.dt
        while not self.down:
            if abort and (why := self.no_go()):
                self.bus.emit("LEG_ABORTED", reason=why, pose=self.robot.pose)
                return False
            arrived = self.robot.step(x, y, dt, self.wind.step(self.bus.t, dt))
            self.bus.t += dt
            self.observe()
            for fn in self.tickers:
                fn()
            if self.robot.battery_wh <= 0:
                self.down = True
                self.bus.emit("FORCED_LANDING", pose=self.robot.pose)
            elif arrived:
                return True
        return False

    def hold(self) -> None:
        """One tick while an agent thinks: the drone hovers where it is, or waits on the pad."""
        dt, r = self.cfg.sim.dt, self.robot
        if not self.down and math.dist((r.pose.x, r.pose.y), self.home) >= 5:
            r.step(r.pose.x, r.pose.y, dt, self.wind.step(self.bus.t, dt))  # hover costs power
            if r.battery_wh <= 0:
                self.down = True
                self.bus.emit("FORCED_LANDING", pose=r.pose)
        self.bus.t += dt
        for fn in self.tickers:
            fn()

    def beliefs(self):
        return aggregate(self.belief, self.world.parcels)

    def emit_beliefs(self) -> None:
        self.bus.emit("BELIEF", parcels=list(self.beliefs().values()))

    def energy_wh(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        t, w = self.bus.t, self.weather
        s = self.wind.mean_mps(t)
        if w:  # plan for the windier of now and next hour's 90th percentile forecast
            s = max(s, float(np.quantile(w.members[:, w.hour(t + 3600)], 0.9)))
        wind = self.wind.vec(s * (1 + self.cfg.weather.gust))
        return 1.15 * self.robot.leg_wh(a, b, wind, 2 * self.cfg.sim.dt)

    def can_reach(self, x: float, y: float) -> bool:
        here = (self.robot.pose.x, self.robot.pose.y)
        need = self.energy_wh(here, (x, y)) + self.energy_wh((x, y), self.home)
        return self.robot.battery_wh - need >= self.cfg.robot.reserve_wh

    def fly_mission(self, waypoints: list[tuple[float, float]]) -> int:
        """Fly waypoints while the battery allows a safe return. Returns waypoints completed."""
        done = 0
        for x, y in waypoints:
            if not self.can_reach(x, y):
                self.bus.emit(
                    "RETURN_TO_HOME", reason="battery_reserve", battery_wh=self.robot.battery_wh
                )
                break
            if not self.fly_to(x, y):
                break
            done += 1
        return done

    def go_home(self) -> None:
        if self.fly_to(*self.home, abort=False):
            self.bus.emit(
                "LANDED", battery_wh=self.robot.battery_wh, distance_m=self.robot.distance_m
            )

    def summary(self) -> dict:
        beliefs = self.beliefs()
        truth = parcel_truth(self.world)
        rows, err = [], []
        gate = triggers.gate()
        for p in self.world.parcels:
            b = beliefs[p.id]
            est, tru = triggers.score(p, b.loss_mean), triggers.score(p, truth[p.id])
            issued = est.amount if b.confidence >= gate else 0.0
            err.append(abs(issued - tru.amount) / p.sum_insured)
            rows.append(
                {
                    "id": p.id,
                    "est": round(b.loss_mean, 4),
                    "truth": round(truth[p.id], 4),
                    "conf": round(b.confidence, 3),
                    "pay_est": est.amount,
                    "pay_true": tru.amount,
                }
            )
        return {
            "distance_m": round(self.robot.distance_m, 1),
            "battery_wh": round(self.robot.battery_wh, 2),
            "payout_mae_pct": round(100 * float(sum(err) / len(err)), 3),
            "loss_mae": round(float(sum(abs(r["est"] - r["truth"]) for r in rows) / len(rows)), 4),
            "min_conf": round(min(b.confidence for b in beliefs.values()), 3),
            "unsettled": sum(r["conf"] < gate and r["truth"] >= 0.15 for r in rows),
            "parcels": rows,
        }

    def fly(self, waypoints: list[tuple[float, float]]) -> bool:
        for x, y in waypoints:
            if not self.fly_to(x, y):
                return False
            self.bus.emit("WAYPOINT_REACHED", x=x, y=y)
        return True
