"""Non LLM policies. The benchmark runs these, and the agent falls back to baseline()."""

import os
from contextlib import suppress
from dataclasses import dataclass
from functools import partial

from kipimo.episode import Episode
from kipimo.payout import triggers
from kipimo.payout.approval import Actuary
from kipimo.payout.claim import ClaimRefused, PayoutEngine
from kipimo.planning import infogain, lawnmower
from kipimo.planning.envelope import Envelope
from kipimo.types import Objective
from kipimo.uplink.link import LinkModel
from kipimo.uplink.queue import Outbox
from kipimo.uplink.station import Uplink


@dataclass
class Stack:
    ep: Episode
    env: Envelope
    planner: infogain.InfoGainPlanner
    engine: PayoutEngine
    uplink: Uplink
    actuary: Actuary | None = None


def build(ep: Episode) -> Stack:
    env = ep.env
    planner = infogain.InfoGainPlanner(ep.world, env, ep.robot.footprint_half_m(), ep.cfg)
    engine = PayoutEngine(ep)
    u = ep.cfg.uplink
    link = LinkModel(ep.rng.spawn(1)[0], u.state, u.loss_pct, u.latency_ms)
    uplink = Uplink(ep.bus, engine.signer, link, Outbox(u.db), u.schedule)

    def on_event(ev: dict) -> None:
        if ev["kind"] == "CLAIM_ISSUED":
            uplink.enqueue(engine.issued[ev["data"]["claim"]["parcel_id"]][0])

    ep.bus.subscribers.append(on_event)
    ep.tickers.append(uplink.pump)
    st = Stack(ep, env, planner, engine, uplink)
    who = (
        os.environ.get("KIPIMO_ACTUARY") or ep.cfg.approval.actuary
    )  # a deployment sets it from the login
    st.actuary = Actuary(st, ep.cfg.approval.mode, who)  # after the uplink
    return st


def set_flight(st: Stack, altitude_m: float, speed_mps: float, why: str = "") -> dict:
    from kipimo.robot.sensors import camera_quality

    r = st.ep.robot
    alt, v = r.set_flight(altitude_m, speed_mps)
    st.planner.retune(r.footprint_half_m(), v)
    q = round(camera_quality(r, st.ep.sensor.degraded), 3)
    st.ep.bus.emit("FLIGHT_SET", altitude_m=alt, speed_mps=v, quality=q, reason=why)
    return {"altitude_m": alt, "speed_mps": v, "camera_quality": q}


def wait(st: Stack, minutes: float, why: str = "") -> dict:
    """Hold at the pad: time passes, texts arrive and the uplink drains; nothing is observed."""
    bus, n, w = st.ep.bus, int(max(1, min(minutes, 240))), st.ep.weather
    bus.emit("WAITED", minutes=n, reason=why, wind=round(w.wind(bus.t), 1) if w else None)
    for _ in range(n):
        bus.t += 60
        for fn in st.ep.tickers:
            fn()
    return {"waited_min": n, "wind_now": round(w.wind(bus.t), 1) if w else None}


def unflyable(st: Stack) -> str:
    w = st.ep.weather
    if w and w.wind(st.ep.bus.t) > w.limit:
        return f"wind {w.wind(st.ep.bus.t):.1f} m/s over the {w.limit:.0f} m/s limit"
    return ""


def wait_for_window(st: Stack, who: str) -> None:
    w = st.ep.weather
    if not unflyable(st):
        return
    h = w.window(st.ep.bus.t)
    target = (h if h is not None else w.actual.shape[0]) * 3600
    while unflyable(st) and st.ep.bus.t < target + 3 * 3600:
        wait(st, max(10, (target - st.ep.bus.t) / 60), f"{who}: waiting for the flight window")


def adapt(st: Stack, degraded: bool) -> None:
    """Rule response to a camera fault: fly lower and slower to recover quality."""
    cfg = st.ep.cfg.robot
    alt, v = (25.0, 5.0) if degraded else (cfg.altitude_m, cfg.speed_mps)
    set_flight(st, alt, v, "rules: camera degraded" if degraded else "rules: camera restored")


def go_home(st: Stack) -> None:
    r = st.ep.robot
    for wp in st.env.route((r.pose.x, r.pose.y), st.ep.home):
        st.ep.fly_to(*wp, abort=False)
    st.ep.go_home()


def issue_all(st: Stack, who: str) -> None:
    for p in st.ep.world.parcels:
        sc = st.engine.score(p.id)
        if sc["amount"] > 0 and not sc["issued"]:
            with suppress(ClaimRefused):
                st.engine.issue(p.id, f"{who}: survey complete, trigger met")


def run_lawnmower(st: Stack, overlap: float = 0.1) -> None:
    """Boustrophedon sweep. Negative overlap leaves gaps between lanes to save distance."""
    wait_for_window(st, "lawnmower")
    routed, prev = [], st.ep.home
    for wp in lawnmower.plan(st.ep.world, st.ep.robot.footprint_half_m(), overlap):
        routed += st.env.route(prev, wp)
        prev = wp
    st.ep.fly_mission(routed)
    go_home(st)
    issue_all(st, "lawnmower")


def unsettled(st: Stack) -> list[str]:
    """Parcels below the gate whose interval reaches a payout trigger."""
    b, gate = st.ep.beliefs(), triggers.gate()
    return [
        p.id
        for p in st.ep.world.parcels
        if b[p.id].confidence < gate and triggers.score(p, b[p.id].loss_ci95[1]).amount > 0
    ]


def baseline(st: Stack, who: str = "infogain") -> None:
    """Value weighted survey of every parcel, then settle anything payable, then claim."""
    ep, gate = st.ep, triggers.gate()
    wait_for_window(st, who)
    obj = Objective([p.id for p in ep.world.parcels], gate, 1e9, f"{who}: all parcels")
    fix = partial(adapt, st)
    ep.bus.emit("SURVEY_END", reason=infogain.survey(ep, st.planner, st.env, obj, on_degrade=fix))
    for _ in range(3):
        short = unsettled(st)
        if not short:
            break
        obj = Objective(short, gate, 1e9, f"{who}: settle payable parcels below the gate")
        ep.bus.emit(
            "SURVEY_END",
            reason=infogain.survey(ep, st.planner, st.env, obj, True, on_degrade=fix),
        )
    go_home(st)
    issue_all(st, who)
    for _ in range(2):  # an adjuster challenge asks for more evidence: look harder, then ask again
        open_ = {p for p, v in st.engine.challenged.items() if not v.get("escalated")}
        if not open_:
            break
        ids = sorted(open_)
        obj = Objective(ids, 0.95, 1e9, f"{who}: answer adjuster challenges")
        ep.bus.emit("SURVEY_END", reason=infogain.survey(ep, st.planner, st.env, obj, True, fix))
        go_home(st)
        issue_all(st, who)
