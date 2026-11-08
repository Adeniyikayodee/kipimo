import numpy as np

from kipimo import config
from kipimo.agent import policies
from kipimo.episode import Episode
from kipimo.planning import infogain
from kipimo.types import Objective
from kipimo.world.surge import DRY, breach


def test_a_breach_floods_the_lowest_dry_fields_for_real():
    ep = Episode(config.load("demo_flood"))
    w = ep.world
    n = w.size * w.size
    before = {p.id: w.loss.reshape(n)[p.cells].mean() for p in w.parcels}
    pids = breach(w)
    assert pids and all(before[pid] < DRY for pid in pids)
    for pid in pids:
        cells = w.parcel(pid).cells
        assert (
            w.loss.reshape(n)[cells].mean() > before[pid] and w.water.reshape(n)[cells].min() >= 0.9
        )
    assert breach(w, set(pids)) != pids  # a second breach reaches other fields


def test_the_radar_pass_raises_the_estimate_and_the_drone_replans_toward_it():
    st = policies.build(Episode(config.load("demo_flood")))
    ep = st.ep
    t_surge = {}

    def tick():
        if not ep.surged and ep.bus.t >= 120:
            t_surge["t"], t_surge["pids"] = ep.bus.t, None
            before = {p.id: ep.beliefs()[p.id].loss_mean for p in ep.world.parcels}
            t_surge["pids"] = ep.surge()
            t_surge["before"] = before

    ep.tickers.append(tick)
    obj = Objective([p.id for p in ep.world.parcels], 0.8, 1e9, "test")
    infogain.survey(ep, st.planner, st.env, obj)
    pids = t_surge["pids"]
    after = ep.beliefs()
    assert all(after[pid].loss_mean > t_surge["before"][pid] for pid in pids)
    kinds = [e["kind"] for e in ep.bus.events]
    assert "RIVER_SURGE" in kinds and "SURGE_REPLAN" in kinds
    i = kinds.index("SURGE_REPLAN")
    assert ep.bus.events[i]["t"] - t_surge["t"] <= ep.cfg.sim.dt  # stops on the next tick
    flown = {e["data"]["parcel"] for e in ep.bus.events[i:] if e["kind"] == "PLAN_LANE"}
    assert flown & set(pids)  # the new plan goes to the flooded fields
    assert np.isfinite([after[pid].confidence for pid in pids]).all()


def test_forgetting_returns_the_belief_to_never_observed():
    from kipimo.robot.sensors import SimSensor

    for model in ("gp", "grid"):
        cfg = config.load("demo_flood", belief={"model": model})
        fresh, ep = Episode(cfg), Episode(cfg)
        sensor = SimSensor(ep.world, np.random.default_rng(1))
        cells = ep.world.parcels[0].cells
        for x in (100.0, 160.0, 220.0):
            ep.robot.pose = type(ep.robot.pose)(x, 120.0, 0.0)
            ep.belief.update(sensor.observe(ep.robot, 0.0))
        ep.belief.forget(np.arange(ep.world.size * ep.world.size))
        a, b = fresh.beliefs(), ep.beliefs()
        for p in ep.world.parcels:
            assert abs(a[p.id].loss_mean - b[p.id].loss_mean) < 1e-9, (model, p.id)
            assert abs(a[p.id].confidence - b[p.id].confidence) < 1e-9, (model, p.id)
        del cells


def test_after_a_breach_paid_amounts_follow_the_new_ground():
    from kipimo.run import run

    holder = {}

    def setup(st):
        holder["st"] = st

        def tick():
            if not st.ep.surged and st.ep.bus.t >= 1020:
                st.ep.surge()

        st.ep.tickers.append(tick)

    s = run(config.load("demo_flood"), "infogain", None, setup=setup)
    st = holder["st"]
    paid = {pid: c.amount for pid, (c, _) in st.engine.issued.items()}
    for r in s["parcels"]:
        if r["id"] in st.ep.surged:
            assert paid.get(r["id"], 0) == r["pay_true"], r["id"]
