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
