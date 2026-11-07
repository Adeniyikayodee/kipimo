import numpy as np

from kipimo import config
from kipimo.agent import policies
from kipimo.belief.gp import sigma_meas
from kipimo.episode import Episode
from kipimo.robot.sensors import footprint_cells, quality
from kipimo.run import run
from kipimo.types import Objective, Observation, Pose

CFG = config.load("demo_flood", belief={"model": "gp"})


def _stack():
    return policies.build(Episode(CFG))


def _fly(st, lane, q, rng):
    ep, r = st.ep, CFG.robot
    w, half = ep.world, ep.robot.footprint_half_m()
    steps = max(1, int(lane.length / (r.speed_mps * CFG.sim.dt)))
    for t in np.linspace(0, 1, steps + 1):
        x, y = np.add(lane.a, t * np.subtract(lane.b, lane.a))
        ids = footprint_cells(w, x, y, half)
        noisy = np.clip(w.loss.reshape(-1)[ids] + rng.normal(0, sigma_meas(q), len(ids)), 0, 1)
        f32 = noisy.astype(np.float32)
        ep.belief.update(Observation(0.0, Pose(x, y, 0), ids, f32, f32, q))


def test_forecast_matches_flying_the_lane():
    st, rng = _stack(), np.random.default_rng(0)
    q = quality(CFG.robot.altitude_m, CFG.robot.speed_mps)
    obj = Objective([p.id for p in st.ep.world.parcels], 0.8, 1e9, "test")
    fc = st.planner.forecast(st.ep.belief, obj, q)
    lane = max(st.planner.lanes, key=lambda ln: ln.length)
    fc.commit(lane)
    _fly(st, lane, q, rng)
    _, actual = st.ep.belief.parcel_moments(st.ep.world.parcels)
    k = fc.rows[lane.parcel_id]
    assert abs(fc.width[k] - actual[k]) / actual[k] < 0.2
    assert np.corrcoef(fc.width, actual)[0, 1] > 0.95


def test_one_lane_narrows_every_parcel():
    st = _stack()
    q = quality(CFG.robot.altitude_m, CFG.robot.speed_mps)
    obj = Objective([p.id for p in st.ep.world.parcels], 0.8, 1e9, "test")
    fc = st.planner.forecast(st.ep.belief, obj, q)
    before = fc.width.copy()
    fc.commit(st.planner.lanes[0])
    assert (fc.width < before).all()


def test_gp_mission_settles_every_payable_parcel():
    s = run(CFG, "infogain")
    assert s["unsettled"] == 0 and s["missed_claims"] == 0
    assert s["distance_m"] < run(config.load("demo_flood"), "lawnmower")["distance_m"]
