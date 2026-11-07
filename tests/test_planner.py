import json

from kipimo import config
from kipimo.config import EnvelopeCfg, Zone
from kipimo.planning.envelope import Envelope, seg_dist
from kipimo.run import run
from kipimo.world.generate import generate

W = generate(7)


def test_infogain_beats_lawnmower_on_demo_seed(tmp_path):
    cfg = config.load("demo_flood")
    lm = run(cfg, "lawnmower")
    ig = run(cfg, "infogain", tmp_path / "e.jsonl")
    assert ig["distance_m"] <= 0.7 * lm["distance_m"]
    assert ig["payout_mae_pct"] <= lm["payout_mae_pct"] + 0.5
    assert ig["unsettled"] == 0
    paid = [r for r in ig["parcels"] if r["pay_est"] > 0]
    assert paid and all(r["conf"] >= 0.8 for r in paid)


def test_flight_never_enters_no_fly_zone(tmp_path):
    cfg = config.load("demo_flood")
    z = cfg.envelope.no_fly[0]
    for policy in ("lawnmower", "infogain"):
        run(cfg, policy, tmp_path / f"{policy}.jsonl")
        poses = [
            json.loads(x)["data"]["pose"]
            for x in (tmp_path / f"{policy}.jsonl").read_text().splitlines()
            if '"OBSERVATION"' in x
        ]
        assert min(((p["x"] - z.x) ** 2 + (p["y"] - z.y) ** 2) ** 0.5 for p in poses) > z.r


def test_envelope_rejections_have_reasons():
    env = Envelope(W, EnvelopeCfg(no_fly=[Zone(x=500, y=500, r=50)]))
    assert env.check_point((5000, 5000)).reason == "outside_geofence"
    assert env.check_point((510, 500)).reason == "inside_no_fly_zone"
    assert env.check_point((300, 300)).ok
    v = env.check_battery(20.0, 15.0, 12.0)
    assert not v.ok and v.reason.startswith("battery_reserve")


def test_route_skirts_zone():
    env = Envelope(W, EnvelopeCfg(no_fly=[Zone(x=500, y=500, r=50)]))
    pts = [(300.0, 500.0), *env.route((300.0, 500.0), (700.0, 500.0))]
    for a, b in zip(pts, pts[1:], strict=False):
        assert seg_dist((500, 500), a, b)[0] > 50


def test_degraded_sensor_triggers_replan_and_more_flight(tmp_path):
    from kipimo.agent import policies
    from kipimo.bus import load
    from kipimo.episode import Episode

    def go(degrade_at):
        ep = Episode(config.load("demo_flood"), out=tmp_path / f"{degrade_at}.jsonl")
        if degrade_at:
            ep.tickers.append(lambda: setattr(ep.sensor, "degraded", ep.bus.t > degrade_at))
        policies.baseline(policies.build(ep))
        ep.bus.close()
        return ep, [e["kind"] for e in load(tmp_path / f"{degrade_at}.jsonl")]

    clean, _ = go(0)
    dusty, kinds = go(300)
    assert "DEGRADED_SENSOR_REPLAN" in kinds
    assert dusty.robot.distance_m > clean.robot.distance_m


def test_lanes_cross_parcels_split_by_the_channel():
    run(config.load("demo_flood", seed=0), "lawnmower")


def test_survey_hands_a_camera_change_to_the_agent():
    from kipimo import config
    from kipimo.agent import policies
    from kipimo.episode import Episode
    from kipimo.planning import infogain
    from kipimo.types import Objective

    st = policies.build(Episode(config.load("demo_flood")))
    ep = st.ep
    flips = {"n": 0}

    def tick():
        flips["n"] += 1
        if flips["n"] == 200:
            ep.sensor.degraded = True

    ep.tickers.append(tick)
    obj = Objective([p.id for p in ep.world.parcels], 0.8, 1e9, "test")
    assert infogain.survey(ep, st.planner, st.env, obj, stop_on_camera=True) == "camera_changed"
    assert any(e["kind"] == "DEGRADED_SENSOR_REPLAN" for e in ep.bus.events)
