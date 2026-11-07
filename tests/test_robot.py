import hashlib
import json
import math

from kipimo import config
from kipimo.episode import Episode


def _run(tmp_path, name):
    ep = Episode(config.load("demo_flood"), out=tmp_path / name)
    ok = ep.fly([(500, 100), (800, 800), (200, 600)])
    ep.bus.close()
    return ep, ok


def test_fixed_waypoints_log_and_battery(tmp_path):
    ep, ok = _run(tmp_path, "a.jsonl")
    assert ok
    evs = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    obs = [e for e in evs if e["kind"] == "OBSERVATION"]
    assert len(obs) > 100
    batt = [e["data"]["battery_wh"] for e in obs]
    assert all(b2 < b1 for b1, b2 in zip(batt, batt[1:], strict=False))
    assert sum(e["kind"] == "WAYPOINT_REACHED" for e in evs) == 3


def test_event_log_is_byte_identical(tmp_path):
    _run(tmp_path, "a.jsonl")
    _run(tmp_path, "b.jsonl")
    h = [hashlib.sha256((tmp_path / f).read_bytes()).hexdigest() for f in ("a.jsonl", "b.jsonl")]
    assert h[0] == h[1]


def test_degraded_camera_recovers_lower_and_slower():
    from kipimo.config import RobotCfg
    from kipimo.robot.kinematics import Robot
    from kipimo.robot.sensors import camera_quality
    from kipimo.types import Pose

    r = Robot(RobotCfg(), Pose(0, 0, 0))
    assert camera_quality(r, True) == 0.3
    assert r.set_flight(5, 50) == (20.0, 10.0)
    r.set_flight(25, 5)
    assert camera_quality(r, True) > 0.35 and camera_quality(r, False) > 0.9


def _fly(wind, target=(400.0, 0.0)):
    from kipimo.config import RobotCfg
    from kipimo.robot.kinematics import Robot
    from kipimo.types import Pose

    r, t = Robot(RobotCfg(), Pose(0, 0, 0)), 0
    while not r.step(*target, 1.0, wind):
        t += 1
    return r, t


def test_power_curve_is_cheapest_in_forward_flight():
    from kipimo.config import RobotCfg
    from kipimo.robot.kinematics import power_w

    p = {v: power_w(RobotCfg(), v) for v in (0, 8, 10, 15)}
    assert round(p[0]) == RobotCfg().hover_w and round(p[8]) == 253
    assert p[10] < p[8] < p[0] and p[15] > p[10]


def test_thrust_tilt_caps_acceleration():
    from kipimo.config import RobotCfg
    from kipimo.robot.kinematics import G, Robot
    from kipimo.types import Pose

    r = Robot(RobotCfg(), Pose(0, 0, 0))
    r.step(400, 0, 1.0)
    assert 0 < r.air[0] <= G * math.tan(math.radians(25)) + 1e-9


def test_wind_changes_cost_and_crosswind_still_arrives():
    calm, t0 = _fly((0.0, 0.0))
    gale, t1 = _fly((-10.0, 0.0))  # needs 18 m/s of air, past the 15 m/s limit
    tail, _ = _fly((6.0, 0.0))  # 2 m/s of air, so the rotors work near hover
    cross, _ = _fly((0.0, 6.0))
    assert t1 > 1.5 * t0 and gale.battery_wh < calm.battery_wh
    assert tail.battery_wh < calm.battery_wh
    assert math.dist((cross.pose.x, cross.pose.y), (400, 0)) < 2.0


def test_wind_past_max_airspeed_stalls_progress():
    from kipimo.config import RobotCfg
    from kipimo.robot.kinematics import cruise

    assert cruise(RobotCfg(), 8, (1, 0), (-4, 0)) == (8, 12)
    assert cruise(RobotCfg(), 8, (1, 0), (-16, 0))[0] == 0


def test_footprint_off_the_map_is_empty():
    from kipimo.robot.sensors import footprint_cells

    w = Episode(config.load("demo_flood")).world
    assert all(footprint_cells(w, x, y, 20).size == 0 for x, y in [(-500, -500), (9e4, 9e4)])


def test_empty_battery_forces_one_landing():
    ep = Episode(config.load("demo_flood"))
    ep.robot.battery_wh = 1.0
    assert not ep.fly_to(800, 800)
    assert ep.down and not ep.fly_to(0, 0)
    assert [e["kind"] for e in ep.bus.events].count("FORCED_LANDING") == 1


def test_legs_abort_in_wind_over_the_limit():
    ep = Episode(config.load("storm_flood"))
    assert ep.no_go() == "wind" and not ep.fly_to(500, 500)
    assert ep.bus.events[-1]["kind"] == "LEG_ABORTED"


def test_the_home_pad_is_inside_the_geofence():
    cfg = config.load(None, seed=202, world={"flood_stage": [1.0, 1.7]})
    ep = Episode(cfg)
    assert ep.env.inside_fence(ep.home) and not ep.no_go()
