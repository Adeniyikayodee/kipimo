from kipimo import config
from kipimo.agent import policies
from kipimo.agent.model import ScriptedModel
from kipimo.agent.tools import Tools
from kipimo.bus import load
from kipimo.episode import Episode
from kipimo.run import run

STORM = config.load("demo_flood", weather={"enabled": True})


def test_envelope_refuses_to_fly_into_the_storm_and_names_the_window():
    tools = Tools(policies.build(Episode(STORM)))
    c = tools.call("get_conditions", {})
    assert (
        c["weather"]["wind_mps"] > c["weather"]["limit_mps"] and c["weather"]["next_window_in_min"]
    )
    out = tools.call(
        "survey", dict(parcel_ids=["P07"], target_confidence=0.8, max_distance_m=500, reason="x")
    )
    assert out["accepted"] is False and out["reason"].startswith("wind_above_limit")


def test_rules_wait_for_the_window_then_fly(tmp_path):
    run(STORM, "infogain", tmp_path / "e.jsonl")
    evs = load(tmp_path / "e.jsonl")
    first_obs = next(e["t"] for e in evs if e["kind"] == "OBSERVATION")
    assert any(e["kind"] == "WAITED" for e in evs) and first_obs >= 3 * 3600


def test_rules_fly_lower_when_the_camera_degrades():
    st = policies.build(Episode(config.load("demo_flood")))
    st.ep.sensor.degraded = True
    policies.adapt(st, True)
    assert st.ep.robot.altitude_m == 25 and st.planner.half_m < 30


def test_agent_waits_with_the_wait_tool(tmp_path):
    def script(m, results):
        steps = [
            [("get_conditions", {})],
            [("wait", dict(minutes=200, reason="storm"))],
            [
                (
                    "survey",
                    dict(
                        parcel_ids=["P07"], target_confidence=0.8, max_distance_m=2000, reason="x"
                    ),
                )
            ],
        ]
        return steps[m.n] if m.n < len(steps) else [("report", dict(summary="done"))]

    run(STORM, "agents", tmp_path / "e.jsonl", {"field": ScriptedModel(script)})
    evs = load(tmp_path / "e.jsonl")
    accepted = [e for e in evs if e["kind"] == "OBJECTIVE_ACCEPTED"]
    assert accepted and accepted[0]["t"] >= 3 * 3600
