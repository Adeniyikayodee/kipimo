from kipimo import config
from kipimo.agent.model import ScriptedModel
from kipimo.bus import Bus, load
from kipimo.run import run


def demo_script(agent, results):
    if agent.n == 0:
        return [("get_situation", {})]
    if agent.n == 1:
        top = [p for p in results[0]["parcels"] if p["payout_at_estimate"] > 0][:4]
        agent.targets = [p["id"] for p in top]
        return [
            (
                "survey",
                dict(
                    parcel_ids=agent.targets,
                    target_confidence=0.8,
                    max_distance_m=10000,
                    reason="highest money at risk",
                ),
            )
        ]
    if agent.n == 2:
        return [
            ("issue_claim", dict(parcel_id=p, justification="trigger met")) for p in agent.targets
        ]
    if agent.n == 3:
        agent.refused = [
            r["score"]["parcel_id"]
            for r in results
            if not r["issued"] and r["reason"] == "insufficient_confidence"
        ]
        if agent.refused:
            return [
                (
                    "survey",
                    dict(
                        parcel_ids=agent.refused,
                        target_confidence=0.8,
                        max_distance_m=3000,
                        reason="refused on confidence",
                    ),
                )
            ]
        return [("report", dict(summary="done"))]
    if agent.n == 4:
        return [
            ("issue_claim", dict(parcel_id=p, justification="re-surveyed")) for p in agent.refused
        ]
    return [("report", dict(summary="paid what was due"))]


def _kinds(path):
    return [e["kind"] for e in load(path)]


def test_agent_demo_beat_refused_then_retasked(tmp_path):
    s = run(
        config.load("demo_flood"),
        "agents",
        tmp_path / "e.jsonl",
        {"field": ScriptedModel(demo_script)},
    )
    evs = load(tmp_path / "e.jsonl")
    kinds = [e["kind"] for e in evs]
    assert kinds.count("OBJECTIVE_ACCEPTED") >= 2
    refusals = [e for e in evs if e["kind"] == "CLAIM_REFUSED"]
    assert any(e["data"]["reason"] == "insufficient_confidence" for e in refusals)
    assert kinds.count("CLAIM_ISSUED") >= 2
    assert "AGENT_REPORT" in kinds and "AGENT_FALLBACK" not in kinds
    assert s["claims"] >= 2


def test_budget_exhaustion_falls_back(tmp_path):
    s = run(
        config.load("demo_flood"),
        "agents",
        tmp_path / "e.jsonl",
        {"field": ScriptedModel(lambda a, r: [("get_situation", {})])},
    )
    kinds = _kinds(tmp_path / "e.jsonl")
    assert kinds.count("AGENT_TOOL_CALL") == config.load("demo_flood").agent.max_tool_calls
    assert "AGENT_BUDGET_EXHAUSTED" in kinds and s["claims"] >= 2


def test_envelope_rejects_bad_objective(tmp_path):
    def script(agent, results):
        if agent.n == 0:
            return [
                (
                    "survey",
                    dict(parcel_ids=[], target_confidence=0.8, max_distance_m=500, reason="x"),
                )
            ]
        agent.result = results[0]
        return [("report", dict(summary="stop"))]

    c = ScriptedModel(script)
    run(config.load("demo_flood"), "agents", tmp_path / "e.jsonl", {"field": c})
    assert c.result == {"accepted": False, "reason": "empty_objective"}
    assert "OBJECTIVE_REJECTED" in _kinds(tmp_path / "e.jsonl")


def test_claims_that_passed_the_gate_are_paid_when_the_agent_stops(tmp_path):
    from kipimo.episode import Episode

    ids = [p.id for p in Episode(config.load("demo_flood")).world.parcels]

    def survey_then_stop(m, results):
        args = {"parcel_ids": ids, "target_confidence": 0.9, "max_distance_m": 9e3, "reason": "x"}
        return [] if m.n > 2 else [("survey", args)]

    s = run(
        config.load("demo_flood"),
        "agents",
        tmp_path / "e.jsonl",
        {"field": ScriptedModel(survey_then_stop)},
    )
    passed = sum(r["pay_est"] > 0 and r["conf"] >= 0.8 for r in s["parcels"])
    assert s["claims"] == passed >= 2


def test_a_spent_mission_deadline_stops_any_role():
    import time
    from types import SimpleNamespace as NS

    from kipimo.agent.runtime import Role

    box, model = NS(schemas=[], done=False), ScriptedModel(lambda m, r: [])
    assert Role("farmer", model, "", box, Bus(), 5, 60, time.monotonic()).wake("hi") == "budget"


def test_tool_errors_name_the_problem_and_priors_hide_the_truth():
    from kipimo.agent import policies
    from kipimo.agent.tools import Tools
    from kipimo.episode import Episode

    tools = Tools(policies.build(Episode(config.load("demo_flood"))))
    assert tools.call("get_priors", {"parcel_id": "P99"}) == {"error": "unknown parcel id P99"}
    before = tools.call("get_priors", {"parcel_id": "P01"})
    tools.st.ep.world.loss[:] = 0
    assert tools.call("get_priors", {"parcel_id": "P01"}) == before


def test_time_in_tools_does_not_spend_the_agents_budget():
    import time
    from types import SimpleNamespace as NS

    from kipimo.agent.runtime import Role

    slow = NS(schemas=[], done=False, call=lambda n, a: time.sleep(1.0) or {"ok": True})
    model = ScriptedModel(lambda m, r: [] if m.n >= 2 else [("wait", {})])
    assert (
        Role("field", model, "", slow, Bus(), 5, 11.5).wake("go") == "ended"
    )  # 2 s in tools would leave under 10


def test_no_gemini_key_falls_back(tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = run(config.load("demo_flood"), "agents", tmp_path / "e.jsonl")
    fb = [e for e in load(tmp_path / "e.jsonl") if e["kind"] == "AGENT_FALLBACK"]
    assert fb and fb[0]["data"]["reason"] == "no_client" and s["claims"] >= 2


def test_old_large_tool_replies_shrink_and_recent_ones_stay():
    from kipimo.agent.model import Call
    from kipimo.agent.runtime import STUB, compact

    big, small = {"rows": list(range(500))}, {"ok": True}
    msgs = [{"role": "user", "text": "go"}]
    for i in range(4):
        msgs.append({"role": "assistant", "turn": None})
        msgs.append(
            {
                "role": "tool",
                "results": [(Call(f"c{i}", "get", {}), big), (Call(f"s{i}", "x", {}), small)],
            }
        )
    seen = compact(msgs)
    tool = [m for m in seen if m["role"] == "tool"]
    assert [r for _, r in tool[0]["results"]] == [STUB, small]  # old: big shrinks, small stays
    assert tool[-1]["results"][0][1] is big and tool[-2]["results"][0][1] is big
    assert msgs[2]["results"][0][1] is big  # the original conversation is untouched


def test_sim_time_moves_while_a_slow_model_thinks():
    import time
    from types import SimpleNamespace as NS

    from kipimo.agent.runtime import Role

    class Slow:
        name = "slow"

        def turn(self, *a):
            time.sleep(0.3)
            from kipimo.agent.model import Turn

            return Turn(stop="end")

    ticks = {"n": 0}

    def idle():
        ticks["n"] += 1
        time.sleep(0.01)

    role = Role("field", Slow(), "", NS(schemas=[], done=False), Bus(), 5, 60, idle=idle)
    assert role.wake("go") == "ended" and ticks["n"] > 5


def test_hovering_costs_battery_and_waiting_on_the_pad_does_not():
    from kipimo.episode import Episode

    ep = Episode(config.load("demo_flood"))
    full = ep.robot.battery_wh
    ep.hold()
    assert ep.robot.battery_wh == full and ep.bus.t == ep.cfg.sim.dt  # on the pad
    ep.fly_to(200.0, 200.0)
    before, t0 = ep.robot.battery_wh, ep.bus.t
    for _ in range(10):
        ep.hold()
    assert ep.robot.battery_wh < before and ep.bus.t == t0 + 10 * ep.cfg.sim.dt


def test_survey_waits_for_the_flight_to_match_the_camera():
    from kipimo.agent import policies
    from kipimo.agent.tools import Tools
    from kipimo.episode import Episode

    st = policies.build(Episode(config.load("demo_flood")))
    tools, pid = Tools(st), st.ep.world.parcels[0].id
    ask = dict(parcel_ids=[pid], target_confidence=0.8, max_distance_m=500, reason="t")
    st.ep.sensor.degraded = True  # blurred while the agent was thinking
    out = tools.call("survey", ask)
    assert out["stop_reason"] == "camera_changed" and not out["accepted"]
    tools.call("set_flight", dict(altitude_m=25, speed_mps=4, reason="see clearly"))
    assert tools.call("survey", ask)["accepted"]
    st.ep.sensor.degraded = False  # camera back, still flying low
    assert tools.call("survey", ask)["stop_reason"] == "camera_changed"
