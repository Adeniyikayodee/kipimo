from kipimo import config
from kipimo.agent import dispatch
from kipimo.agent.model import ModelError, ScriptedModel
from kipimo.bus import Bus
from kipimo.planning.district import place_hubs, reach_km, villages

CFG = config.load()
R = reach_km(CFG.robot)
VS = villages(1)
D = dispatch.District(VS, place_hubs(VS, 3, R)[0], R, CFG.robot.speed_mps * 3.6)


def test_assign_is_held_to_reach_and_uniqueness():
    box = dispatch.DispatchTools(D)
    vid = next(v for v in D.vs if D.reach(v))
    far = next((h for h in D.hubs if h not in D.reach(vid)), None)
    assert box.call("assign", dict(village_id=vid, hub_id=D.reach(vid)[0], reason="x"))["queued"]
    assert "error" in box.call("assign", dict(village_id=vid, hub_id=D.reach(vid)[0], reason="x"))
    if far:
        other = next(v for v in D.vs if v != vid and far not in D.reach(v))
        assert "error" in box.call("assign", dict(village_id=other, hub_id=far, reason="x"))


def test_greedy_schedules_every_reachable_village_once():
    q = dispatch.greedy(D)
    planned = [v for vs in q.values() for v in vs]
    assert sorted(planned) == sorted(v for v in D.vs if D.reach(v))
    assert all(h in D.reach(v) for h, vs in q.items() for v in vs)


def test_agent_plans_and_falls_back():
    def script(m, results):
        if m.n == 0:
            return [("get_district", {})]
        if m.n == 1:
            vs = results[0]["villages"]
            return [
                (
                    "assign",
                    dict(village_id=v["id"], hub_id=next(iter(v["hubs_in_reach"])), reason="x"),
                )
                for v in vs
                if v["hubs_in_reach"]
            ]
        return [("finish", dict(summary="done"))]

    q, by = dispatch.plan(D, ScriptedModel(script, "disp"), Bus(None))
    assert by == "disp" and any(q.values())

    class Down:
        name = "down"

        def turn(self, *a):
            raise ModelError("connection")

    assert dispatch.plan(D, Down(), Bus(None))[1] == "rules"


def test_villages_the_agent_skips_still_get_a_drone():
    def one_then_finish(m, results):
        if m.n == 0:
            return [("get_district", {})]
        v = next(v for v in results[0]["villages"] if v["hubs_in_reach"]) if m.n == 1 else None
        if v:
            hub = next(iter(v["hubs_in_reach"]))
            return [("assign", dict(village_id=v["id"], hub_id=hub, reason="x"))]
        return [("finish", dict(summary="done"))]

    q, _ = dispatch.plan(D, ScriptedModel(one_then_finish, "disp"), Bus(None))
    assert sorted(v for vs in q.values() for v in vs) == sorted(v for v in D.vs if D.reach(v))
