from kipimo import config
from kipimo.agent import mission, policies
from kipimo.agent.model import ScriptedModel
from kipimo.episode import Episode

CFG = config.load("demo_flood", belief={"model": "gp"})


def _mission(monkeypatch, models):
    st = policies.build(Episode(CFG))
    monkeypatch.setattr(mission.review, "holding", lambda f: ["A3_tier_edge"])
    return mission.Mission(st, models), st


def test_three_challenges_then_a_person_decides(monkeypatch):
    m, st = _mission(monkeypatch, {})
    pid, sc = st.ep.world.parcels[0].id, {"amount": 100, "confidence": 0.85}
    verdicts = [m.review(pid, sc, "more coverage now") for _ in range(4)]
    assert all(v["rule_id"] == "A3_tier_edge" for v in verdicts)
    assert verdicts[-1]["escalated"] and not verdicts[0].get("escalated")
    st.engine.challenged[pid] = verdicts[-1]
    assert m.desk.status(pid)["under_review"]
    assert any(e["kind"] == "CLAIM_ESCALATED" for e in st.ep.bus.events)


def test_adjuster_keeps_one_conversation_per_claim(monkeypatch):
    def script(model, results):
        pid = model.brief.split("for ")[1].split(":")[0].split(".")[0]
        if not results:
            return [("get_dossier", dict(parcel_id=pid))]
        if model.brief.startswith("Round"):
            return [("approve", dict(parcel_id=pid, note="evidence improved"))]
        return [("challenge", dict(parcel_id=pid, rule_id="A3_tier_edge", note="need 90%"))]

    m, st = _mission(monkeypatch, {"adjuster": ScriptedModel(script, "adj")})
    pid, sc = st.ep.world.parcels[0].id, {"amount": 100, "confidence": 0.85}
    first = m.review(pid, sc, "trigger met")
    second = m.review(pid, sc, "flew again, confidence 0.92")
    role = m.cases[pid]["role"]
    briefs = [x["text"] for x in role.messages if x["role"] == "user"]
    assert first["rule_id"] == "A3_tier_edge" and second is None
    assert len(briefs) == 2 and briefs[1].startswith("Round 2")
