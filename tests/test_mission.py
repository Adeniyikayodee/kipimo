import re

from kipimo import config
from kipimo.agent.farmer import read
from kipimo.agent.model import ScriptedModel
from kipimo.run import run

CFG = config.load("demo_flood", belief={"model": "gp"}, agent={"sms_profile": "adversarial"})


def _step(m):
    """Turn index within the current brief, so one scripted model can serve many wakes."""
    if m.brief != getattr(m, "last", None):
        m.last, m.k = m.brief, 0
    m.k += 1
    return m.k - 1, re.search(r"P\d+", m.brief).group(0) if "P" in m.brief else None


def farmer(m, results):
    k, pid = _step(m)
    if k == 0 and "mission is over" not in m.brief:
        r = read(m.brief.split(": ", 1)[1])
        args = dict(parcel_id=pid, kind=r["kind"], severity=r["severity"], where=r["where"])
        args |= dict(flood_hours_ago=r["hours"], suspicious=r["suspicious"], note="x")
        return [("record_report", args)]
    if k <= 1:
        return [("send_sms", dict(parcel_id=pid, text="You will get KES 99,999 today"))]
    return []


def adjuster(m, results):
    k, pid = _step(m)
    if k == 0:
        return [("get_dossier", dict(parcel_id=pid))]
    if k == 1:
        return [("challenge", dict(parcel_id=pid, rule_id="A1_coverage", note="x"))]
    return [("approve", dict(parcel_id=pid, note="ok"))]


def field(m, results):
    if m.n == 0:
        return [("get_situation", {})]
    if m.n == 1:
        m.ids = [p["id"] for p in results[0]["parcels"] if p["payout_at_estimate"] > 0][:6]
        return [
            (
                "survey",
                dict(parcel_ids=m.ids, target_confidence=0.85, max_distance_m=9000, reason="x"),
            )
        ]
    if m.n == 2:
        return [("issue_claim", dict(parcel_id=p, justification="x")) for p in m.ids]
    return [("report", dict(summary="done"))]


def _kinds(s):
    return [e["kind"] for e in s]


def test_rules_mission_is_deterministic_and_reviews_every_claim(tmp_path):
    a = run(CFG, "agents", tmp_path / "a.jsonl", model={})
    b = run(CFG, "agents", tmp_path / "b.jsonl", model={})
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    from kipimo.bus import load

    evs = load(tmp_path / "a.jsonl")
    reviewed = {
        e["data"]["parcel"]
        for e in evs
        if e["kind"] == "CLAIM_REVIEW" and e["data"]["decision"] == "approve"
    }
    issued = {e["data"]["claim"]["parcel_id"] for e in evs if e["kind"] == "CLAIM_ISSUED"}
    assert issued and issued <= reviewed
    attacks = {
        e["data"]["parcel"]
        for e in evs
        if e["kind"] == "SMS_RECEIVED" and e["data"]["label"]["kind"] == "injection"
    }
    assert not any(
        e["data"]["doubted"]
        for e in evs
        if e["kind"] == "FARMER_REPORT"
        and e["data"]["parcel"] in attacks
        and e["data"]["suspicious"]
    )
    assert a["claims"] == b["claims"] > 0


def test_scripted_agents_are_held_to_the_rules(tmp_path):
    from kipimo.bus import load

    models = {
        "field": ScriptedModel(field, "field-llm"),
        "farmer": ScriptedModel(farmer, "farmer-llm"),
        "adjuster": ScriptedModel(adjuster, "adjuster-llm"),
    }
    s = run(CFG, "agents", tmp_path / "e.jsonl", model=models)
    evs = load(tmp_path / "e.jsonl")
    kinds = _kinds(evs)
    assert "AGENT_FALLBACK" not in kinds and s["claims"] >= 2
    assert "SMS_REJECTED" in kinds  # quoted an unissued amount: blocked
    assert not any("99,999" in e["data"]["text"] for e in evs if e["kind"] == "SMS_SENT")
    assert any(e["data"]["by"] == "farmer-llm" for e in evs if e["kind"] == "FARMER_REPORT")
    reviews = [e["data"] for e in evs if e["kind"] == "CLAIM_REVIEW"]
    assert reviews and all(r["by"] == "adjuster-llm" for r in reviews)
    if "CHALLENGE_REJECTED" in kinds:  # ungrounded challenges never block a claim
        assert all(r["decision"] == "approve" or r["rule_id"] in r["holding"] for r in reviews)
