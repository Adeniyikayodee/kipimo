from kipimo import config
from kipimo.agent import policies
from kipimo.episode import Episode
from kipimo.payout import review
from kipimo.payout.claim import ClaimRefused


def _stack():
    return policies.build(Episode(config.load("demo_flood")))


def test_rules_hold_from_evidence_only():
    f = {"coverage": 0.1, "report_gap": 0, "edge": 0.2, "confidence": 0.85}
    assert review.holding(f) == ["A1_coverage"]
    f = {"coverage": 0.4, "report_gap": 2, "edge": 0.01, "confidence": 0.85}
    assert review.holding(f) == ["A2_report_conflict", "A3_tier_edge"]
    assert review.holding({"coverage": 0.9, "report_gap": 0, "edge": 0.2, "confidence": 0.95}) == []
    hairline = {"coverage": 0.9, "report_gap": 0, "edge": 0.0052, "confidence": 0.92}  # P03 live
    assert review.holding(hairline) == ["A4_tier_hairline"]
    assert review.holding(hairline | {"confidence": 0.95}) == []


def test_reviewer_challenge_blocks_signing_until_answered():
    st = _stack()
    calls = []

    def reviewer(pid, sc, why):
        calls.append(pid)
        return {"rule_id": "A3_tier_edge", "by": "test"} if len(calls) == 1 else None

    st.engine.reviewer = reviewer
    policies.baseline(st)
    first = calls[0]
    assert first in st.engine.issued and not st.engine.challenged
    assert any(e["kind"] == "CLAIM_CHALLENGED" for e in st.ep.bus.events)


def test_challenge_is_a_refusal_the_agent_can_read():
    st = _stack()
    policies.baseline(st)
    pid = next(iter(st.engine.issued))
    st.engine.issued.clear()
    st.engine.reviewer = lambda pid, sc, why: {"rule_id": "A1_coverage", "by": "test"}
    try:
        st.engine.issue(pid, "x")
    except ClaimRefused as e:
        assert e.reason == "challenged" and e.detail["rule_id"] == "A1_coverage"
    else:
        raise AssertionError
