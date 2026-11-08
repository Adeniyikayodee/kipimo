from kipimo import config
from kipimo.agent import policies
from kipimo.audit import audit
from kipimo.episode import Episode
from kipimo.payout.approval import CHECKS, FLAGS
from kipimo.run import run


def _run(tmp_path, mode="human", name="run.jsonl"):
    cfg = config.load("demo_flood", approval={"mode": mode})
    holder = {}
    s = run(cfg, "infogain", tmp_path / name, setup=lambda st: holder.update(st=st), close=False)
    st = holder["st"]
    st.uplink.drain(cfg.sim.dt)
    return s, st


def _kinds(st, kind):
    return [e["data"] for e in st.ep.bus.events if e["kind"] == kind]


def test_every_flag_has_a_check():
    assert [r["id"] for r in FLAGS] == list(CHECKS)


def test_human_mode_holds_every_claim_for_the_actuary(tmp_path):
    s, st = _run(tmp_path)
    cases = _kinds(st, "CLAIM_AWAITING_APPROVAL")
    assert s["claims"] > 0 and len(cases) == len(st.engine.issued)
    assert not _kinds(st, "ACTUARY_DECISION") and not st.actuary.paid


def test_signed_decisions_pay_and_pass_the_audit(tmp_path):
    _, st = _run(tmp_path)
    first, second, *rest = sorted(st.actuary.cases)
    assert st.actuary.decide(first, "approve", "a. actuary", "looks right")
    assert (
        st.actuary.decide(second, "reject", "a. actuary", "") is None
    )  # a rejection needs a reason
    assert st.actuary.decide(second, "reject", "a. actuary", "tier edge, resurvey first")
    assert st.actuary.decide(first, "approve", "a. actuary") is None  # decided once
    st.ep.bus.close()
    r = audit(tmp_path / "run.jsonl")
    assert r["problems"] == [] and r["approved"] == 1 and r["awaiting"] == len(rest)
    assert r["approved_total"] == st.engine.issued[st.actuary.decided[first]["parcel_id"]][0].amount


def test_audit_catches_a_decision_on_a_forged_claim(tmp_path):
    from kipimo.audit import _decision_problem
    from kipimo.payout.approval import claim_hash

    _, st = _run(tmp_path)
    case = next(iter(st.actuary.cases))
    st.actuary.decide(case, "approve", "a. actuary")
    ev = _kinds(st, "ACTUARY_DECISION")[0]
    claims = {d["claim"]["claim_id"]: d["claim"] for d in _kinds(st, "CLAIM_ISSUED")}
    key = st.actuary.signer.public_hex
    assert _decision_problem(ev, claims, set(), key) == ""
    assert _decision_problem(ev, claims, {case}, key) == "decided twice"
    forged = {k: dict(c, amount=c["amount"] * 2) for k, c in claims.items()}
    assert claim_hash(forged[ev["decision"]["claim_id"]]) != ev["decision"]["claim_hash"]
    assert "not in the log" in _decision_problem(ev, forged, set(), key)
    assert _decision_problem(ev, claims, set(), st.engine.signer.public_hex).startswith("bad")


def test_auto_mode_passes_clean_claims_and_leaves_flagged_ones(tmp_path):
    _, st = _run(tmp_path, "auto")
    cases = _kinds(st, "CLAIM_AWAITING_APPROVAL")
    clean = {c["case_id"] for c in cases if not c["flags"]}
    assert {d["decision"]["case_id"] for d in _kinds(st, "ACTUARY_DECISION")} == clean
    assert set(st.actuary.cases) == {c["case_id"] for c in cases} - clean


def test_approve_clean_leaves_flagged_claims_for_a_look(tmp_path):
    _, st = _run(tmp_path)
    flagged = {k for k, c in st.actuary.cases.items() if c["flags"]}
    n = st.actuary.approve_clean("a. actuary")
    assert set(st.actuary.cases) == flagged and n == len(st.actuary.decided)


def test_releasing_a_hold_issues_the_claim_and_pays_once_it_lands():
    cfg = config.load("demo_flood")
    st = policies.build(Episode(cfg))
    st.engine.reviewer = lambda pid, sc, why: {"rule_id": "A3_tier_edge", "note": "hold"}
    policies.baseline(st)  # the adjuster holds every claim
    assert not st.engine.issued
    pid = sorted(st.engine.challenged)[0]
    st.ep.bus.emit("CLAIM_ESCALATED", parcel=pid, rule_id="A3_tier_edge", rounds=3)
    case = f"H-{pid}"
    assert st.actuary.cases[case]["flags"][-1] == "F6_adjuster_hold"
    st.actuary.decide(case, "approve", "a. actuary", "field photos settle the tier")
    assert pid in st.engine.issued
    st.uplink.drain(cfg.sim.dt)
    assert pid in st.actuary.paid  # the person already ruled: no second sign-off


def test_a_released_hold_still_cannot_pass_the_gate():
    cfg = config.load("demo_flood")
    st = policies.build(Episode(cfg))  # nothing surveyed: every parcel is below the gate
    pid = st.ep.world.parcels[0].id
    st.ep.bus.emit("CLAIM_ESCALATED", parcel=pid, rule_id="A1_coverage", rounds=3)
    st.actuary.decide(f"H-{pid}", "approve", "a. actuary", "pay it")
    assert pid not in st.engine.issued
    assert _kinds(st, "CLAIM_REFUSED")[-1]["parcel"] == pid


def test_a_claim_within_a_point_of_the_payout_line_is_flagged():
    assert CHECKS["F7_hairline"]({"edge": 0.0052})
    assert not CHECKS["F7_hairline"]({"edge": 0.02})
