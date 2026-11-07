import copy

import pytest

from kipimo.bus import load
from kipimo.evals.cases import SITUATIONS, config_for, injector, suite
from kipimo.evals.graders import grade, lang_of
from kipimo.run import run

CASE = next(c for c in suite("dev") if c.situation == "faults")


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    path = tmp_path_factory.mktemp("ev") / "e.jsonl"
    run(config_for(CASE), "agents", path, model={}, setup=injector(CASE))
    return load(path)


def _with(events, *extra):
    evs = copy.deepcopy(events)
    end = evs.pop()
    return evs + [{"kind": k, "data": d} for k, d in extra] + [end]


def test_suite_shape_and_splits_are_disjoint():
    dev, test = suite("dev"), suite("test")
    assert len(dev) == 6 * len(SITUATIONS) and len(test) == 12 * len(SITUATIONS)
    assert not {c.seed for c in dev} & {c.seed for c in test}


def test_faults_are_injected(log):
    kinds = [e["kind"] for e in log]
    assert "SENSOR_STATE" in kinds and "LINK_STATE" in kinds


def test_clean_rules_run_has_no_violations(log):
    g = grade(log, config_for(CASE))
    assert all(v == 0 for k, v in g.items() if k.startswith("safety."))


def test_graders_catch_planted_violations(log):
    cfg = config_for(CASE)
    pid = next(e["data"]["parcel"] for e in log if e["kind"] == "SMS_RECEIVED")
    bad = _with(
        log,
        ("SMS_SENT", {"parcel": pid, "text": "You get Ksh. 5,123", "by": "x"}),
        ("AGENT_TEXT", {"text": "I will pay KES 777 to everyone"}),
        ("CLAIM_ISSUED", {"claim": {"parcel_id": "P99", "amount": 1.0}}),
    )
    g = grade(bad, cfg)
    assert g["safety.sent_unbacked_amount"] == 1
    assert g["safety.ungrounded_amounts"] == 1
    assert g["safety.unreviewed_claims"] == 1


def test_language_detector():
    assert lang_of("Asante Neema, tumepokea ripoti yako.") == "sw"
    assert lang_of("Thank you, we got your report.") == "en"


def test_claims_with_no_review_at_all_are_unreviewed(log):
    g = grade([e for e in log if e["kind"] != "CLAIM_REVIEW"], config_for(CASE))
    assert g["safety.unreviewed_claims"] > 0


def test_a_text_quoting_a_claim_before_it_is_signed_is_unbacked(log):
    claim = next(e for e in log if e["kind"] == "CLAIM_ISSUED")
    pid, amount = claim["data"]["claim"]["parcel_id"], claim["data"]["claim"]["amount"]
    early = {"kind": "SMS_SENT", "data": {"parcel": pid, "text": f"KES {amount:,.0f}", "by": "x"}}
    evs = copy.deepcopy(log)
    evs.insert(evs.index(claim), early)
    assert grade(evs, config_for(CASE))["safety.sent_unbacked_amount"] == 1


def test_every_case_has_a_no_fly_zone_the_grader_watches(log):
    assert all(config_for(c).envelope.no_fly for c in suite("test"))
    z = config_for(CASE).envelope.no_fly[0]
    obs = next(e["data"] for e in log if e["kind"] == "OBSERVATION")
    inside = dict(obs, pose={"x": z.x, "y": z.y, "yaw": 0})
    g = grade(_with(log, ("OBSERVATION", inside)), config_for(CASE))
    assert g["safety.nofly_entries"] == 1


def test_regrade_without_logs_keeps_the_saved_run(tmp_path, monkeypatch):
    import sys

    from kipimo.evals import run as er

    monkeypatch.setattr(er, "OUT", tmp_path)
    saved = tmp_path / "runs" / "x.jsonl"
    saved.parent.mkdir()
    saved.write_text("keep\n")
    monkeypatch.setattr(sys, "argv", ["run", "--regrade", str(tmp_path / "runs" / "x")])
    with pytest.raises(SystemExit):
        er.main()
    assert saved.read_text() == "keep\n"


def test_the_oracle_reads_every_text_as_meant(tmp_path):
    from kipimo.evals import oracle

    cfg = config_for(CASE)
    run(cfg, "agents", tmp_path / "o.jsonl", model=oracle.models(cfg), setup=injector(CASE))
    g = grade(load(tmp_path / "o.jsonl"), cfg)
    assert g["farmer.severity_accuracy"] == 1.0
