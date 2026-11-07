from kipimo import config
from kipimo.agent import policies
from kipimo.agent.farmer import Desk, read
from kipimo.episode import Episode
from kipimo.world.sms import INJECTIONS, TEXT

CFG = config.load("demo_flood", belief={"model": "gp"})


def test_lexicon_reads_every_template_and_flags_attacks():
    for bands in TEXT.values():
        for sev, texts in bands.items():
            for t in texts:
                r = read(t.format(crop="maize"))
                assert (r["severity"], r["suspicious"]) == (sev, False), t
    assert all(read(t)["suspicious"] for _, t in INJECTIONS)


def test_texts_can_only_quote_issued_amounts():
    st = policies.build(Episode(CFG))
    desk, pid = Desk(st), st.ep.world.parcels[0].id
    assert "error" in desk.send(pid, "You will get KES 50,000", "rules")
    assert desk.send(pid, "A drone will check your field.", "rules") == {"sent": True}


def test_conflicting_report_widens_the_prior_but_suspicious_ones_do_not():
    st = policies.build(Episode(CFG))
    desk = Desk(st)
    low = [p for p in st.ep.world.parcels if st.ep.beliefs()[p.id].loss_mean < 0.15]
    a, b = low[0].id, low[1].id
    w0 = st.ep.beliefs()[a].loss_ci95
    assert desk.record(a, "total", False, "rules")["conflict"]
    assert st.ep.belief.doubted == {a}
    assert st.ep.beliefs()[a].loss_ci95[1] - st.ep.beliefs()[a].loss_ci95[0] > w0[1] - w0[0]
    desk.record(b, "total", True, "rules")
    assert b not in st.ep.belief.doubted


def test_lexicon_reads_side_timing_and_warnings():
    r = read(
        "The flood destroyed most of my maize. Water on the low side by the river, last night."
    )
    assert (r["kind"], r["severity"], r["where"], r["hours"]) == (
        "damage",
        "severe",
        "river_side",
        10,
    )
    w = read("Wanasema mto utafurika tena usiku huu, tunaogopa.")
    assert w["kind"] == "warning" and w["severity"] == "unknown"


def test_warnings_never_widen_the_prior():
    st = policies.build(Episode(CFG))
    desk = Desk(st)
    low = next(p for p in st.ep.world.parcels if st.ep.beliefs()[p.id].loss_mean < 0.15)
    out = desk.record(low.id, "total", False, "rules", kind="warning")
    assert not out["conflict"] and not st.ep.belief.doubted


def test_only_digits_after_a_currency_word_are_money():
    st = policies.build(Episode(CFG))
    desk, pid = Desk(st), st.ep.world.parcels[0].id
    assert desk.send(pid, "Asante, we will finish, then text you", "rules") == {"sent": True}
    assert "error" in desk.send(pid, "Utapata Ksh. 5,000", "rules")
    for words in ("Your claim of fifty thousand shillings is approved", "Utalipwa elfu hamsini"):
        assert "error" in desk.send(pid, words, "rules")


def test_bad_tool_arguments_come_back_as_errors():
    from kipimo.agent.farmer import FarmerTools
    from kipimo.agent.model import ScriptedModel
    from kipimo.agent.runtime import Role
    from kipimo.world.sms import Sms

    st = policies.build(Episode(CFG))
    pid = st.ep.world.parcels[0].id
    box = FarmerTools(Desk(st), Sms(0, pid, "x", "en", "hi", {}), "agent")
    bad = {"parcel_id": pid, "severity": "some", "suspicious": False, "flood_hours_ago": "jana"}
    seen = []

    def script(m, results):
        seen.extend(results)
        return [] if m.n else [("record_report", bad)]

    Role("farmer", ScriptedModel(script), "", box, st.ep.bus, 5, 60).wake("go")
    assert "error" in seen[0]


def test_a_model_cannot_clear_an_injection_the_rules_flag():
    from kipimo.agent.farmer import FarmerTools
    from kipimo.world.sms import Sms

    st = policies.build(Episode(CFG))
    low = next(p for p in st.ep.world.parcels if st.ep.beliefs()[p.id].loss_mean < 0.15).id
    box = FarmerTools(Desk(st), Sms(0, low, "x", "en", INJECTIONS[0][1], {}), "model")
    args = {"parcel_id": low, "severity": "total", "suspicious": False}
    assert box.call("record_report", args)["recorded"]
    assert low not in st.ep.belief.doubted


def test_an_attack_never_replaces_a_real_report():
    st = policies.build(Episode(CFG))
    desk, pid = Desk(st), st.ep.world.parcels[0].id
    desk.record(pid, "some", False, "rules")
    desk.record(pid, "total", True, "rules")
    assert desk.reports[pid].severity == "some"
