from kipimo.world.generate import generate, parcel_truth
from kipimo.world.sms import BANDS, band, inbox

W = generate(7)


def test_inbox_is_deterministic_and_labelled():
    a, b = inbox(W, 7, "adversarial"), inbox(W, 7, "adversarial")
    assert a == b and a
    truth = parcel_truth(W)
    for m in a:
        assert m.label["true"] == band(truth[m.parcel_id])
        if m.label["kind"] == "honest":
            assert m.label["claimed"] == m.label["true"]
    assert {m.lang for m in a} == {"en", "sw"}
    assert sum(m.label["kind"] == "injection" for m in a) == 2
    assert [m.t for m in a] == sorted(m.t for m in a)


def test_calm_profile_has_no_attacks():
    assert all(m.label["kind"] != "injection" for m in inbox(W, 7, "calm"))


def test_bands_follow_payout_thresholds():
    assert [band(x) for x in (0.0, 0.2, 0.4, 0.9)] == list(BANDS)
