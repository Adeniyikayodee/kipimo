import copy

import pytest

from kipimo import config
from kipimo.bus import load, verify_chain
from kipimo.episode import Episode
from kipimo.payout.claim import InsufficientConfidence, PayoutEngine, verify_claim
from kipimo.run import run


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    path = tmp_path_factory.mktemp("a") / "e.jsonl"
    s = run(config.load("demo_flood"), "infogain", path)
    return s, load(path)


def test_chain_verifies(log):
    assert verify_chain(log[1])


@pytest.mark.parametrize("i", [0, 5, 200, -1])
def test_tampering_any_event_breaks_chain(log, i):
    evs = copy.deepcopy(log[1])
    evs[i]["data"]["tampered"] = 1
    assert not verify_chain(evs)


def test_deleting_or_reordering_breaks_chain(log):
    evs = log[1]
    assert not verify_chain(evs[:10] + evs[11:])
    assert not verify_chain([evs[1], evs[0], *evs[2:]])


def test_claims_are_signed_and_verifiable(log):
    s, evs = log
    issued = [e["data"] for e in evs if e["kind"] == "CLAIM_ISSUED"]
    assert len(issued) == s["claims"] >= 2
    for c in issued:
        assert verify_claim(c["claim"], c["sig"], c["pubkey"])
        forged = dict(c["claim"], amount=c["claim"]["amount"] * 10)
        assert not verify_claim(forged, c["sig"], c["pubkey"])


def test_low_confidence_claim_is_refused_and_logged():
    ep = Episode(config.load("demo_flood"))
    eng = PayoutEngine(ep)
    pid = max(ep.world.parcels, key=lambda p: eng.score(p.id)["amount"]).id
    assert eng.score(pid)["gate"] == "fail"
    with pytest.raises(InsufficientConfidence):
        eng.issue(pid, "trying before any survey")
    last = ep.bus.events[-1]
    assert last["kind"] == "CLAIM_REFUSED" and last["data"]["reason"] == "insufficient_confidence"
    assert not eng.issued


def test_audit_cli_passes_then_fails_on_tamper(tmp_path):
    from kipimo.audit import audit

    path = tmp_path / "e.jsonl"
    run(config.load("demo_flood"), "infogain", path)
    r = audit(path)
    assert not r["problems"] and r["claims"] >= 2
    lines = path.read_text().splitlines()
    i = len(lines) // 2
    lines[i] = lines[i].replace('"t":', '"t":9', 1)
    assert '"t":9' in lines[i]
    path.write_text("\n".join(lines) + "\n")
    assert "hash chain broken" in audit(path)["problems"]


def _resign(evs, signer):
    """An attacker rewrites the claims, re-signs them and rebuilds the chain."""
    from kipimo.bus import GENESIS, event_hash

    prev = GENESIS
    for e in evs:
        if e["kind"] == "CLAIM_ISSUED":
            c = e["data"]["claim"] = dict(e["data"]["claim"], amount=999999.0)
            e["data"]["sig"], e["data"]["pubkey"] = signer.sign(c), signer.public_hex
        e["prev"] = prev
        e["hash"] = prev = event_hash(e)
    return evs


@pytest.mark.parametrize("attacker_seed", [99, 7])  # a foreign key, then the leaked demo key
def test_audit_catches_forged_claims_on_a_rebuilt_chain(log, tmp_path, attacker_seed):
    from kipimo.audit import audit
    from kipimo.bus import canonical
    from kipimo.payout.claim import Signer

    evs = _resign(copy.deepcopy(log[1]), Signer(attacker_seed))
    path = tmp_path / "f.jsonl"
    path.write_text("".join(canonical(e).decode() + "\n" for e in evs))
    assert verify_chain(evs)
    problems = audit(path)["problems"]
    assert problems and all("hash chain" not in p for p in problems)


def test_audit_refuses_empty_torn_and_truncated_logs(log, tmp_path):
    from kipimo.audit import audit
    from kipimo.bus import canonical

    lines = [canonical(e).decode() for e in log[1]]
    for name, text in [("empty", ""), ("torn", "\n".join(lines)[:-5])]:
        (tmp_path / name).write_text(text)
        assert audit(tmp_path / name)["problems"]
    (tmp_path / "cut").write_text("\n".join(lines[:50]) + "\n")
    assert audit(tmp_path / "cut", head=log[1][-1]["hash"])["problems"]
