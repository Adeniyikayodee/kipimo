import cbor2
import pytest

from kipimo import config
from kipimo.agent import policies
from kipimo.bus import load
from kipimo.episode import Episode
from kipimo.uplink import codec
from kipimo.uplink.queue import Outbox
from kipimo.uplink.station import Receiver


def _stack(tmp_path, **uplink):
    cfg = config.load("demo_flood", uplink=uplink or None)
    return policies.build(Episode(cfg, out=tmp_path / "e.jsonl"))


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    st = _stack(tmp_path_factory.mktemp("u"), schedule=[{"t": 0, "state": "offline"}])
    policies.baseline(st)
    return st


def test_packet_fits_one_sms_and_verifies(offline_run):
    st = offline_run
    claims = [c for c, _ in st.engine.issued.values()]
    assert len(claims) >= 2
    for c in claims:
        pkt = codec.encode(c, st.engine.signer)
        assert len(pkt) < codec.SMS_LIMIT
        d = codec.verify(pkt, st.engine.signer.public_hex)
        assert (d["parcel_id"], d["tier_pct"]) == (c.parcel_id, round(c.tier * 100))
        assert d["loss_bp"] == round(c.loss_mean * 10**4)
        raw, sig = cbor2.loads(pkt)
        body = cbor2.loads(raw)
        body[6] = 50 if body[6] == 100 else 100  # forge the tier
        forged = cbor2.dumps([cbor2.dumps(body), sig])
        assert codec.verify(forged, st.engine.signer.public_hex) is None


def test_replays_and_garbage_settle_nothing_twice(offline_run):
    st = offline_run
    claim = next(iter(st.engine.issued.values()))[0]
    pkt = codec.encode(claim, st.engine.signer)
    r = Receiver(st.engine.signer.public_hex)
    for p in (pkt, pkt, b"\x00", cbor2.dumps([1, 2, 3]), cbor2.dumps([pkt, b"x"])):
        r.receive(p)
    assert (len(r.settled), r.duplicates, r.rejected) == (1, 1, 3)


def test_offline_then_online_flushes_each_claim_exactly_once(offline_run):
    st = offline_run
    n = len(st.engine.issued)
    assert st.uplink.outbox.depth() == n and not st.uplink.receiver.settled
    t_off = st.ep.bus.t
    st.uplink.schedule = [{"t": t_off + 60, "state": "online"}]
    st.uplink.drain(1.0)
    assert st.ep.bus.t >= t_off + 60
    assert st.uplink.outbox.depth() == 0
    assert len(st.uplink.receiver.settled) == n and st.uplink.receiver.duplicates == 0
    queued = [e["data"]["claim_id"] for e in st.ep.bus.events if e["kind"] == "UPLINK_QUEUED"]
    assert [f"E{e}-{p}" for e, p in st.uplink.receiver.settled] == queued


def test_lossy_link_resends_and_receiver_dedupes(tmp_path):
    st = _stack(tmp_path, state="degraded", loss_pct=50.0)
    policies.baseline(st)
    st.uplink.drain(1.0)
    kinds = [e["kind"] for e in load(tmp_path / "e.jsonl")]
    assert "UPLINK_RETRY" in kinds
    assert len(st.uplink.receiver.settled) == len(st.engine.issued)
    assert st.uplink.outbox.depth() == 0


def test_outbox_survives_restart_and_dedupes(tmp_path):
    db = tmp_path / "q.sqlite"
    q = Outbox(db)
    assert q.put("C1", b"a") and q.put("C2", b"b") and not q.put("C1", b"a")
    q.mark(q.pending()[0][0], True)
    q2 = Outbox(db)
    assert [c for _, c, _ in q2.pending()] == ["C2"] and q2.depth() == 1


def test_outbox_refuses_a_second_packet_for_one_claim():
    o = Outbox()
    assert o.put("E7-P01", b"a") and not o.put("E7-P01", b"a")
    with pytest.raises(ValueError):
        o.put("E7-P01", b"b")
