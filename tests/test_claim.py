from dataclasses import replace

from kipimo import config
from kipimo.agent import policies
from kipimo.episode import Episode
from kipimo.payout import triggers


def _stack_with(loss: float, conf: float = 0.9):
    st = policies.build(Episode(config.load("demo_flood")))
    real, p = st.ep.beliefs, st.ep.world.parcels[0]
    st.ep.beliefs = lambda: real() | {p.id: replace(real()[p.id], loss_mean=loss, confidence=conf)}
    return st, p


def test_claims_are_scored_on_the_loss_they_sign():
    st, p = _stack_with(0.34996)
    claim, _ = st.engine.issue(p.id, "test")
    assert claim.loss_mean == 0.35 and claim.tier == triggers.score(p, 0.35).tier == 0.5


def test_signing_keys_come_from_the_environment_when_set(monkeypatch):
    import pytest
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from kipimo.payout.claim import Signer

    demo = Signer(7).public_hex
    raw = bytes(range(32))
    monkeypatch.setenv("KIPIMO_SIGNING_KEY", raw.hex())
    want = Ed25519PrivateKey.from_private_bytes(raw).public_key().public_bytes_raw().hex()
    assert Signer(7).public_hex == want != demo
    assert Signer(7, "actuary").public_hex != want  # the actuary keeps its own key
    monkeypatch.setenv("KIPIMO_ACTUARY_KEY", "abcd")
    with pytest.raises(ValueError):
        Signer(7, "actuary")
