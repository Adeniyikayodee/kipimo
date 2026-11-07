"""Claim packet: one SMS carrying a claim and an ed25519 signature over its exact body bytes.

packet = CBOR [body, sig64], body = CBOR bytes of
[version, event, parcel, loss_bp, confidence_pct, trigger_id, tier_pct, t_s, ruleset4, belief8]
Amounts are not sent: the insurer derives them from tier and the policy's sum insured.
"""

import re

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from kipimo.payout.claim import Claim, Signer

VERSION = 2
SMS_LIMIT = 140
FIELDS = "version event parcel loss_bp confidence_pct trigger_id tier_pct t_s".split()
LIMITS = (VERSION, 2**32, 10**4, 10**4, 100, 2**16, 100, 2**32)  # inclusive maxima


def encode(claim: Claim, signer: Signer) -> bytes:
    body = cbor2.dumps(
        [
            VERSION,
            claim.event,
            int(re.fullmatch(r"P(\d{2,})", claim.parcel_id)[1]),
            round(claim.loss_mean * 10**4),
            round(claim.confidence * 100),
            claim.trigger_id,
            round(claim.tier * 100),
            int(claim.t),
            bytes.fromhex(claim.ruleset),
            bytes.fromhex(claim.belief_hash)[:8],
        ]
    )
    return cbor2.dumps([body, signer.key.sign(body)])


def decode(packet: bytes) -> dict:
    """Fields of a well formed packet, unverified. ValueError on anything else."""
    try:
        raw, sig = cbor2.loads(packet)
        *ints, ruleset, belief = cbor2.loads(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"malformed packet: {e}") from e
    ok = len(ints) == len(FIELDS) and ints[0] == VERSION
    ok = ok and all(type(v) is int and 0 <= v <= m for v, m in zip(ints, LIMITS, strict=True))
    lens = [(ruleset, 4), (belief, 8), (sig, 64)]
    if not ok or not all(isinstance(b, bytes) and len(b) == n for b, n in lens):
        raise ValueError("malformed packet")
    d = dict(zip(FIELDS, ints, strict=True))
    d["parcel_id"], d["ruleset"] = f"P{d['parcel']:02d}", ruleset.hex()
    return d | {"raw": raw, "sig": sig}


def verify(packet: bytes, public_hex: str) -> dict | None:
    """Decoded fields if well formed and signed by the key, else None."""
    try:
        d = decode(packet)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(d["sig"], d["raw"])
        return d
    except (ValueError, InvalidSignature):
        return None
