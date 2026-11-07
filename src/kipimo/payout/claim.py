import hashlib
import os
from dataclasses import asdict, dataclass

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from kipimo.bus import _plain, canonical
from kipimo.payout import triggers


class ClaimRefused(Exception):
    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class InsufficientConfidence(ClaimRefused):
    pass


@dataclass(frozen=True)
class Claim:
    claim_id: str
    event: int  # the flood this claim settles: one claim per event and parcel
    parcel_id: str
    holder: str
    crop: str
    sum_insured: float
    ruleset: str
    loss_mean: float
    loss_ci95: tuple[float, float]
    confidence: float
    trigger_id: int
    tier: float
    amount: float
    belief_hash: str
    t: float
    justification: str

    def doc(self) -> dict:
        d = asdict(self)
        d["loss_ci95"] = list(d["loss_ci95"])
        return d


KEY_ENV = {"demo": "KIPIMO_SIGNING_KEY", "actuary": "KIPIMO_ACTUARY_KEY"}


class Signer:
    """The key named in KIPIMO_SIGNING_KEY or KIPIMO_ACTUARY_KEY (64 hex characters), else a demo
    key derived from the seed. A deployment would keep the key in secure hardware."""

    def __init__(self, seed: int, role: str = "demo"):
        given = os.environ.get(KEY_ENV.get(role, ""), "").strip()
        raw = (
            bytes.fromhex(given)
            if given
            else hashlib.sha256(f"kipimo-{role}-key-{seed}".encode()).digest()
        )
        if len(raw) != 32:
            raise ValueError(f"{KEY_ENV[role]} must be 32 bytes as 64 hex characters")
        self.key = Ed25519PrivateKey.from_private_bytes(raw)
        self.public_hex = self.key.public_key().public_bytes_raw().hex()

    def sign(self, doc: dict) -> str:
        return self.key.sign(canonical(_plain(doc))).hex()  # exactly as the bus logs it


def verify_claim(doc: dict, sig_hex: str, public_hex: str) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_hex)).verify(
            bytes.fromhex(sig_hex), canonical(doc)
        )
        return True
    except (InvalidSignature, ValueError, TypeError):  # bad signature, bad hex or a bad key
        return False


def belief_hash(n: np.ndarray, sx: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in (n, sx):  # little endian doubles, no negative zero: same bytes on any machine
        h.update((np.round(a, 6) + 0.0).astype("<f8").tobytes())
    return h.hexdigest()


class PayoutEngine:
    """Deterministic claim issuance. The agent can ask; only this decides."""

    def __init__(self, ep):
        self.ep = ep
        self.signer = Signer(ep.cfg.seed)
        self.issued: dict[str, tuple[Claim, str]] = {}
        self.reviewer = None  # optional (parcel_id, score, justification) -> challenge or None
        self.challenged: dict[str, dict] = {}

    def score(self, parcel_id: str) -> dict:
        p = self.ep.world.parcel(parcel_id)
        b = self.ep.beliefs()[parcel_id]
        loss, conf = round(b.loss_mean, 4), round(b.confidence, 3)  # decide on what gets signed
        s = triggers.score(p, loss)
        gate = triggers.gate()
        return {
            "parcel_id": parcel_id,
            "loss_mean": loss,
            "loss_ci95": [round(b.loss_ci95[0], 4), round(b.loss_ci95[1], 4)],
            "confidence": conf,
            "trigger_id": s.trigger_id,
            "tier": s.tier,
            "amount": s.amount,
            "gate": "pass" if conf >= gate else "fail",
            "min_confidence": gate,
            "issued": parcel_id in self.issued,
        }

    def issue(self, parcel_id: str, justification: str, released_by: str = "") -> tuple[Claim, str]:
        """released_by names the person who overruled an adjuster hold. Nothing overrules the
        gate or the trigger table."""
        bus = self.ep.bus
        sc = self.score(parcel_id)
        if parcel_id in self.issued:
            reason = "already_issued"
        elif sc["gate"] == "fail":
            reason = "insufficient_confidence"
        elif sc["amount"] <= 0:
            reason = "no_trigger"
        else:
            reason = ""
        if reason:
            bus.emit("CLAIM_REFUSED", parcel=parcel_id, reason=reason, score=sc)
            cls = InsufficientConfidence if reason == "insufficient_confidence" else ClaimRefused
            raise cls(reason, sc)
        review = self.reviewer and not released_by
        verdict = self.reviewer(parcel_id, sc, justification) if review else None
        if verdict:
            self.challenged[parcel_id] = verdict
            bus.emit("CLAIM_CHALLENGED", parcel=parcel_id, **verdict)
            raise ClaimRefused("challenged", sc | verdict)
        self.challenged.pop(parcel_id, None)
        p = self.ep.world.parcel(parcel_id)
        g = self.ep.belief
        claim = Claim(
            claim_id=f"E{self.ep.cfg.seed}-{parcel_id}",
            event=self.ep.cfg.seed,
            parcel_id=parcel_id,
            holder=p.holder,
            crop=p.crop,
            sum_insured=p.sum_insured,
            ruleset=triggers.version(),
            loss_mean=sc["loss_mean"],
            loss_ci95=tuple(sc["loss_ci95"]),
            confidence=sc["confidence"],
            trigger_id=sc["trigger_id"],
            tier=sc["tier"],
            amount=sc["amount"],
            belief_hash=belief_hash(*g.evidence(p.cells)),
            t=round(bus.t, 3),
            justification=justification,
        )
        sig = self.signer.sign(claim.doc())
        self.issued[parcel_id] = (claim, sig)
        bus.emit("CLAIM_ISSUED", claim=claim.doc(), sig=sig, pubkey=self.signer.public_hex)
        return claim, sig
