"""Actuary sign-off, the human in the loop.

A claim reaches the actuary when its packet lands at the insurer, so the drone never waits on
a person. Money moves only on a decision signed with the actuary's own key and tied to the
hash of the exact claim. The flags in approval.yaml show where to look first; they never decide.

The actuary may release a claim the adjuster still holds after three rounds. Nobody can
release a claim below the confidence gate or change an amount: that stays with the engine.
"""

import hashlib

import yaml

from kipimo.bus import _plain, canonical
from kipimo.config import CONFIG_DIR
from kipimo.payout import review, triggers
from kipimo.payout.claim import ClaimRefused, Signer

FLAGS = yaml.safe_load((CONFIG_DIR / "approval.yaml").read_text())
TEXT = {r["id"]: r["text"] for r in FLAGS}
CHECKS = {
    "F1_large": lambda f: f["share"] >= 0.5,
    "F2_near_gate": lambda f: f["confidence"] - f["gate"] < 0.05,
    "F3_report_conflict": lambda f: f["report_gap"] >= 2,
    "F4_challenged": lambda f: f["challenges"] > 0,
    "F5_model_review": lambda f: f["model_review"],
    "F6_adjuster_hold": lambda f: f["hold"],
}
DECISIONS = ("approve", "reject")


def claim_hash(doc: dict) -> str:
    return hashlib.sha256(canonical(_plain(doc))).hexdigest()


class Actuary:
    def __init__(self, st, mode: str = "human", name: str = "duty actuary"):
        self.st, self.bus, self.mode, self.name = st, st.ep.bus, mode, name
        self.signer = Signer(st.ep.cfg.seed, "actuary")
        self.cases: dict[str, dict] = {}  # open, by case id
        self.decided: dict[str, dict] = {}
        self.paid: set[str] = set()  # parcels whose claim the actuary approved
        self.released: dict[str, tuple[str, str]] = {}  # parcel -> (by, note) of a lifted hold
        self.seen: dict[str, dict] = {}  # parcel -> review history from the bus
        self.bus.subscribers.append(self._on)

    def _on(self, ev: dict) -> None:
        k, d = ev["kind"], ev["data"]
        if k == "CLAIM_CHALLENGED":
            self._seen(d["parcel"])["challenges"] += 1
        elif k == "CLAIM_REVIEW" and d["decision"] == "approve" and d.get("by") != "rules":
            self._seen(d["parcel"])["model_review"] = True
        elif k == "CLAIM_ESCALATED":
            self._open(f"H-{d['parcel']}", d["parcel"], "hold", d["rule_id"])
        elif k == "UPLINK_SENT":
            pid = next(
                p for p, (c, _) in self.st.engine.issued.items() if c.claim_id == d["claim_id"]
            )
            self._open(d["claim_id"], pid, "claim")

    def _seen(self, pid: str) -> dict:
        return self.seen.setdefault(pid, {"challenges": 0, "model_review": False})

    def facts(self, pid: str, hold: bool) -> dict:
        p, f = self.st.ep.world.parcel(pid), review.facts(self.st, pid)
        issued = self.st.engine.issued.get(pid)
        c = issued[0].doc() if issued else self.st.engine.score(pid)
        return {
            "holder": p.holder,
            "crop": p.crop,
            "sum_insured": p.sum_insured,
            "amount": c["amount"],
            "share": round(c["amount"] / p.sum_insured, 3),
            "tier": c["tier"],
            "trigger_id": c["trigger_id"],
            "loss_mean": c["loss_mean"],
            "loss_ci95": list(c["loss_ci95"]),
            "confidence": c["confidence"],
            "gate": triggers.gate(),
            "coverage": f["coverage"],
            "farmer_report": f["farmer_report"],
            "report_gap": f["report_gap"],
            "hold": hold,
        } | self._seen(pid)

    def _open(self, case_id: str, pid: str, subject: str, rule_id: str = "") -> None:
        if case_id in self.cases or case_id in self.decided:
            return
        f = self.facts(pid, subject == "hold")
        flags = [r["id"] for r in FLAGS if CHECKS[r["id"]](f)]
        case = {"case_id": case_id, "subject": subject, "parcel": pid, "flags": flags, "facts": f}
        self.cases[case_id] = case | {"rule_id": rule_id}
        self.bus.emit(
            "CLAIM_AWAITING_APPROVAL",
            **case,
            rule_id=rule_id,
            flag_text=[TEXT[i] for i in flags],
            mode=self.mode,
        )
        if subject == "claim" and pid in self.released:
            by, note = self.released[pid]
            self.decide(case_id, "approve", by, f"released over the adjuster hold: {note}")
        elif self.mode == "auto" and not flags:
            self.decide(case_id, "approve", "rules", "no flags")

    def decide(self, case_id: str, decision: str, by: str, note: str = "") -> dict | None:
        """None when the case is not open, the decision is unknown, or a rejection has no reason."""
        case, note = self.cases.get(case_id), note.strip()[:300]
        if case is None or decision not in DECISIONS or (decision == "reject" and not note):
            return None
        pid = case["parcel"]
        issued = self.st.engine.issued.get(pid) if case["subject"] == "claim" else None
        doc = {
            "case_id": case_id,
            "subject": case["subject"],
            "parcel_id": pid,
            "claim_id": issued[0].claim_id if issued else None,
            "claim_hash": claim_hash(issued[0].doc()) if issued else None,
            "decision": decision,
            "by": by,
            "note": note,
            "flags": case["flags"],
            "t": round(self.bus.t, 3),
        }
        del self.cases[case_id]
        self.decided[case_id] = doc
        sig = self.signer.sign(doc)
        self.bus.emit("ACTUARY_DECISION", decision=doc, sig=sig, pubkey=self.signer.public_hex)
        if decision == "approve" and case["subject"] == "hold":
            self.released[pid] = (by, note)
            try:
                self.st.engine.issue(pid, f"released by {by}: {note}", released_by=by)
            except ClaimRefused:
                pass  # the engine logged why: the gate and the table still bind
        elif decision == "approve":
            self.paid.add(pid)
            self._tell_farmer(pid)
        return doc

    def approve_clean(self, by: str) -> int:
        clean = [k for k, c in self.cases.items() if c["subject"] == "claim" and not c["flags"]]
        for k in clean:
            self.decide(k, "approve", by, "no flags, approved together")
        return len(clean)

    def _tell_farmer(self, pid: str) -> None:
        desk = getattr(self.st, "desk", None)
        if desk is not None:
            from kipimo.agent import farmer

            farmer.rules_reply(desk, pid, self.st.ep.world.parcel(pid).holder)
