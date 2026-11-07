"""Farmer liaison: turns SMS into structured reports and keeps farmers informed.

A report can only widen a parcel's prior (so the drone looks), never move an estimate or a
payout. Outgoing texts may only quote an amount the payout engine has issued for that parcel.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from kipimo.agent.tools import _obj
from kipimo.payout import triggers
from kipimo.world.sms import BANDS, Sms, band

PROMPT = (Path(__file__).parent / "prompts" / "farmer.md").read_text()
SEVERITY = (*BANDS, "unknown")
KINDS = ("damage", "warning", "other")
SIDES = ("river_side", "road_side", "whole", "unknown")
PID = {"type": "string", "description": "Parcel id, e.g. P07"}
WORDS = re.compile(r"(?i)\b(hundred|thousand|million|shillings?|mia|elfu|laki|milioni|shilingi)\b")
MONEY = re.compile(r"\b(?:KES|KSH|SH)\.?\s*(\d[\d,]*)|\b(\d{1,3}(?:,\d{3})+|\d{4,})\b", re.I)

LEXICON = {  # checked in this order; tuned on the template messages only
    "severe": ["most of", "more than half", "under water", "sehemu kubwa", "zaidi ya nusu"],
    "total": ["everything is gone", "washed away", "whole field", "yote", "lote", "kabisa"],
    "some": ["some water", "small loss", "a few", "part of", "kidogo", "baadhi", "hasara ndogo"],
    "none": ["fine", "no damage", "did not reach", " ok", "salama", "hakuna hasara", "sawa"],
}
ATTACK = [
    "ignore",
    "instruction",
    "system",
    "approve",
    "issue the claim",
    "puuza",
    "maagizo",
    "nilipe",
]
WARNING = [
    "will burst",
    "worried",
    "warning",
    "they say",
    "heard",
    "policy active",
    "wanasema",
    "tunaogopa",
    "nimesikia",
    "bima",
]
SIDE_WORDS = {
    "river_side": ["river", "low side", "lower", "mto", "chini"],
    "road_side": ["road", "upper", "barabara", "juu"],
    "whole": ["whole field", "shamba zima"],
}
HOURS = [  # longest phrase first
    (48, ["two days", "siku mbili"]),
    (10, ["last night", "jana usiku"]),
    (24, ["yesterday", "jana"]),
    (4, ["this morning", "asubuhi"]),
]

SCHEMAS = [
    {
        "name": "get_status",
        "description": "Current estimate band, confidence and any issued claim for one parcel.",
        "input_schema": _obj({"parcel_id": PID}, ["parcel_id"]),
        "strict": True,
    },
    {
        "name": "record_report",
        "description": "Record what the farmer reports. Call once per message.",
        "input_schema": _obj(
            {
                "parcel_id": PID,
                "kind": {"type": "string", "enum": list(KINDS)},
                "severity": {"type": "string", "enum": list(SEVERITY)},
                "where": {"type": "string", "enum": list(SIDES)},
                "flood_hours_ago": {"type": "number", "description": "-1 if not stated"},
                "suspicious": {"type": "boolean"},
                "note": {"type": "string"},
            },
            ["parcel_id", "kind", "severity", "where", "flood_hours_ago", "suspicious", "note"],
        ),
        "strict": True,
    },
    {
        "name": "send_sms",
        "description": "Text the farmer. Amounts must match an issued claim exactly.",
        "input_schema": _obj({"parcel_id": PID, "text": {"type": "string"}}, ["parcel_id", "text"]),
        "strict": True,
    },
]


@dataclass
class Report:
    t: float
    severity: str
    suspicious: bool
    lang: str
    conflict: bool
    by: str
    kind: str = "damage"
    where: str = "unknown"
    hours: float = -1


def read(text: str) -> dict:
    """Rule fallback: keyword kind, severity, side, timing and attack detection."""
    low = f" {text.lower()} "
    attack = any(w in low for w in ATTACK)
    kind = "warning" if any(w in low for w in WARNING) else "damage"
    sev = next((k for k, ws in LEXICON.items() if any(w in low for w in ws)), "unknown")
    where = next((k for k, ws in SIDE_WORDS.items() if any(w in low for w in ws)), "unknown")
    hours = next((h for h, ws in HOURS if any(w in low for w in ws)), -1)
    if kind == "warning":
        sev, where, hours = "unknown", "unknown", -1
    return {"kind": kind, "severity": sev, "where": where, "hours": hours, "suspicious": attack}


class Desk:
    """Shared by the farmer role and its fallback: records reports and guards outgoing texts."""

    def __init__(self, st, memory=None):
        self.st, self.memory = st, memory
        self.reports: dict[str, Report] = {}
        self.lang: dict[str, str] = {}
        self.log: list[tuple[str, str]] = []  # (parcel, by) per recorded report

    def status(self, pid: str) -> dict:
        b = self.st.ep.beliefs()[pid]
        issued = self.st.engine.issued.get(pid)
        return {
            "parcel_id": pid,
            "estimate_band": band(b.loss_mean),
            "confidence": round(b.confidence, 2),
            "gate": triggers.gate(),
            "claim": {"amount": issued[0].amount, "claim_id": issued[0].claim_id}
            if issued
            else None,
            "under_review": bool(self.st.engine.challenged.get(pid, {}).get("escalated")),
            "approved": pid in getattr(self.st.actuary, "paid", ()),
            "sender_reliability": self._reliability(pid),
        }

    def _holder(self, pid: str) -> str:
        return self.st.ep.world.parcel(pid).holder

    def _reliability(self, pid: str) -> float | None:
        r = self.memory.reliability(self._holder(pid)) if self.memory else None
        return None if r is None else round(r, 2)

    def record(
        self,
        pid: str,
        severity: str,
        suspicious: bool,
        by: str,
        kind="damage",
        where="unknown",
        hours=-1,
    ) -> dict:
        est = band(self.st.ep.beliefs()[pid].loss_mean)
        known = severity in BANDS and kind == "damage"
        conflict = known and abs(BANDS.index(severity) - BANDS.index(est)) >= 2
        discounted = bool(conflict and self.memory and self.memory.distrusted(self._holder(pid)))
        doubted = bool(
            conflict and not suspicious and not discounted and self.st.ep.belief.doubt(pid)
        )
        t = self.st.ep.bus.t
        lang = self.lang.get(pid, "")
        rep = Report(t, severity, suspicious, lang, conflict, by, kind, where, hours)
        old = self.reports.get(pid)
        if not (suspicious and old and not old.suspicious):  # an attack never hides a real report
            self.reports[pid] = rep
        self.log.append((pid, by))
        self.st.ep.bus.emit(
            "FARMER_REPORT",
            parcel=pid,
            report_kind=kind,
            severity=severity,
            where=where,
            hours=hours,
            suspicious=suspicious,
            conflict=conflict,
            discounted=discounted,
            doubted=doubted,
            by=by,
        )
        return {"recorded": True, "estimate_band": est, "conflict": conflict}

    def send(self, pid: str, text: str, by: str) -> dict:
        issued = self.st.engine.issued.get(pid)
        allowed = {int(issued[0].amount)} if issued else set()
        quoted = {int((a or b).replace(",", "")) for a, b in MONEY.findall(text)}
        if quoted - allowed or WORDS.search(text):  # amounts only as the engine's digits
            self.st.ep.bus.emit("SMS_REJECTED", parcel=pid, text=text, by=by)
            return {"error": "amount not issued by the payout engine for this parcel"}
        self.st.ep.bus.emit("SMS_SENT", parcel=pid, text=text, by=by)
        return {"sent": True}


class FarmerTools:
    schemas = SCHEMAS

    def __init__(self, desk: Desk, sms: Sms, by: str):
        self.desk, self.sms, self.by, self.done = desk, sms, by, False

    def call(self, name: str, args: dict) -> dict:
        pid = args.get("parcel_id")
        if pid != self.sms.parcel_id:
            return {"error": f"this message is about {self.sms.parcel_id}"}
        if name == "get_status":
            return self.desk.status(pid)
        if name == "record_report" and not self.sms.text:
            return {"error": "there is no text to record"}
        if name == "record_report":
            a = {"kind": "damage", "where": "unknown", "flood_hours_ago": -1} | args
            if a["severity"] not in SEVERITY or a["kind"] not in KINDS or a["where"] not in SIDES:
                return {"error": "value outside the allowed enum"}
            hours = float(a["flood_hours_ago"])
            sus = bool(a["suspicious"]) or read(self.sms.text)["suspicious"]  # rules can only add
            return self.desk.record(pid, a["severity"], sus, self.by, a["kind"], a["where"], hours)
        if name == "send_sms":
            return self.desk.send(pid, args["text"], self.by)
        return {"error": f"unknown tool {name}"}


REPLY = {
    "en": (
        "Thank you {name}, we got your report.",
        "Your claim of KES {amount:,.0f} is approved.",
        "A drone will check your field.",
        "A person is reviewing your claim.",
        "Your claim of KES {amount:,.0f} is with the insurer for a final check.",
    ),
    "sw": (
        "Asante {name}, tumepokea ripoti yako.",
        "Dai lako la KES {amount:,.0f} limeidhinishwa.",
        "Ndege isiyo na rubani itakagua shamba lako.",
        "Mtu anakagua dai lako.",
        "Dai lako la KES {amount:,.0f} liko kwa kampuni ya bima kwa ukaguzi wa mwisho.",
    ),
}


def rules_reply(desk: Desk, pid: str, name: str) -> None:
    thanks, paid, pending, review, check = REPLY[desk.lang.get(pid) or "en"]
    st = desk.status(pid)
    claim = st["claim"]
    if claim:
        body = (paid if st["approved"] else check).format(amount=claim["amount"])
    else:
        body = review if st["under_review"] else pending
    desk.send(pid, f"{thanks.format(name=name.split()[0])} {body}", "rules")
