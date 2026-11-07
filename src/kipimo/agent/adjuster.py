"""Insurer's adjuster: reviews each claim that passes the gate, approves or challenges it.

A challenge must cite an audit rule that holds on the evidence; ungrounded ones are rejected.
Approving while a rule holds is allowed but logged, so evals can count audit misses.
"""

from pathlib import Path

from kipimo.agent.farmer import PID
from kipimo.agent.tools import _obj
from kipimo.payout import review

PROMPT = (Path(__file__).parent / "prompts" / "adjuster.md").read_text()
NOTE = {"type": "string"}
SCHEMAS = [
    {
        "name": "get_dossier",
        "description": "Evidence for one claim and the audit rules.",
        "input_schema": _obj({"parcel_id": PID}, ["parcel_id"]),
        "strict": True,
    },
    {
        "name": "approve",
        "description": "Approve the claim for signing.",
        "input_schema": _obj({"parcel_id": PID, "note": NOTE}, ["parcel_id", "note"]),
        "strict": True,
    },
    {
        "name": "challenge",
        "description": "Challenge the claim under an audit rule that holds on the evidence.",
        "input_schema": _obj(
            {"parcel_id": PID, "rule_id": {"type": "string"}, "note": NOTE},
            ["parcel_id", "rule_id", "note"],
        ),
        "strict": True,
    },
]


class AdjusterTools:
    schemas = SCHEMAS

    def __init__(self, st, pid: str):
        self.st, self.pid = st, pid
        self.facts = review.facts(st, pid)
        self.verdict: dict | None = None
        self.decided = False

    @property
    def done(self) -> bool:
        return self.decided

    def call(self, name: str, args: dict) -> dict:
        if args.get("parcel_id") != self.pid:
            return {"error": f"this review is for {self.pid}"}
        if name == "get_dossier":
            return self.facts | {"rules": review.RULES}
        if name == "approve":
            self.decided = True
            return {"approved": True}
        if name == "challenge":
            if args["rule_id"] not in review.holding(self.facts):
                self.st.ep.bus.emit("CHALLENGE_REJECTED", parcel=self.pid, rule_id=args["rule_id"])
                return {"error": "ungrounded: that rule does not hold on this evidence"}
            self.verdict, self.decided = {"rule_id": args["rule_id"], "note": args["note"]}, True
            return {"challenged": True}
        return {"error": f"unknown tool {name}"}


def rules_verdict(st, pid: str) -> dict | None:
    held = review.holding(review.facts(st, pid))
    return {"rule_id": held[0], "note": "audit rule holds"} if held else None
