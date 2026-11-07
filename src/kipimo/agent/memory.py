"""Memory that outlives a mission: how each farmer's reports compared with what the drone
measured, and short notes the agents chose to keep. One bounded JSON file.

Only settled evidence teaches: a report is scored against the final estimate of a parcel that
passed the confidence gate and was not challenged. Reliability is the Beta(1, 1) posterior
mean of not overstating, and needs MIN_REPORTS before it can distrust anyone.
"""

import json
from pathlib import Path

from kipimo.payout import triggers
from kipimo.world.sms import BANDS, band

MAX_NOTES = 20
DISTRUST = 0.5  # below this, a farmer's report no longer sends the drone
MIN_REPORTS = 3


class Memory:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        data = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        self.farmers: dict[str, dict] = data.get("farmers", {})
        self.notes: list[dict] = data.get("notes", [])

    def reliability(self, holder: str) -> float | None:
        f = self.farmers.get(holder)
        if not f or not f["reports"]:
            return None
        return (f["reports"] - f["overstated"] + 1) / (f["reports"] + 2)

    def distrusted(self, holder: str) -> bool:
        r = self.reliability(holder)
        return r is not None and self.farmers[holder]["reports"] >= MIN_REPORTS and r < DISTRUST

    def note(self, text: str, by: str) -> dict:
        self.notes = (self.notes + [{"by": by, "text": text[:300]}])[-MAX_NOTES:]
        return {"kept": True, "notes": len(self.notes)}

    def learn(self, st, reports: dict) -> dict:
        """Score each damage report against a settled estimate, then save."""
        beliefs, gate, seen = st.ep.beliefs(), triggers.gate(), 0
        for pid, rep in reports.items():
            b = beliefs[pid]
            if (
                rep.suspicious
                or rep.kind != "damage"
                or rep.severity not in BANDS
                or b.confidence < gate
                or pid in st.engine.challenged
            ):
                continue
            holder = st.ep.world.parcel(pid).holder
            f = self.farmers.setdefault(holder, {"reports": 0, "overstated": 0})
            f["reports"] += 1
            f["overstated"] += BANDS.index(rep.severity) > BANDS.index(band(b.loss_mean))
            seen += 1
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"farmers": self.farmers, "notes": self.notes}, indent=1)
            )
        return {"scored": seen, "farmers": len(self.farmers), "notes": len(self.notes)}
