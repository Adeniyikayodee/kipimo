"""Insurer audit policy: rules are data, and whether one holds is computed from the evidence."""

import yaml

from kipimo.config import CONFIG_DIR
from kipimo.payout import triggers
from kipimo.world.sms import BANDS, band

RULES = yaml.safe_load((CONFIG_DIR / "audit.yaml").read_text())
CHECKS = {
    "A1_coverage": lambda f: f["coverage"] < 0.25,
    "A2_report_conflict": lambda f: f["report_gap"] >= 2 and f["coverage"] < 0.5,
    "A3_tier_edge": lambda f: f["edge"] < 0.03 and f["confidence"] < 0.9,
}


def facts(st, pid: str) -> dict:
    p, b = st.ep.world.parcel(pid), st.ep.beliefs()[pid]
    rep = getattr(st, "desk", None) and st.desk.reports.get(pid)
    known = rep and rep.severity in BANDS and not rep.suspicious
    return {
        "parcel_id": pid,
        "loss": round(b.loss_mean, 3),
        "confidence": round(b.confidence, 3),
        "coverage": round(b.cells_observed / b.cells_total, 3),
        "edge": round(min(abs(b.loss_mean - x) for x in triggers.boundaries(p.crop)), 3),
        "estimate_band": band(b.loss_mean),
        "farmer_report": rep.severity if rep else None,
        "report_gap": abs(BANDS.index(rep.severity) - BANDS.index(band(b.loss_mean)))
        if known
        else 0,
    }


def holding(f: dict) -> list[str]:
    return [r["id"] for r in RULES if CHECKS[r["id"]](f)]
