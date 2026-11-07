"""A farmer reader that reads every text exactly as its sender meant it, from the hidden label.

No model can beat it, so rules against oracle is the most that reading texts can be worth.
Field and adjuster stay on rules, so the difference is the reading alone.
"""

from kipimo.agent.model import ScriptedModel
from kipimo.world import sms
from kipimo.world.generate import generate

KIND = {"warning": "warning", "injection": "other"}


def models(cfg) -> dict:
    world = generate(cfg.seed, cfg.world)
    a = cfg.agent
    said = {
        (m.parcel_id, repr(m.text)): m.label
        for m in sms.inbox(world, cfg.seed, a.sms_profile, a.sms_style)
    }

    def read(m: ScriptedModel, results: list) -> list:
        hit = next(
            ((p, lab) for (p, t), lab in said.items() if p in m.brief and t in m.brief), None
        )
        if results or not hit:
            return []
        pid, lab = hit
        args = {
            "parcel_id": pid,
            "severity": lab["claimed"] or "unknown",
            "suspicious": lab["kind"] == "injection",
            "kind": KIND.get(lab["kind"], "damage"),
            "where": lab["where"] or "unknown",
            "flood_hours_ago": -1 if lab["hours"] is None else lab["hours"],
        }
        return [("record_report", args)]

    return {"farmer": ScriptedModel(read, "oracle")}
