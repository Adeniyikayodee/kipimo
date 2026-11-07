"""Dispatch: several villages flood and each hub has one drone. Who goes where, in what order.

Graded on money left waiting: owed payout times hours until each reached village is settled,
with owed money out of every hub's reach reported separately. The rules
fallback is greedy list scheduling by expected payout per hour of drone time; the naive baseline
sends each village to its nearest hub, nearest first.
"""

import math
from pathlib import Path

from kipimo.agent.model import ModelError
from kipimo.agent.runtime import Role
from kipimo.agent.tools import _obj
from kipimo.planning.district import Village, mission_config
from kipimo.run import run

PROMPT = (Path(__file__).parent / "prompts" / "dispatcher.md").read_text()
RECHARGE_H, MISSION_H = 0.75, 0.6
SCHEMAS = [
    {
        "name": "get_district",
        "description": "Flooded villages, expected payouts, and which hubs reach each and how far.",
        "input_schema": _obj({}, []),
        "strict": True,
    },
    {
        "name": "assign",
        "description": "Queue a village on a hub. A hub works its queue in assignment order.",
        "input_schema": _obj(
            {
                "village_id": {"type": "string"},
                "hub_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            ["village_id", "hub_id", "reason"],
        ),
        "strict": True,
    },
    {
        "name": "finish",
        "description": "End planning with a one sentence summary.",
        "input_schema": _obj({"summary": {"type": "string"}}, ["summary"]),
        "strict": True,
    },
]


class District:
    def __init__(
        self, vs: list[Village], hubs: list[tuple[float, float]], radius: float, kmh: float
    ):
        self.vs = {v.id: v for v in vs if v.flooded and v.expected > 0}
        self.hubs = {f"H{i}": h for i, h in enumerate(hubs)}
        self.radius, self.kmh = radius, kmh
        self._hours: dict[str, float] = {}

    def km(self, vid: str, hid: str) -> float:
        v = self.vs[vid]
        return math.dist((v.x, v.y), self.hubs[hid])

    def mission_h(self, vid: str, models=None) -> float:
        """Survey time does not depend on dispatch order, so each village flies once."""
        key = (vid, id(models))
        if key not in self._hours:
            s = run(
                mission_config(self.vs[vid]), "agents", model=models if models is not None else {}
            )
            self._hours[key] = s["duration_s"] / 3600
        return self._hours[key]

    def reach(self, vid: str) -> list[str]:
        return [h for h in self.hubs if self.km(vid, h) <= self.radius]


class DispatchTools:
    schemas = SCHEMAS

    def __init__(self, d: District):
        self.d, self.queues, self.done = d, {h: [] for h in d.hubs}, False

    def call(self, name: str, args: dict) -> dict:
        d = self.d
        if name == "get_district":
            return {
                "villages": [
                    {
                        "id": v.id,
                        "expected_payout": round(v.expected),
                        "hubs_in_reach": {h: round(d.km(v.id, h), 1) for h in d.reach(v.id)},
                    }
                    for v in d.vs.values()
                ],
                "hubs": list(d.hubs),
                "drone_kmh": round(d.kmh, 1),
                "survey_hours": MISSION_H,
                "recharge_hours": RECHARGE_H,
            }
        if name == "assign":
            vid, hid = args["village_id"], args["hub_id"]
            if vid not in d.vs or hid not in d.hubs:
                return {"error": "unknown village or hub"}
            if hid not in d.reach(vid):
                return {"error": f"{hid} cannot reach {vid} on one battery"}
            if any(vid in q for q in self.queues.values()):
                return {"error": f"{vid} is already assigned"}
            self.queues[hid].append(vid)
            return {"queued": True, "position": len(self.queues[hid])}
        if name == "finish":
            self.done = True
            return {"planned": sum(map(len, self.queues.values()))}
        return {"error": f"unknown tool {name}"}


def greedy(d: District) -> dict[str, list[str]]:
    """List scheduling: next, the hub and village with the most expected money per finish hour."""
    free, queues, left = dict.fromkeys(d.hubs, 0.0), {h: [] for h in d.hubs}, set(d.vs)
    while left:
        best = None
        for vid in sorted(left):  # fixed order, so ties never depend on the hash seed
            for hid in d.reach(vid):
                done_h = free[hid] + d.km(vid, hid) / d.kmh + MISSION_H
                score = d.vs[vid].expected / done_h
                if best is None or score > best[0]:
                    best = (score, vid, hid, done_h)
        if best is None:
            break
        _, vid, hid, done_h = best
        queues[hid].append(vid)
        free[hid] = done_h + d.km(vid, hid) / d.kmh + RECHARGE_H
        left.discard(vid)
    return queues


def nearest(d: District) -> dict[str, list[str]]:
    queues = {h: [] for h in d.hubs}
    for vid in d.vs:
        hs = d.reach(vid)
        if hs:
            queues[min(hs, key=lambda h: d.km(vid, h))].append(vid)
    return {h: sorted(q, key=lambda v: d.km(v, h)) for h, q in queues.items()}


def execute(d: District, queues: dict[str, list[str]], models=None) -> dict:
    """Fly each hub's queue in sim time, running every village's mission for real."""
    settled = {}
    for hid, q in queues.items():
        clock = 0.0
        for vid in q:
            out = d.km(vid, hid) / d.kmh
            settled[vid] = clock + out + d.mission_h(vid, models)
            clock = settled[vid] + out + RECHARGE_H
    owed = {vid: v.owed for vid, v in d.vs.items()}
    return {
        "settled_h": {v: round(h, 2) for v, h in settled.items()},
        "kes_hours_waiting": round(sum(owed[v] * h for v, h in settled.items())),
        "owed_unreached": round(sum(owed[v] for v in owed if v not in settled)),
        "makespan_h": round(max(settled.values(), default=0.0), 2),
        "villages": len(owed),
        "settled": len(settled),
    }


def plan(d: District, model, bus) -> tuple[dict[str, list[str]], str]:
    """The dispatcher agent plans; on any failure or an empty plan, the greedy rules do."""
    if model is None:
        return greedy(d), "rules"
    box = DispatchTools(d)
    try:
        Role("dispatcher", model, PROMPT, box, bus, 40, 90).wake(
            f"{len(d.vs)} villages flooded, {len(d.hubs)} hubs. Plan the drones."
        )
    except ModelError as e:
        bus.emit("AGENT_FALLBACK", role="dispatcher", reason=e.reason)
        return greedy(d), "rules"
    if not any(box.queues.values()):
        return greedy(d), "rules"
    planned = {v for q in box.queues.values() for v in q}
    left = [v for v in sorted(set(d.vs) - planned) if d.reach(v)]
    for vid in left:  # a village the agent skipped still gets its nearest drone
        box.queues[min(d.reach(vid), key=lambda h: d.km(vid, h))].append(vid)
    if left:
        bus.emit("DISPATCH_FILLED", villages=left)
    return box.queues, model.name
