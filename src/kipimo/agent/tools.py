"""The agent's six tools. Each returns compact JSON. None of them moves the robot directly
or computes a payout: survey hands an objective to the planner, claims go through the engine."""

import math

import numpy as np

from kipimo.agent import policies
from kipimo.agent.policies import Stack
from kipimo.belief.parcels import prior_width
from kipimo.payout import triggers
from kipimo.payout.claim import ClaimRefused
from kipimo.planning import infogain
from kipimo.robot.sensors import camera_quality
from kipimo.types import Objective


def _obj(props: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


PID = {"type": "string", "description": "Parcel id, e.g. P07"}

SCHEMAS: list[dict] = [
    {
        "name": "get_situation",
        "description": (
            "Current belief for every insured parcel: estimated loss, 95% interval, confidence, "
            "coverage, sum insured, the payout the deterministic engine would give at the "
            "current estimate, whether the confidence gate passes, and money_at_risk "
            "(expected payout error plus payable money blocked by the gate). Also robot "
            "battery and distance. This is your only view of the world."
        ),
        "input_schema": _obj({}, []),
        "strict": True,
    },
    {
        "name": "get_priors",
        "description": (
            "Remote sensing context for one parcel: rainfall anomaly, NDVI change (null until "
            "a clear optical pass after the flood), radar flood fraction and age of the last "
            "satellite pass. Cached offline."
        ),
        "input_schema": _obj({"parcel_id": PID}, ["parcel_id"]),
        "strict": True,
    },
    {
        "name": "survey",
        "description": (
            "Task the robot with an objective. The safety envelope accepts or rejects it, then "
            "the planner flies value weighted survey lanes over those parcels until each reaches "
            "target_confidence or the distance budget runs out. One objective is one battery "
            "sortie, capped at the sortie limit. Returns what was flown and the new confidence "
            "per parcel. You set goals; you never choose waypoints."
        ),
        "input_schema": _obj(
            {
                "parcel_ids": {"type": "array", "items": PID},
                "target_confidence": {"type": "number", "description": "0.5 to 0.95"},
                "max_distance_m": {"type": "number"},
                "reason": {"type": "string", "description": "One sentence justification"},
            },
            ["parcel_ids", "target_confidence", "max_distance_m", "reason"],
        ),
        "strict": True,
    },
    {
        "name": "score_payout",
        "description": (
            "Deterministic payout score for one parcel at the current estimate: trigger id, "
            "tier, amount and gate status. No claim is issued."
        ),
        "input_schema": _obj({"parcel_id": PID}, ["parcel_id"]),
        "strict": True,
    },
    {
        "name": "issue_claim",
        "description": (
            "Ask the payout engine to issue a signed claim. The engine refuses with "
            "insufficient_confidence below the gate, no_trigger when nothing is payable, "
            "or already_issued. A refusal is final for the current evidence: survey again."
        ),
        "input_schema": _obj(
            {"parcel_id": PID, "justification": {"type": "string"}},
            ["parcel_id", "justification"],
        ),
        "strict": True,
    },
    {
        "name": "get_conditions",
        "description": (
            "Flying conditions now: wind and the drone's limit, the forecast ensemble's share of "
            "calm members for the next hours and the next safe window, camera state and quality, "
            "flight altitude and speed, battery, uplink. Use it before tasking and after faults."
        ),
        "input_schema": _obj({}, []),
        "strict": True,
    },
    {
        "name": "set_flight",
        "description": (
            "Set altitude (20 to 60 m) and speed (3 to 10 m/s). Lower and slower sees better, "
            "which helps most when the camera is degraded, but covers less ground per metre."
        ),
        "input_schema": _obj(
            {
                "altitude_m": {"type": "number"},
                "speed_mps": {"type": "number"},
                "reason": {"type": "string"},
            },
            ["altitude_m", "speed_mps", "reason"],
        ),
        "strict": True,
    },
    {
        "name": "wait",
        "description": (
            "Hold the drone at the pad for some minutes (max 240), e.g. for a flight window or "
            "for signal. Texts still arrive and queued claims still send. Only possible at the pad."
        ),
        "input_schema": _obj(
            {"minutes": {"type": "number"}, "reason": {"type": "string"}}, ["minutes", "reason"]
        ),
        "strict": True,
    },
    {
        "name": "recall",
        "description": (
            "What earlier missions taught: each farmer's reliability (share of reports not "
            "overstated, once settled) and notes agents chose to keep."
        ),
        "input_schema": _obj({}, []),
        "strict": True,
    },
    {
        "name": "remember",
        "description": "Keep one short lesson for future missions, e.g. a radar bias you saw.",
        "input_schema": _obj({"note": {"type": "string"}}, ["note"]),
        "strict": True,
    },
    {
        "name": "report",
        "description": "Write the episode summary to the audit trail. Ends the episode.",
        "input_schema": _obj({"summary": {"type": "string"}}, ["summary"]),
        "strict": True,
    },
]


class UnknownParcel(Exception):
    pass


class Tools:
    schemas = SCHEMAS

    def __init__(self, st: Stack):
        self.st = st
        self.done = False

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except UnknownParcel as e:
            return {"error": f"unknown parcel id {e}"}
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def _parcel(self, pid: str):
        p = next((p for p in self.st.ep.world.parcels if p.id == pid), None)
        if p is None:
            raise UnknownParcel(pid)
        return p

    def t_get_situation(self) -> dict:
        ep = self.st.ep
        beliefs = ep.beliefs()
        gate = triggers.gate()
        w0 = prior_width(ep.belief.cfg)
        rows = []
        for p in ep.world.parcels:
            b = beliefs[p.id]
            width = b.loss_ci95[1] - b.loss_ci95[0]
            sc = triggers.score(p, b.loss_mean)
            risk = infogain.money_at_risk(p, b.loss_mean, width, w0, gate)
            rows.append(
                {
                    "id": p.id,
                    "crop": p.crop,
                    "sum_insured": p.sum_insured,
                    "loss": round(b.loss_mean, 3),
                    "ci95": [round(b.loss_ci95[0], 3), round(b.loss_ci95[1], 3)],
                    "confidence": round(b.confidence, 2),
                    "coverage": round(b.cells_observed / b.cells_total, 2),
                    "payout_at_estimate": sc.amount,
                    "gate": "pass" if b.confidence >= gate else "fail",
                    "money_at_risk": round(risk),
                    "claimed": p.id in self.st.engine.issued,
                    **self._news(p.id),
                }
            )
        rows.sort(key=lambda r: -r["money_at_risk"])
        r = ep.robot
        return {
            "parcels": rows,
            "robot": {
                "battery_wh": round(r.battery_wh, 1),
                "distance_m": round(r.distance_m),
                "sensor": "degraded" if ep.sensor.degraded else "ok",
            },
            "gate_min_confidence": gate,
            "sortie_limit_m": ep.cfg.agent.max_sortie_m,
            "claims_issued": len(self.st.engine.issued),
        }

    def _news(self, pid: str) -> dict:
        desk = getattr(self.st, "desk", None)
        rep = desk.reports.get(pid) if desk else None
        rep = rep if rep and not rep.suspicious else None
        ch = self.st.engine.challenged.get(pid)
        return {
            "farmer_report": rep.severity if rep else None,
            "report_conflict": bool(rep and rep.conflict),
            "challenged": ch["rule_id"] if ch else None,
        }

    def t_get_conditions(self) -> dict:
        ep, w, r = self.st.ep, self.st.ep.weather, self.st.ep.robot
        t = ep.bus.t
        weather = None
        if w:
            h = w.hour(t)
            nxt = w.window(t)
            weather = {
                "wind_mps": round(w.wind(t), 1),
                "limit_mps": w.limit,
                "calm_share_next_hours": [round(w.safe_share(k), 2) for k in range(h, h + 6)],
                "next_window_in_min": None if nxt is None else max(0, round((nxt * 3600 - t) / 60)),
            }
        return {
            "sim_time_min": round(t / 60),
            "weather": weather,
            "camera": {
                "degraded": ep.sensor.degraded,
                "quality": round(camera_quality(r, ep.sensor.degraded), 3),
            },
            "flight": {
                "altitude_m": r.altitude_m,
                "speed_mps": r.speed_mps,
                "footprint_m": round(2 * r.footprint_half_m()),
                "power_w": round(r.power),
            },
            "wind_at_drone": {"mps": round(math.hypot(*r.wind), 1), "from_deg": ep.wind.from_deg()},
            "battery_wh": round(r.battery_wh, 1),
            "at_pad": math.dist((r.pose.x, r.pose.y), ep.home) < 5,
            "uplink": self.st.uplink.link.state,
        }

    def t_set_flight(self, altitude_m: float, speed_mps: float, reason: str) -> dict:
        return policies.set_flight(self.st, altitude_m, speed_mps, reason)

    def t_wait(self, minutes: float, reason: str) -> dict:
        ep = self.st.ep
        if math.dist((ep.robot.pose.x, ep.robot.pose.y), ep.home) >= 5:
            return {"error": "the drone is airborne; finish or end the sortie first"}
        return policies.wait(self.st, minutes, reason)

    def t_recall(self) -> dict:
        m = getattr(self.st, "memory", None)
        if not m:
            return {"farmers": {}, "notes": []}
        rel = {h: round(m.reliability(h), 2) for h in m.farmers if m.reliability(h) is not None}
        return {"farmers": rel, "notes": m.notes}

    def t_remember(self, note: str) -> dict:
        m = getattr(self.st, "memory", None)
        return m.note(note, "field") if m else {"error": "no memory configured"}

    def t_get_priors(self, parcel_id: str) -> dict:
        ep = self.st.ep
        p = self._parcel(parcel_id)
        i = ep.world.parcels.index(p)
        rng = np.random.default_rng([ep.cfg.seed, 99, i])
        water = float(ep.world.water.reshape(-1)[p.cells].mean())
        prior = float(ep.world.prior.reshape(-1)[p.cells].mean())
        return {
            "parcel_id": parcel_id,
            "crop": p.crop,
            "planted_day": p.planted_day,
            "rainfall_anomaly_mm": round(60 + 180 * water + rng.normal(0, 25)),
            "ndvi_delta": None,  # the last clear optical pass predates the flood
            "radar_loss_prior": round(prior, 3),
            "last_optical_pass_days": int(rng.integers(6, 15)),
            "optical_cloud_cover": round(float(rng.uniform(0.6, 0.95)), 2),
            "last_radar_pass_days": int(rng.integers(1, 4)),
        }

    def _estimate_m(self, obj: Objective) -> float:
        ep = self.st.ep
        r = ep.robot
        q = camera_quality(r, ep.sensor.degraded)
        here = (r.pose.x, r.pose.y)
        batch = self.st.planner.plan_batch(ep.belief, here, obj, q, until_target=True)
        d = 0.0
        for _, a, b, _ in batch:
            d += math.dist(here, a) + math.dist(a, b)
            here = b
        return d

    def t_survey(
        self, parcel_ids: list[str], target_confidence: float, max_distance_m: float, reason: str
    ) -> dict:
        st, ep = self.st, self.st.ep
        for pid in parcel_ids:
            self._parcel(pid)
        cap = ep.cfg.agent.max_sortie_m
        obj = Objective(
            sorted(set(parcel_ids)),
            float(min(max(target_confidence, 0.5), 0.95)),
            float(min(max_distance_m, cap)),
            reason,
        )
        verdict = self._check(obj)
        if verdict:
            ep.bus.emit("OBJECTIVE_REJECTED", objective=obj, reason=verdict)
            return {"accepted": False, "reason": verdict}
        r = ep.robot
        if ep.sensor.degraded == (r.altitude_m >= ep.cfg.robot.altitude_m):
            # the camera changed while the agent was thinking: let it set the flight first
            q = round(camera_quality(r, ep.sensor.degraded), 3)
            ep.bus.emit("DEGRADED_SENSOR_REPLAN", degraded=ep.sensor.degraded, quality=q)
            return {
                "accepted": False,
                "stop_reason": "camera_changed",
                "camera": {"degraded": ep.sensor.degraded, "quality": q},
                "altitude_m": r.altitude_m,
                "speed_mps": r.speed_mps,
            }
        est = self._estimate_m(obj)
        risk = st.planner.risk_table(
            ep.belief, Objective([p.id for p in ep.world.parcels], 0.8, 0, "")
        )
        ranked = sorted(risk, key=lambda k: -risk[k])
        ep.bus.emit(
            "OBJECTIVE_ACCEPTED",
            objective=obj,
            estimated_distance_m=round(est),
            risk={k: round(risk[k]) for k in ranked},
        )
        start = ep.robot.distance_m
        stop = infogain.survey(ep, st.planner, st.env, obj, until_target=True, stop_on_camera=True)
        ep.bus.emit("SURVEY_END", reason=stop)
        beliefs = ep.beliefs()
        return {
            "accepted": True,
            "estimated_distance_m": round(est),
            "flown_m": round(ep.robot.distance_m - start),
            "stop_reason": stop,
            "capped_at_m": obj.max_distance_m,
            "parcels": [
                {
                    "id": pid,
                    "confidence": round(beliefs[pid].confidence, 2),
                    "loss": round(beliefs[pid].loss_mean, 3),
                    "reached_target": beliefs[pid].confidence >= obj.target_confidence,
                }
                for pid in obj.parcel_ids
            ],
        }

    def _check(self, obj: Objective) -> str:
        ep, env = self.st.ep, self.st.env
        if not obj.parcel_ids:
            return "empty_objective"
        n = ep.world.size
        for pid in obj.parcel_ids:
            if not any(lane.parcel_id == pid for lane in self.st.planner.lanes):
                return f"no_safe_lane: {pid} lies inside a no fly zone"
            rows, cols = np.divmod(self._parcel(pid).cells, n)
            c = ep.world.cell_m
            v = env.check_point(((cols.mean() + 0.5) * c, (rows.mean() + 0.5) * c))
            if not v.ok:
                return f"{v.reason}: {pid}"
        wind = policies.unflyable(self.st)
        if wind:
            return f"wind_above_limit: {wind}; call get_conditions for the next window"
        r = ep.robot
        need = (
            ep.energy_wh((r.pose.x, r.pose.y), ep.home)
            + r.wh_per_m(math.hypot(*ep.wind.mean(ep.bus.t))) * obj.max_distance_m
        )
        v = env.check_battery(r.battery_wh, need, r.cfg.reserve_wh)
        return "" if v.ok else v.reason

    def t_score_payout(self, parcel_id: str) -> dict:
        self._parcel(parcel_id)
        return self.st.engine.score(parcel_id)

    def t_issue_claim(self, parcel_id: str, justification: str) -> dict:
        self._parcel(parcel_id)
        try:
            claim, sig = self.st.engine.issue(parcel_id, justification)
        except ClaimRefused as e:
            return {"issued": False, "reason": e.reason, "score": e.detail}
        return {
            "issued": True,
            "claim_id": claim.claim_id,
            "amount": claim.amount,
            "tier": claim.tier,
            "signature": sig[:16] + "...",
            "packet_bytes": self.st.uplink.sizes.get(claim.claim_id),
            "uplink_queue_depth": self.st.uplink.outbox.depth(),
        }

    def t_report(self, summary: str) -> dict:
        self.st.ep.bus.emit("AGENT_REPORT", summary=summary)
        self.done = True
        return {"logged": True}
