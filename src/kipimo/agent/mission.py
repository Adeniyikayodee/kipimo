"""Three agents with different stakes, woken by events on one bus.

    field     flies and gathers evidence, asks for payouts
    farmer    reads farmers' texts into reports, keeps farmers informed
    adjuster  the insurer's sceptic: challenges claims under written audit rules

Models propose, rules dispose: the confidence gate and the audit checks are code. Each role
falls back to its rule policy on its own when its model fails, so a mission always finishes.
"""

import time
from collections import deque
from pathlib import Path

from kipimo.agent import adjuster, farmer, policies
from kipimo.agent.memory import Memory
from kipimo.agent.model import ModelError, make_model
from kipimo.agent.runtime import Role
from kipimo.agent.tools import Tools
from kipimo.payout import review
from kipimo.world import sms

FIELD_PROMPT = (Path(__file__).parent / "prompts" / "field.md").read_text()
BRIEF = (
    "Flood reported along the river on day 63. {n} insured parcels, total sum insured {si:,.0f}. "
    "The robot is charged at the home pad. Assess the damage and settle the claims that are due."
)
ROLES = ("field", "farmer", "adjuster")
MAX_ROUNDS = 3  # adjuster rounds per claim before a person decides


class Mission:
    def __init__(self, st: policies.Stack, models=None):
        self.st, self.bus, self.cfg = st, st.ep.bus, st.ep.cfg.agent
        self.deadline = time.monotonic() + self.cfg.mission_seconds
        self.models = self._models(models)
        self.memory = st.memory = Memory(self.cfg.memory or None)
        self.desk = st.desk = farmer.Desk(st, self.memory)
        self.inbox = deque(
            sms.inbox(st.ep.world, st.ep.cfg.seed, self.cfg.sms_profile, self.cfg.sms_style)
        )
        self.news: list[str] = []
        self.cases: dict[str, dict] = {}
        st.engine.reviewer = self.review
        st.ep.tickers.append(self.pump)
        self.bus.subscribers.append(self._watch)

    def _models(self, models) -> dict:
        if isinstance(models, dict):
            return {r: models.get(r) for r in ROLES}
        if models is not None:
            return dict.fromkeys(ROLES, models)
        spec = {r: (self.cfg.roles.get(r, self.cfg.model), self.cfg.thinking.get(r)) for r in ROLES}
        built: dict[tuple, object] = {}
        for key in dict.fromkeys(spec.values()):
            try:
                built[key] = make_model(*key)
            except ModelError as e:
                who = [r for r in ROLES if spec[r] == key]
                self.bus.emit(
                    "AGENT_FALLBACK",
                    role="all" if len(who) == 3 else ",".join(who),
                    reason=e.reason,
                )
                built[key] = None
        return {r: built[spec[r]] for r in ROLES}

    def _down(self, role: str, why: str, detail: str = "") -> None:
        if self.models[role] is not None:
            self.bus.emit("AGENT_FALLBACK", role=role, reason=why, detail=detail)
        self.models[role] = None

    def _role(self, name: str, prompt: str, box, calls: int) -> Role:
        m, cfg = self.models[name], self.cfg
        idle = self.st.ep.hold if self.st.ep.live_time else None
        return Role(name, m, prompt, box, self.bus, calls, cfg.max_seconds, self.deadline, idle)

    def _watch(self, ev: dict) -> None:
        d = ev["data"]
        if ev["kind"] == "FARMER_REPORT" and d["conflict"] and not d["suspicious"]:
            self.news.append(
                f"{d['parcel']}: farmer reports {d['severity']}, which conflicts with the estimate"
            )
        elif ev["kind"] == "CLAIM_CHALLENGED" and not d.get("escalated"):
            self.news.append(f"{d['parcel']}: adjuster challenged the claim under {d['rule_id']}")

    # farmer
    def pump(self) -> None:
        while self.inbox and self.inbox[0].t <= self.bus.t:
            self.handle(self.inbox.popleft())

    def handle(self, m: sms.Sms) -> None:
        self.desk.lang[m.parcel_id] = m.lang
        self.bus.emit("SMS_RECEIVED", parcel=m.parcel_id, lang=m.lang, text=m.text, label=m.label)
        n = len(self.desk.log)
        if self.models["farmer"]:
            box = farmer.FarmerTools(self.desk, m, self.models["farmer"].name)
            brief = f"Text from {m.sender}, policyholder of {m.parcel_id}: {m.text!r}"
            try:
                self._role("farmer", farmer.PROMPT, box, self.cfg.role_calls).wake(brief)
            except ModelError as e:
                self._down("farmer", e.reason, e.detail)
        if len(self.desk.log) == n:  # the model did not record it: rules do, and evals see who
            r = farmer.read(m.text)
            self.desk.record(
                m.parcel_id,
                r["severity"],
                r["suspicious"],
                "rules",
                r["kind"],
                r["where"],
                r["hours"],
            )

    def close_farmers(self) -> None:
        while self.inbox:
            self.handle(self.inbox.popleft())
        for pid in sorted(self.desk.reports):
            name = self.st.ep.world.parcel(pid).holder
            if self.models["farmer"]:
                fake = sms.Sms(self.bus.t, pid, name, self.desk.lang.get(pid, "en"), "", {})
                box = farmer.FarmerTools(self.desk, fake, self.models["farmer"].name)
                brief = (
                    f"The mission is over. Text {name} ({pid}) a short status in their language."
                )
                try:
                    self._role("farmer", farmer.PROMPT, box, 2).wake(brief)
                    continue
                except ModelError as e:
                    self._down("farmer", e.reason, e.detail)
            farmer.rules_reply(self.desk, pid, name)

    # adjuster
    def review(self, pid: str, score: dict, why: str = "") -> dict | None:
        """One adjuster conversation per claim. Past MAX_ROUNDS a holding rule escalates."""
        held = review.holding(review.facts(self.st, pid))
        case = self.cases.setdefault(pid, {"rounds": 0, "role": None})
        case["rounds"] += 1
        n, verdict, by, m = case["rounds"], None, "rules", self.models["adjuster"]
        if n > MAX_ROUNDS and held:
            verdict = {
                "rule_id": held[0],
                "note": "unresolved after negotiation",
                "escalated": True,
            }
            self.bus.emit("CLAIM_ESCALATED", parcel=pid, rule_id=held[0], rounds=n - 1)
            return verdict | {"by": "rules"}
        if m:
            box = adjuster.AdjusterTools(self.st, pid)
            if case["role"] is None:
                case["role"] = self._role("adjuster", adjuster.PROMPT, box, 4 * MAX_ROUNDS)
                brief = f"Claim for {pid}: KES {score['amount']}, confidence {score['confidence']}."
            else:
                brief = f"Round {n} for {pid}. The field agent asks again: {why!r}."
            case["role"].box = box
            try:
                case["role"].wake(f"{brief} Field agent's case: {why!r}" if n == 1 else brief)
            except ModelError as e:
                self._down("adjuster", e.reason, e.detail)
            if box.decided:
                verdict, by = box.verdict, m.name
        if by == "rules":
            verdict = adjuster.rules_verdict(self.st, pid)
        decision = "challenge" if verdict else "approve"
        self.bus.emit(
            "CLAIM_REVIEW",
            parcel=pid,
            decision=decision,
            by=by,
            holding=held,
            round=n,
            **(verdict or {}),
        )
        return verdict and verdict | {"by": by}

    # field
    def fly(self) -> str:
        if not self.models["field"]:
            policies.baseline(self.st, who="fallback")
            return "rules"
        tools = Tools(self.st)
        role = self._role("field", FIELD_PROMPT, tools, self.cfg.max_tool_calls)
        si = sum(p.sum_insured for p in self.st.ep.world.parcels)
        brief = BRIEF.format(n=len(self.st.ep.world.parcels), si=si)
        self.bus.emit(
            "AGENT_START", model=self.models["field"].name, max_tool_calls=self.cfg.max_tool_calls
        )
        for _ in range(self.cfg.field_wakes):
            self.news.clear()
            try:
                why = role.wake(brief)
            except ModelError as e:
                self._down("field", e.reason, e.detail)
                why = "error"
            if why in ("budget", "refusal", "cut", "error"):
                if why != "error":
                    kind = "AGENT_BUDGET_EXHAUSTED" if why == "budget" else "AGENT_FALLBACK"
                    self.bus.emit(kind, role="field", reason=why)
                policies.baseline(self.st, who="fallback")
                return why
            self.pump()
            if not self.news:
                break
            tools.done = False
            brief = "New since your last report:\n" + "\n".join(dict.fromkeys(self.news))
        self.bus.emit("AGENT_END", role="field", tool_calls=role.calls)
        policies.go_home(self.st)
        policies.issue_all(self.st, "rules")
        return "ended"

    def run(self) -> str:
        why = self.fly()
        self.close_farmers()
        self.bus.emit("MEMORY_UPDATED", **self.memory.learn(self.st, self.desk.reports))
        return why


def run_mission(st: policies.Stack, models=None) -> str:
    return Mission(st, models).run()
