"""One runtime for every agent role: a model, a prompt, a toolbox and a hard budget.

Each wake appends a brief and runs turns until the model stops calling tools, the toolbox says
it is done, or the budget runs out. Every turn, call and result is an event tagged with the role.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

BAD_ARGS = (KeyError, ValueError, TypeError, IndexError, AttributeError)
MIN_TURN_S = 10.0  # the Gemini API refuses any request deadline under 10 s
KEEP_RESULTS = 2  # tool replies the model sees in full; older ones shrink to a stub
STUB_OVER = 600  # characters of JSON above which an older reply is shrunk
STUB = {"elided": "an older result; call the tool again for fresh data"}


def compact(messages: list[dict], keep: int = KEEP_RESULTS) -> list[dict]:
    """The conversation the model sees: recent tool replies in full, older large ones as stubs.

    The log keeps every reply; only the prompt shrinks, so long missions stay fast and cheap.
    """
    tool = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool[:-keep] if keep else tool)
    out = []
    for i, m in enumerate(messages):
        if i in old:
            res = [
                (c, STUB if len(json.dumps(r, default=str)) > STUB_OVER else r)
                for c, r in m["results"]
            ]
            m = {"role": "tool", "results": res}
        out.append(m)
    return out


class Role:
    def __init__(
        self, name, model, prompt, toolbox, bus, max_calls, max_seconds, deadline=None, idle=None
    ):
        self.name, self.model, self.prompt, self.box, self.bus = name, model, prompt, toolbox, bus
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.deadline = deadline or float("inf")  # time.monotonic() shared by a whole mission
        self.messages: list[dict] = []
        self.calls = 0
        self.spent = 0.0  # own model seconds: other roles' turns inside our tools don't count
        self.idle = idle  # called repeatedly while the model thinks, so sim time keeps moving

    def _ask(self, messages: list[dict], left: float):
        if self.idle is None:
            return self.model.turn(self.prompt, messages, self.box.schemas, left)
        pool = ThreadPoolExecutor(1)
        try:
            job = pool.submit(self.model.turn, self.prompt, messages, self.box.schemas, left)
            while not job.done():
                self.idle()
            return job.result()
        finally:
            pool.shutdown(wait=False)  # a stopped run never waits on the network

    def wake(self, brief: str) -> str:
        """Returns ended, done, budget, refusal or cut. Raises ModelError for the caller."""
        self.messages.append({"role": "user", "text": brief})
        while True:
            t0 = time.monotonic()
            left = min(self.max_seconds - self.spent, self.deadline - t0)
            if left < MIN_TURN_S or self.calls >= self.max_calls:
                return "budget"
            turn = self._ask(compact(self.messages), left)
            self.spent += time.monotonic() - t0
            self.messages.append({"role": "assistant", "turn": turn})
            ev = {"role": self.name, "usage": turn.usage}
            self.bus.emit("AGENT_TURN", calls=len(turn.calls), stop=turn.stop, **ev)
            if turn.thinking:
                self.bus.emit("AGENT_THINKING", text=turn.thinking, **ev)
            if turn.text:
                self.bus.emit("AGENT_TEXT", text=turn.text, **ev)
            if turn.stop in ("refusal", "cut"):
                if turn.calls:  # answer them, so a later wake sends a valid conversation
                    results = [(c, {"error": "turn stopped"}) for c in turn.calls]
                    self.messages.append({"role": "tool", "results": results})
                return turn.stop
            if not turn.calls:
                return "ended"
            results = []
            for c in turn.calls:
                if self.calls >= self.max_calls:
                    out = {"error": "tool call budget exhausted"}
                else:
                    self.calls += 1
                    self.bus.emit(
                        "AGENT_TOOL_CALL", name=c.name, input=c.args, n=self.calls, role=self.name
                    )
                    try:
                        out = self.box.call(c.name, dict(c.args))
                    except BAD_ARGS as e:  # the model's arguments, never the mission, fail
                        out = {"error": f"bad call: {e!r}"}
                    self.bus.emit("AGENT_TOOL_RESULT", name=c.name, result=out, role=self.name)
                results.append((c, out))
            self.messages.append({"role": "tool", "results": results})
            if self.box.done:
                return "done"
