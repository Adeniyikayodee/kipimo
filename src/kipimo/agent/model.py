"""Provider neutral model interface. Agents talk to a Model, never to an SDK.

Messages are neutral too: {"role": "user", "text"}, {"role": "assistant", "turn"} or
{"role": "tool", "results": [(Call, dict)]}. Adapters translate both ways.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    thinking: str = ""
    calls: list[Call] = field(default_factory=list)
    stop: str = "end"  # end | tool_use | refusal | cut (max tokens or any other stop)
    usage: dict = field(default_factory=dict)
    raw: object = None


class ModelError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class GeminiModel:
    """Gemini adapter. Automatic function calling is off: our loop owns budgets and checks."""

    BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}

    def __init__(self, model: str, client=None, thinking: str | None = None):
        from google import genai

        self.name, self.thinking = model, thinking
        try:
            self.client = client or genai.Client()
        except Exception as e:  # missing key or bad environment
            raise ModelError("no_client", str(e)[:200]) from e

    @staticmethod
    def _contents(messages: list[dict]) -> list:
        from google.genai import types

        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                out.append(m["turn"].raw)
            else:
                parts = [
                    types.Part(
                        function_response=types.FunctionResponse(id=c.id, name=c.name, response=r)
                    )
                    for c, r in m["results"]
                ]
                out.append(types.Content(role="user", parts=parts))
        return out

    def turn(self, system: str, messages: list[dict], tools: list[dict], timeout: float) -> Turn:
        from google.genai import errors, types

        decls = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters_json_schema=t["input_schema"],
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_level=self.thinking
            ),
            http_options=types.HttpOptions(timeout=int(max(timeout, 10.0) * 1000)),
        )
        t0 = time.monotonic()
        try:
            r = self.client.models.generate_content(
                model=self.name, contents=self._contents(messages), config=config
            )
        except errors.APIError as e:
            raise ModelError(f"http_{e.code}", str(getattr(e, "message", "") or e)[:300]) from e
        except Exception as e:  # transport failures surface as httpx errors
            raise ModelError("connection", str(e)[:200]) from e
        if not r.candidates:
            return Turn(stop="refusal")
        cand = r.candidates[0]
        t = Turn(raw=cand.content, stop="end")
        for i, part in enumerate(cand.content.parts or []):
            if part.function_call:
                fc = part.function_call
                t.calls.append(Call(fc.id or f"g{i}", fc.name, dict(fc.args or {})))
            elif part.text and part.thought:
                t.thinking += part.text
            elif part.text:
                t.text += part.text
        reason = getattr(cand.finish_reason, "name", str(cand.finish_reason))
        if reason in self.BLOCKED:
            t.stop = "refusal"
        else:
            t.stop = ("tool_use" if t.calls else "end") if reason == "STOP" else "cut"
        u = r.usage_metadata
        t.usage = {
            "in": getattr(u, "prompt_token_count", 0) or 0,
            "out": getattr(u, "candidates_token_count", 0) or 0,
            "s": round(time.monotonic() - t0, 3),
        }
        return t


def make_model(name: str, thinking: str | None = None):
    return GeminiModel(name, thinking=thinking)


Script = Callable[["ScriptedModel", list[dict]], list[tuple[str, dict]]]


class ScriptedModel:
    """Test and eval double: a function of the latest tool results decides the next calls."""

    def __init__(self, script: Script, name: str = "scripted"):
        self.script, self.name, self.n, self.brief = script, name, 0, ""

    def turn(self, system: str, messages: list[dict], tools: list[dict], timeout: float) -> Turn:
        last = messages[-1]
        results = [r for _, r in last["results"]] if last["role"] == "tool" else []
        self.brief = last.get("text", self.brief)
        calls = [
            Call(f"c{self.n}_{i}", n, a) for i, (n, a) in enumerate(self.script(self, results))
        ]
        self.n += 1
        return Turn(text="next step", calls=calls, stop="tool_use" if calls else "end")
