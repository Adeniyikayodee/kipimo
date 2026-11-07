from types import SimpleNamespace as NS

from google.genai import errors, types

from kipimo import config
from kipimo.agent.model import GeminiModel, ModelError
from kipimo.agent.tools import SCHEMAS
from kipimo.bus import load
from kipimo.run import run


class FakeModels:
    """Answers like Gemini: one get_situation call, then a report."""

    def __init__(self):
        self.seen = []

    def generate_content(self, model, contents, config):
        self.seen.append((contents, config))
        if len(self.seen) == 1:
            part = types.Part(
                function_call=types.FunctionCall(id="a", name="get_situation", args={})
            )
        else:
            fr = contents[-1].parts[0].function_response
            assert fr.id == "a" and "parcels" in fr.response
            part = types.Part(
                function_call=types.FunctionCall(name="report", args={"summary": "ok"})
            )
        cand = types.Candidate(
            content=types.Content(role="model", parts=[part]), finish_reason="STOP"
        )
        return types.GenerateContentResponse(candidates=[cand])


def test_gemini_adapter_runs_the_agent_loop(tmp_path):
    fake = FakeModels()
    model = GeminiModel("gemini-robotics-er-2-preview", client=NS(models=fake))
    run(config.load("demo_flood"), "agents", tmp_path / "e.jsonl", {"field": model})
    kinds = [e["kind"] for e in load(tmp_path / "e.jsonl")]
    assert "AGENT_REPORT" in kinds and "AGENT_FALLBACK" not in kinds
    cfg = fake.seen[0][1]
    assert cfg.automatic_function_calling.disable
    names = [d.name for d in cfg.tools[0].function_declarations]
    assert names == [s["name"] for s in SCHEMAS]


def test_gemini_api_error_maps_to_fallback_reason():
    class Boom:
        def generate_content(self, **_):
            raise errors.ClientError(429, {"error": {"message": "quota"}})

    model = GeminiModel("gemini-robotics-er-2-preview", client=NS(models=Boom()))
    try:
        model.turn("s", [{"role": "user", "text": "hi"}], SCHEMAS, 5)
    except ModelError as e:
        assert e.reason == "http_429"
    else:
        raise AssertionError("expected ModelError")


def test_a_cut_turn_falls_back_and_answers_its_calls():
    from kipimo.agent.runtime import Role
    from kipimo.bus import Bus

    class Cut:
        def generate_content(self, **_):
            part = types.Part(function_call=types.FunctionCall(name="get_situation", args={}))
            content = types.Content(role="model", parts=[part])
            cand = types.Candidate(content=content, finish_reason="MAX_TOKENS")
            return types.GenerateContentResponse(candidates=[cand])

    model = GeminiModel("gemini-robotics-er-2-preview", client=NS(models=Cut()))
    role = Role("field", model, "", NS(schemas=SCHEMAS, done=False), Bus(), 5, 60)
    assert role.wake("go") == "cut" and role.messages[-1]["role"] == "tool"


def _capture():
    seen = []

    class Once:
        def generate_content(self, model, contents, config):
            seen.append(config)
            cand = types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="done")]),
                finish_reason="STOP",
            )
            return types.GenerateContentResponse(candidates=[cand])

    return seen, NS(models=Once())


def test_requests_never_ask_gemini_for_a_deadline_under_10_seconds():
    seen, client = _capture()
    GeminiModel("gemini-4-argon", client=client).turn(
        "s", [{"role": "user", "text": "hi"}], SCHEMAS, 3
    )
    assert seen[0].http_options.timeout >= 10_000  # the API answers 400 below 10 s


def test_role_hands_over_instead_of_calling_with_under_10_seconds_left():
    from kipimo.agent.runtime import Role
    from kipimo.bus import Bus

    seen, client = _capture()
    role = Role(
        "field",
        GeminiModel("gemini-4-argon", client=client),
        "",
        NS(schemas=SCHEMAS, done=False),
        Bus(),
        5,
        9.5,
    )
    assert role.wake("go") == "budget" and not seen


def test_thinking_level_and_api_error_text_reach_their_places():
    seen, client = _capture()
    GeminiModel("gemini-3.8-flash", client=client, thinking="low").turn(
        "s", [{"role": "user", "text": "hi"}], SCHEMAS, 60
    )
    assert seen[0].thinking_config.thinking_level.value.lower() == "low"

    class Bad:
        def generate_content(self, **_):
            raise errors.ClientError(
                400, {"error": {"message": "Minimum allowed deadline is 10s."}}
            )

    try:
        GeminiModel("gemini-4-argon", client=NS(models=Bad())).turn(
            "s", [{"role": "user", "text": "hi"}], SCHEMAS, 60
        )
    except ModelError as e:
        assert e.reason == "http_400" and "10s" in e.detail
    else:
        raise AssertionError("expected ModelError")
