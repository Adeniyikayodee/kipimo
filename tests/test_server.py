import json
import math

from fastapi.testclient import TestClient

from kipimo.server.app import app, session


def test_dashboard_streams_a_full_mission():
    session.speed = 5000
    c = TestClient(app)
    assert "Kipimo" in c.get("/").text
    for name in ("app.css", "app.js", "map3d.js"):
        assert c.get(f"/{name}").status_code == 200
    assert c.get("/secrets.txt").status_code == 404
    assert "demo_flood" in c.get("/api/scenarios").json()
    kinds = []
    with c.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["kind"] == "HELLO"
        ws.send_text(json.dumps({"cmd": "start", "scenario": "demo_flood", "policy": "infogain"}))
        while "EPISODE_END" not in kinds:
            ev = json.loads(ws.receive_text())
            kinds.append(ev["kind"])
            if ev["kind"] == "FRAME":
                frame = ev["data"]
        while (ev := json.loads(ws.receive_text()))["kind"] != "CLAIM_AWAITING_APPROVAL":
            kinds.append(ev["kind"])
        case = ev["data"]["case_id"]
        ws.send_text(json.dumps({"cmd": "decide", "case_id": case, "decision": "approve"}))
        while (ev := json.loads(ws.receive_text()))["kind"] != "ACTUARY_DECISION":
            pass
        assert ev["data"]["decision"]["case_id"] == case
        ws.send_text(json.dumps({"cmd": "link", "state": "offline"}))
        while "LINK_STATE" not in kinds:
            kinds.append(json.loads(ws.receive_text())["kind"])
        ws.send_text(json.dumps({"cmd": "reset"}))
    for k in ("WORLD_INFO", "FRAME", "OBSERVATION", "PLAN_BATCH", "CLAIM_ISSUED", "UPLINK_SENT"):
        assert k in kinds, k
    assert frame["perception"]["p50_ms"] > 0 and frame["uplink"]["state"] == "online"
    assert len(frame["situation"]["parcels"]) >= 12
    assert frame["conditions"]["camera"]["quality"] > 0 and frame["conditions"]["weather"] is None


def test_replay_streams_a_recording_and_rejects_bad_paths():
    import pytest

    from kipimo.server.app import recording

    session.speed = 5000
    c = TestClient(app)
    recs = c.get("/api/recordings").json()
    name = next(r["name"] for r in recs if r["name"].startswith("recordings/"))
    kinds = []
    with c.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps({"cmd": "start", "scenario": "demo_flood", "policy": f"replay:{name}"})
        )
        while "EPISODE_END" not in kinds:
            kinds.append(json.loads(ws.receive_text())["kind"])
        ws.send_text(json.dumps({"cmd": "reset"}))
    assert {"SESSION_RESET", "WORLD_INFO", "FRAME", "CLAIM_ISSUED"} <= set(kinds)
    for bad in ("recordings/../../pyproject.toml", "etc/passwd", "runs/../events.jsonl"):
        with pytest.raises((ValueError, KeyError)):
            recording(bad)


def test_district_endpoint_places_hubs_and_flies_both_plans():
    c = TestClient(app)
    d = c.get("/api/district?seed=11").json()
    assert len(d["hubs"]) == 3 and 0 < d["covered"] <= 1
    g, n = d["plans"]["greedy"], d["plans"]["nearest"]
    assert (
        set(g["settled_h"]) == set(n["settled_h"])
        and g["kes_hours_waiting"] <= n["kes_hours_waiting"]
    )


def test_evals_endpoint_summarises_every_committed_run():
    d = TestClient(app).get("/api/evals").json()
    names = {s["name"] for s in d["suites"]}
    assert "rules-default-test" in names and d["districts"]
    s = next(s for s in d["suites"] if s["name"] == "rules-default-test")
    assert s["summary"]["safety.pass_k"][0] == 1.0 and s["cases"] == 60


def test_foreign_pages_and_bad_commands_are_refused():
    import pytest
    from starlette.websockets import WebSocketDisconnect

    c = TestClient(app)
    with (
        pytest.raises(WebSocketDisconnect),
        c.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws,
    ):
        ws.receive_text()
    bad = [
        {"cmd": "speed", "value": float("nan")},
        {"cmd": "degrade"},
        {"cmd": "link", "state": "sideways"},
        {"cmd": "start", "scenario": "/tmp/x", "policy": "infogain"},
        {"cmd": "start", "scenario": "demo_flood", "policy": "replay:nope/x.jsonl"},
        {"cmd": "decide", "case_id": "E7-P00", "decision": "pay double"},
        {"cmd": "decide", "case_id": ["E7-P00"], "decision": "approve"},
        {"cmd": "decide", "case_id": "E7-P00", "decision": "reject", "note": "x" * 301},
    ]
    with c.websocket_connect("/ws") as ws:
        ws.receive_text()
        for msg in bad:
            ws.send_text(json.dumps(msg))
            while json.loads(ws.receive_text())["kind"] != "COMMAND_ERROR":
                pass  # history of an earlier run streams first
    assert math.isfinite(session.speed)


def test_a_token_guards_the_live_connection(monkeypatch):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("KIPIMO_TOKEN", "s3cret")
    c = TestClient(app)
    with pytest.raises(WebSocketDisconnect), c.websocket_connect("/ws") as ws:
        ws.receive_text()
    with c.websocket_connect("/ws?token=s3cret") as ws:
        assert json.loads(ws.receive_text())["kind"] == "HELLO"
