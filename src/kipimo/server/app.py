"""Dashboard server. One process: the sim runs in a thread, every bus event streams over a
WebSocket, and browser commands are applied inside the sim thread between ticks."""

import asyncio
import base64
import dataclasses
import functools
import hmac
import json
import math
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kipimo import config
from kipimo.agent import dispatch, policies
from kipimo.agent.tools import Tools
from kipimo.bus import load
from kipimo.config import ROOT
from kipimo.payout import approval
from kipimo.perception.model import INT8
from kipimo.planning import district
from kipimo.run import run
from kipimo.uplink import link
from kipimo.world import terrain
from kipimo.world.generate import EVENT_DAY

STATIC = Path(__file__).parent / "static"
RUNS = ROOT / "runs"
RECORDINGS = ROOT / "demo" / "recordings"
FRAME_EVERY_S = 8.0
POLICIES = ("lawnmower", "infogain", "agents")
BACKLOG = 20_000  # events a client may fall behind before it is dropped
GRID = 64
RELIEF = 128


class Halt(Exception):
    pass


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def _pool(a: np.ndarray, n: int) -> np.ndarray:
    k = a.shape[0] // n
    return a.reshape(n, k, n, k).mean(axis=(1, 3))


def hillshade(z: np.ndarray, cell_m: float, exaggerate: float = 4.0) -> np.ndarray:
    """Lambertian shade lit from the north west at 45 degrees, 0..1."""
    dy, dx = np.gradient(z * exaggerate, cell_m)
    n = np.stack([-dx, -dy, np.ones_like(z)]) / np.sqrt(dx**2 + dy**2 + 1)
    light = np.array([-1.0, -1.0, np.sqrt(2.0)]) / 2.0
    return np.clip(np.tensordot(light, n, 1), 0, 1)


def agent_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def world_info(st: policies.Stack) -> dict:
    ep = st.ep
    w = ep.world
    n, c = w.size, w.cell_m
    pm = w.parcel_map.astype(np.int16)
    parcels = []
    for p in w.parcels:
        rows, cols = np.divmod(p.cells, n)
        parcels.append(
            {
                "id": p.id,
                "holder": p.holder,
                "crop": p.crop,
                "age_days": EVENT_DAY - p.planted_day,
                "sum_insured": p.sum_insured,
                "cx": float((cols.mean() + 0.5) * c),
                "cy": float((rows.mean() + 0.5) * c),
            }
        )
    return {
        "span_m": n * c,
        "map_px": n,
        "parcel_map": _b64(np.where(pm < 0, 255, pm)),
        "truth": _b64(_pool(w.loss, GRID) * 255),
        "grid": GRID,
        "relief": _b64(hillshade(_pool(w.elevation, RELIEF), w.size * c / RELIEF) * 255),
        "relief_grid": RELIEF,
        "height": _b64(_pool(np.maximum(w.elevation, 0), RELIEF) / w.elevation.max() * 255),
        "height_max": float(w.elevation.max()),
        "stage": w.stage,
        "imagery": w.rect and {"url": f"imagery/{terrain.IMAGERY.name}", "rect": w.rect},
        "site": w.site,
        "parcels": parcels,
        "no_fly": [z.model_dump() for z in ep.cfg.envelope.no_fly],
        "home": list(ep.home),
        "fence": list(st.env.fence),
        "half_m": ep.robot.footprint_half_m(),
        "fov_tan": math.tan(math.radians(ep.cfg.robot.fov_deg) / 2),
        "sortie_m": ep.cfg.agent.max_sortie_m,
        "sensor": ep.cfg.sim.sensor,
        "approval": ep.cfg.approval.mode,  # EPISODE_START goes out before the browser subscribes
    }


def belief_frame(st: policies.Stack, tools: Tools) -> dict:
    ep = st.ep
    n = ep.world.size
    g = ep.belief
    w = g.weight()
    sit = tools.t_get_situation()
    risk = {r["id"]: r["money_at_risk"] for r in sit["parcels"]}
    voi = np.zeros(n * n)
    for p in ep.world.parcels:
        voi[p.cells] = risk[p.id] / len(p.cells) * (1 - w[p.cells])
    voi = _pool(voi.reshape(n, n), GRID)
    peak = voi.max()
    perception = getattr(ep.sensor, "backend", None)
    stats = perception.stats() if hasattr(perception, "stats") else None
    u = st.uplink
    return {
        "mean": _b64(_pool(g.mean().reshape(n, n), GRID) * 255),
        "observed": _b64(_pool(w.reshape(n, n), GRID) * 255),
        "voi": _b64(voi / peak * 255 if peak > 0 else voi),
        "situation": sit,
        "conditions": tools.t_get_conditions(),
        "perception": stats,
        "uplink": {
            "state": u.link.state,
            "depth": u.outbox.depth(),
            "bytes_sent": u.link.sent_bytes,
            "settled": len(u.receiver.settled),
            "packet_bytes": max(u.sizes.values(), default=0),
        },
    }


class Session:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: dict[asyncio.Queue, WebSocket] = {}
        self.run_id = 0  # events from an older run are dropped, never shown on a newer one
        self.history: list[dict] = []
        self.commands: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self.halt = threading.Event()
        self.speed = 10.0
        self.paused = False
        self.status = {"running": False, "scenario": "demo_flood", "policy": "infogain"}
        self.out = ROOT / "events.jsonl"

    def _push(self, ev: dict, rid: int | None = None) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._fanout, ev, self.run_id if rid is None else rid)

    def _fanout(self, ev: dict, rid: int) -> None:
        """Event loop only."""
        if rid != self.run_id:
            return
        if ev["kind"] == "SESSION_RESET":
            self.history = []
        self.history.append(ev)
        for q, sock in list(self.clients.items()):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:  # a stalled browser must not hold the server's memory
                del self.clients[q]
                asyncio.ensure_future(sock.close(code=1013))

    def start(self, scenario: str, policy: str) -> None:
        """Called off the event loop. Raises ValueError for a bad request or a stuck run."""
        if scenario not in scenarios():
            raise ValueError(f"unknown scenario {scenario}")
        if policy.startswith("replay:"):
            path = recording(policy.removeprefix("replay:"))
            target, args = self._replay, (path,)
        elif policy in POLICIES:
            target, args = self._run, (scenario, policy)
        else:
            raise ValueError(f"unknown policy {policy}")
        if not self.stop():
            raise ValueError("the previous run is still stopping")
        self.halt.clear()
        self.paused = False
        self.run_id += 1
        self.status = {"running": True, "scenario": scenario, "policy": policy}
        if target == self._replay:
            self.status["replay"] = path.stem
        self._push({"kind": "SESSION_RESET", "data": dict(self.status), "seq": -1, "t": 0})
        push = functools.partial(self._push, rid=self.run_id)
        self.thread = threading.Thread(target=target, args=(*args, push), daemon=True)
        self.thread.start()

    def stop(self) -> bool:
        """True once no run is alive."""
        if self.thread and self.thread.is_alive():
            self.halt.set()
            self.thread.join(timeout=5)
        return not (self.thread and self.thread.is_alive())

    def _failed(self, e: Exception, push) -> None:
        push({"kind": "SESSION_ERROR", "data": {"error": repr(e)[:300]}, "seq": -1, "t": 0})

    def _apply(self, st: policies.Stack) -> None:
        while not self.commands.empty():
            cmd = self.commands.get_nowait()
            if cmd["cmd"] == "link":
                st.uplink.set_state(cmd["state"])
            elif cmd["cmd"] == "degrade":
                st.ep.sensor.degraded = cmd["on"]
                st.ep.bus.emit("SENSOR_STATE", degraded=st.ep.sensor.degraded)
            elif cmd["cmd"] == "surge":
                st.ep.surge()
            elif cmd["cmd"] == "decide":  # a stale or repeated click finds no open case
                st.actuary.decide(cmd["case_id"], cmd["decision"], st.actuary.name, cmd["note"])
            elif cmd["cmd"] == "approve_clean":
                st.actuary.approve_clean(st.actuary.name)

    def _run(self, scenario: str, policy: str, push=None, idle: bool = True, after=None) -> None:
        push = push or self._push
        sensor = "onnx" if INT8.exists() else "classical"
        extra = {
            "belief": {"model": "gp"},
            "agent": {"sms_profile": "adversarial", "sms_style": "freeform"},
        }
        extra = extra if policy == "agents" else {}
        holder: dict = {}

        def tick() -> None:
            st, tools = holder["st"], holder["tools"]
            if self.halt.is_set():
                raise Halt
            self._apply(st)
            while self.paused and not self.halt.is_set():
                time.sleep(0.05)
                self._apply(st)
            time.sleep(cfg.sim.dt / max(self.speed, 0.1))
            if st.ep.bus.t - holder.get("last", -1e9) >= FRAME_EVERY_S:
                holder["last"] = st.ep.bus.t
                st.ep.bus.emit("FRAME", **belief_frame(st, tools))

        def setup(st: policies.Stack) -> None:
            holder["st"], holder["tools"] = st, Tools(st)
            st.ep.live_time = True  # the drone hovers and texts arrive while an agent thinks
            st.ep.bus.subscribers.append(push)
            st.ep.bus.emit("WORLD_INFO", **world_info(st))
            st.ep.bus.emit("FRAME", **belief_frame(st, holder["tools"]))
            st.ep.tickers.append(tick)

        try:
            cfg = config.load(scenario, sim={"sensor": sensor}, **extra)
            self.out.unlink(missing_ok=True)  # scratch; each finished run is kept in runs/
            run(cfg, policy, self.out, setup=setup, close=False)
            st = holder["st"]
            if st.uplink.link.state != "offline":  # the saved run ends with claims delivered
                st.uplink.drain(cfg.sim.dt)
            if after:
                after(st)
            st.ep.bus.emit("FRAME", **belief_frame(st, holder["tools"]))
            st.ep.bus.flush()
            RUNS.mkdir(exist_ok=True)
            shutil.copy(self.out, RUNS / f"{scenario}_{policy}.jsonl")
            holder["last"] = float("inf")  # idle: frames only when the uplink changes
            seen = (st.uplink.link.state, st.uplink.outbox.depth())
            saved = st.ep.bus.seq
            while idle and not self.halt.is_set():  # link toggles and sign-offs still land
                st.ep.bus.t += cfg.sim.dt
                st.uplink.pump()
                tick()
                now = (st.uplink.link.state, st.uplink.outbox.depth())
                if now != seen:
                    seen = now
                    st.ep.bus.emit("FRAME", **belief_frame(st, holder["tools"]))
                if st.ep.bus.seq != saved:  # the saved run keeps every later decision
                    saved = st.ep.bus.seq
                    st.ep.bus.flush()
                    shutil.copy(self.out, RUNS / f"{scenario}_{policy}.jsonl")
        except Halt:
            pass
        except Exception as e:  # the browser hears about it; the server lives on
            self._failed(e, push)
            raise
        finally:
            if "st" in holder:
                holder["st"].ep.bus.close()
            self.status["running"] = False

    def _replay(self, path: Path, push=None) -> None:
        """Re-streams a recorded run at the chosen speed. No model, no network, no sim."""
        push, prev = push or self._push, None
        try:
            for ev in load(path):
                if self.halt.is_set():
                    return
                while self.paused and not self.halt.is_set():
                    time.sleep(0.05)
                if prev is not None and ev["t"] > prev:
                    time.sleep(min(2.0, (ev["t"] - prev) / max(self.speed, 0.1)))
                prev = ev["t"]
                push(ev)
        except Exception as e:
            self._failed(e, push)
            raise
        finally:
            self.status["running"] = False


def recordings() -> list[dict]:
    out = []
    for folder, label in ((RECORDINGS, "recorded"), (RUNS, "last run")):
        for p in sorted(folder.glob("*.jsonl")):
            out.append(
                {
                    "name": f"{folder.name}/{p.name}",
                    "label": f"{p.stem.replace('_', ' ')} ({label})",
                }
            )
    return out


def recording(name: str) -> Path:
    folder, _, file = name.partition("/")
    base = {RECORDINGS.name: RECORDINGS, RUNS.name: RUNS}.get(folder)
    if base is None:
        raise ValueError(name)
    path = (base / file).resolve()
    if path.parent != base.resolve() or path.suffix != ".jsonl":
        raise ValueError(name)
    return path


session = Session()
app = FastAPI(title="Kipimo")
app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")
app.mount("/imagery", StaticFiles(directory=terrain.IMAGERY.parent), name="imagery")


ASSETS = ("app.css", "app.js", "map3d.js")  # the page, its styles and its two scripts


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/{name}", include_in_schema=False)
def asset(name: str) -> FileResponse:
    if name not in ASSETS:
        raise HTTPException(404)
    return FileResponse(STATIC / name)


@app.get("/api/scenarios")
def scenarios() -> list[str]:
    names = sorted(p.stem for p in (config.CONFIG_DIR / "scenarios").glob("*.yaml"))
    return [n for n in names if not n.startswith("bench")]


@app.get("/api/recordings")
def list_recordings() -> list[dict]:
    return recordings()


@functools.lru_cache(maxsize=8)
def district_plan(seed: int, hubs: int = 3) -> dict:
    """Hubs placed by maximal covering, then both dispatch plans flown for real."""
    robot = config.load().robot
    r = district.reach_km(robot)
    vs = district.villages(seed)
    sites, covered = district.place_hubs(vs, hubs, r)
    d = dispatch.District(vs, sites, r, robot.speed_mps * 3.6)
    plans = {"greedy": dispatch.greedy(d), "nearest": dispatch.nearest(d)}
    return {
        "span_km": district.SPAN_KM,
        "reach_km": round(r, 2),
        "covered": round(covered, 3),
        "villages": [dataclasses.asdict(v) for v in vs],
        "hubs": [{"id": h, "x": x, "y": y} for h, (x, y) in d.hubs.items()],
        "plans": {k: {"queues": q} | dispatch.execute(d, q) for k, q in plans.items()},
    }


@app.get("/api/district")
def get_district(seed: int = 11) -> dict:
    return district_plan(seed)


def _clean(x):
    """JSON has no NaN: missing metrics become null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return None if isinstance(x, float) and math.isnan(x) else x


@app.get("/api/evals")
def get_evals() -> dict:
    """Every committed eval run, summarised the same way as its report."""
    from kipimo.evals import district as devals
    from kipimo.evals.run import OUT, summarise

    suites, districts = [], []
    for path in sorted((OUT / "runs").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if path.stem.startswith("district"):
            districts.append({"name": path.stem, "summary": devals.summary(rows)})
        else:
            cases = len({r["case"] for r in rows})
            suites.append({"name": path.stem, "cases": cases, "summary": summarise(rows)})
    return _clean({"suites": suites, "districts": districts, "agent_available": agent_available()})


@app.get("/api/status")
def status() -> dict:
    return {**session.status, "agent_available": agent_available(), "speed": session.speed}


async def command(msg: dict) -> None:
    """One browser command, checked before it touches the sim. ValueError if malformed."""
    cmd = msg.get("cmd")
    if cmd == "start":
        await asyncio.to_thread(session.start, str(msg["scenario"]), str(msg["policy"]))
    elif cmd == "speed" and math.isfinite(v := float(msg["value"])):
        session.speed = min(max(v, 1.0), 20.0)
    elif cmd == "pause" and isinstance(msg["on"], bool):
        session.paused = msg["on"]
    elif cmd == "reset":
        await asyncio.to_thread(session.stop)
    elif (cmd == "link" and msg["state"] in link.STATES) or (
        cmd == "degrade" and isinstance(msg["on"], bool)
    ):
        session.commands.put(msg)
    elif cmd == "surge":
        session.commands.put({"cmd": "surge"})
    elif cmd == "decide" and msg["decision"] in approval.DECISIONS:
        case, note = msg["case_id"], msg.get("note", "")
        if not (isinstance(case, str) and isinstance(note, str) and len(note) <= 300):
            raise ValueError("bad decision")
        session.commands.put(
            {"cmd": cmd, "case_id": case, "decision": msg["decision"], "note": note}
        )
    elif cmd == "approve_clean":
        session.commands.put({"cmd": cmd})
    else:
        raise ValueError(f"bad command {cmd}")


def same_origin(sock: WebSocket) -> bool:
    """Browsers send Origin; any other page could otherwise drive this server."""
    origin = sock.headers.get("origin")
    return origin is None or origin.split("://", 1)[-1] == sock.headers.get("host")


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    token = os.environ.get("KIPIMO_TOKEN", "")
    if not same_origin(sock) or (
        token and not hmac.compare_digest(sock.query_params.get("token", ""), token)
    ):
        await sock.close(code=1008)
        return
    await sock.accept()
    session.loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(BACKLOG)
    for ev in session.history:
        q.put_nowait(ev)
    session.clients[q] = sock
    hello = {**session.status, "agent_available": agent_available(), "speed": session.speed}
    await sock.send_text(json.dumps({"kind": "HELLO", "data": hello}))

    async def sender() -> None:
        while True:
            await asyncio.wait_for(sock.send_text(json.dumps(await q.get())), 10)

    async def receiver() -> None:
        while True:
            try:
                await command(json.loads(await sock.receive_text()))
            except (ValueError, KeyError, TypeError) as e:
                await sock.send_text(json.dumps({"kind": "COMMAND_ERROR", "data": str(e)}))

    tasks = {asyncio.create_task(sender()), asyncio.create_task(receiver())}
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        session.clients.pop(q, None)
