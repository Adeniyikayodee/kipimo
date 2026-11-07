import argparse
from collections.abc import Callable
from pathlib import Path

from kipimo import config
from kipimo.agent import policies
from kipimo.episode import Episode
from kipimo.weather import floodhub


def run(
    cfg: config.Config,
    policy: str,
    out: Path | None = None,
    model=None,
    setup: Callable[[policies.Stack], None] | None = None,
    close: bool = True,
    param: float | None = None,
) -> dict:
    ep = Episode(cfg, out=out)
    try:
        return _run(ep, cfg, policy, model, setup, close, param)
    finally:
        if close:
            ep.bus.close()


def _run(ep, cfg, policy, model, setup, close, param) -> dict:
    ep.bus.emit("EPISODE_START", policy=policy, seed=cfg.seed, approval=cfg.approval.mode)
    st = policies.build(ep)
    if setup:
        setup(st)
    if not floodhub.alert(ep):
        pass  # the river stays below danger: no flight
    elif policy == "lawnmower":
        policies.run_lawnmower(st, 0.1 if param is None else param)
    elif policy == "infogain":
        policies.baseline(st)
    elif policy == "agents":
        from kipimo.agent.mission import run_mission

        run_mission(st, models=model)
    else:
        raise SystemExit(f"unknown policy {policy}")
    if close:
        st.uplink.drain(cfg.sim.dt)
    ep.emit_beliefs()
    s = ep.summary()
    s["claims"] = len(st.engine.issued)
    s["settled"] = len(st.uplink.receiver.settled)
    s["approved"] = len(st.actuary.paid)
    s["awaiting_approval"] = len(st.actuary.cases)
    s["queue_depth"] = st.uplink.outbox.depth()
    s["bytes_sent"] = st.uplink.link.sent_bytes
    s["packet_bytes_max"] = max(st.uplink.sizes.values(), default=0)
    issued = [e["t"] for e in ep.bus.events if e["kind"] == "CLAIM_ISSUED"]
    s["first_claim_s"] = min(issued) if issued else None
    s["duration_s"] = round(ep.bus.t, 1)
    paid = {pid: c.amount for pid, (c, _) in st.engine.issued.items()}  # what was signed
    s["owed"] = sum(r["pay_true"] > 0 for r in s["parcels"])
    s["missed_claims"] = sum(r["pay_true"] > 0 and r["id"] not in paid for r in s["parcels"])
    s["wrong_claims"] = sum(paid.get(r["id"], r["pay_true"]) != r["pay_true"] for r in s["parcels"])
    ep.bus.emit("EPISODE_END", **{k: v for k, v in s.items() if k != "parcels"})
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="demo_flood")
    ap.add_argument("--policy", default="lawnmower")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", type=Path, default=Path("events.jsonl"))
    a = ap.parse_args()
    s = run(config.load(a.scenario, seed=a.seed), a.policy, a.out)
    for r in s["parcels"]:
        print(r)
    print({k: v for k, v in s.items() if k != "parcels"})


if __name__ == "__main__":
    main()
