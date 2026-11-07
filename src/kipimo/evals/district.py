"""District dispatch eval: the same flooded districts under each dispatch policy.

python -m kipimo.evals.district --split test [--live MODEL]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from kipimo import config
from kipimo.agent import dispatch
from kipimo.agent.model import make_model
from kipimo.bus import Bus
from kipimo.evals.run import OUT, _fmt, ci
from kipimo.planning.district import place_hubs, reach_km, villages

SPLITS = {"dev": range(600, 605), "test": range(500, 512)}  # villages: seed * 100 + i, unused
HUBS = 3


def one(job: tuple[int, str | None]) -> list[dict]:
    seed, live = job
    cfg = config.load()
    r = reach_km(cfg.robot)
    vs = villages(seed)
    hubs, covered = place_hubs(vs, HUBS, r)
    d = dispatch.District(vs, hubs, r, cfg.robot.speed_mps * 3.6)
    plans = {"nearest": dispatch.nearest(d), "greedy": dispatch.greedy(d)}
    if live:
        plans["agent"], _ = dispatch.plan(d, make_model(live), Bus(None))
    rows = []
    for k, q in plans.items():
        res = {m: v for m, v in dispatch.execute(d, q).items() if m != "settled_h"}
        rows.append({"seed": seed, "policy": k, "covered": round(covered, 3)} | res)
    return rows


def summary(rows: list[dict]) -> dict:
    """Per policy: mean KES-hours waiting, paired reduction against nearest, unreached, makespan."""
    by = {
        p: {r["seed"]: r for r in rows if r["policy"] == p}
        for p in dict.fromkeys(r["policy"] for r in rows)
    }
    base = by["nearest"]
    return {
        p: {
            "kes_hours": float(np.mean([rs[s]["kes_hours_waiting"] for s in base])),
            "less_than_nearest": ci(
                np.array(
                    [1 - rs[s]["kes_hours_waiting"] / base[s]["kes_hours_waiting"] for s in base]
                )
            ),
            "owed_unreached": float(np.mean([rs[s]["owed_unreached"] for s in base])),
            "makespan_h": float(np.mean([rs[s]["makespan_h"] for s in base])),
            "districts": len(base),
        }
        for p, rs in by.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--live", default=None, help="model name for the dispatcher agent")
    a = ap.parse_args()
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(one, [(s, a.live) for s in SPLITS[a.split]]) for r in rs]
    sm = summary(rows)
    n = sm["nearest"]["districts"]
    lines = [f"# District dispatch eval: {a.split}", "", f"{n} districts, {HUBS} hubs each.", ""]
    lines += [
        "| policy | KES-hours waiting | vs nearest (paired) | owed unreached | makespan h |",
        "|---|---|---|---|---|",
    ]
    for p, m in sm.items():
        less, un = _fmt(m["less_than_nearest"]), m["owed_unreached"]
        lines.append(
            f"| {p} | {m['kes_hours']:,.0f} | {less} less | {un:,.0f} | {m['makespan_h']:.2f} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    name = f"district-{a.live or 'rules'}-{a.split}"
    (OUT / "reports" / f"{name}.md").write_text(md)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(md)


if __name__ == "__main__":
    main()
