"""30 seeded scenarios, three policies, each swept over its own distance knob.

    make bench   ->  bench/results.csv and bench/frontier.png

"agent" here is the agent's stack with the non LLM policy choosing objectives, which is also
the agent's fallback. Calling Claude 30 times would be neither headless nor reproducible.
"""

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import yaml

from kipimo import config
from kipimo.config import ROOT
from kipimo.run import run

OUT = ROOT / "bench"
SWEEPS = {
    "lawnmower": [0.1, -0.5, -1.0, -1.5, -2.0],  # lane overlap; negative leaves gaps
    "agent": [0.05, 0.1, 0.2, 0.4, 0.8],  # stop when a metre buys less risk reduction than this
}
OPERATING = {"lawnmower": 0.1, "agent": 0.2}
FIELDS = [
    "scenario",
    "seed",
    "policy",
    "param",
    "distance_m",
    "payout_mae_pct",
    "loss_mae",
    "claims",
    "owed",
    "missed_claims",
    "wrong_claims",
    "min_conf",
    "first_claim_s",
]


def scenarios() -> list[tuple[str, int]]:
    out = []
    for name in ("bench_heavy", "bench_corridor"):
        lo, hi = yaml.safe_load((config.CONFIG_DIR / "scenarios" / f"{name}.yaml").read_text())[
            "seeds"
        ]
        out += [(name, s) for s in range(lo, hi + 1)]
    return out


def one(job: tuple[str, int, str, float]) -> dict:
    scenario, seed, policy, param = job
    raw = yaml.safe_load((config.CONFIG_DIR / "scenarios" / f"{scenario}.yaml").read_text())
    cfg = config.load(None, seed=seed, world=raw.get("world"))
    if policy == "agent":
        cfg = cfg.model_copy(
            update={"planner": cfg.planner.model_copy(update={"min_gain_per_m": param})}
        )
        s = run(cfg, "infogain")
    else:
        s = run(cfg, policy, param=param)
    return {"scenario": scenario, "seed": seed, "policy": policy, "param": param} | {
        k: s[k] for k in FIELDS[4:]
    }


def main() -> None:
    jobs = [
        (sc, seed, pol, p) for sc, seed in scenarios() for pol, ps in SWEEPS.items() for p in ps
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = list(pool.map(one, jobs, chunksize=4))
    OUT.mkdir(exist_ok=True)
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs over {len(scenarios())} scenarios in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
