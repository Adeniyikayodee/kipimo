"""Run the suite, grade every case, summarise with bootstrap CIs, compare runs paired by case.

python -m kipimo.evals.run --agents rules --split test
python -m kipimo.evals.run --agents live --model gemini-4-argon --trials 3
python -m kipimo.evals.run --regrade evals/runs/<name>          # no model calls
python -m kipimo.evals.run --compare evals/runs/A.jsonl evals/runs/B.jsonl
"""

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from kipimo.bus import load
from kipimo.config import ROOT
from kipimo.evals import oracle
from kipimo.evals.cases import SITUATIONS, Case, config_for, injector, suite
from kipimo.evals.graders import grade
from kipimo.run import run

OUT = ROOT / "evals"
HEADLINE = [
    "outcome.payout_mae_pct",
    "outcome.owed_unpaid",
    "outcome.distance_km",
    "farmer.severity_accuracy",
    "farmer.attack_recall",
    "farmer.where_accuracy",
    "adjuster.audit_misses",
    "safety.flew_in_wind",
]


def job(a: tuple[Case, int, str, str | None, Path, Path | None]) -> dict:
    case, trial, agents, model, logs, memory = a
    cfg = config_for(case, model)
    if memory:
        cfg.agent.memory = str(memory)
    if agents == "live":  # a slow model must not be cut off by the demo's wall clock
        cfg.agent.mission_seconds = 1800
    path = logs / f"{case.id}_t{trial}.jsonl"
    path.unlink(missing_ok=True)
    models = {"rules": {}, "oracle": oracle.models(cfg) if agents == "oracle" else None}
    run(cfg, "agents", path, model=models.get(agents), setup=injector(case))
    return _row(case, trial, load(path), cfg)


def _row(case: Case, trial: int, events: list[dict], cfg) -> dict:
    meta = {"case": case.id, "family": case.family, "situation": case.situation, "trial": trial}
    return meta | grade(events, cfg)


def _per_case(rows: list[dict]) -> dict[str, dict[str, float]]:
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    keys = [k for k in rows[0] if "." in k]
    return {
        c: {
            k: np.nanmean([r[k] for r in rs]) if any(not math.isnan(r[k]) for r in rs) else math.nan
            for k in keys
        }
        for c, rs in sorted(by.items())  # sorted: a regrade resamples exactly like the run
    }


def ci(v: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    v = v[~np.isnan(v)]
    if not len(v):
        return math.nan, math.nan, math.nan
    boot = v[np.random.default_rng(0).integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarise(rows: list[dict]) -> dict:
    cases = _per_case(rows)
    keys = list(next(iter(cases.values())))
    out = {k: ci(np.array([c[k] for c in cases.values()])) for k in keys}
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(all(r[k] == 0 for k in keys if k.startswith("safety.")))
    out["safety.pass_k"] = (float(np.mean([all(v) for v in by_case.values()])),) * 3
    return out


def compare(a: list[dict], b: list[dict]) -> dict:
    ca, cb = _per_case(a), _per_case(b)
    shared = sorted(set(ca) & set(cb))
    return {k: ci(np.array([cb[c][k] - ca[c][k] for c in shared])) for k in ca[shared[0]]}


def _fmt(t: tuple) -> str:
    m, lo, hi = t
    return "n/a" if math.isnan(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def report(name: str, rows: list[dict]) -> str:
    s = summarise(rows)
    n_cases, trials = len({r["case"] for r in rows}), max(r["trial"] for r in rows) + 1
    lines = [
        f"# Eval: {name}",
        "",
        f"{n_cases} cases x {trials} trials. Mean [95% bootstrap CI] over cases.",
        "",
    ]
    lines += ["| metric | value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in s.items()]
    lines += [
        "",
        "## By situation",
        "",
        "| situation | " + " | ".join(HEADLINE) + " |",
        "|---" * (len(HEADLINE) + 1) + "|",
    ]
    for sit in [x for x in SITUATIONS if any(r["situation"] == x for r in rows)]:
        sub = summarise([r for r in rows if r["situation"] == sit])
        lines.append(f"| {sit} | " + " | ".join(f"{sub[k][0]:.3f}" for k in HEADLINE) + " |")
    return "\n".join(lines) + "\n"


def pick(cases: list[Case], floods: int | None) -> list[Case]:
    """The first `floods` seeds of each family, every situation: a fixed, paired subset."""
    if not floods:
        return cases
    keep = {
        f: sorted({c.seed for c in cases if c.family == f})[:floods] for f in ("broad", "corridor")
    }
    return [c for c in cases if c.seed in keep[c.family]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", choices=["rules", "oracle", "live"], default="rules")
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--regrade", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--memory", action="store_true", help="run cases in order, memory carried")
    ap.add_argument("--floods", type=int, help="first N floods of each family, every situation")
    a = ap.parse_args()
    if a.compare:
        rows = [[json.loads(x) for x in p.read_text().splitlines()] for p in a.compare]
        for k, v in compare(*rows).items():
            print(f"{k:32s} {_fmt(v)}")
        return
    tag = ("-memory" if a.memory else "") + (f"-{a.floods}floods" if a.floods else "")
    model = (a.model or "default").replace(":", "-")
    name = a.regrade.name if a.regrade else f"{a.agents}-{model}-{a.split}{tag}"
    logs = OUT / "runs" / name
    logs.mkdir(parents=True, exist_ok=True)
    cases = {c.id: c for s in ("dev", "test") for c in suite(s)}
    if a.regrade:
        rows = []
        for p in sorted(logs.glob("*.jsonl")):
            cid, t = p.stem.rsplit("_t", 1)
            rows.append(_row(cases[cid], int(t), load(p), config_for(cases[cid])))
        if not rows:  # never overwrite a saved run with nothing
            raise SystemExit(f"no per case logs in {logs}; they are local to the machine that ran")
        order = list(cases)
        rows.sort(key=lambda r: (order.index(r["case"]), r["trial"]))
    else:
        mem = logs / "memory.json" if a.memory else None
        if mem:
            mem.unlink(missing_ok=True)
        jobs = [
            (c, t, a.agents, a.model, logs, mem)
            for c in pick(suite(a.split), a.floods)
            for t in range(a.trials)
        ]
        if mem:  # sequential: each mission learns from the ones before
            rows = [job(j) for j in jobs]
        else:
            local = a.agents != "live"
            pool = ProcessPoolExecutor if local else ThreadPoolExecutor
            n = a.workers or (os.cpu_count() if local else 2)
            with pool(max_workers=n) as ex:
                rows = list(ex.map(job, jobs))
    (OUT / "runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "reports").mkdir(exist_ok=True)
    md = report(name, rows)
    (OUT / "reports" / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
