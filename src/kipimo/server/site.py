"""Build the dashboard as a static site: seeded runs replay in the browser, district plans
and eval summaries are precomputed. No server, no model, no key.

uv run python -m kipimo.server.site --out site
"""

import argparse
import json
import shutil
from pathlib import Path

from kipimo.server.app import ASSETS, STATIC, Session, district_plan, get_evals
from kipimo.world import terrain

RUNS = [
    ("demo_flood", "agents", "AI agents, demo flood"),
    ("storm_flood", "agents", "AI agents, storm"),
    ("demo_flood", "infogain", "Planner, demo flood"),
]
ACTUARY = "demo actuary (scripted)"  # a static page has no person to click, so a script signs off
REVIEW_S = 90.0  # sim seconds the scripted actuary spends on each claim


def sign_off(st) -> None:
    """Approve every claim, one at a time, the way the dashboard's actuary would."""
    a, dt = st.actuary, st.ep.cfg.sim.dt
    for _ in range(3):  # a released hold lands as a new claim
        for case in sorted(a.cases):
            st.ep.bus.t += REVIEW_S
            a.decide(case, "approve", ACTUARY, "evidence and flags checked")
        st.uplink.drain(dt)
        if not a.cases:
            return


DISTRICTS = range(11, 23)  # the seeds the district view offers


def record(scenario: str, policy: str, dest: Path) -> None:
    s = Session()
    s.speed, s.out = 1e9, dest
    s._run(scenario, policy, idle=False, after=sign_off)


def build(out: Path) -> None:
    data = out / "data"
    (data / "runs").mkdir(parents=True, exist_ok=True)
    page = (STATIC / "index.html").read_text().replace("<html ", '<html data-site="static" ', 1)
    (out / "index.html").write_text(page)
    (out / ".nojekyll").touch()
    for name in ASSETS:
        shutil.copy(STATIC / name, out / name)
    shutil.copytree(STATIC / "vendor", out / "vendor", dirs_exist_ok=True)
    (out / "imagery").mkdir(exist_ok=True)
    shutil.copy(terrain.IMAGERY, out / "imagery")
    listed = []
    for scenario, policy, label in RUNS:
        name = f"{scenario}_{policy}.jsonl"
        record(scenario, policy, data / "runs" / name)
        if policy == "agents" and "AGENT_TOOL_CALL" not in (data / "runs" / name).read_text():
            label = label.replace("AI agents", "Agent roles run by rules (no model key)")
        listed.append({"name": name, "label": f"{label}, actuary sign-off scripted"})

    def dump(name: str, obj) -> None:
        (data / name).write_text(json.dumps(obj, separators=(",", ":")))

    dump("recordings.json", listed)
    dump("evals.json", get_evals())
    for seed in DISTRICTS:
        dump(f"district-{seed}.json", district_plan(seed))
    mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"built {out} ({mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("site"))
    build(ap.parse_args().out)
