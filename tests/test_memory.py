from kipimo import config
from kipimo.agent.memory import Memory
from kipimo.bus import load
from kipimo.run import run
from kipimo.world.sms import habitual


def test_reliability_is_a_beta_posterior():
    m = Memory(None)
    assert m.reliability("Esther") is None and not m.distrusted("Esther")
    m.farmers["Esther"] = {"reports": 1, "overstated": 1}
    assert not m.distrusted("Esther")  # one report is not a habit
    m.farmers["Esther"] = {"reports": 4, "overstated": 4}
    assert m.reliability("Esther") == 1 / 6 and m.distrusted("Esther")


def test_memory_learns_habits_across_missions_and_stops_wasted_flights(tmp_path):
    path = tmp_path / "memory.json"
    for seed in range(300, 308):
        cfg = config.load(None, seed=seed, belief={"model": "gp"}, agent={"memory": str(path)})
        run(cfg, "agents", tmp_path / f"{seed}.jsonl", model={})
    m = Memory(path)
    scored = {h: m.reliability(h) for h in m.farmers if m.farmers[h]["reports"] >= 2}
    worst = min(scored, key=scored.get)
    assert habitual(worst)
    discounted = [
        e
        for s in range(304, 308)
        for e in load(tmp_path / f"{s}.jsonl")
        if e["kind"] == "FARMER_REPORT" and e["data"].get("discounted")
    ]
    assert discounted
    assert m.note("radar runs high on rice", "field")["kept"]
