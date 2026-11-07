from kipimo import config
from kipimo.bus import load
from kipimo.config import FloodCfg
from kipimo.run import run
from kipimo.weather import floodhub


def test_the_river_passes_danger_at_night_days_ahead():
    for seed in range(40):
        g = floodhub.gauge(seed, FloodCfg())
        h = floodhub.crossing(g)
        assert h and h >= 18 and (g.issued_hour + h) % 24 in {21, 22, 23, 0, 1, 2, 3, 4, 5}


def test_no_flood_no_flight(tmp_path):
    cfg = config.load("demo_flood", flood={"peak_m": 3.5})
    run(cfg, "infogain", tmp_path / "e.jsonl")
    kinds = [e["kind"] for e in load(tmp_path / "e.jsonl")]
    assert "FLOOD_WATCH" in kinds and "OBSERVATION" not in kinds and "CLAIM_ISSUED" not in kinds


def test_a_storm_books_the_first_calm_window(tmp_path):
    run(config.load("storm_flood"), "infogain", tmp_path / "e.jsonl")
    booked = next(e["data"] for e in load(tmp_path / "e.jsonl") if e["kind"] == "FLIGHT_BOOKED")
    assert booked["in_h"] > 0 and booked["reason"] == "first calm window"
