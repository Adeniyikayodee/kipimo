import json

from kipimo import config
from kipimo.payout import triggers
from kipimo.run import run
from kipimo.types import Parcel


def test_lawnmower_completes_and_logs_mae(tmp_path):
    s = run(config.load("demo_flood"), "lawnmower", tmp_path / "e.jsonl")
    end = json.loads((tmp_path / "e.jsonl").read_text().splitlines()[-1])
    assert end["kind"] == "EPISODE_END" and "payout_mae_pct" in end["data"]
    assert s["min_conf"] > 0.8 and s["payout_mae_pct"] < 7


def test_lawnmower_returns_home_on_low_battery(tmp_path):
    cfg = config.load("demo_flood", robot={"battery_wh": 40.0})
    run(cfg, "lawnmower", tmp_path / "e.jsonl")
    kinds = [json.loads(x)["kind"] for x in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert "RETURN_TO_HOME" in kinds and "FORCED_LANDING" not in kinds


def test_trigger_tiers():
    p = Parcel("X", None, "h", 1000.0, "maize", 0)
    assert triggers.score(p, 0.10).amount == 0
    assert triggers.score(p, 0.15).amount == 200
    assert triggers.score(p, 0.40).tier == 0.5
    assert triggers.score(p, 0.99).amount == 1000


def test_full_overlap_is_refused():
    import pytest

    from kipimo.planning.lawnmower import plan
    from kipimo.world.generate import generate

    with pytest.raises(ValueError):
        plan(generate(7), 20.0, 1.0)
