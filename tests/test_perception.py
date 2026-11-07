import numpy as np
import pytest

from kipimo import config
from kipimo.perception import classical
from kipimo.perception.model import INT8
from kipimo.robot.kinematics import Robot
from kipimo.robot.sensors import PerceptionSensor
from kipimo.types import Pose
from kipimo.world.generate import generate, parcel_truth

needs_model = pytest.mark.skipif(not INT8.exists(), reason="run make train first")
W = generate(7, config.load("demo_flood").world)
CFG = config.load()


def _paired_observations(backends, n=40):
    out = {k: [] for k in backends}
    rng = np.random.default_rng(3)
    poses = rng.uniform(100, 900, (n, 2))
    for name, fn in backends.items():
        s = PerceptionSensor(W, np.random.default_rng(5), fn, name)
        for x, y in poses:
            r = Robot(CFG.robot, Pose(float(x), float(y), 0.0))
            out[name].append(s.observe(r, 0.0))
    return out


def test_classical_observations_track_truth():
    obs = _paired_observations({"classical": classical.predict})["classical"]
    err = np.mean([np.abs(o.crop_loss - W.loss.reshape(-1)[o.cell_ids]).mean() for o in obs])
    assert err < 0.12


@needs_model
def test_onnx_within_ten_percent_of_classical_and_small_and_fast():
    from kipimo.perception.infer import OnnxPerception

    onnx = OnnxPerception()
    obs = _paired_observations({"classical": classical.predict, "onnx": onnx})
    diffs, errs = [], {"classical": [], "onnx": []}
    for a, b in zip(obs["classical"], obs["onnx"], strict=True):
        assert np.array_equal(a.cell_ids, b.cell_ids)
        diffs.append(np.abs(a.crop_loss - b.crop_loss).mean())
        truth = W.loss.reshape(-1)[a.cell_ids]
        errs["classical"].append(np.abs(a.crop_loss - truth).mean())
        errs["onnx"].append(np.abs(b.crop_loss - truth).mean())
    st = onnx.stats()
    print(
        f"\nonnx vs classical {np.mean(diffs):.3f}  vs truth: onnx {np.mean(errs['onnx']):.3f} "
        f"classical {np.mean(errs['classical']):.3f}  {st}"
    )
    assert np.mean(diffs) < 0.10
    assert np.mean(errs["onnx"]) <= np.mean(errs["classical"])
    assert st["model_mb"] < 5.0


@needs_model
def test_onnx_episode_payouts():
    from kipimo.run import run

    s = run(config.load("demo_flood", sim={"sensor": "onnx"}), "infogain")
    assert s["payout_mae_pct"] < 7 and s["claims"] >= 2
    truth = parcel_truth(W)
    assert max(abs(r["est"] - truth[r["id"]]) for r in s["parcels"]) < 0.15
