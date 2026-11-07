import numpy as np

from kipimo import config
from kipimo.belief.grid import BeliefGrid
from kipimo.belief.parcels import aggregate
from kipimo.robot.sensors import quality
from kipimo.types import Observation, Pose
from kipimo.world.generate import generate, parcel_truth

CFG = config.load()
W = generate(7)


def _obs(ids, x, q):
    z = np.zeros(len(ids), np.float32)
    return Observation(0.0, Pose(0, 0, 0), ids, z, np.full(len(ids), x, np.float32), q)


def test_confidence_monotone_with_fixed_quality_observations():
    rng = np.random.default_rng(0)
    for trial in range(20):
        g = BeliefGrid(W.prior, CFG.belief)
        p = W.parcels[trial % len(W.parcels)]
        q = float(rng.uniform(0.2, 1.0))
        last = aggregate(g, (p,))[p.id].confidence
        assert 0 <= last < 0.8
        for _ in range(30):
            ids = rng.choice(p.cells, size=int(rng.integers(20, 300)), replace=False)
            g.update(_obs(ids, float(rng.uniform(0, 1)), q))
            c = aggregate(g, (p,))[p.id].confidence
            assert c > last
            last = c


def test_full_coverage_estimate_close_to_truth():
    rng = np.random.default_rng(1)
    g = BeliefGrid(W.prior, CFG.belief)
    q = quality(CFG.robot.altitude_m, CFG.robot.speed_mps)
    ids = np.flatnonzero(W.parcel_map.reshape(-1) >= 0).astype(np.int32)
    loss = W.loss.reshape(-1)[ids]
    for _ in range(8):
        noisy = np.clip(loss + rng.normal(0, 0.03 + 0.15 * (1 - q), len(ids)), 0, 1)
        g.update(Observation(0.0, Pose(0, 0, 0), ids, noisy, noisy.astype(np.float32), q))
    truth = parcel_truth(W)
    for pid, b in aggregate(g, W.parcels).items():
        assert abs(b.loss_mean - truth[pid]) < 0.05, pid
        assert b.confidence > 0.8
        assert b.loss_ci95[0] <= b.loss_mean <= b.loss_ci95[1]


def test_degraded_quality_counts_for_less():
    p = W.parcels[0]
    good, bad = BeliefGrid(W.prior, CFG.belief), BeliefGrid(W.prior, CFG.belief)
    for _ in range(5):
        good.update(_obs(p.cells, 0.4, 0.7))
        bad.update(_obs(p.cells, 0.4, 0.3))
    assert aggregate(good, (p,))[p.id].confidence > aggregate(bad, (p,))[p.id].confidence
