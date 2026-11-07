import time

import numpy as np

from kipimo import config
from kipimo.belief.gp import GPBelief, sigma_meas
from kipimo.types import Observation, Pose
from kipimo.world.generate import generate, parcel_truth

CFG = config.load()
Q = 0.77


def _look(g, w, cells, rng, frames=8):
    for _ in range(frames):
        y = np.clip(w.loss.reshape(-1)[cells] + rng.normal(0, sigma_meas(Q), len(cells)), 0, 1)
        g.update(
            Observation(0.0, Pose(0, 0, 0), cells, y.astype(np.float32), y.astype(np.float32), Q)
        )


def _band(p, n, rng):
    rows = p.cells // n
    y0 = rng.integers(rows.min(), rows.max() + 1)
    return p.cells[np.abs(rows - y0) < 8]


def test_intervals_are_calibrated_on_partial_surveys():
    z = []
    for s in range(4000, 4012):
        w, rng = generate(s, CFG.world), np.random.default_rng(s)
        g = GPBelief(w, CFG.belief)
        for p in w.parcels:
            if rng.random() < 0.6:
                _look(g, w, _band(p, w.size, rng), rng)
        m, width = g.parcel_moments(w.parcels)
        t = parcel_truth(w)
        z += list((np.array([t[p.id] for p in w.parcels]) - m) / (width / 3.92))
    cover = np.mean(np.abs(z) < 1.96)
    assert 0.9 <= cover <= 0.99, cover


def test_looking_at_some_fields_informs_the_others():
    w, rng = generate(4100, CFG.world), np.random.default_rng(0)
    g = GPBelief(w, CFG.belief)
    seen, unseen = w.parcels[::2], w.parcels[1::2]
    _, w0 = g.parcel_moments(unseen)
    for p in seen:
        _look(g, w, p.cells, rng, frames=4)
    _, w1 = g.parcel_moments(unseen)
    assert (w1 < w0).all()


def test_full_survey_is_close_and_confident():
    w, rng = generate(7, CFG.world), np.random.default_rng(1)
    g = GPBelief(w, CFG.belief)
    for p in w.parcels:
        _look(g, w, p.cells, rng)
    m, width = g.parcel_moments(w.parcels)
    t = np.array([parcel_truth(w)[p.id] for p in w.parcels])
    assert np.abs(m - t).max() < 0.05
    assert (width < 0.6 * 0.2).all()  # passes the 0.8 confidence gate


def test_update_is_cheap_and_independent_of_map_size():
    w, rng = generate(7, CFG.world), np.random.default_rng(2)
    g = GPBelief(w, CFG.belief)
    cells = w.parcels[0].cells[:60]
    t = time.perf_counter()
    _look(g, w, cells, rng, frames=200)
    assert (time.perf_counter() - t) / 200 < 0.002
    assert g.lam.shape[0] == 3 + len(w.parcels)


def test_doubt_widens_the_range_but_never_moves_the_estimate():
    w = generate(7, CFG.world)
    g = GPBelief(w, CFG.belief)
    p = w.parcels[3]
    m0, w0 = g.parcel_moments((p,))
    assert g.doubt(p.id) and not g.doubt(p.id)
    m1, w1 = g.parcel_moments((p,))
    assert w1[0] > w0[0] * 1.2 and abs(m1[0] - m0[0]) < 1e-12
