from kipimo.config import WeatherCfg
from kipimo.weather.forecast import make

CFG = WeatherCfg(enabled=True)


def test_disabled_by_default():
    assert make(7, WeatherCfg()) is None


def test_storm_blocks_the_first_hours_and_a_window_follows():
    w = make(7, CFG)
    assert w.members.shape == (CFG.members, CFG.hours) and w.actual.shape == (CFG.hours,)
    assert w.safe_share(1) < 0.5
    start = w.window(0, need_h=1)
    assert start is not None and start >= 2
    assert w.safe_share(start) >= 0.95


def test_seeded():
    a, b = make(3, CFG), make(3, CFG)
    assert (a.members == b.members).all() and (a.actual == b.actual).all()


def test_gusts_are_seeded_and_scale_with_the_mean():
    import numpy as np

    from kipimo.weather.wind import Wind

    def series(seed):
        w = Wind(seed, lambda t: 5.0, 0.2, 4.0)
        return np.array([w.step(t, 1.0) for t in range(4000)]), w

    a, w = series(1)
    assert np.array_equal(a, series(1)[0]) and not np.array_equal(a, series(2)[0])
    assert np.allclose(a.mean(0), w.mean(0), atol=0.3)
    assert 0.8 < (a - a.mean(0)).std(0).mean() < 1.2  # 20% of 5 m/s
