import numpy as np

from kipimo.world.generate import generate, parcel_truth


def test_deterministic():
    a, b = generate(7), generate(7)
    for f in ("elevation", "water", "loss", "parcel_map", "prior"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert [p.sum_insured for p in a.parcels] == [p.sum_insured for p in b.parcels]
    assert not np.array_equal(generate(8).loss, a.loss)


def test_parcels_and_loss_shape():
    for seed in range(5):
        w = generate(seed)
        assert 12 <= len(w.parcels) <= 20
        truth = np.array(list(parcel_truth(w).values()))
        assert truth.min() < 0.1 and truth.max() > 0.35


def test_flood_is_spatially_coherent():
    w = generate(7)
    lag = np.corrcoef(w.water[:, :-1].ravel(), w.water[:, 1:].ravel())[0, 1]
    assert lag > 0.95
    assert 0.1 < (w.water > 0.5).mean() < 0.5


def test_no_empty_or_sliver_parcels():
    for seed in [*range(100, 115), *range(200, 215)]:
        w = generate(seed)
        assert all(len(p.cells) >= 200 for p in w.parcels)
        assert int(w.parcel_map.max()) + 1 == len(w.parcels)


def test_tana_terrain_is_real_seeded_and_keeps_farms_off_the_channel():
    import numpy as np

    from kipimo.config import WorldCfg
    from kipimo.world.generate import generate

    cfg = WorldCfg(terrain="tana")
    a, b = generate(3, cfg), generate(3, cfg)
    assert a.site.startswith("Tana River") and np.array_equal(a.elevation, b.elevation)
    assert not np.array_equal(a.elevation, generate(4, cfg).elevation)
    assert 0.15 < (a.water > 0.5).mean() < 0.85
    assert (a.parcel_map[a.elevation < 0.3] == -1).all()
    left, top, right, bottom = a.rect
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1 and generate(3).rect is None
