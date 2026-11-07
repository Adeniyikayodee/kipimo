import math

import numpy as np

from kipimo import config
from kipimo.planning.district import place_hubs, reach_km, sites, villages


def test_reach_matches_the_battery():
    assert 5.0 < reach_km(config.load().robot) < 5.6


def test_greedy_hubs_cover_far_more_than_random_sites():
    vs, r = villages(1), reach_km(config.load().robot)
    hubs, covered = place_hubs(vs, 4, r)
    rng, cands, total = np.random.default_rng(0), sites(), sum(v.insured for v in vs)
    rand = np.mean(
        [
            sum(v.insured for v in vs if any(math.dist((v.x, v.y), cands[i]) <= r for i in pick))
            / total
            for pick in (rng.choice(len(cands), 4, replace=False) for _ in range(100))
        ]
    )
    assert len(hubs) == 4 and covered > rand + 0.3
