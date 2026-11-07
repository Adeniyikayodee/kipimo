from kipimo.bench.run import one, scenarios


def test_thirty_disjoint_scenarios():
    sc = scenarios()
    assert len(sc) == 30 and len(set(sc)) == 30
    assert not {s for _, s in sc} & {7, 11, *range(1000, 1040)}


def test_agent_stack_frontier_on_six_floods():
    seeds = [("bench_heavy", s) for s in range(100, 103)] + [
        ("bench_corridor", s) for s in range(200, 203)
    ]

    def mean(policy, param, key):
        return sum(one((sc, s, policy, param))[key] for sc, s in seeds) / len(seeds)

    full = mean("lawnmower", 0.1, "distance_m")
    assert mean("agent", 0.2, "distance_m") < 0.7 * full
    assert mean("agent", 0.2, "payout_mae_pct") < mean("lawnmower", -1.0, "payout_mae_pct")
    assert sum(one((sc, s, "agent", 0.2))["missed_claims"] for sc, s in seeds) == 0
