from itertools import combinations

from kipimo.belief import fit
from kipimo.bench.run import scenarios
from kipimo.evals import district
from kipimo.evals.cases import suite
from kipimo.perception.model import TRAIN_SEEDS
from kipimo.server.site import DISTRICTS


def _villages(seeds) -> set[int]:
    return {s * 100 + i for s in seeds for i in range(20)}


def test_world_seeds_never_cross_purposes():
    spaces = {
        "bench": {s for _, s in scenarios()},
        "evals": {c.seed for split in ("dev", "test") for c in suite(split)},
        "district evals": _villages(s for sp in district.SPLITS.values() for s in sp),
        "train": set(TRAIN_SEEDS),
        "calibration": set(fit.SEEDS),
        "gp test": set(range(4000, 4012)),
        "demo": {7, 11} | _villages(DISTRICTS),
    }
    for a, b in combinations(spaces, 2):
        assert not spaces[a] & spaces[b], (a, b, sorted(spaces[a] & spaces[b])[:5])


def test_same_seed_same_log_under_any_hash_seed(tmp_path):
    import hashlib
    import os
    import subprocess
    import sys

    def digest(hashseed: str) -> str:
        out = tmp_path / f"{hashseed}.jsonl"
        env = {k: v for k, v in os.environ.items() if "KEY" not in k and "TOKEN" not in k}
        env["PYTHONHASHSEED"] = hashseed  # no keys: the rule agents run, offline
        cmd = [sys.executable, "-m", "kipimo.run", "--policy", "agents", "--out", str(out)]
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        return hashlib.sha256(out.read_bytes()).hexdigest()

    assert digest("1") == digest("999")
