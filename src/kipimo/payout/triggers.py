import hashlib
import re
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import get_args

import yaml

from kipimo.config import CONFIG_DIR
from kipimo.types import Crop, Parcel

RULE = re.compile(r"^\s*loss_mean\s*>=\s*([0-9.]+)\s*$")


@dataclass(frozen=True)
class Rule:
    trigger_id: int
    threshold: float
    pays: float


@dataclass(frozen=True)
class Score:
    trigger_id: int  # 0 means no trigger fired
    tier: float
    amount: float


class _Strict(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [k.value for k, _ in node.value]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate key in {keys}")
        return super().construct_mapping(node, deep)


@cache
def load(path: Path = CONFIG_DIR / "triggers.yaml") -> tuple[dict[str, tuple[Rule, ...]], float]:
    """Rules per crop, highest threshold first, and the confidence gate. Refuses a bad file."""
    raw = yaml.load(path.read_text(), _Strict)
    gate = float(raw.pop("gate")["min_confidence"])
    if set(raw) != set(get_args(Crop)) or not 0 < gate <= 1:
        raise ValueError(f"need every crop {get_args(Crop)} and a gate in (0, 1]")
    rules = {}
    for crop, items in raw.items():
        out = []
        for it in items:
            m = RULE.match(it["if"])
            if not m:
                raise ValueError(f"unsupported rule: {it['if']}")
            out.append(Rule(int(it["id"]), float(m.group(1)), float(it["pays"])))
        rules[crop] = rs = tuple(sorted(out, key=lambda r: -r.threshold))
        ok = all(0 < r.threshold <= 1 and 0 < r.pays <= 1 for r in rs)
        ok = ok and all(round(r.pays * 100, 6) % 1 == 0 for r in rs)
        if not ok or any(a.pays <= b.pays or a.threshold == b.threshold for a, b in pairwise(rs)):
            raise ValueError(f"{crop}: thresholds in (0, 1], whole percent pays rising with loss")
    ids = [r.trigger_id for rs in rules.values() for r in rs]
    if 0 in ids or len(ids) != len(set(ids)):
        raise ValueError("trigger ids must be unique and nonzero")
    return rules, gate


def score(parcel: Parcel, loss_mean: float) -> Score:
    return score_crop(parcel.crop, parcel.sum_insured, loss_mean)


def score_crop(crop: str, sum_insured: float, loss_mean: float) -> Score:
    for r in load()[0][crop]:
        if loss_mean >= r.threshold:
            return Score(r.trigger_id, r.pays, round(r.pays * sum_insured, 2))
    return Score(0, 0.0, 0.0)


@cache
def version(path: Path = CONFIG_DIR / "triggers.yaml") -> str:
    """Short hash of the rule file. Claims carry it, so old claims stay interpretable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def boundaries(crop: str) -> list[float]:
    return [r.threshold for r in load()[0][crop]]


def gate() -> float:
    return load()[1]
