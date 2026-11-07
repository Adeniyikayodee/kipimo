import json

import numpy as np
import pytest

from kipimo.bus import Bus
from kipimo.types import Pose


def test_bus_writes_jsonl(tmp_path):
    bus = Bus(tmp_path / "e.jsonl")
    bus.emit("A", pose=Pose(1.0, 2.0, 0.1234567891), v=np.float32(0.5), ids=np.arange(3))
    bus.close()
    ev = json.loads((tmp_path / "e.jsonl").read_text().splitlines()[0])
    assert ev["kind"] == "A" and ev["seq"] == 0
    assert ev["data"]["pose"]["yaw"] == 0.123457
    assert ev["data"]["ids"] == [0, 1, 2]


def test_nested_emits_reach_subscribers_in_seq_order():
    bus = Bus()
    seen = []

    def echo(ev):
        if ev["kind"] == "A":
            bus.emit("B")

    bus.subscribers += [echo, lambda ev: seen.append(ev["seq"])]
    bus.emit("A")
    assert seen == [0, 1]


def test_nan_is_refused():
    with pytest.raises(ValueError):
        Bus().emit("A", v=float("nan"))


def test_a_raising_subscriber_does_not_stop_delivery():
    bus, seen = Bus(), []

    def boom(ev):
        if ev["seq"] == 0:
            raise RuntimeError

    bus.subscribers += [boom, lambda ev: seen.append(ev["seq"])]
    with pytest.raises(RuntimeError):
        bus.emit("A")
    bus.emit("B")
    assert seen[-1] == 1


def test_canonical_json_is_rfc_8785():
    from kipimo.bus import canonical

    nums = [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27, 1e-7, -1.5, 1.0, -0.0, 1e21, 123.456]
    out = "[333333333.3333333,1e+30,4.5,0.002,1e-27,1e-7,-1.5,1,0,1e+21,123.456]"
    assert canonical(nums).decode() == out
    assert canonical({"b": "ñ\n", "a": True}).decode() == '{"a":true,"b":"ñ\\n"}'


def test_an_existing_log_is_never_overwritten(tmp_path):
    Bus(tmp_path / "e.jsonl").close()
    with pytest.raises(FileExistsError):
        Bus(tmp_path / "e.jsonl")
