import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np


def _plain(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float | np.floating):
        return round(float(o), 6)
    if hasattr(o, "__dataclass_fields__"):
        return {k: _plain(getattr(o, k)) for k in o.__dataclass_fields__}
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_plain(v) for v in o]
    return o


GENESIS = "0" * 64


_ESC = re.compile(r'[\x00-\x1f"\\]')  # characters a JSON string must escape


def _num(x: float) -> str:
    """ECMAScript number to string, as RFC 8785 requires."""
    if not math.isfinite(x):
        raise ValueError(f"{x} is not JSON")
    r = repr(x)
    if "e" not in r and x:  # plain decimals print alike, bar Python's trailing .0
        return r[:-2] if r.endswith(".0") else r
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    _, ds, e = Decimal(repr(abs(x))).as_tuple()
    d, n = "".join(map(str, ds)).rstrip("0"), len(ds) + e
    s = "-" if x < 0 else ""
    if 0 < n <= 21:
        return s + d[:n] + "." + d[n:]
    if -6 < n <= 0:
        return s + "0." + "0" * -n + d
    return s + d[0] + ("." + d[1:] if len(d) > 1 else "") + f"e{n - 1:+d}"


def _obj(parts: dict[str, str]) -> str:
    keys = sorted(parts) if all(k.isascii() for k in parts) else sorted(parts, key=_utf16)
    return "{" + ",".join(f"{_jcs(k)}:{parts[k]}" for k in keys) + "}"


def _utf16(k: str) -> bytes:
    return k.encode("utf-16-be")


def _jcs(o: Any) -> str:
    t = type(o)  # exact types first: this runs for every value of every event
    if t is str:
        return json.dumps(o, ensure_ascii=False) if _ESC.search(o) else f'"{o}"'
    if t is float:
        return _num(o)
    if t is int:
        return str(o)
    if t is dict:
        return _obj({k: _jcs(v) for k, v in o.items()})
    if t is list or t is tuple:
        return "[" + ",".join(map(_jcs, o)) + "]"
    if o is None or isinstance(o, bool):
        return json.dumps(o)
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        return _num(o)
    if isinstance(o, str):
        return json.dumps(o, ensure_ascii=False)
    if isinstance(o, dict):
        return _obj({k: _jcs(v) for k, v in o.items()})
    if isinstance(o, list | tuple):
        return "[" + ",".join(map(_jcs, o)) + "]"
    raise TypeError(f"{type(o).__name__} is not JSON")


def canonical(obj: Any) -> bytes:
    """RFC 8785 canonical JSON, so any language can rebuild the bytes behind a hash or signature."""
    return _jcs(obj).encode()


def event_hash(ev: dict) -> str:
    body = {k: v for k, v in ev.items() if k != "hash"}
    return hashlib.sha256(canonical(body)).hexdigest()


def verify_chain(events: list[dict]) -> bool:
    """True if every event hashes correctly and links to its predecessor."""
    prev = GENESIS
    for i, ev in enumerate(events):
        if ev.get("seq") != i or ev.get("prev") != prev or ev.get("hash") != event_hash(ev):
            return False
        prev = ev["hash"]
    return True


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class Bus:
    """Append only event log. Sim time only, never wall clock, so logs are reproducible."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.seq = 0
        self.t = 0.0
        self.head = GENESIS
        self._pending: list[dict] = []
        self._dispatching = False
        self.events: list[dict] = []
        self.subscribers: list[Callable[[dict], None]] = []
        self._fh = path.open("x") if path else None  # never overwrite an audit trail

    def emit(self, kind: str, **data: Any) -> dict:
        ev = {
            "seq": self.seq,
            "t": round(self.t, 3),
            "kind": kind,
            "data": _plain(data),
            "prev": self.head,
        }
        parts = {k: _jcs(v) for k, v in ev.items()}  # each value serialised once
        ev["hash"] = self.head = hashlib.sha256(_obj(parts).encode()).hexdigest()
        line = _obj(parts | {"hash": f'"{ev["hash"]}"'})
        self.seq += 1
        self.events.append(ev)
        if self._fh:
            self._fh.write(line + "\n")
            if kind in ("CLAIM_ISSUED", "ACTUARY_DECISION"):  # signed records survive a crash
                self._fh.flush()
                os.fsync(self._fh.fileno())
        self._pending.append(ev)
        if not self._dispatching:  # events emitted by subscribers are delivered after, in order
            self._dispatching = True
            try:
                while self._pending:
                    e = self._pending.pop(0)
                    for fn in self.subscribers:
                        fn(e)
            finally:
                self._dispatching = False
        return ev

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
