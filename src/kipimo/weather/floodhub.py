"""The trigger: a river gauge forecast passing its danger level books the flight, no farmer needed.

Shaped like a Google Flood Hub gauge forecast: hourly river levels for the coming days against
the gauge's warning and danger levels. Synthetic and seeded until the Flood Forecasting API is
wired in; the trigger does not change. Mission time zero is the hour the river passes danger.
"""

from dataclasses import dataclass

import numpy as np

from kipimo.config import FloodCfg


@dataclass(frozen=True)
class Gauge:
    name: str
    levels: np.ndarray  # metres, hourly from when the forecast was issued
    warning: float
    danger: float
    issued_hour: int  # local clock hour of issue


def gauge(seed: int, cfg: FloodCfg) -> Gauge:
    """Floods come at night: the crossing falls between 21:00 and 05:00 local, days ahead."""
    rng = np.random.default_rng([seed, 17])
    day, night = rng.integers(*cfg.lead_days), rng.uniform(21, 29)
    cross = 24 * day + night - cfg.issued_hour
    h = np.arange(cfg.hours)
    levels = np.clip(cfg.danger + cfg.rise_m_per_h * (h - cross), cfg.base_m, cfg.peak_m)
    return Gauge(cfg.gauge, np.round(levels, 3), cfg.warning, cfg.danger, cfg.issued_hour)


def crossing(g: Gauge) -> int | None:
    """First forecast hour at or above danger."""
    over = np.flatnonzero(g.levels >= g.danger)
    return int(over[0]) if over.size else None


def alert(ep) -> bool:
    """Emit the alert and the booking. False, and no flight, if the river stays below danger."""
    cfg, bus = ep.cfg.flood, ep.bus
    if not cfg.enabled:
        return True
    g = gauge(ep.cfg.seed, cfg)
    h = crossing(g)
    if h is None:
        bus.emit("FLOOD_WATCH", gauge=g.name, peak_m=float(g.levels.max()), danger_m=g.danger)
        return False
    w = ep.weather
    window = (w.window(bus.t) if w else 0) or 0
    bus.emit(
        "FLOOD_ALERT",
        gauge=g.name,
        danger_m=g.danger,
        level_m=float(g.levels[h]),
        lead_h=h,
        local_hour=(g.issued_hour + h) % 24,
    )
    bus.emit("FLIGHT_BOOKED", hub="home", in_h=window, reason="first calm window" if w else "now")
    return True
