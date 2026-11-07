from dataclasses import dataclass, field

import numpy as np

STATES = ("online", "degraded", "offline")


@dataclass
class LinkModel:
    """Cellular uplink. Offline blocks everything; degraded drops packets and acks at random."""

    rng: np.random.Generator
    state: str = "online"
    loss_pct: float = 30.0
    latency_ms: float = 800.0
    sent_bytes: int = 0
    attempts: int = field(default=0)

    def transmit(self, packet: bytes) -> tuple[bool, bool]:
        """Returns (delivered, acked). An ack can be lost after delivery, causing a resend."""
        if self.state == "offline":
            return False, False
        self.attempts += 1
        self.sent_bytes += len(packet)
        if self.state == "online":
            return True, True
        p = self.loss_pct / 100
        delivered = self.rng.random() >= p
        acked = delivered and self.rng.random() >= p
        return delivered, acked
