from kipimo.bus import Bus
from kipimo.payout.claim import Signer
from kipimo.uplink import codec
from kipimo.uplink.link import LinkModel
from kipimo.uplink.queue import Outbox


class Receiver:
    """Insurer gateway. Verifies every packet and settles each event and parcel once.

    The key comes from the signed body, never from the label the packet travelled with.
    """

    def __init__(self, public_hex: str):
        self.public_hex = public_hex
        self.settled: dict[tuple[int, str], dict] = {}
        self.duplicates = 0
        self.rejected = 0

    def receive(self, packet: bytes) -> None:
        d = codec.verify(packet, self.public_hex)
        key = d and (d["event"], d["parcel_id"])
        if not d:
            self.rejected += 1
        elif key in self.settled:
            self.duplicates += 1
        else:
            self.settled[key] = d


class Uplink:
    """Queues claim packets as they are issued and forwards them whenever the link allows."""

    def __init__(self, bus: Bus, signer: Signer, link: LinkModel, outbox: Outbox, schedule=()):
        self.bus = bus
        self.signer = signer
        self.link = link
        self.outbox = outbox
        self.receiver = Receiver(signer.public_hex)
        self.schedule = sorted(schedule, key=lambda s: s["t"])
        self.sizes: dict[str, int] = {}
        self.claims: dict[str, object] = {}

    def enqueue(self, claim) -> int:
        packet = codec.encode(claim, self.signer)
        self.sizes[claim.claim_id] = len(packet)
        if self.outbox.put(claim.claim_id, packet):
            self.bus.emit(
                "UPLINK_QUEUED",
                claim_id=claim.claim_id,
                bytes=len(packet),
                packet=packet.hex(),
                depth=self.outbox.depth(),
            )
        return len(packet)

    def set_state(self, state: str) -> None:
        if state != self.link.state:
            self.link.state = state
            self.bus.emit("LINK_STATE", state=state, depth=self.outbox.depth())

    def pump(self) -> None:
        while self.schedule and self.schedule[0]["t"] <= self.bus.t:
            self.set_state(self.schedule.pop(0)["state"])
        if self.link.state == "offline":
            return
        for seq, claim_id, packet in self.outbox.pending():
            delivered, acked = self.link.transmit(packet)
            if delivered:
                self.receiver.receive(packet)
            self.outbox.mark(seq, acked)
            if not acked:
                self.bus.emit("UPLINK_RETRY", claim_id=claim_id, delivered=delivered)
                return  # keep order; try again next tick
            self.bus.emit(
                "UPLINK_SENT",
                claim_id=claim_id,
                bytes=len(packet),
                depth=self.outbox.depth(),
                total_bytes=self.link.sent_bytes,
            )

    def drain(self, dt: float, max_ticks: int = 600) -> None:
        """Advance sim time at the pad until the queue empties, the schedule permitting."""
        for _ in range(max_ticks):
            if not self.outbox.depth() and not self.schedule:
                return
            if not self.outbox.depth() and self.link.state != "offline":
                return
            self.bus.t += dt
            self.pump()
