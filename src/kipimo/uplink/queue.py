import sqlite3
from pathlib import Path


class Outbox:
    """Store and forward queue in SQLite. Monotonic seq, one row per claim id, survives restarts."""

    def __init__(self, path: Path | str = ":memory:"):
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS outbox ("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " claim_id TEXT UNIQUE NOT NULL,"
            " packet BLOB NOT NULL,"
            " sent INTEGER NOT NULL DEFAULT 0,"
            " attempts INTEGER NOT NULL DEFAULT 0)"
        )
        self.db.commit()

    def put(self, claim_id: str, packet: bytes) -> bool:
        """True if queued, False if this exact packet already was. A different one raises."""
        cur = self.db.execute(
            "INSERT OR IGNORE INTO outbox (claim_id, packet) VALUES (?, ?)", (claim_id, packet)
        )
        self.db.commit()
        if cur.rowcount == 0:
            (old,) = self.db.execute(
                "SELECT packet FROM outbox WHERE claim_id = ?", (claim_id,)
            ).fetchone()
            if old != packet:
                raise ValueError(f"{claim_id} already queued with a different packet")
        return cur.rowcount == 1

    def pending(self) -> list[tuple[int, str, bytes]]:
        return self.db.execute(
            "SELECT seq, claim_id, packet FROM outbox WHERE sent = 0 ORDER BY seq"
        ).fetchall()

    def mark(self, seq: int, sent: bool) -> None:
        self.db.execute(
            "UPDATE outbox SET attempts = attempts + 1, sent = ? WHERE seq = ?", (int(sent), seq)
        )
        self.db.commit()

    def depth(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent = 0").fetchone()[0]
