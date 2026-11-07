"""Verify a recorded run: the hash chain, every claim against the rules and a pinned key, every
SMS packet against its claim, and every actuary decision against the claim it signs off.

uv run python -m kipimo.audit events.jsonl [--pubkey HEX] [--actuary HEX] [--head HASH]

Without keys the demo keys for the run's seed are pinned; a deployment pins the insurer's and
the actuary's.
Pass --head, recorded outside the log, to catch a truncated tail.
"""

import argparse
import json
import sys
from pathlib import Path

from kipimo.bus import verify_chain
from kipimo.payout import triggers
from kipimo.payout.approval import claim_hash
from kipimo.payout.claim import Signer, verify_claim
from kipimo.uplink import codec


def _claim_problem(c: dict, sig: str, logged_key: str, key: str) -> str:
    s = triggers.score_crop(c["crop"], c["sum_insured"], c["loss_mean"])
    if logged_key != key or not verify_claim(c, sig, key):
        return "bad signature"
    if c["ruleset"] != triggers.version():
        return f"signed under unknown ruleset {c['ruleset']}"
    if (s.trigger_id, s.tier, s.amount) != (c["trigger_id"], c["tier"], c["amount"]):
        return "amount does not follow from the rules"
    return "" if c["confidence"] >= triggers.gate() else "below the confidence gate"


def _packet_ok(hex_: str, c: dict | None, key: str) -> bool:
    p = codec.verify(bytes.fromhex(hex_), key)
    want = c and (c["event"], c["parcel_id"], round(c["tier"] * 100), round(c["loss_mean"] * 1e4))
    return bool(p) and want == (p["event"], p["parcel_id"], p["tier_pct"], p["loss_bp"])


def _decision_problem(d: dict, claims: dict, done: set, key: str) -> str:
    doc = d["decision"]
    if d["pubkey"] != key or not verify_claim(doc, d["sig"], key):
        return "bad actuary signature"
    if doc["case_id"] in done:
        return "decided twice"
    c = claims.get(doc["claim_id"]) if doc["subject"] == "claim" else None
    if doc["subject"] == "claim" and (c is None or claim_hash(c) != doc["claim_hash"]):
        return "signs off a claim that is not in the log"
    return "note missing on a rejection" if doc["decision"] == "reject" and not doc["note"] else ""


def audit(
    path: Path, pubkey: str | None = None, head: str | None = None, actuary: str | None = None
) -> dict:
    evs, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        try:
            evs.append(json.loads(line))
        except ValueError:
            problems.append(f"line {i} is torn")
            break
    if not evs:
        problems.append("empty log")
    if not verify_chain(evs):
        problems.append("hash chain broken")
    if head and (not evs or evs[-1]["hash"] != head):
        problems.append("head does not match: log truncated or replaced")
    seed = next((e["data"]["seed"] for e in evs if e["kind"] == "EPISODE_START"), None)
    key = pubkey or (Signer(seed).public_hex if seed is not None else "")
    akey = actuary or (Signer(seed, "actuary").public_hex if seed is not None else "")
    claims: dict[str, dict] = {}
    done: set[str] = set()
    approved: dict[str, float] = {}
    opened: set[str] = set()
    for e in evs:
        d = e["data"]
        if e["kind"] == "CLAIM_ISSUED":
            c = claims[d["claim"]["claim_id"]] = d["claim"]
            if why := _claim_problem(c, d["sig"], d["pubkey"], key):
                problems.append(f"{c['claim_id']}: {why}")
        elif e["kind"] == "UPLINK_QUEUED":
            if not _packet_ok(d["packet"], claims.get(d["claim_id"]), key):
                problems.append(f"{d['claim_id']}: packet does not match its claim")
        elif e["kind"] == "CLAIM_AWAITING_APPROVAL":
            opened.add(d["case_id"])
        elif e["kind"] == "ACTUARY_DECISION":
            doc = d["decision"]
            if why := _decision_problem(d, claims, done, akey):
                problems.append(f"{doc['case_id']}: {why}")
            elif doc["subject"] == "claim" and doc["decision"] == "approve":
                approved[doc["claim_id"]] = claims[doc["claim_id"]]["amount"]
            done.add(doc["case_id"])
    return {
        "events": len(evs),
        "head": evs[-1]["hash"] if evs else None,
        "claims": len(claims),
        "refusals": sum(e["kind"] == "CLAIM_REFUSED" for e in evs),
        "paid_total": round(sum(c["amount"] for c in claims.values()), 2),
        "approved": len(approved),
        "approved_total": round(sum(approved.values()), 2),
        "awaiting": len(opened - done),
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, nargs="?", default=Path("events.jsonl"))
    ap.add_argument("--pubkey")
    ap.add_argument("--head")
    ap.add_argument("--actuary")
    a = ap.parse_args()
    r = audit(a.path, a.pubkey, a.head, a.actuary)
    print(f"{a.path}: {r['events']} events, head {r['head']}")
    print(f"{r['claims']} signed claims, {r['refusals']} refusals, signed {r['paid_total']:,.2f}")
    print(
        f"actuary approved {r['approved']} for {r['approved_total']:,.2f}, "
        f"{r['awaiting']} awaiting a decision"
    )
    print("\n".join(r["problems"]) or "OK")
    sys.exit(1 if r["problems"] else 0)


if __name__ == "__main__":
    main()
