"""Kenyan flood events from Groundsource (Google Research, 2026; CC BY 4.0).

Groundsource used Gemini to turn news reports into 2.6M geo-tagged flood events. We keep the
Kenyan ones as independent evidence of when and where real floods happened, to choose and check
the regions we pull satellite data for. Download the parquet from doi.org/10.5281/zenodo.18647053
into data/groundsource/ first.

    uv run --group geo python -m kipimo.geo.groundsource
"""

import csv
import struct

from kipimo.config import ROOT

SRC = ROOT / "data" / "groundsource" / "groundsource_2026.parquet"
OUT = ROOT / "config" / "regions" / "groundsource_kenya.csv"
KENYA = (33.9, -4.7, 41.9, 5.0)  # lon min, lat min, lon max, lat max
REGIONS = {  # the three study areas, lon min, lat min, lon max, lat max
    "tana_river": (38.5, -2.6, 40.5, 0.5),
    "budalangi": (33.9, -0.2, 34.25, 0.3),
    "kano_plains": (34.7, -0.35, 35.2, 0.0),
}


def bbox(wkb: bytes) -> tuple[float, float, float, float]:
    """Bounding box of a WKB point, polygon or multipolygon, no geometry library needed."""
    xs, ys = [], []

    def geom(o: int) -> int:
        e = "<" if wkb[o] == 1 else ">"
        kind = struct.unpack_from(e + "I", wkb, o + 1)[0] % 1000
        o += 5
        if kind == 1:
            x, y = struct.unpack_from(e + "2d", wkb, o)
            xs.append(x), ys.append(y)
            return o + 16
        if kind == 3:
            (rings,) = struct.unpack_from(e + "I", wkb, o)
            o += 4
            for _ in range(rings):
                (n,) = struct.unpack_from(e + "I", wkb, o)
                pts = struct.unpack_from(e + f"{2 * n}d", wkb, o + 4)
                xs.extend(pts[0::2]), ys.extend(pts[1::2])
                o += 4 + 16 * n
            return o
        if kind in (4, 6, 7):
            (parts,) = struct.unpack_from(e + "I", wkb, o)
            o += 4
            for _ in range(parts):
                o = geom(o)
            return o
        raise ValueError(f"unsupported WKB type {kind}")

    geom(0)
    return min(xs), min(ys), max(xs), max(ys)


def _inside(b, box) -> bool:
    x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def kenya(path=SRC) -> list[dict]:
    import pyarrow.parquet as pq

    f, out = pq.ParquetFile(path), []
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["uuid", "area_km2", "geometry", "start_date", "end_date"])
        for r in t.to_pylist():
            b = bbox(r["geometry"])
            if not _inside(b, KENYA):
                continue
            region = next((k for k, box in REGIONS.items() if _inside(b, box)), "")
            out.append(
                {
                    "uuid": r["uuid"],
                    "start": r["start_date"],
                    "end": r["end_date"],
                    "lon": round((b[0] + b[2]) / 2, 4),
                    "lat": round((b[1] + b[3]) / 2, 4),
                    "area_km2": round(r["area_km2"], 3),
                    "region": region,
                }
            )
    return sorted(out, key=lambda r: (r["start"], r["uuid"]))


def load(path=OUT) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def peak_week(rows: list[dict], region: str, start: str, end: str) -> tuple[str, int]:
    """ISO week with the most reported events in a region and window: where to look for radar."""
    from collections import Counter
    from datetime import date

    weeks = Counter(
        "{}-W{:02d}".format(*date.fromisoformat(r["start"]).isocalendar()[:2])
        for r in rows
        if r["region"] == region and start <= r["start"] <= end
    )
    return weeks.most_common(1)[0] if weeks else ("", 0)


def main() -> None:
    rows = kenya()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        fh.write(
            "# Groundsource (Google Research 2026), CC BY 4.0, doi.org/10.5281/zenodo.18647053\n"
        )
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} Kenyan events -> {OUT.relative_to(ROOT)}")
    for region in REGIONS:
        week, n = peak_week(rows, region, "2024-03-01", "2024-06-15")
        print(f"  {region:12s} 2024 long rains peak {week} ({n} events)")


if __name__ == "__main__":
    main()
