import struct

from kipimo.geo.groundsource import REGIONS, bbox, load, peak_week


def _poly(pts):
    ring = struct.pack("<I", len(pts)) + b"".join(struct.pack("<2d", *p) for p in pts)
    return b"\x01" + struct.pack("<I", 3) + struct.pack("<I", 1) + ring


def test_bbox_reads_polygons_and_multipolygons():
    a = _poly([(34.0, 0.0), (34.1, 0.0), (34.1, 0.2), (34.0, 0.0)])
    b = _poly([(39.0, -1.0), (39.5, -1.0), (39.5, -0.5), (39.0, -1.0)])
    multi = b"\x01" + struct.pack("<I", 6) + struct.pack("<I", 2) + a + b
    assert bbox(a) == (34.0, 0.0, 34.1, 0.2)
    assert bbox(multi) == (34.0, -1.0, 39.5, 0.2)


def test_committed_extract_covers_the_study_regions_in_2024():
    rows = load()
    for region in REGIONS:
        week, n = peak_week(rows, region, "2024-03-01", "2024-06-15")
        assert week.startswith("2024-W") and n >= 3, region
