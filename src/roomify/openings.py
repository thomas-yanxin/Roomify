"""Opening candidates, swing-arc evidence, and wall segments.

Openings are found by SCANNING WALLS, not by morphological gap-closing:
wall segments are derived from the room polygons (shared edges between
adjacent rooms, leftover edges toward everything else), and every interval
along a segment where the solid mask has no wall is an opening candidate.
Closing-based detection was tried first and fails structurally — door gaps
that open onto corridors narrower than the closing kernel get subsumed into
corridor-sized blobs. Scanning also yields ``connects`` and the hosting
wall for free.

The VLM classifies candidates against the legend vocabulary (CV cannot tell
a sliding door from a passage and should not try); a local window hint
(2-3 thin parallel strokes, legend cue B28) covers the no-VLM degradation.

Swing/hinge facts are the opposite: they come EXCLUSIVELY from pixel
evidence (a quarter-disc sector at a jamb, drawn either as a stroked arc or
as a tinted fill), never from the VLM. Plans that draw no swing arc get
``swing=None`` — reporting the unobservable would be invention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from roomify.merge import RoomDraft
from roomify.walls import WallExtraction

EXTERIOR = "exterior"
UNKNOWN = "unknown"  # beyond the wall is inside the building but not a room
MAX_CANDIDATES = 40  # one crop per candidate goes to the VLM; cap the bill

# Sector-occupancy arc test: the swing sector's fill differs from the floor
# beyond it (L1 over BGR medians), and a stroked arc must cover at least
# this fraction of the quarter-circle.
# Measured separations on the reference corpus: speckle-textured rooms fake
# tint deltas up to ~26, real tinted swing sectors start at ~37.
SECTOR_COLOR_DELTA = 30.0
ARC_STROKE_COVERAGE = 0.55  # a drawn arc traces the whole quarter-circle
ARC_STROKE_MARGIN = 0.30  # …and must beat the room's texture baseline
MIN_SECTOR_PX = 100  # smaller samples (tiny rooms) give no verdict
# A swing sector's radius IS the leaf, and no door leaf is 1.4m wide. Beyond
# that the sector test only ever fires on a floor-tint change spanning a wide
# mouth (measured: 1.6-2.4m "arcs" on balcony and corridor openings, while
# every genuinely arc-drawn door on the corpus sits at 0.57-1.33m).
MAX_LEAF_MM = 1400.0


@dataclass(frozen=True)
class ArcEvidence:
    hinge: tuple[float, float]  # working px
    swing: str  # "clockwise" | "counterclockwise" (image coords, y down)
    opens_into: int | None  # room index the sector lies in


@dataclass(frozen=True)
class WallSegment:
    start: tuple[float, float]  # working px, centerline
    end: tuple[float, float]
    thickness_px: float
    rooms: tuple[int | str, int | str]  # room indices / "exterior"


@dataclass(frozen=True)
class OpeningCandidate:
    marker: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 working px
    center: tuple[float, float]
    axis: str  # "h": break in a horizontal wall run; "v": vertical
    width_px: float  # break length along the wall
    kind_hint: str  # "window" | "doorlike"
    connects: tuple[int | str, int | str]  # room indices / "exterior" / "unknown"
    wall_index: int  # index into the wall-segment list
    arc: ArcEvidence | None


def find_openings(
    walls: WallExtraction,
    rooms: list[RoomDraft],
    bgr: np.ndarray,
    door_px: float | None = None,
) -> tuple[list[OpeningCandidate], list[WallSegment]]:
    from roomify.vlm import marker_ids

    if door_px is None:
        x0, y0, x1, y1 = walls.footprint
        door_px = 0.06 * max(x1 - x0, y1 - y0)  # ~900mm on a 10-15m dwelling

    segments = derive_wall_segments(rooms, walls)
    room_masks = _room_masks(rooms, walls.solid.shape)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )

    min_len = max(6.0, 0.5 * door_px)
    max_len = 4.6 * door_px  # wide balcony mouths still count; whole façades don't
    raw: list[OpeningCandidate] = []
    for si, seg in enumerate(segments):
        raw.extend(_scan_segment(si, seg, walls, room_masks, min_len, max_len))

    raw = _dedupe(raw, walls.thickness_px)
    if len(raw) > MAX_CANDIDATES:
        raw.sort(key=lambda c: c.width_px, reverse=True)
        raw = raw[:MAX_CANDIDATES]

    out: list[OpeningCandidate] = []
    ids = marker_ids(len(raw))
    max_leaf_px = door_px * MAX_LEAF_MM / 1000.0
    ordered = sorted(raw, key=lambda c: (c.center[1], c.center[0]))
    for marker, cand in zip(ids, ordered, strict=True):
        kind = "window" if _looks_like_window(cand, walls) else "doorlike"
        arc = None
        if kind == "doorlike" and cand.width_px <= max_leaf_px:
            arc = _detect_arc(cand, walls, room_masks, bgr, binary)
        out.append(replace(cand, marker=marker, kind_hint=kind, arc=arc))
    return out, segments


# ------------------------------------------------------------- wall scanning


def _scan_segment(
    seg_index: int,
    seg: WallSegment,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    min_len: float,
    max_len: float,
) -> list[OpeningCandidate]:
    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    dx, dy = sx1 - sx0, sy1 - sy0
    if abs(dx) < abs(dy) * 0.2 or abs(dy) < abs(dx) * 0.2:
        axis = "h" if abs(dx) >= abs(dy) else "v"
    else:
        return []  # diagonal walls: no opening scan (rare in the domain)

    solid = walls.solid
    h, w = solid.shape
    t = seg.thickness_px
    band = max(3, int(round(1.3 * t)))  # covers the wall body on either side

    if axis == "h":
        c = int(round((sy0 + sy1) / 2))
        lo, hi = int(round(min(sx0, sx1))), int(round(max(sx0, sx1)))
        lo, hi = max(0, lo), min(w - 1, hi)
        strip = solid[max(0, c - band) : min(h, c + band + 1), lo : hi + 1]
        coverage = strip.any(axis=0)
    else:
        c = int(round((sx0 + sx1) / 2))
        lo, hi = int(round(min(sy0, sy1))), int(round(max(sy0, sy1)))
        lo, hi = max(0, lo), min(h - 1, hi)
        strip = solid[lo : hi + 1, max(0, c - band) : min(w, c + band + 1)]
        coverage = strip.any(axis=1)
    assert isinstance(coverage, np.ndarray)
    if coverage.size == 0:
        return []

    connects = _resolve_connects(seg, walls, room_masks)
    runs = _absent_runs(coverage, min_len, max_len)
    # Windows do not always interrupt the wall: some plans draw the window
    # box ON a continuous wall run. Multi-stroke intervals in the lines mask
    # are the second, independent opening signal.
    runs = _merge_runs(runs, _window_runs(walls.lines, axis, c, lo, hi, band, min_len))

    out = []
    for run_lo, run_hi in runs:
        a, b = lo + run_lo, lo + run_hi
        center_along = (a + b) / 2
        if axis == "h":
            bbox = (float(a), float(c - t / 2), float(b), float(c + t / 2))
            center = (center_along, float(c))
        else:
            bbox = (float(c - t / 2), float(a), float(c + t / 2), float(b))
            center = (float(c), center_along)
        out.append(
            OpeningCandidate(
                marker="",
                bbox=bbox,
                center=center,
                axis=axis,
                width_px=float(run_hi - run_lo + 1),
                kind_hint="doorlike",
                connects=connects,
                wall_index=seg_index,
                arc=None,
            )
        )
    return out


def _window_runs(
    lines: np.ndarray, axis: str, c: int, lo: int, hi: int, band: int, min_len: float
) -> list[tuple[int, int]]:
    """Intervals along the segment where ≥2 separated strokes run in the
    wall band — the window signature, valid even on unbroken walls."""
    h, w = lines.shape
    if axis == "h":
        strip = lines[max(0, c - band) : min(h, c + band + 1), lo : hi + 1] > 0
        strokes = strip.astype(np.int8)
        transitions = (np.diff(strokes, axis=0) == 1).sum(axis=0) + strokes[0]
    else:
        strip = (lines[lo : hi + 1, max(0, c - band) : min(w, c + band + 1)] > 0).T
        strokes = strip.astype(np.int8)
        transitions = (np.diff(strokes, axis=0) == 1).sum(axis=0) + strokes[0]
    multi = (transitions >= 2).astype(np.int8)
    out = []
    for start, end, value in _runs(multi):
        if value == 1 and end - start + 1 >= min_len:
            out.append((start, end))
    return out


def _merge_runs(
    a: list[tuple[int, int]], b: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Union of two interval lists, coalescing overlaps."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(a + b):
        if merged and lo <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _absent_runs(
    coverage: np.ndarray, min_len: float, max_len: float
) -> list[tuple[int, int]]:
    """Intervals (inclusive) without wall. Wall specks ≤2px (JPEG noise)
    do not split a run; absent slivers ≤2px are jamb noise, not openings."""
    values = coverage.astype(np.int8)
    # absorb tiny wall specks into surrounding absence
    runs = _runs(values)
    for start, end, value in runs:
        if value == 1 and end - start + 1 <= 2:
            values[start : end + 1] = 0
    out = []
    for start, end, value in _runs(values):
        if value == 0 and min_len <= end - start + 1 <= max_len:
            out.append((start, end))
    return out


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    boundaries = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries - 1, [len(values) - 1]))
    return [(int(s), int(e), int(values[s])) for s, e in zip(starts, ends, strict=True)]


def _resolve_connects(
    seg: WallSegment,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
) -> tuple[int | str, int | str]:
    """Room indices pass through; the non-room side of a leftover edge is
    "exterior" only when it actually leaves the footprint — otherwise it is
    circulation space no room polygon covered, reported as "unknown"."""
    a, b = seg.rooms
    if isinstance(a, int) and isinstance(b, int):
        return (a, b)
    room = a if isinstance(a, int) else b
    mask = room_masks[room] if isinstance(room, int) else None
    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    mid = ((sx0 + sx1) / 2, (sy0 + sy1) / 2)
    probe_distance = 2.5 * seg.thickness_px
    outward = _room_outward_normal(seg, mask)
    if outward is None:
        far: int | str = UNKNOWN
    else:
        fx, fy = (
            mid[0] + outward[0] * probe_distance,
            mid[1] + outward[1] * probe_distance,
        )
        x0, y0, x1, y1 = walls.footprint
        inside_footprint = x0 + 2 <= fx <= x1 - 2 and y0 + 2 <= fy <= y1 - 2
        far = UNKNOWN if inside_footprint else EXTERIOR
    return (room, far) if isinstance(seg.rooms[0], int) else (far, room)


def _room_outward_normal(seg: WallSegment, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    dx, dy = sx1 - sx0, sy1 - sy0
    length = float(np.hypot(dx, dy))
    if length == 0:
        return None
    normal = np.array([-dy / length, dx / length])
    mid = np.array([(sx0 + sx1) / 2, (sy0 + sy1) / 2])
    probe_distance = 2.5 * seg.thickness_px
    for sign in (1, -1):
        px, py = mid + sign * normal * probe_distance
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
            return -sign * normal
    return None


def _dedupe(candidates: list[OpeningCandidate], thickness: float) -> list[OpeningCandidate]:
    """The same physical opening can be scanned from two rooms' leftover
    edges: same axis, along-intervals overlapping, centerlines within a wall
    of each other. Keep one, preferring room-room over room-unknown. Nearby
    PERPENDICULAR candidates are different openings and are never merged."""

    def rank(c: OpeningCandidate) -> tuple[int, float]:
        known = sum(1 for side in c.connects if isinstance(side, int))
        return (-known, -c.width_px)

    def along_interval(c: OpeningCandidate) -> tuple[float, float, float]:
        x0, y0, x1, y1 = c.bbox
        if c.axis == "h":
            return x0, x1, (y0 + y1) / 2
        return y0, y1, (x0 + x1) / 2

    kept: list[OpeningCandidate] = []
    for cand in sorted(candidates, key=rank):
        c_lo, c_hi, c_perp = along_interval(cand)
        duplicate = False
        for other in kept:
            if other.axis != cand.axis:
                continue
            o_lo, o_hi, o_perp = along_interval(other)
            if abs(c_perp - o_perp) > 2.0 * thickness:
                continue
            overlap = min(c_hi, o_hi) - max(c_lo, o_lo)
            if overlap > 0.5 * min(c_hi - c_lo, o_hi - o_lo):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _room_masks(rooms: list[RoomDraft], shape: tuple[int, int]) -> list[np.ndarray | None]:
    masks: list[np.ndarray | None] = []
    for room in rooms:
        if room.source == "vlm" or len(room.polygon) < 3:
            masks.append(None)
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(room.polygon).astype(np.int32)], 255)
        masks.append(mask)
    return masks


def _looks_like_window(candidate: OpeningCandidate, walls: WallExtraction) -> bool:
    """Windows carry 2-3 thin strokes ALONG the wall axis across the break
    (legend cue B28). Count distinct stroke rows/columns in the lines mask.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in candidate.bbox)
    h, w = walls.lines.shape
    pad = 2
    crop = walls.lines[
        max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)
    ]
    if crop.size == 0:
        return False
    # per-row (h) / per-column (v) fill fraction along the wall direction
    coverage = (crop > 0).mean(axis=1 if candidate.axis == "h" else 0)
    stroke_rows = coverage >= 0.5
    runs = int(np.count_nonzero(np.diff(np.concatenate(([0], stroke_rows.view(np.int8)))) == 1))
    return runs >= 2


# ---------------------------------------------------------------- swing arcs


def _detect_arc(
    candidate: OpeningCandidate,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    bgr: np.ndarray,
    binary: np.ndarray,
) -> ArcEvidence | None:
    """Quarter-disc evidence at either jamb, on either side of the wall.

    Two independent detectors, either suffices:
    - tinted sector: the disc's fill differs in color from the same room's
      floor just beyond it (listing plans draw the sweep as a lighter tint);
    - stroked arc: dark non-wall pixels concentrated on the quarter-circle
      at radius = opening width.
    Both compare strictly WITHIN the room the sector opens into — comparing
    across rooms would mistake a flooring change for a door arc (measured to
    produce 100% false positives otherwise). Ambiguous evidence yields None:
    swing is then honestly unresolved rather than invented.
    """
    x0, y0, x1, y1 = candidate.bbox
    radius = candidate.width_px
    if radius < 8:
        return None
    if candidate.axis == "h":
        jambs = [(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)]
        normals = [(0.0, -1.0), (0.0, 1.0)]
    else:
        jambs = [((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)]
        normals = [(-1.0, 0.0), (1.0, 0.0)]

    hits = []  # (jamb_idx, normal_idx, room_idx, score)
    for j, jamb in enumerate(jambs):
        other = jambs[1 - j]
        along = np.array([other[0] - jamb[0], other[1] - jamb[1]])
        norm = float(np.linalg.norm(along))
        if norm == 0:
            continue
        along = along / norm
        for k, normal_t in enumerate(normals):
            normal = np.array(normal_t)
            room_idx = _room_at(
                room_masks,
                (
                    jamb[0] + (along[0] + normal[0]) * radius * 0.45,
                    jamb[1] + (along[1] + normal[1]) * radius * 0.45,
                ),
            )
            if room_idx is None:
                continue
            mask = room_masks[room_idx]
            assert mask is not None
            valid = (mask > 0) & (walls.solid == 0)
            tinted, fill = _sector_color_delta(bgr, valid, jamb, along, normal, radius)
            stroked = _arc_stroke_coverage(binary, valid, jamb, along, normal, radius)
            baseline = _arc_stroke_coverage(binary, valid, jamb, along, normal, radius * 1.3)
            # Text and speckle textures light up any circle equally; a real
            # drawn arc lights up only the circle at the leaf radius.
            stroke_hit = (
                stroked >= ARC_STROKE_COVERAGE and stroked - baseline >= ARC_STROKE_MARGIN
            )
            if tinted >= SECTOR_COLOR_DELTA or stroke_hit:
                # fill breaks the jamb tie: both jambs' sectors overlap the
                # same tinted disc, but only the true hinge's annulus is
                # covered edge to edge.
                hits.append((j, k, room_idx, fill if tinted >= SECTOR_COLOR_DELTA else stroked))

    if not hits:
        return None
    hits.sort(key=lambda item: item[3], reverse=True)
    j, k, room_idx, _ = hits[0]
    jamb, normal = jambs[j], np.array(normals[k])
    other = jambs[1 - j]
    along = np.array([other[0] - jamb[0], other[1] - jamb[1]])
    along = along / np.linalg.norm(along)
    # The closed leaf lies along the wall (toward the far jamb); a positive
    # cross product means it sweeps clockwise (image coords, y down) to
    # reach the sector side.
    cross = along[0] * normal[1] - along[1] * normal[0]
    swing = "clockwise" if cross > 0 else "counterclockwise"
    return ArcEvidence(hinge=(float(jamb[0]), float(jamb[1])), swing=swing, opens_into=room_idx)


def _room_at(room_masks: list[np.ndarray | None], point: tuple[float, float]) -> int | None:
    for idx, mask in enumerate(room_masks):
        if mask is None:
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            return idx
    return None


def _sector_mask(
    shape: tuple[int, int],
    hinge: tuple[float, float],
    along: np.ndarray,
    normal: np.ndarray,
    r_inner: float,
    r_outer: float,
) -> np.ndarray:
    h, w = shape
    x_lo = max(0, int(hinge[0] - r_outer - 1))
    x_hi = min(w, int(hinge[0] + r_outer + 2))
    y_lo = max(0, int(hinge[1] - r_outer - 1))
    y_hi = min(h, int(hinge[1] + r_outer + 2))
    mask = np.zeros(shape, dtype=bool)
    if x_hi <= x_lo or y_hi <= y_lo:
        return mask
    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dx, dy = xs - hinge[0], ys - hinge[1]
    dist = np.hypot(dx, dy)
    a = dx * along[0] + dy * along[1]  # component along the closed leaf
    b = dx * normal[0] + dy * normal[1]  # component toward the sector side
    mask[y_lo:y_hi, x_lo:x_hi] = (dist >= r_inner) & (dist <= r_outer) & (a > 0) & (b > 0)
    return mask


def _sector_color_delta(
    bgr: np.ndarray,
    valid: np.ndarray,
    hinge: tuple[float, float],
    along: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> tuple[float, float]:
    """(median color delta sector-vs-beyond, fraction of the sector that is
    actually tinted). The fraction discriminates the true hinge: seen from
    the wrong jamb the tinted disc only partially covers the sector."""
    inner = (
        _sector_mask(bgr.shape[:2], hinge, along, normal, radius * 0.2, radius * 0.85) & valid
    )
    control = (
        _sector_mask(bgr.shape[:2], hinge, along, normal, radius * 1.15, radius * 1.6) & valid
    )
    if inner.sum() < MIN_SECTOR_PX or control.sum() < MIN_SECTOR_PX:
        return 0.0, 0.0
    inner_median = np.median(bgr[inner], axis=0)
    control_median = np.median(bgr[control], axis=0)
    delta = float(np.abs(inner_median - control_median).sum())
    if delta == 0:
        return 0.0, 0.0
    to_inner = np.abs(bgr[inner].astype(np.int16) - inner_median).sum(axis=1)
    to_control = np.abs(bgr[inner].astype(np.int16) - control_median).sum(axis=1)
    fill = float((to_inner < to_control).mean())
    return delta, fill


def _arc_stroke_coverage(
    binary: np.ndarray,
    valid: np.ndarray,
    hinge: tuple[float, float],
    along: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> float:
    """Fraction of the quarter-circle (at the opening-width radius) covered
    by dark stroke pixels, sampled only where the room's floor actually is."""
    angles = np.linspace(0.08, np.pi / 2 - 0.08, 32)
    h, w = binary.shape
    hit = usable = 0
    for angle in angles:
        direction = along * np.cos(angle) + normal * np.sin(angle)
        px = int(round(hinge[0] + direction[0] * radius))
        py = int(round(hinge[1] + direction[1] * radius))
        if not (0 <= px < w and 0 <= py < h) or not valid[py, px]:
            continue
        usable += 1
        if binary[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2].any():
            hit += 1
    if usable < len(angles) * 0.5:  # most of the circle is off-room: no verdict
        return 0.0
    return hit / usable


# ------------------------------------------------------------- wall segments


def derive_wall_segments(
    rooms: list[RoomDraft], walls: WallExtraction
) -> list[WallSegment]:
    """Centerline wall segments from room-polygon edges.

    Room polygons trace inner wall faces, so two adjacent rooms leave a
    2×(half thickness) gap between their facing edges — the wall. Facing
    edge pairs merge into one shared segment on the centerline; leftover
    edge intervals face the exterior or uncovered circulation space.
    Diagonal edges are emitted as-is (rare in the domain; refine when a
    corpus needs it). Exterior leftovers move half a measured wall thickness
    outward and extend at both ends so adjacent facade centerlines still meet.
    """
    t = walls.thickness_px
    room_masks = _room_masks(rooms, walls.solid.shape)
    edges = []  # (room_idx, axis, fixed_coord, lo, hi)
    diagonals: list[WallSegment] = []
    for idx, room in enumerate(rooms):
        if room.source == "vlm":
            continue  # bbox geometry would fabricate walls
        ring = room.polygon
        for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if max(dx, dy) < t:  # sub-thickness jog, not a wall
                continue
            if dy <= 0.09 * dx:  # horizontal edge
                edges.append((idx, "h", (a[1] + b[1]) / 2, min(a[0], b[0]), max(a[0], b[0])))
            elif dx <= 0.09 * dy:  # vertical edge
                edges.append((idx, "v", (a[0] + b[0]) / 2, min(a[1], b[1]), max(a[1], b[1])))
            elif float(np.hypot(dx, dy)) >= 2.5 * t:
                # Short diagonals are contour-rounding residue around door
                # notches, not architecture — only walls longer than a few
                # thicknesses can be genuine chamfers.
                diagonals.append(
                    WallSegment(
                        start=(float(a[0]), float(a[1])),
                        end=(float(b[0]), float(b[1])),
                        thickness_px=t,
                        rooms=(idx, EXTERIOR),
                    )
                )

    segments: list[WallSegment] = list(diagonals)
    for i, (room_i, axis, coord_i, lo_i, hi_i) in enumerate(edges):
        shared: list[tuple[float, float]] = []
        for j, (room_j, axis_j, coord_j, lo_j, hi_j) in enumerate(edges):
            if j == i or axis_j != axis or room_j == room_i:
                continue
            if abs(coord_j - coord_i) > 2.5 * t:
                continue
            lo, hi = max(lo_i, lo_j), min(hi_i, hi_j)
            if hi - lo < max(6.0, t):
                continue
            if j > i:  # emit each shared wall once
                center = (coord_i + coord_j) / 2
                segments.append(_segment(axis, center, lo, hi, t, (room_i, room_j)))
            shared.append((lo, hi))
        for lo, hi in _subtract_intervals((lo_i, hi_i), shared):
            if hi - lo >= max(6.0, 1.5 * t):
                segments.append(_segment(axis, coord_i, lo, hi, t, (room_i, EXTERIOR)))
    resolved = [replace(seg, rooms=_resolve_connects(seg, walls, room_masks)) for seg in segments]
    centered = [_center_exterior_segment(seg, room_masks) for seg in resolved]
    return _close_tee_junctions(_fuse_collinear_segments(centered, t), t)


def _fuse_collinear_segments(segments: list[WallSegment], t: float) -> list[WallSegment]:
    """Merge same-axis segments that are really one wall.

    Sub-thickness face jitter in the traced polygons splits a straight wall
    into two pieces offset by a few pixels; renderers show the offset as a
    crack. Pieces with the same rooms on both sides, nearly the same
    centerline (≤0.6t apart) and no real gap between them (≤1.5t) fuse into
    one segment at the length-weighted centerline. Genuine recesses are a
    full thickness deep or more and stay split."""

    def geometry(seg: WallSegment) -> tuple[str, float, float, float] | None:
        dx, dy = seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]
        if abs(dy) <= 0.09 * abs(dx):
            return ("h", (seg.start[1] + seg.end[1]) / 2,
                    min(seg.start[0], seg.end[0]), max(seg.start[0], seg.end[0]))
        if abs(dx) <= 0.09 * abs(dy):
            return ("v", (seg.start[0] + seg.end[0]) / 2,
                    min(seg.start[1], seg.end[1]), max(seg.start[1], seg.end[1]))
        return None

    out: list[WallSegment] = []
    pool = list(segments)
    while pool:
        seg = pool.pop(0)
        info = geometry(seg)
        if info is None:
            out.append(seg)
            continue
        axis, fixed, lo, hi = info
        weight = hi - lo
        merged = True
        while merged:
            merged = False
            for other in pool:
                o_info = geometry(other)
                if (
                    o_info is None
                    or o_info[0] != axis
                    or set(other.rooms) != set(seg.rooms)
                    or abs(o_info[1] - fixed) > 0.6 * t
                    or o_info[2] > hi + 1.5 * t
                    or o_info[3] < lo - 1.5 * t
                ):
                    continue
                o_weight = o_info[3] - o_info[2]
                fixed = (fixed * weight + o_info[1] * o_weight) / max(weight + o_weight, 1e-6)
                lo, hi = min(lo, o_info[2]), max(hi, o_info[3])
                weight += o_weight
                pool.remove(other)
                merged = True
                break
        out.append(_segment(axis, fixed, lo, hi, seg.thickness_px, seg.rooms))
    return out


def _close_tee_junctions(segments: list[WallSegment], t: float) -> list[WallSegment]:
    """Extend wall ends to meet the centerline of a crossing perpendicular
    wall. Segments derived from room-polygon edges stop at the polygon's
    inner corners, leaving half-thickness notches at every T and L junction;
    renderers show them as broken walls."""
    reach = 1.4 * t

    def geometry(seg: WallSegment) -> tuple[str, float, float, float] | None:
        dx, dy = seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]
        if abs(dy) <= 0.09 * abs(dx):
            return ("h", (seg.start[1] + seg.end[1]) / 2,
                    min(seg.start[0], seg.end[0]), max(seg.start[0], seg.end[0]))
        if abs(dx) <= 0.09 * abs(dy):
            return ("v", (seg.start[0] + seg.end[0]) / 2,
                    min(seg.start[1], seg.end[1]), max(seg.start[1], seg.end[1]))
        return None

    infos = [geometry(seg) for seg in segments]
    out: list[WallSegment] = []
    for seg, info in zip(segments, infos, strict=True):
        if info is None:
            out.append(seg)
            continue
        axis, fixed, lo, hi = info
        for other in infos:
            if other is None or other[0] == axis:
                continue
            o_fixed, o_lo, o_hi = other[1], other[2], other[3]
            if not (o_lo - 0.6 * t <= fixed <= o_hi + 0.6 * t):
                continue  # the perpendicular wall doesn't cross our line
            if lo - reach <= o_fixed < lo:
                lo = o_fixed
            if hi < o_fixed <= hi + reach:
                hi = o_fixed
        out.append(
            _segment(axis, fixed, lo, hi, seg.thickness_px, seg.rooms)
            if (lo, hi) != (info[2], info[3])
            else seg
        )
    return out


def _center_exterior_segment(seg: WallSegment, room_masks: list[np.ndarray | None]) -> WallSegment:
    if EXTERIOR not in seg.rooms:
        return seg
    room = next((side for side in seg.rooms if isinstance(side, int)), None)
    mask = room_masks[room] if room is not None else None
    outward = _room_outward_normal(seg, mask)
    if outward is None:
        return seg

    start = np.asarray(seg.start, dtype=np.float64)
    end = np.asarray(seg.end, dtype=np.float64)
    along = end - start
    length = float(np.linalg.norm(along))
    if length == 0:
        return seg
    along /= length
    half = seg.thickness_px / 2
    start = start + outward * half - along * half
    end = end + outward * half + along * half
    return replace(
        seg,
        start=(float(start[0]), float(start[1])),
        end=(float(end[0]), float(end[1])),
    )


def measure_bay_protrusion(
    seg: WallSegment,
    bbox: tuple[float, float, float, float],
    walls: WallExtraction,
    rooms: list[RoomDraft],
    bgr: np.ndarray,
) -> list[tuple[float, float]] | None:
    """Measure a rectangular/trapezoidal bay outline from its facade strokes.

    The host opening supplies the base. The farthest parallel structural run
    with two connecting side strokes supplies the outer edge; unrelated
    dimension lines therefore do not become bay windows.
    """
    if EXTERIOR not in seg.rooms:
        return None
    room = next((side for side in seg.rooms if isinstance(side, int)), None)
    masks = _room_masks(rooms, walls.union.shape)
    mask = masks[room] if room is not None else None
    outward = _room_outward_normal(seg, mask)
    if outward is None:
        return None

    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    horizontal = abs(sx1 - sx0) >= abs(sy1 - sy0)
    if horizontal and abs(outward[1]) < 0.9:
        return None
    if not horizontal and abs(outward[0]) < 0.9:
        return None

    x0, y0, x1, y1 = bbox
    base_lo, base_hi = (x0, x1) if horizontal else (y0, y1)
    base_coord = (sy0 + sy1) / 2 if horizontal else (sx0 + sx1) / 2
    width = base_hi - base_lo
    if width <= 0:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raw_strokes = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    structural = (walls.union > 0) | (raw_strokes > 0)
    h, w = structural.shape
    min_depth = max(4, int(round(1.25 * seg.thickness_px)))
    max_depth = int(round(min(max(6 * seg.thickness_px, 0.75 * width), 0.25 * max(h, w))))
    pad = int(round(seg.thickness_px))
    along_lo = max(0, int(round(base_lo)) - pad)
    along_hi = min((w if horizontal else h) - 1, int(round(base_hi)) + pad)
    min_run = max(6, int(round(0.35 * width)))
    close_width = max(3, int(round(seg.thickness_px / 2)))
    candidates: list[tuple[int, int, int]] = []

    for depth in range(min_depth, max_depth + 1):
        coord = int(round(base_coord + depth * (outward[1] if horizontal else outward[0])))
        if horizontal:
            if not 0 <= coord < h:
                break
            values = structural[coord, along_lo : along_hi + 1]
        else:
            if not 0 <= coord < w:
                break
            values = structural[along_lo : along_hi + 1, coord]
        closed = cv2.morphologyEx(
            values.astype(np.uint8)[None, :],
            cv2.MORPH_CLOSE,
            np.ones((1, close_width), np.uint8),
        ).ravel()
        for run_lo, run_hi, value in _runs(closed):
            if value and run_hi - run_lo + 1 >= min_run:
                candidates.append((depth, along_lo + run_lo, along_lo + run_hi))

    if not candidates:
        return None

    # One physical outer stroke occupies several adjacent raster rows. Use
    # its middle rather than its far edge to avoid another half-stroke bias.
    groups: list[list[tuple[int, int, int]]] = []
    for candidate in candidates:
        if groups and candidate[0] <= groups[-1][-1][0] + 1:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    for group in reversed(groups):
        median_depth = float(np.median([item[0] for item in group]))
        outer_lo = float(np.median([item[1] for item in group]))
        outer_hi = float(np.median([item[2] for item in group]))
        outer_coord = base_coord + median_depth * (outward[1] if horizontal else outward[0])
        if horizontal:
            base_a, base_b = (base_lo, base_coord), (base_hi, base_coord)
            outer_a, outer_b = (outer_lo, outer_coord), (outer_hi, outer_coord)
        else:
            base_a, base_b = (base_coord, base_lo), (base_coord, base_hi)
            outer_a, outer_b = (outer_coord, outer_lo), (outer_coord, outer_hi)
        side_coverage = min(
            _stroke_coverage(structural, base_a, outer_a),
            _stroke_coverage(structural, base_b, outer_b),
        )
        if side_coverage >= 0.35:
            return [base_a, base_b, outer_b, outer_a]
    return None


def _stroke_coverage(
    mask: np.ndarray, start: tuple[float, float], end: tuple[float, float]
) -> float:
    length = max(2, int(round(np.hypot(end[0] - start[0], end[1] - start[1]))))
    xs = np.linspace(start[0], end[0], length + 1)
    ys = np.linspace(start[1], end[1], length + 1)
    h, w = mask.shape
    hits = 0
    for x, y in zip(xs, ys, strict=True):
        xi, yi = int(round(x)), int(round(y))
        if mask[max(0, yi - 2) : min(h, yi + 3), max(0, xi - 2) : min(w, xi + 3)].any():
            hits += 1
    return hits / len(xs)


def _segment(
    axis: str,
    coord: float,
    lo: float,
    hi: float,
    thickness: float,
    rooms: tuple[int | str, int | str],
) -> WallSegment:
    if axis == "h":
        return WallSegment((lo, coord), (hi, coord), thickness, rooms)
    return WallSegment((coord, lo), (coord, hi), thickness, rooms)


def _subtract_intervals(
    base: tuple[float, float], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces = [base]
    for h_lo, h_hi in sorted(holes):
        next_pieces = []
        for lo, hi in pieces:
            if h_hi <= lo or h_lo >= hi:
                next_pieces.append((lo, hi))
                continue
            if h_lo > lo:
                next_pieces.append((lo, h_lo))
            if h_hi < hi:
                next_pieces.append((h_hi, hi))
        pieces = next_pieces
    return pieces
