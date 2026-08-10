"""Room extraction: enclosed voids in the wall mask become room polygons.

Rooms are holes in the contour hierarchy of the (gap-closed) wall mask.
A small 5px close is the primary sealing strategy — listing-style plans
draw thin sill/window strokes across every opening, and those strokes are
already in the union mask. Plans without such strokes leave door gaps open,
so a fallback ladder sweeps increasing directional close sizes; directional
(N,3)/(3,N) kernels bridge gaps along walls without merging parallel walls
the way a square kernel would.

Polygons follow the INNER-FACE (net floor area) convention: hole contours
trace the inner wall faces directly. Chinese listing plans print per-room
areas in exactly this net convention (verified against a reviewed ground
truth: the dimension-chain rectangle of a "9.65㎡" bedroom measures 10.71㎡ —
printed labels are net, not centerline), and the scale calibration and
deviation flags depend on comparing like with like.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from roomify.walls import WallExtraction

MIN_COMPACTNESS = 0.03  # 4πA/P²; furniture outlines and slivers score lower
MIN_ASPECT = 0.1  # min-area-rect aspect; rooms are not 10:1 threads
MAX_AREA_RATIO = 0.6  # of image; larger = the exterior leaked in
MIN_COVERAGE = 0.55  # rooms-to-silhouette ratio below this = rooms went missing
MAX_LARGEST_RATIO = 0.65  # one room above this = door gaps merged the voids
# A 900mm door is 6-10% of a typical dwelling's dimension, so the sweep must
# reach past 0.10 to bridge bare (sill-less) doorways.
FALLBACK_GAP_RATIOS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
COLLINEAR_ANGLE_DEG = 8.0
COLLINEAR_MIN_EDGE_RATIO = 0.02  # of perimeter


@dataclass(frozen=True)
class CVRoom:
    polygon: np.ndarray  # (N, 2) float64, open ring, working px, inner-face
    area_px: float
    perimeter_px: float
    edge_lengths_px: list[float]
    seed: tuple[float, float]  # a point guaranteed inside the visible room
    # True when part of this room's boundary is a dashed zone divider, not a
    # physical wall (玄关/走廊 functional splits): geometry is measured, the
    # boundary evidence is weaker.
    zone_bounded: bool = False


@dataclass(frozen=True)
class RoomDetection:
    rooms: list[CVRoom]
    strategy: str  # "close5" or "directional_close(<gap>px)"


def _sealed(union: np.ndarray, angles: tuple[float, ...], close) -> np.ndarray:
    """Apply a closing recipe axis-aligned AND in each rotated wall frame.

    Directional closes only bridge gaps along the axes; a diamond wing's
    doorways lie along its own wall directions, so the same recipe runs in
    a frame where that family is axis-aligned and the result is unioned
    back. Nearest-neighbour warps keep the mask binary; the 1px jaggies
    they introduce are below the polygon simplification tolerance.
    """
    work = close(union)
    if not angles:
        return work
    h, w = union.shape
    for angle in angles:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        rotated = cv2.warpAffine(
            union, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0
        )
        closed_back = cv2.warpAffine(
            close(rotated),
            cv2.invertAffineTransform(m),
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        work = cv2.bitwise_or(work, closed_back)
    return work


def _building_area(
    union: np.ndarray, footprint: tuple[int, int, int, int]
) -> float:
    """Estimate the building silhouette area for candidate scoring.

    Facade openings (balcony fronts, window walls, glazing on diagonal
    facades) can leave the raw wall contour open. A large close is useful for
    estimating a coverage denominator, but its contour is deliberately never
    inserted into the enclosure mask: doing so would turn an open courtyard
    or deep exterior notch into invented room geometry.
    """
    x0, y0, x1, y1 = footprint
    # 0.2 of the footprint: must exceed the widest facade mouth (a double
    # entry door with its swing arcs spans ~0.15), while staying below deep
    # courtyard notches. ponytail: single fixed ratio; a per-mouth adaptive
    # close is the upgrade path if a corpus plan ever exceeds it.
    gap = max(21, int(0.20 * max(x1 - x0, y1 - y0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    blob = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.contourArea(c) for c in contours))


def detect_rooms(walls: WallExtraction, min_area_px: float | None = None) -> RoomDetection:
    h, w = walls.union.shape
    if min_area_px is None:
        min_area_px = max(500.0, 0.001 * h * w)

    building_area = _building_area(walls.union, walls.footprint)
    union = walls.union
    if walls.zones is not None:
        # dashed zone dividers seal like walls; the rooms they bound carry
        # zone_bounded=True so downstream can flag the weaker evidence
        union = cv2.bitwise_or(union, walls.zones)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    x0, y0, x1, y1 = walls.footprint
    max_dim = max(x1 - x0, y1 - y0)
    candidates: list[RoomDetection] = []
    work = _sealed(
        union, walls.angles, lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel5)
    )
    candidates.append(
        RoomDetection(rooms=_holes_to_rooms(work, walls, min_area_px), strategy="close5")
    )
    for ratio in FALLBACK_GAP_RATIOS:
        gap = max(6, int(ratio * max_dim))

        def directional(mask: np.ndarray, gap: int = gap) -> np.ndarray:
            out = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 3))
            )
            return cv2.morphologyEx(
                out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap))
            )

        work = _sealed(union, walls.angles, directional)
        candidates.append(
            RoomDetection(
                rooms=_holes_to_rooms(work, walls, min_area_px),
                strategy=f"directional_close({gap}px)",
            )
        )

    # Among complete-looking candidates: big closes both split merged voids
    # (good) and eat floor area off small rooms (bad), so first demand
    # near-best coverage, then take the most SIGNIFICANT rooms (≥1% of the
    # silhouette — slivers can't buy a bigger close), ties to the smallest
    # close (least invented mask material).
    def coverage(det: RoomDetection) -> float:
        return sum(r.area_px for r in det.rooms) / max(1.0, building_area)

    def significant(det: RoomDetection) -> int:
        return sum(1 for r in det.rooms if r.area_px >= 0.01 * building_area)

    complete = [c for c in candidates if _looks_complete(c.rooms, building_area)]
    if complete:
        floor = max(coverage(c) for c in complete) - 0.05
        return max((c for c in complete if coverage(c) >= floor), key=significant)
    return max(candidates, key=lambda c: len(c.rooms))


def _looks_complete(rooms: list[CVRoom], building_area: float) -> bool:
    """A plausible segmentation: rooms cover most of the building silhouette
    and no single 'room' spans it — an oversized largest void means door gaps
    merged several real rooms (the case the fallback ladder exists to fix).
    """
    if not rooms:
        return False
    building_area = max(1.0, building_area)
    coverage = sum(r.area_px for r in rooms) / building_area
    largest_ratio = rooms[0].area_px / building_area  # rooms sorted desc
    return coverage >= MIN_COVERAGE and largest_ratio <= MAX_LARGEST_RATIO


def _holes_to_rooms(
    work: np.ndarray, walls: WallExtraction, min_area_px: float
) -> list[CVRoom]:
    h, w = work.shape
    contours, hierarchy = cv2.findContours(work, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    rooms: list[CVRoom] = []
    for i, contour in enumerate(contours):
        # Odd depth = a void inside wall material. Even depths are wall
        # regions themselves (depth 2 is e.g. a column standing in a room).
        depth, j = 0, i
        while hierarchy[0][j][3] != -1:
            j = hierarchy[0][j][3]
            depth += 1
        if depth % 2 == 0:
            continue

        area = cv2.contourArea(contour)
        if area < min_area_px or area > MAX_AREA_RATIO * h * w:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4 * np.pi * area / perimeter**2 < MIN_COMPACTNESS:
            continue
        _, (bw, bh), _ = cv2.minAreaRect(contour)
        if max(bw, bh) <= 0 or min(bw, bh) / max(bw, bh) < MIN_ASPECT:
            continue

        room = _polygonize(contour, walls.thickness_px)
        if room is not None:
            if walls.zones is not None:
                room = replace(room, zone_bounded=_touches_zone(room, walls.zones))
            rooms.append(room)

    rooms.sort(key=lambda r: r.area_px, reverse=True)
    return rooms


def _touches_zone(room: CVRoom, zones: np.ndarray) -> bool:
    """True when a meaningful share of the boundary rides a dashed divider."""
    near = cv2.dilate(zones, np.ones((7, 7), np.uint8))
    h, w = near.shape
    ring = np.vstack([room.polygon, room.polygon[:1]])
    hits = total = 0
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:], strict=False):
        steps = max(2, int(np.hypot(x1 - x0, y1 - y0) / 4))
        for t in np.linspace(0, 1, steps):
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            total += 1
            if 0 <= x < w and 0 <= y < h and near[y, x]:
                hits += 1
    return total > 0 and hits / total >= 0.12


def _polygonize(contour: np.ndarray, wall_thickness: float) -> CVRoom | None:
    approx = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2).astype(np.float64)
    if len(approx) < 3:
        return None

    cleaned = _remove_spikes(_merge_collinear(approx), max_len=3.5 * wall_thickness)
    squared = _square_corners(cleaned, max_cut=max(6.0, 2.9 * wall_thickness))
    # squaring can leave a vertex exactly on the line of its neighbours
    inner = Polygon(_merge_collinear(squared))
    if not inner.is_valid:
        inner = make_valid(inner)
        if isinstance(inner, MultiPolygon):
            inner = max(inner.geoms, key=lambda g: g.area)
        if not isinstance(inner, Polygon) or inner.is_empty:
            return None

    ring = np.asarray(inner.exterior.coords[:-1], dtype=np.float64)
    if len(ring) < 3:
        return None

    closed = np.vstack([ring, ring[:1]])
    edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    seed = inner.representative_point()
    return CVRoom(
        polygon=ring,
        area_px=float(inner.area),
        perimeter_px=float(edges.sum()),
        edge_lengths_px=[float(e) for e in edges],
        seed=(float(seed.x), float(seed.y)),
    )


def _remove_spikes(points: np.ndarray, max_len: float) -> np.ndarray:
    """Delete needle spurs: a short edge pair that goes out and nearly
    straight back (a thin stroke — e.g. a door leaf — poking into the room
    mask leaves a zero-width spike, which downstream reads as a wall)."""
    pts = points.copy()
    for _ in range(20):
        n = len(pts)
        if n <= 4:
            break
        changed = False
        for i in range(n):
            tip = (i + 1) % n
            e1 = pts[tip] - pts[i]
            e2 = pts[(i + 2) % n] - pts[tip]
            l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
            if l1 == 0 or l2 == 0 or max(l1, l2) > max_len:
                continue
            if float(np.dot(e1, e2)) / (l1 * l2) <= -0.95:
                pts = np.delete(pts, tip, axis=0)
                changed = True
                break
        if not changed:
            break
    return pts


def _square_corners(points: np.ndarray, max_cut: float) -> np.ndarray:
    """Replace short corner-cut runs with sharp right-angle corners.

    Rooms in this domain are rectilinear; contour tracing rounds their
    corners into one-to-three sub-wall-thickness jog edges which then leak
    downstream as spurious diagonal wall stubs. Any short run bridging a
    long horizontal edge and a long vertical edge is such a cut: all its
    vertices collapse to the corner the walls actually form. Genuine
    chamfers and bay trapezoids have longer edges and survive untouched.
    """

    def axis(p: np.ndarray, q: np.ndarray) -> str | None:
        dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
        if dy <= 0.15 * dx:
            return "h"
        if dx <= 0.15 * dy:
            return "v"
        return None

    pts = points.copy()
    for _ in range(30):
        n = len(pts)
        if n <= 4:
            break
        lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        changed = False
        for i in range(n):  # i = a long bounding edge; the run starts after it
            if lengths[i] <= max_cut:
                continue
            run: list[int] = []
            j = (i + 1) % n
            while lengths[j] <= max_cut and len(run) < 3:
                run.append(j)
                j = (j + 1) % n
            if not run or lengths[j] <= max_cut:
                continue
            first_axis, last_axis = axis(pts[i], pts[(i + 1) % n]), axis(pts[j], pts[(j + 1) % n])
            if first_axis is None or last_axis is None or first_axis == last_axis:
                continue
            span = float(np.linalg.norm(pts[j] - pts[(i + 1) % n]))
            if span > 1.6 * max_cut:
                continue
            if all(axis(pts[k], pts[(k + 1) % n]) is not None for k in run):
                continue  # pure axis-aligned steps are handled elsewhere
            # corner = x from the vertical bounding edge, y from the horizontal
            h_end, v_end = (pts[(i + 1) % n], pts[j]) if first_axis == "h" else (
                pts[j], pts[(i + 1) % n],
            )
            corner = np.array([v_end[0], h_end[1]])
            first_between = (i + 1) % n
            drop = {first_between, *run, j}
            kept = [
                corner if k == first_between else pts[k]
                for k in range(n)
                if k not in drop or k == first_between
            ]
            pts = np.asarray(kept, dtype=np.float64)
            changed = True
            break
        if not changed:
            break
    return pts


def _merge_collinear(points: np.ndarray) -> np.ndarray:
    """Drop a vertex only when it is BOTH nearly straight AND adjacent to a
    short edge. Angle alone would flatten genuine shallow bays; length alone
    would eat real short walls at sharp corners. Iterates to a fixed point.
    """
    pts = points.copy()
    for _ in range(100):
        n = len(pts)
        if n <= 4:
            break
        perimeter = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum()
        min_edge = COLLINEAR_MIN_EDGE_RATIO * perimeter
        keep = np.ones(n, dtype=bool)
        changed = False
        for i in range(n):
            if not keep[i]:
                continue
            prev_pt, cur, next_pt = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            a, b = cur - prev_pt, next_pt - cur
            len_a, len_b = np.linalg.norm(a), np.linalg.norm(b)
            if len_a == 0 or len_b == 0:
                keep[i] = False
                changed = True
                continue
            cos = np.clip(np.dot(a, b) / (len_a * len_b), -1.0, 1.0)
            deviation = np.degrees(np.arccos(cos))  # 0 = perfectly straight
            if deviation < COLLINEAR_ANGLE_DEG and min(len_a, len_b) < min_edge:
                keep[i] = False
                changed = True
        pts = pts[keep]
        if not changed:
            break
    return pts
