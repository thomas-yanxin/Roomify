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

from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from roomify.walls import WallExtraction

MIN_COMPACTNESS = 0.03  # 4πA/P²; furniture outlines and slivers score lower
MIN_ASPECT = 0.1  # min-area-rect aspect; rooms are not 10:1 threads
MAX_AREA_RATIO = 0.6  # of image; larger = the exterior leaked in
MIN_COVERAGE = 0.55  # rooms-to-footprint ratio below this = rooms went missing
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


@dataclass(frozen=True)
class RoomDetection:
    rooms: list[CVRoom]
    strategy: str  # "close5" or "directional_close(<gap>px)"


def detect_rooms(walls: WallExtraction, min_area_px: float | None = None) -> RoomDetection:
    h, w = walls.union.shape
    if min_area_px is None:
        min_area_px = max(500.0, 0.001 * h * w)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    work = cv2.morphologyEx(walls.union, cv2.MORPH_CLOSE, kernel5)
    rooms = _holes_to_rooms(work, walls, min_area_px)
    best = RoomDetection(rooms=rooms, strategy="close5")
    if _looks_complete(rooms, walls):
        return best

    x0, y0, x1, y1 = walls.footprint
    max_dim = max(x1 - x0, y1 - y0)
    for ratio in FALLBACK_GAP_RATIOS:
        gap = max(6, int(ratio * max_dim))
        work = cv2.morphologyEx(
            walls.union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 3))
        )
        work = cv2.morphologyEx(
            work, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap))
        )
        rooms = _holes_to_rooms(work, walls, min_area_px)
        candidate = RoomDetection(rooms=rooms, strategy=f"directional_close({gap}px)")
        if _looks_complete(rooms, walls):
            return candidate
        if len(rooms) > len(best.rooms):
            best = candidate
    return best


def _looks_complete(rooms: list[CVRoom], walls: WallExtraction) -> bool:
    """A plausible segmentation: rooms cover most of the footprint and no
    single 'room' spans it — an oversized largest void means door gaps merged
    several real rooms (the case the fallback ladder exists to fix).
    """
    if not rooms:
        return False
    x0, y0, x1, y1 = walls.footprint
    footprint_area = max(1.0, float(x1 - x0) * float(y1 - y0))
    coverage = sum(r.area_px for r in rooms) / footprint_area
    largest_ratio = rooms[0].area_px / footprint_area  # rooms sorted desc
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

        room = _polygonize(contour)
        if room is not None:
            rooms.append(room)

    rooms.sort(key=lambda r: r.area_px, reverse=True)
    return rooms


def _polygonize(contour: np.ndarray) -> CVRoom | None:
    approx = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2).astype(np.float64)
    if len(approx) < 3:
        return None

    inner = Polygon(_merge_collinear(approx))
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
