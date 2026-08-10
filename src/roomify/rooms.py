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
# A 900mm door is 6-10% of a typical dwelling's dimension, so the sweep must
# reach past 0.10 to bridge bare (sill-less) doorways.
FALLBACK_GAP_RATIOS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
COLLINEAR_ANGLE_DEG = 8.0
COLLINEAR_MIN_EDGE_RATIO = 0.02  # of perimeter
# How much of a sub-void's boundary may sit off the drawn ink before its
# split is judged invented rather than found (see _composite). One doorway
# is ~3% of a room's perimeter; a close-fabricated divider costs 28-40%.
DECOMPOSE_INK_SLACK = 0.15


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
    # True when the room was recovered from floor space no candidate claimed
    # (decor strokes — railings, wardrobe lines — had chopped it below the
    # size filters). Real measured pixels, weaker segmentation evidence.
    recovered: bool = False


@dataclass(frozen=True)
class RoomDetection:
    rooms: list[CVRoom]
    strategy: str  # "close5" or "directional_close(<gap>px)"


def _sealed(
    union: np.ndarray, angles: tuple[float, ...], close, thickness_px: float
) -> np.ndarray:
    """Apply a closing recipe axis-aligned AND in each rotated wall frame.

    Directional closes only bridge gaps along the axes; a diamond wing's
    doorways lie along its own wall directions, so the same recipe runs in
    a frame where that family is axis-aligned and the result is unioned
    back. Nearest-neighbour warps keep the mask binary; the 1px jaggies
    they introduce are below the polygon simplification tolerance.

    Only WALL-THIN additions are accepted from a rotated pass. In a rotated
    frame the kernel bridges gaps along the family's walls (the intent) but
    ALSO cuts a chord across every 90° corner it meets, filling the triangle
    behind it — every room in the plan loses its corners, and the ones with
    interior strokes fragment (measured: 23k px of a 700px plan, room areas
    30-50% under their printed labels). A bridged wall gap is as thick as
    the wall; a corner chord is half the kernel deep.
    """
    work = close(union)
    if not angles:
        return work
    h, w = union.shape
    max_width = max(6.0, 2.5 * thickness_px)
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
        added = cv2.bitwise_and(closed_back, cv2.bitwise_not(union))
        work = cv2.bitwise_or(work, _thin_additions(added, max_width))
    return work


def _thin_additions(added: np.ndarray, max_width: float) -> np.ndarray:
    """Keep only components no thicker than ``max_width``.

    Thickness = twice the largest inscribed radius (the distance transform's
    peak, padded so the border does not truncate it) — exact for strips and
    unfooled by a long thin wedge.

    ponytail: a shallow corner chord thinner than a wall still gets through
    (small kernels on high-res inputs). It costs a sliver, not a room; gate
    on the added region's ORIENTATION against the family angle if a corpus
    ever needs more.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        component = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8)
        padded = cv2.copyMakeBorder(component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        radius = float(cv2.distanceTransform(padded, cv2.DIST_L2, 5).max())
        keep[i] = 2.0 * radius <= max_width
    return np.where(keep[labels], np.uint8(255), np.uint8(0))


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
    return _silhouette(union, footprint)[1]


def _silhouette(
    union: np.ndarray, footprint: tuple[int, int, int, int]
) -> tuple[np.ndarray, float]:
    """Filled building silhouette mask and its area."""
    x0, y0, x1, y1 = footprint
    # 0.2 of the footprint: must exceed the widest facade mouth (a double
    # entry door with its swing arcs spans ~0.15), while staying below deep
    # courtyard notches. ponytail: single fixed ratio; a per-mouth adaptive
    # close is the upgrade path if a corpus plan ever exceeds it.
    gap = max(21, int(0.20 * max(x1 - x0, y1 - y0)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    blob = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(union)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask, float(sum(cv2.contourArea(c) for c in contours))


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
    # Door leaves + swing arcs are the drawing's own strokes across
    # sill-less doorways, but on tinted-sector plans they excise the swing
    # sector from its room. So they seal a PARALLEL candidate set only, and
    # the selector below arbitrates: sill-sealed plans keep the plain
    # variant, arc-style leaky plans win with the hinted one.
    variants: list[tuple[np.ndarray, np.ndarray | None, str]] = [(union, None, "")]
    if walls.door_hints is not None:
        variants.append(
            (cv2.bitwise_or(union, walls.door_hints), walls.door_hints, "+door_hints")
        )
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    x0, y0, x1, y1 = walls.footprint
    max_dim = max(x1 - x0, y1 - y0)
    candidates: list[RoomDetection] = []
    for base, hints, tag in variants:
        work = _sealed(
            base,
            walls.angles,
            lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel5),
            walls.thickness_px,
        )
        candidates.append(
            RoomDetection(
                rooms=_holes_to_rooms(work, walls, min_area_px, hints),
                strategy=f"close5{tag}",
            )
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

            work = _sealed(base, walls.angles, directional, walls.thickness_px)
            candidates.append(
                RoomDetection(
                    rooms=_holes_to_rooms(work, walls, min_area_px, hints),
                    strategy=f"directional_close({gap}px){tag}",
                )
            )

    detection = _composite(candidates, building_area, union)
    recovered = _recover_uncovered(detection.rooms, walls, min_area_px)
    if recovered:
        rooms = sorted(detection.rooms + recovered, key=lambda r: -r.area_px)
        detection = RoomDetection(rooms=rooms, strategy=detection.strategy)
    return detection


def _recover_uncovered(
    rooms: list[CVRoom], walls: WallExtraction, min_area_px: float
) -> list[CVRoom]:
    """Claim floor space that no candidate turned into a room.

    Every patch of the building silhouette is either wall or floor, and
    every floor patch belongs to some room — decor strokes (balcony
    railings, wardrobe fronts) chop narrow rooms into slivers below the
    size filters and the space simply vanishes. Recover it: take the
    silhouette interior minus walls minus chosen rooms, absorb the decor
    strokes with a small close, and keep patches that are room-sized AND
    mostly bounded by real drawn structure (an exterior notch bridged by
    the silhouette close is bounded by the hull, not by ink, and stays
    out). Recovered rooms carry ``recovered=True`` — real measured pixels,
    weaker segmentation evidence.
    """
    silhouette, _ = _silhouette(walls.union, walls.footprint)
    interior = cv2.bitwise_and(
        silhouette, cv2.bitwise_not(cv2.dilate(walls.union, np.ones((5, 5), np.uint8)))
    )
    # exterior veto: anything reachable from the image border through the
    # (lightly sealed) mask is outside — an open courtyard's mouth carries
    # no ink, so its interior floods from the border and must not be
    # reclaimed, however wall-bounded its other three sides are
    sealed = cv2.morphologyEx(
        walls.union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    n_out, out_labels = cv2.connectedComponents(
        (sealed == 0).astype(np.uint8), connectivity=4
    )
    border = np.unique(
        np.concatenate(
            [out_labels[0, :], out_labels[-1, :], out_labels[:, 0], out_labels[:, -1]]
        )
    )
    exterior = np.isin(out_labels, border[border != 0])
    interior = cv2.bitwise_and(
        interior, cv2.bitwise_not(exterior.astype(np.uint8) * 255)
    )
    covered = np.zeros_like(silhouette)
    for room in rooms:
        cv2.fillPoly(covered, [np.round(room.polygon).astype(np.int32)], 255)
    uncovered = cv2.bitwise_and(
        interior, cv2.bitwise_not(cv2.dilate(covered, np.ones((7, 7), np.uint8)))
    )
    uncovered = cv2.morphologyEx(
        uncovered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    near_union = cv2.dilate(walls.union, np.ones((11, 11), np.uint8))
    out: list[CVRoom] = []
    contours, _ = cv2.findContours(uncovered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2 * min_area_px:  # recovery earns no benefit of the doubt
            continue
        pts = contour.reshape(-1, 2)
        on_ink = sum(1 for x, y in pts if near_union[int(y), int(x)])
        if on_ink < 0.7 * len(pts):
            continue  # bounded by the silhouette hull, not by drawn strokes
        recovered_room = _polygonize(contour, walls.thickness_px)
        if recovered_room is not None:
            out.append(replace(recovered_room, recovered=True))
    return out


def _composite(
    candidates: list[RoomDetection], building_area: float, union: np.ndarray
) -> RoomDetection:
    """Per-void composition across all sealing candidates.

    One close size cannot serve every room: a dot-hatched bathroom only
    survives the 5px close (anything bigger solidifies its texture), while
    the open-plan zones next door only split under a 23px close. So instead
    of electing one candidate, cluster the rooms of ALL candidates by IoU
    and pick each cluster on its own: clusters seen at many rungs are real
    (texture artifacts and merge accidents are rung-specific), the version
    from the smallest close carries the least invented mask, and overlap
    beats supersets — a merged mega-void loses to the split pieces that
    appear more consistently.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    clusters: list[dict] = []
    for cand in candidates:
        for room in cand.rooms:
            try:
                poly = ShapelyPolygon(room.polygon)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area <= 0:
                    continue
            except Exception:
                continue
            for cl in clusters:
                inter = poly.intersection(cl["poly"]).area
                if inter / max(poly.union(cl["poly"]).area, 1.0) >= 0.55:
                    cl["count"] += 1
                    if room.area_px > cl["room"].area_px:
                        cl["room"], cl["poly"], cl["src"] = room, poly, cand.strategy
                    break
            else:
                clusters.append(
                    {"poly": poly, "room": room, "count": 1, "src": cand.strategy}
                )

    clusters.sort(key=lambda c: (-c["count"], -c["room"].area_px))
    chosen: list[dict] = []
    for cl in clusters:
        room = cl["room"]
        # tiny AND ragged = wall pocket, not a room (real tiny rooms —
        # AC platforms, shafts — are compact boxes)
        compactness = 4.0 * np.pi * room.area_px / max(room.perimeter_px**2, 1.0)
        if room.area_px < 0.01 * building_area and compactness < 0.25:
            continue
        overlap = sum(cl["poly"].intersection(c["poly"]).area for c in chosen)
        if overlap > 0.25 * cl["poly"].area:
            continue
        chosen.append(cl)

    # Decompose merges the vote couldn't settle: when a chosen void is
    # explained by ≥2 significant unchosen sub-voids (each ≥80% inside it,
    # jointly ≥70% of its area), the split MAY be the better reading — but
    # only if those pieces trace real structure rather than a divider a big
    # close invented while bridging. Drawn ink settles it: splitting on a
    # real wall or dashed divider costs a piece only the doorway it bridges
    # (measured 0.97 vs a 1.00 parent), while an invented divider leaves a
    # third to a half of the piece's boundary floating (0.60, 0.51).
    # Without this a count-11 bedroom was shredded by count-2 artifacts.
    unchosen = [cl for cl in clusters if cl not in chosen]
    final: list[dict] = []
    for cl in chosen:
        if cl["poly"].area < 0.08 * building_area:
            # only merge-scale voids qualify — a small room's "sub-voids"
            # are texture accidents, not structure
            final.append(cl)
            continue
        pieces = [
            p
            for p in unchosen
            if p["count"] >= 2
            and p["room"].area_px >= 0.01 * building_area
            and p["poly"].intersection(cl["poly"]).area >= 0.8 * p["poly"].area
        ]
        placed: list[dict] = []
        for p in sorted(pieces, key=lambda p: -p["room"].area_px):
            if all(
                p["poly"].intersection(q["poly"]).area <= 0.25 * p["poly"].area
                for q in placed
            ):
                placed.append(p)
        if (
            len(placed) >= 2
            and sum(p["poly"].area for p in placed) >= 0.7 * cl["poly"].area
            and max(_boundary_fraction(p["room"], union) for p in placed)
            >= _boundary_fraction(cl["room"], union) - DECOMPOSE_INK_SLACK
        ):
            final.extend(placed)
        else:
            final.append(cl)
    chosen = final
    rooms = sorted((c["room"] for c in chosen), key=lambda r: -r.area_px)
    # honest reporting: when every chosen void already existed at the plain
    # 5px close, no fallback material was needed and the pipeline's
    # room_gap_fallback warning must stay silent
    strategy = (
        "close5"
        if all(c["src"].startswith("close5") for c in chosen)
        else "composite"
    )
    return RoomDetection(rooms=rooms, strategy=strategy)


def _holes_to_rooms(
    work: np.ndarray,
    walls: WallExtraction,
    min_area_px: float,
    hints: np.ndarray | None = None,
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
            # a door-scale pocket bounded largely by hint strokes is a swing
            # sector carved off its room, not a room
            if (
                hints is not None
                and room.area_px < 3500
                and _boundary_fraction(room, hints) >= 0.25
            ):
                continue
            if walls.zones is not None:
                room = replace(room, zone_bounded=_touches_zone(room, walls.zones))
            rooms.append(room)

    rooms.sort(key=lambda r: r.area_px, reverse=True)
    return rooms


def _boundary_fraction(room: CVRoom, mask: np.ndarray) -> float:
    """Share of the room's boundary that rides within 3px of ``mask``."""
    near = cv2.dilate(mask, np.ones((7, 7), np.uint8))
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
    return hits / total if total else 0.0


def _touches_zone(room: CVRoom, zones: np.ndarray) -> bool:
    """True when a meaningful share of the boundary rides a dashed divider."""
    return _boundary_fraction(room, zones) >= 0.12


def _polygonize(contour: np.ndarray, wall_thickness: float) -> CVRoom | None:
    approx = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2).astype(np.float64)
    if len(approx) < 3:
        return None

    cleaned = _remove_spikes(_merge_collinear(approx), max_len=3.5 * wall_thickness)
    squared = _merge_collinear(  # squaring can leave collinear vertices
        _square_corners(cleaned, max_cut=max(6.0, 2.9 * wall_thickness))
    )
    if len(squared) < 3:  # cleaning can degenerate a sliver contour entirely
        return None
    inner = Polygon(squared)
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
