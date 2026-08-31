"""Fusing CV geometry with VLM semantics, and estimating the px→mm scale.

Room recall is max(CV, VLM): CV polygons get VLM names by marker id, the
VLM can veto a non-room, and rooms only the VLM saw enter as bounding boxes
marked ``source="vlm"``. Matching is strictly id-keyed — position-zipping
mislabels every room the moment the model reorders its answer.

Scale estimation combines two independent, individually flawed sources:

- Dimension chains give per-axis scales (which anisotropically resized
  exports need) but a single OCR slip corrupts a whole chain, and chains
  measure axis-to-axis — the footprint extent minus one wall thickness.
- Printed room areas give a robust magnitude — the MEDIAN of area ratios
  over many rooms shrugs off individual misreads — but only the geometric
  mean sqrt(sx·sy), never the axis split.

So: chains propose per-axis candidates, printed areas elect the candidate
pair whose product matches the median ratio. Each source alone still
degrades gracefully (with the appropriate confidence and warnings).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from roomify.rooms import CVRoom
from roomify.schema import EXTERIOR, OUTDOOR_ROOM_TYPES, ParseWarning, Scale, Unresolved
from roomify.vlm import (
    ChainRead,
    ExtraRoom,
    OpeningsRead,
    RoomEntry,
    RoomRead,
    SpatialRoomRead,
)
from roomify.walls import WallExtraction

if TYPE_CHECKING:  # openings imports merge; annotate without the cycle
    from roomify.openings import OpeningCandidate, WallSegment

MIN_ROOM_SQM = 0.5  # below this a nameless room is segmentation junk
DEVIATION_FLAG_THRESHOLD = 0.10
MIN_ROOMS_FOR_AREA_SCALE = 3
AGREE_HIGH = 0.05  # relative disagreement (on the area scale) between sources
AGREE_MEDIUM = 0.15


@dataclass
class RoomDraft:
    """A room mid-pipeline, in WORKING pixels (converted at assembly)."""

    polygon: np.ndarray
    area_px: float
    perimeter_px: float
    edge_lengths_px: list[float]
    seed: tuple[float, float]
    source: str  # "cv" | "vlm" | "cv+vlm"
    confidence: float
    name: str | None = None
    room_type: str = "unknown_space"
    printed_area_sqm: float | None = None
    marker: str | None = None  # overlay marker id, for warnings/debug
    zone_bounded: bool = False  # boundary includes a dashed zone divider
    recovered: bool = False  # reclaimed from uncovered floor space
    spatially_grounded: bool = False  # printed label position lies in this polygon
    vlm_vetoed: bool = False  # semantic veto, accepted only with physical size support


@dataclass
class ScaleDraft:
    """Scale in WORKING px per mm; converted alongside geometry at assembly."""

    px_per_mm_x: float
    px_per_mm_y: float
    method: str
    confidence: str
    px_per_mm_from_areas: float | None
    n_rooms_used: int
    n_chain_values_used: int

    def to_schema(self, to_original: float) -> Scale:
        # working px × to_original = original px, so px-per-mm grows by the
        # same factor when expressed in original-image pixels.
        return Scale(
            px_per_mm_x=self.px_per_mm_x * to_original,
            px_per_mm_y=self.px_per_mm_y * to_original,
            method=self.method,  # type: ignore[arg-type]
            confidence=self.confidence,  # type: ignore[arg-type]
            px_per_mm_from_areas=(
                self.px_per_mm_from_areas * to_original if self.px_per_mm_from_areas else None
            ),
            anisotropy=self.px_per_mm_y / self.px_per_mm_x - 1,
            n_rooms_used=self.n_rooms_used,
            n_chain_values_used=self.n_chain_values_used,
        )


def pixels_per_mm_in_direction(dx: float, dy: float, scale: ScaleDraft) -> float:
    """Directional scale for anisotropically resized drawings."""
    length = math.hypot(dx, dy)
    if length == 0:
        return math.sqrt(scale.px_per_mm_x * scale.px_per_mm_y)
    mm_per_px = math.hypot(
        dx / length / scale.px_per_mm_x,
        dy / length / scale.px_per_mm_y,
    )
    return 1.0 / mm_per_px


@dataclass
class MergeOutcome:
    rooms: list[RoomDraft]
    warnings: list[ParseWarning] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)
    zone_boundaries: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=list
    )


def _with_polygon(room: RoomDraft, polygon, **changes) -> RoomDraft:
    ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    edges = np.linalg.norm(np.diff(np.vstack((ring, ring[:1])), axis=0), axis=1)
    seed = polygon.representative_point()
    return replace(
        room,
        polygon=ring,
        area_px=float(polygon.area),
        perimeter_px=float(edges.sum()),
        edge_lengths_px=[float(edge) for edge in edges],
        seed=(float(seed.x), float(seed.y)),
        **changes,
    )


def _split_claimed_extra(
    rooms: list[RoomDraft],
    bbox: tuple[float, float, float, float],
    label_point: tuple[float, float] | None = None,
    expected_area_px: float | None = None,
):
    """Split a wall-shaped alcove that the VLM found inside a larger CV room."""
    from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
    from shapely.ops import split, unary_union

    target = box(*bbox)
    anchor = Point(label_point) if label_point is not None else target.centroid
    best = None
    for room_index, room in enumerate(rooms):
        parent = Polygon(room.polygon)
        if not parent.is_valid:
            parent = parent.buffer(0)
        if parent.is_empty or (
            not parent.intersects(target)
            and parent.distance(anchor) > math.sqrt(expected_area_px or target.area)
        ):
            continue
        min_x, min_y, max_x, max_y = parent.bounds
        for x, y in parent.exterior.coords[:-1]:
            for end in (
                (min_x - 1, y),
                (max_x + 1, y),
                (x, min_y - 1),
                (x, max_y + 1),
            ):
                pieces = [
                    piece
                    for piece in split(parent, LineString(((x, y), end))).geoms
                    if isinstance(piece, Polygon) and piece.area > 1
                ]
                if len(pieces) < 2:
                    continue
                for piece in pieces:
                    intersection = piece.intersection(target).area
                    overlap = intersection / max(min(piece.area, target.area), 1.0)
                    area_fit = min(piece.area, target.area) / max(piece.area, target.area)
                    rest = unary_union(
                        [candidate for candidate in pieces if candidate != piece]
                    )
                    if not isinstance(rest, Polygon):
                        continue
                    if expected_area_px:
                        area_error = abs(math.log(piece.area / expected_area_px))
                        distance = piece.distance(anchor) / math.sqrt(expected_area_px)
                        if (
                            area_error > math.log(2)
                            or distance > 0.75
                            or rest.area < 0.5 * expected_area_px
                        ):
                            continue
                        score = (-(area_error + distance), -piece.area)
                    else:
                        if (
                            overlap < 0.25
                            or area_fit < 0.5
                            or rest.area < 0.5 * target.area
                        ):
                            continue
                        score = (overlap * area_fit, -piece.area)
                    shared = piece.boundary.intersection(rest.boundary)
                    lines = (
                        [shared]
                        if isinstance(shared, LineString)
                        else list(shared.geoms)
                        if isinstance(shared, MultiLineString)
                        else []
                    )
                    boundary = max(lines, key=lambda line: line.length, default=None)
                    if boundary is None or boundary.length <= 1:
                        continue
                    if best is None or score > best[0]:
                        best = (score, room_index, rest, piece, boundary)
    return None if best is None else best[1:]


def ground_room_inventory(
    cv_rooms: list[CVRoom],
    marker_read: RoomRead | None,
    spatial_read: SpatialRoomRead | None,
    image_shape: tuple[int, int],
) -> RoomRead | None:
    """Attach inventory labels by their printed position, never list order."""
    if spatial_read is None or not spatial_read.rooms:
        return marker_read

    from shapely.geometry import Point, Polygon

    marker_read = marker_read or RoomRead()
    h, w = image_shape
    polygons = [Polygon(room.polygon) for room in cv_rooms]

    def same_area(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return True
        return math.isclose(left, right, rel_tol=0.02, abs_tol=0.05)

    groups: dict[int, list[int]] = {}
    unmatched: set[int] = set(range(len(spatial_read.rooms)))
    for item_index, item in enumerate(spatial_read.rooms):
        y0, x0, y1, x1 = item.label_box_2d
        point = Point((x0 + x1) * w / 2000.0, (y0 + y1) * h / 2000.0)
        hits = [
            room_index
            for room_index, polygon in enumerate(polygons)
            if not polygon.is_empty and polygon.buffer(1).covers(point)
        ]
        if hits:
            groups.setdefault(min(hits, key=lambda i: polygons[i].area), []).append(item_index)

    assigned: dict[int, int] = {}
    for room_index, item_indices in groups.items():
        marker = marker_read.rooms.get(str(room_index + 1))
        corroborated = [
            item_index
            for item_index in item_indices
            if marker is not None and marker.name == spatial_read.rooms[item_index].name
        ]
        candidates = corroborated or item_indices
        chosen = max(
            candidates,
            key=lambda i: (
                spatial_read.rooms[i].printed_area_sqm or 0.0,
                (spatial_read.rooms[i].room_box_2d[2] - spatial_read.rooms[i].room_box_2d[0])
                * (
                    spatial_read.rooms[i].room_box_2d[3]
                    - spatial_read.rooms[i].room_box_2d[1]
                ),
            ),
        )
        assigned[room_index] = chosen
        unmatched.discard(chosen)

    ratios = [
        cv_rooms[room_index].area_px / item.printed_area_sqm
        for room_index, item_index in assigned.items()
        if (item := spatial_read.rooms[item_index]).printed_area_sqm
    ]
    median_ratio = statistics.median(ratios) if len(ratios) >= 3 else None
    if median_ratio:
        for item_index in sorted(unmatched):
            item = spatial_read.rooms[item_index]
            if not item.printed_area_sqm:
                continue
            expected = item.printed_area_sqm * median_ratio
            available = [index for index in range(len(cv_rooms)) if index not in assigned]
            if not available:
                break
            best = min(
                available,
                key=lambda index: abs(math.log(cv_rooms[index].area_px / expected)),
            )
            if abs(math.log(cv_rooms[best].area_px / expected)) <= math.log(RECONCILE_RATIO):
                assigned[best] = item_index
                unmatched.discard(item_index)

    entries: dict[str, RoomEntry] = {}
    for room_index, item_index in assigned.items():
        item = spatial_read.rooms[item_index]
        marker = marker_read.rooms.get(str(room_index + 1))
        printed = item.printed_area_sqm
        if marker is not None and marker.name == item.name and marker.printed_area_sqm:
            if printed is None or median_ratio is None:
                printed = marker.printed_area_sqm
            else:
                printed = min(
                    (printed, marker.printed_area_sqm),
                    key=lambda area: abs(
                        math.log(cv_rooms[room_index].area_px / area / median_ratio)
                    ),
                )
        room_type = item.room_type
        if room_type == "unknown_space" and marker is not None and marker.name == item.name:
            room_type = marker.room_type
        entries[str(room_index + 1)] = RoomEntry(
            name=item.name,
            room_type=room_type,
            printed_area_sqm=printed,
            confidence=item.confidence,
            not_a_room=item.not_a_room,
            spatially_grounded=True,
        )

    spatial_pairs = {
        (item.name, round(item.printed_area_sqm, 2))
        for item in spatial_read.rooms
        if item.name and item.printed_area_sqm
    }
    assigned_pairs = {
        (item.name, round(item.printed_area_sqm, 2))
        for item_index in assigned.values()
        if (item := spatial_read.rooms[item_index]).name and item.printed_area_sqm
    }
    for marker_id, marker_entry in marker_read.rooms.items():
        if marker_id in entries or marker_entry.not_a_room:
            continue
        if not marker_entry.name and marker_entry.room_type == "unknown_space":
            continue
        if marker_entry.printed_area_sqm and (
            marker_entry.name,
            round(marker_entry.printed_area_sqm, 2),
        ) in assigned_pairs:
            continue
        entries[marker_id] = marker_entry

    extras: list[ExtraRoom] = []
    consumed_marker_extras: set[int] = set()
    for item_index in sorted(unmatched):
        item = spatial_read.rooms[item_index]
        if item.not_a_room:
            continue
        corroborated_extras = [
            (index, extra)
            for index, extra in enumerate(marker_read.extra_rooms)
            if same_area(extra.printed_area_sqm, item.printed_area_sqm)
            and (
                extra.room_type == item.room_type
                or "unknown_space" in (extra.room_type, item.room_type)
            )
        ]
        marker_extra = (
            corroborated_extras[0][1] if len(corroborated_extras) == 1 else None
        )
        if marker_extra is not None:
            consumed_marker_extras.add(corroborated_extras[0][0])
        extras.append(
            ExtraRoom(
                name=marker_extra.name if marker_extra and marker_extra.name else item.name,
                room_type=(
                    marker_extra.room_type
                    if marker_extra and marker_extra.room_type != "unknown_space"
                    else item.room_type
                ),
                printed_area_sqm=item.printed_area_sqm or (
                    marker_extra.printed_area_sqm if marker_extra else None
                ),
                confidence=max(item.confidence, marker_extra.confidence if marker_extra else 0),
                box_2d=item.room_box_2d,
                label_box_2d=item.label_box_2d,
                expected_area_px=(
                    item.printed_area_sqm * median_ratio
                    if item.printed_area_sqm and median_ratio
                    else None
                ),
                spatially_grounded=True,
            )
        )
    spatial_type_areas = {
        (item.room_type, round(item.printed_area_sqm, 2))
        for item in spatial_read.rooms
        if item.printed_area_sqm
    }
    for index, extra in enumerate(marker_read.extra_rooms):
        pair = (
            (extra.name, round(extra.printed_area_sqm, 2))
            if extra.name and extra.printed_area_sqm
            else None
        )
        type_area = (
            (extra.room_type, round(extra.printed_area_sqm, 2))
            if extra.printed_area_sqm
            else None
        )
        if (
            index not in consumed_marker_extras
            and pair not in spatial_pairs
            and type_area not in spatial_type_areas
        ):
            extras.append(extra)
    return RoomRead(rooms=entries, extra_rooms=extras)


def merge_rooms(
    cv_rooms: list[CVRoom],
    read: RoomRead | None,
    image_shape: tuple[int, int],
    free_floor: np.ndarray | None = None,
) -> MergeOutcome:
    """CV polygons + VLM semantics table → room drafts.

    Marker ids are 1-based strings matching render_room_overlay's numbering.

    ``free_floor`` is the floor inside the building that no CV polygon
    claims (``rooms.uncovered_floor``). A normal extra room must place its
    center on that floor. A three-sided labelled alcove may already be part
    of a larger CV polygon; in that case an orthogonal reflex-vertex cut
    separates it without overlapping or inventing a wall.
    """
    out = MergeOutcome(rooms=[])
    entries = read.rooms if read is not None else {}

    for i, cv_room in enumerate(cv_rooms):
        marker = str(i + 1)
        entry = entries.get(marker)
        draft = RoomDraft(
            polygon=cv_room.polygon,
            area_px=cv_room.area_px,
            perimeter_px=cv_room.perimeter_px,
            edge_lengths_px=cv_room.edge_lengths_px,
            seed=cv_room.seed,
            source="cv",
            confidence=0.3,
            marker=marker,
            zone_bounded=cv_room.zone_bounded,
            recovered=cv_room.recovered,
        )
        if entry is None:
            if read is not None:
                out.warnings.append(
                    ParseWarning(
                        code="room_semantics_missing",
                        message=f"VLM returned no entry for marker {marker}; kept CV geometry",
                        ref=marker,
                    )
                )
            out.unresolved.append(
                Unresolved(path=f"rooms/{marker}/name", reason="no VLM reading for this room")
            )
            out.rooms.append(draft)
            continue
        if entry.not_a_room:
            out.warnings.append(
                ParseWarning(
                    code="room_vetoed",
                    message=(
                        f"VLM judged marker {marker} not to be a room; semantics ignored "
                        "and CV geometry kept for physical validation"
                    ),
                    ref=marker,
                )
            )
            out.unresolved.append(
                Unresolved(
                    path=f"rooms/{marker}/name",
                    reason="VLM room veto needs physical validation",
                )
            )
            out.rooms.append(replace(draft, vlm_vetoed=True))
            continue
        out.rooms.append(
            replace(
                draft,
                source="cv+vlm",
                confidence=entry.confidence,
                name=entry.name,
                room_type=entry.room_type,
                printed_area_sqm=entry.printed_area_sqm,
                spatially_grounded=entry.spatially_grounded,
            )
        )

    h, w = image_shape
    for j, extra in enumerate(read.extra_rooms if read is not None else []):
        marker = f"vlm_{j + 1}"
        if extra.not_a_room:
            out.warnings.append(
                ParseWarning(
                    code="room_vetoed",
                    message="VLM judged an unclaimed labelled region not to be a room",
                    ref=marker,
                )
            )
            continue
        y0, x0, y1, x1 = (v / 1000.0 for v in extra.box_2d)
        polygon = np.array(
            [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
            dtype=np.float64,
        )
        width, height = (x1 - x0) * w, (y1 - y0) * h
        if free_floor is not None:
            free_box = free_floor[
                max(0, int(y0 * h)) : int(y1 * h) + 1,
                max(0, int(x0 * w)) : int(x1 * w) + 1,
            ]
            box_height, box_width = free_box.shape
            middle = (
                free_box[
                    box_height // 3 : max(box_height // 3 + 1, 2 * box_height // 3),
                    box_width // 3 : max(box_width // 3 + 1, 2 * box_width // 3),
                ]
                if free_box.size
                else free_box
            )
            if middle.size == 0 or not middle.any():
                label_point = None
                if extra.label_box_2d is not None:
                    ly0, lx0, ly1, lx1 = extra.label_box_2d
                    label_point = ((lx0 + lx1) * w / 2000, (ly0 + ly1) * h / 2000)
                claimed = (
                    _split_claimed_extra(
                        out.rooms,
                        (x0 * w, y0 * h, x1 * w, y1 * h),
                        label_point,
                        extra.expected_area_px,
                    )
                    if extra.expected_area_px is not None
                    else None
                )
                if claimed is not None:
                    parent_index, parent, extra_polygon, boundary = claimed
                    parent_room = out.rooms[parent_index]
                    out.rooms[parent_index] = _with_polygon(
                        parent_room, parent, zone_bounded=True
                    )
                    extra_room = _with_polygon(
                        out.rooms[parent_index],
                        extra_polygon,
                        source="cv+vlm",
                        confidence=min(extra.confidence, 0.6),
                        name=extra.name,
                        room_type=extra.room_type,
                        printed_area_sqm=extra.printed_area_sqm,
                        marker=marker,
                        zone_bounded=True,
                        recovered=False,
                        spatially_grounded=extra.spatially_grounded,
                    )
                    out.rooms.append(extra_room)
                    coords = list(boundary.coords)
                    out.zone_boundaries.append((tuple(coords[0]), tuple(coords[-1])))
                    out.warnings.append(
                        ParseWarning(
                            code="room_split_from_open_zone",
                            message=(
                                f"room {extra.name or marker} was separated from a larger "
                                "CV region at its wall-defined open edge"
                            ),
                            ref=marker,
                        )
                    )
                    continue
                from shapely.geometry import Point, Polygon

                box_area = width * height
                claimed_box = Polygon(polygon)
                new_labelled_room = bool(
                    extra.spatially_grounded
                    and extra.name
                    and extra.printed_area_sqm
                    and extra.expected_area_px
                    and label_point
                    and 0.5 <= box_area / extra.expected_area_px <= 2.0
                    and not any(
                        Polygon(room.polygon).covers(Point(label_point))
                        for room in out.rooms
                    )
                    and all(
                        Polygon(room.polygon).intersection(claimed_box).area
                        <= 0.25 * box_area
                        for room in out.rooms
                    )
                )
                if not new_labelled_room:
                    out.warnings.append(
                        ParseWarning(
                            code="room_already_measured",
                            message=(
                                f"VLM reported {extra.name or 'an extra room'} where no "
                                "unclaimed floor remains; it is a zone of an "
                                "already-measured room, not a room of its own"
                            ),
                            ref=marker,
                        )
                    )
                    continue
        if not (
            extra.spatially_grounded
            and extra.name
            and extra.printed_area_sqm
            and extra.expected_area_px
        ):
            out.warnings.append(
                ParseWarning(
                    code="room_without_physical_evidence",
                    message=(
                        f"VLM suggested {extra.name or 'an extra room'}, but no printed "
                        "label with area anchors missing floor geometry; it is not emitted"
                    ),
                    ref=marker,
                )
            )
            continue
        out.rooms.append(
            RoomDraft(
                polygon=polygon,
                area_px=float(width * height),
                perimeter_px=float(2 * (width + height)),
                edge_lengths_px=[float(width), float(height), float(width), float(height)],
                seed=(float((x0 + x1) / 2 * w), float((y0 + y1) / 2 * h)),
                source="vlm",
                confidence=min(extra.confidence, 0.5),
                name=extra.name,
                room_type=extra.room_type,
                printed_area_sqm=extra.printed_area_sqm,
                marker=marker,
                spatially_grounded=extra.spatially_grounded,
            )
        )
        out.warnings.append(
            ParseWarning(
                code="room_geometry_is_bbox",
                message=(
                    f"room {extra.name or marker} was found only by the VLM; its geometry is "
                    "an approximate bounding box"
                ),
                ref=marker,
            )
        )
    return out


RECONCILE_RATIO = 1.4  # a pairing is a misfit beyond ±40% of expectation
RECONCILE_FIT = 1.25  # a repair must land within ±25% to be adopted
MERGE_SUM_TOL = 0.20  # fragment sums must match the label within ±20%


def reconcile_rooms(
    rooms: list[RoomDraft],
    scale: ScaleDraft | None,
    wall_thickness_px: float,
) -> tuple[list[RoomDraft], list[ParseWarning]]:
    """Repair label↔polygon pairing using the printed areas as ink truth.

    The VLM reads printed labels reliably but attaches them to overlay
    markers noisily once fragments multiply. Measured pixel areas are exact,
    so a pairing whose measured/printed ratio strays far from the plan-wide
    median is a wrong ATTACHMENT, not a wrong measurement. Three repairs,
    all confined to the misfit subset (well-fitting rooms are never touched)
    and all logged:

    1. reassign labels among misfit rooms (labels move with their printed
       value — name and area are one piece of ink);
    2. merge ADJACENT misfit/unnamed fragments whose joint area matches an
       otherwise-unplaceable label (a room chopped by a texture accident);
    3. restore a label in place when no better geometry fits it.
    """
    import itertools

    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    warnings: list[ParseWarning] = []
    if scale is None or len(rooms) < 2:
        return rooms, warnings
    px_per_sqm = scale.px_per_mm_x * scale.px_per_mm_y * 1e6

    def misfit_of(area_px: float, printed: float) -> float:
        return abs(math.log(area_px / max(printed * px_per_sqm, 1e-9)))

    log_bad = math.log(RECONCILE_RATIO)
    log_fit = math.log(RECONCILE_FIT)

    # cv-measured rooms only: bbox geometry cannot anchor a repair
    misfits = [
        i
        for i, r in enumerate(rooms)
        if r.source != "vlm"
        and r.printed_area_sqm
        and not r.spatially_grounded
        and misfit_of(r.area_px, r.printed_area_sqm) > log_bad
    ]
    unnamed = [
        i
        for i, r in enumerate(rooms)
        if r.source != "vlm" and not r.printed_area_sqm and not r.name
    ]
    if not misfits:
        return rooms, warnings

    labels: list[tuple[str | None, str, float, int]] = []
    for i in misfits:
        printed_area = rooms[i].printed_area_sqm
        assert printed_area is not None
        labels.append((rooms[i].name, rooms[i].room_type, printed_area, i))
    slots = sorted(set(misfits) | set(unnamed))

    polys: dict[int, ShapelyPolygon] = {}
    for i in slots:
        try:
            p = ShapelyPolygon(rooms[i].polygon)
            polys[i] = p if p.is_valid else p.buffer(0)
        except Exception:
            polys[i] = ShapelyPolygon()

    out = list(rooms)
    for i in misfits:  # detach the misfit labels; slots start clean
        out[i] = replace(
            out[i], name=None, printed_area_sqm=None, room_type="unknown_space", confidence=0.3
        )

    gap = max(8.0, 1.5 * wall_thickness_px)
    used: set[int] = set()
    for name, room_type, printed, origin in labels:
        expected = printed * px_per_sqm
        free = [i for i in slots if i not in used]
        # single-room fit first
        best_i, best_m = None, log_fit
        for i in free:
            m = misfit_of(out[i].area_px, printed)
            if m < best_m:
                best_i, best_m = i, m
        # fragment merge: pairs/triples of mutually adjacent free rooms
        best_group = None
        best_group_m = math.inf
        if best_i is None:
            for size in (2, 3):
                for combo in itertools.combinations(free, size):
                    total = sum(out[i].area_px for i in combo)
                    group_m = abs(math.log(max(total, 1e-9) / expected))
                    if group_m > math.log(1 + MERGE_SUM_TOL):
                        continue
                    if (
                        all(
                            polys[a].distance(polys[b]) <= gap
                            for a, b in itertools.combinations(combo, 2)
                        )
                        and group_m < best_group_m
                    ):
                        best_group = combo
                        best_group_m = group_m
        if best_i is not None:
            out[best_i] = replace(
                out[best_i],
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=0.6,
            )
            used.add(best_i)
            warnings.append(
                ParseWarning(
                    code="label_reassigned",
                    message=f"printed label {name or '?'} ({printed}㎡) re-attached "
                    "to the room whose measured area matches it",
                    ref=out[best_i].marker,
                )
            )
        elif best_group is not None:
            merged_poly = (
                unary_union([polys[i] for i in best_group]).buffer(gap / 2).buffer(-gap / 2)
            )
            if merged_poly.geom_type == "MultiPolygon":
                merged_poly = max(merged_poly.geoms, key=lambda g: g.area)
            ring = np.asarray(merged_poly.exterior.coords[:-1], dtype=np.float64)
            closed = np.vstack([ring, ring[:1]])
            edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
            seed = merged_poly.representative_point()
            keeper = min(best_group)
            out[keeper] = replace(
                out[keeper],
                polygon=ring,
                area_px=float(merged_poly.area),
                perimeter_px=float(edges.sum()),
                edge_lengths_px=[float(e) for e in edges],
                seed=(float(seed.x), float(seed.y)),
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=0.5,
                zone_bounded=any(out[i].zone_bounded for i in best_group),
            )
            used.update(best_group)
            for i in best_group:
                if i != keeper:
                    out[i] = None  # type: ignore[call-overload]
            warnings.append(
                ParseWarning(
                    code="rooms_merged_for_label",
                    message=(
                        f"{len(best_group)} adjacent fragments merged: their joint "
                        f"area matches the printed label {name or '?'} ({printed}㎡)"
                    ),
                    ref=out[keeper].marker,
                )
            )
        elif origin not in used and out[origin] is not None:
            # No better home anywhere: the label was PRINTED inside this
            # room, and that spatial certainty outranks the area mismatch
            # (0.01㎡ duct labels, glazing-eaten balconies). Restore the
            # original pairing untouched — the deviation flag still tells
            # the truth about the mismatch.
            out[origin] = replace(
                out[origin],
                name=name,
                room_type=room_type,
                printed_area_sqm=printed,
                source="cv+vlm",
                confidence=rooms[origin].confidence,
            )
            used.add(origin)
        else:
            warnings.append(
                ParseWarning(
                    code="label_unplaced",
                    message=(
                        f"printed label {name or '?'} ({printed}㎡) matches no "
                        "measured room or adjacent fragment group; left unassigned"
                    ),
                )
            )

    return [r for r in out if r is not None], warnings


def estimate_scale(
    chains: list[ChainRead],
    walls: WallExtraction,
    rooms: list[RoomDraft],
) -> tuple[ScaleDraft | None, list[ParseWarning]]:
    warnings: list[ParseWarning] = []
    x0, y0, x1, y1 = walls.footprint
    # Chains measure between wall AXES: the outer-face footprint extent is
    # one wall thickness (half at each end) larger than the chain span.
    extent_x = (x1 - x0) - walls.thickness_px
    extent_y = (y1 - y0) - walls.thickness_px

    x_candidates = _axis_candidates(chains, ("top", "bottom"), extent_x)
    y_candidates = _axis_candidates(chains, ("left", "right"), extent_y)

    # Median of per-room area ratios: px² per mm². VLM bbox geometry is never
    # calibration evidence; solid-wall CV rooms are preferred over dashed
    # functional zones when enough of them exist.
    primary_ratios = [
        r.area_px / (r.printed_area_sqm * 1e6)
        for r in rooms
        if r.printed_area_sqm and r.source != "vlm" and not r.zone_bounded
    ]
    # Some listing exports draw every room split as a dashed functional
    # divider. Excluding all of them throws away the only independent scale
    # evidence, even when the labelled polygons agree closely.
    ratios = (
        primary_ratios
        if len(primary_ratios) >= MIN_ROOMS_FOR_AREA_SCALE
        else [
            r.area_px / (r.printed_area_sqm * 1e6)
            for r in rooms
            if r.printed_area_sqm and r.source != "vlm"
        ]
    )
    area_scale_sq = statistics.median(ratios) if len(ratios) >= MIN_ROOMS_FOR_AREA_SCALE else None
    from_areas = area_scale_sq**0.5 if area_scale_sq else None

    if x_candidates and y_candidates and area_scale_sq:
        best = min(
            ((sx, nx, sy, ny) for sx, nx in x_candidates for sy, ny in y_candidates),
            key=lambda c: abs(c[0] * c[2] - area_scale_sq),
        )
        sx, nx, sy, ny = best
        disagreement = abs(sx * sy / area_scale_sq - 1)
        # A degenerate pairing can nail the area PRODUCT with an absurd
        # per-axis split (seen: 11:1 when a lone chain matched the wrong
        # extent on a derotated plan); real listing exports stay ≤ ~1.15.
        anisotropy = max(sx, sy) / max(min(sx, sy), 1e-12)
        if disagreement > AGREE_MEDIUM or anisotropy > 1.35:
            # No credible chain pair — typical of diagonal units, where
            # dimension chains only span the rectilinear wing while the
            # footprint bbox covers the whole plan. The area median is
            # self-consistent with the polygons being measured; a wrong
            # per-axis split is worse than an isotropic assumption.
            warnings.append(
                ParseWarning(
                    code="scale_disagreement",
                    message=(
                        f"dimension chains disagree with printed areas "
                        f"(area {disagreement:.0%}, axis ratio "
                        f"{anisotropy:.2f}); using the printed-area scale "
                        "(equal axes assumed)"
                    ),
                )
            )
            assert from_areas is not None
            return (
                ScaleDraft(
                    px_per_mm_x=from_areas,
                    px_per_mm_y=from_areas,
                    method="printed_areas",
                    confidence="medium" if len(ratios) >= 5 else "low",
                    px_per_mm_from_areas=from_areas,
                    n_rooms_used=len(ratios),
                    n_chain_values_used=0,
                ),
                warnings,
            )
        confidence = "high" if disagreement <= AGREE_HIGH else "medium"
        # Chains determine anisotropy; printed areas determine magnitude.
        normalise = math.sqrt(area_scale_sq / (sx * sy))
        sx, sy = sx * normalise, sy * normalise
        return (
            ScaleDraft(
                px_per_mm_x=sx,
                px_per_mm_y=sy,
                method="dimension_chains+printed_areas",
                confidence=confidence,
                px_per_mm_from_areas=from_areas,
                n_rooms_used=len(ratios),
                n_chain_values_used=nx + ny,
            ),
            warnings,
        )

    if x_candidates and y_candidates:
        # No printed areas to elect a pair: the chain with the largest mm sum
        # (the smallest px/mm candidate) is likeliest to span the full extent.
        sx, nx = min(x_candidates, key=lambda c: c[0])
        sy, ny = min(y_candidates, key=lambda c: c[0])
        return (
            ScaleDraft(
                px_per_mm_x=sx,
                px_per_mm_y=sy,
                method="dimension_chains",
                confidence="medium",
                px_per_mm_from_areas=None,
                n_rooms_used=0,
                n_chain_values_used=nx + ny,
            ),
            warnings,
        )

    if area_scale_sq and (x_candidates or y_candidates):
        # One measured axis plus the area-scale product determines the
        # missing axis; do not throw away anisotropy evidence.
        if x_candidates:
            sx, nx = min(x_candidates, key=lambda c: c[0])
            sy, ny = area_scale_sq / sx, 0
        else:
            sy, ny = min(y_candidates, key=lambda c: c[0])
            sx, nx = area_scale_sq / sy, 0
        # Same degenerate-pairing guard as above: a lone chain matched to
        # the wrong extent forces the derived axis absurdly far away (seen:
        # 11:1 on a derotated plan). Real exports stay ≤ ~1.15.
        if max(sx, sy) / max(min(sx, sy), 1e-12) <= 1.35:
            return (
                ScaleDraft(
                    px_per_mm_x=sx,
                    px_per_mm_y=sy,
                    method="dimension_chains+printed_areas",
                    confidence="medium" if len(ratios) >= 5 else "low",
                    px_per_mm_from_areas=from_areas,
                    n_rooms_used=len(ratios),
                    n_chain_values_used=nx + ny,
                ),
                warnings,
            )
        warnings.append(
            ParseWarning(
                code="scale_disagreement",
                message=(
                    "single dimension chain contradicts the printed-area scale "
                    f"(axis ratio {max(sx, sy) / max(min(sx, sy), 1e-12):.1f}); "
                    "using the printed-area scale (equal axes assumed)"
                ),
            )
        )

    if from_areas:
        warnings.append(
            ParseWarning(
                code="isotropy_assumed",
                message=(
                    "scale derived from printed areas only; equal horizontal/vertical "
                    "scale assumed, per-axis lengths may be off on anisotropically "
                    "resized images"
                ),
            )
        )
        return (
            ScaleDraft(
                px_per_mm_x=from_areas,
                px_per_mm_y=from_areas,
                method="printed_areas",
                confidence="medium" if len(ratios) >= 5 else "low",
                px_per_mm_from_areas=from_areas,
                n_rooms_used=len(ratios),
                n_chain_values_used=0,
            ),
            warnings,
        )

    warnings.append(
        ParseWarning(
            code="no_scale",
            message="no readable dimension chains or printed areas; output is pixel-only",
        )
    )
    return None, warnings


def _axis_candidates(
    chains: list[ChainRead], sides: tuple[str, str], extent_px: float
) -> list[tuple[float, int]]:
    """(px_per_mm, n_values) per usable chain on this axis."""
    if extent_px <= 0:
        return []
    candidates = []
    for chain in chains:
        if chain.side in sides and chain.values_mm:
            total_mm = sum(chain.values_mm)
            if total_mm > 0:
                candidates.append((extent_px / total_mm, len(chain.values_mm)))
    return candidates


@dataclass
class OpeningDraft:
    """An opening mid-pipeline, in WORKING pixels."""

    marker: str
    element_type: str
    raw_text: str | None
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    axis: str
    width_px: float
    wall_index: int
    connects: tuple[int | str, int | str]
    swing: str | None
    hinge: tuple[float, float] | None
    source: str
    confidence: float


@dataclass
class ElementDraft:
    """A VLM-only element with box_2d geometry, in WORKING pixels."""

    element_type: str
    raw_text: str | None
    bbox: tuple[float, float, float, float]
    source: str
    confidence: float


# Types whose leaves swing on a hinge — the only ones arc evidence applies to.
_SWINGING = {"single_door", "double_door", "folding_door"}
_WINDOWS = {
    "window",
    "casement_window",
    "sliding_window",
    "fixed_window",
    "bay_window",
    "floor_to_ceiling_window",
    "blind_window",
}
_OPEN_PLAN_ROOMS = {
    "living_room",
    "living_dining",
    "dining_room",
    "kitchen",
    "hallway",
    "entrance",
}
# Legend elements that legitimately stand free of walls; anything else the
# VLM reports as an "extra" is an opening claim without usable geometry.
_FREE_STANDING = {
    "stair",
    "railing",
    "elevator",
    "escalator",
    "equipment_platform",
    "column",
    "chimney",
    "unknown_symbol",
}


def _both_sides_are_rooms(connects: tuple[int | str, int | str], rooms: list[RoomDraft]) -> bool:
    return all(isinstance(side, int) and 0 <= side < len(rooms) for side in connects)


def _both_sides_indoor(connects: tuple[int | str, int | str], rooms: list[RoomDraft]) -> bool:
    return _both_sides_are_rooms(connects, rooms) and all(
        rooms[side].room_type not in OUTDOOR_ROOM_TYPES
        for side in connects
        if isinstance(side, int)
    )


def _window_conflicts_with_privacy_room(
    connects: tuple[int | str, int | str], rooms: list[RoomDraft]
) -> bool:
    return _both_sides_indoor(connects, rooms) and any(
        rooms[side].room_type in {"bedroom", "bathroom"}
        for side in connects
        if isinstance(side, int)
    )


def _both_sides_open_plan(connects: tuple[int | str, int | str], rooms: list[RoomDraft]) -> bool:
    return _both_sides_indoor(connects, rooms) and all(
        rooms[side].room_type in _OPEN_PLAN_ROOMS for side in connects if isinstance(side, int)
    )


# Beyond this an interior break is not credible as one opening: the corpus's
# genuine room-to-room mouths top out around 2m, while every wider one sits
# on a wall the ``solid`` mask lost. Openings are scanned against ``solid``,
# which by construction drops partitions thinner than ~5px, so on plans that
# draw thin partitions a whole wall reads as absent (measured on fp6: a 3.7m
# "passage" between a living-dining space and a bedroom). Pixels cannot
# settle it — at that stroke width a partition and a glazing band are
# identical — so the span is reported with its class unresolved rather than
# asserted. ponytail: a width rule; the real fix is a solid mask that keeps
# thin partitions, which needs a corpus that draws them at more than 4px.
MAX_INTERIOR_SPAN_MM = 2500.0

# A door is something you walk through, so its width is bounded by what a
# leaf can be built as — four sliding leaves at ~900mm is the widest thing
# in the domain. Past that the drawing is a glazed facade or a whole wall
# run the scan merged, not a door: the corpus's real balcony doors stop at
# 2957mm and the next one is 4085mm, so the bound sits in an empty band.
# Single-leaf width is NOT bounded here — the measurement is the break in
# the wall, frame and reveal included, and single_door widths run
# continuously from 1112 to 1375mm with no gap to cut at.
MAX_DOOR_SPAN_MM = 3500.0
# Below this, a classified door/passage is a drawing notch rather than a
# route a resident can use. The corpus's smallest credible opening is 470mm.
MIN_WALK_THROUGH_SPAN_MM = 450.0
# A plain or threshold-drawn opening below this span is a single-leaf door;
# wider parallel tracks are sliding leaves and wider plain gaps are passages.
MAX_SINGLE_DOOR_SPAN_MM = 1400.0
MIN_WALK_THROUGH_THICKNESSES = 2.5
MAX_SINGLE_DOOR_THICKNESSES = 7.5
MAX_DOOR_THICKNESSES = 16.0
MAX_EXTERIOR_WINDOW_THICKNESSES = 30.0
MAX_SHALLOW_EDGE_DEPTH_THICKNESSES = 15.0
MIN_GLAZED_FACADE_FRONTAGE_SHARE = 0.50
MIN_RAILING_FRONTAGE_SHARE = 0.75
MIN_WHOLE_HOST_OPENING_SHARE = 0.80
# Below this share a CV "room" is commonly furniture or a door-sweep fragment;
# keep its openings unresolved until room semantics confirm it.
MIN_OPENING_ROOM_SHARE = 0.015
_WALK_THROUGH = {
    "passage",
    "single_door",
    "double_door",
    "sliding_door",
    "folding_door",
}


def _opening_span_mm(
    candidate: OpeningCandidate,
    scale: ScaleDraft | None,
    segments: list[WallSegment] | None,
) -> float | None:
    if scale is None:
        return None
    if candidate.axis == "d" and segments is not None and 0 <= candidate.wall_index < len(segments):
        segment = segments[candidate.wall_index]
        per_mm = pixels_per_mm_in_direction(
            segment.end[0] - segment.start[0],
            segment.end[1] - segment.start[1],
            scale,
        )
    else:
        per_mm = scale.px_per_mm_x if candidate.axis == "h" else scale.px_per_mm_y
    return candidate.width_px / per_mm if per_mm > 0 else None


def _cv_opening_type(
    candidate: OpeningCandidate,
    rooms: list[RoomDraft],
    scale: ScaleDraft | None,
    segments: list[WallSegment] | None,
) -> str:
    """Resolve common symbols from tracks, span, and the two wall sides."""
    span_mm = _opening_span_mm(candidate, scale, segments)
    host = (
        segments[candidate.wall_index]
        if segments is not None and 0 <= candidate.wall_index < len(segments)
        else None
    )
    thicknesses = candidate.width_px / host.thickness_px if host is not None else None
    if span_mm is not None:
        walkable = MIN_WALK_THROUGH_SPAN_MM <= span_mm <= MAX_DOOR_SPAN_MM
        single = walkable and span_mm <= MAX_SINGLE_DOOR_SPAN_MM
        if thicknesses is not None and thicknesses < MIN_WALK_THROUGH_THICKNESSES:
            walkable = single = False
    elif thicknesses is not None:
        walkable = MIN_WALK_THROUGH_THICKNESSES <= thicknesses <= MAX_DOOR_THICKNESSES
        single = walkable and thicknesses <= MAX_SINGLE_DOOR_THICKNESSES
    else:
        walkable = single = False

    too_wide_for_door = (
        span_mm > MAX_DOOR_SPAN_MM
        if span_mm is not None
        else thicknesses is not None and thicknesses > MAX_DOOR_THICKNESSES
    )

    total_room_area = sum(room.area_px for room in rooms)
    connects_rooms = _both_sides_are_rooms(candidate.connects, rooms) and all(
        rooms[side].area_px >= MIN_OPENING_ROOM_SHARE * total_room_area
        for side in candidate.connects
        if isinstance(side, int)
    )
    shallow_frontage_share = 0.0
    if host is not None and candidate.axis in {"h", "v"}:
        for side in candidate.connects:
            if not isinstance(side, int):
                continue
            room_width, room_height = np.ptp(rooms[side].polygon, axis=0)
            frontage, depth = (
                (room_width, room_height)
                if candidate.axis == "h"
                else (room_height, room_width)
            )
            if (
                depth <= MAX_SHALLOW_EDGE_DEPTH_THICKNESSES * host.thickness_px
                and candidate.width_px <= 1.2 * frontage
            ):
                shallow_frontage_share = max(
                    shallow_frontage_share, candidate.width_px / max(frontage, 1.0)
                )
    if candidate.kind_hint == "double_door":
        return "double_door"
    if candidate.kind_hint == "passage":
        return "passage"
    if (
        EXTERIOR in candidate.connects
        and too_wide_for_door
        and any(
            isinstance(side, int)
            and 0 <= side < len(rooms)
            and rooms[side].room_type in OUTDOOR_ROOM_TYPES
            for side in candidate.connects
        )
    ):
        return "railing"
    if candidate.kind_hint == "window":
        if (
            EXTERIOR in candidate.connects
            and too_wide_for_door
            and shallow_frontage_share >= MIN_RAILING_FRONTAGE_SHARE
        ):
            # A long open edge around a shallow exterior strip is a balcony
            # railing, not a wall opening or glazing symbol.
            return "railing"
        if connects_rooms and too_wide_for_door:
            if shallow_frontage_share >= MIN_GLAZED_FACADE_FRONTAGE_SHARE:
                return "floor_to_ceiling_window"
            # A whole missing partition between deep rooms can look like a
            # multi-track window; do not turn it into an interior facade.
            return "unknown_symbol"
        if (
            EXTERIOR in candidate.connects
            and thicknesses is not None
            and thicknesses > MAX_EXTERIOR_WINDOW_THICKNESSES
        ):
            # A whole balcony edge is a railing or glazed facade; the generic
            # window class would invent a wall opening where CV cannot choose.
            return "unknown_symbol"
        if connects_rooms and single:
            return "single_door"
        if connects_rooms and walkable:
            return "sliding_door"
        return "window"
    if connects_rooms and single:
        return "single_door"
    if single and candidate.arc is not None:
        return "single_door"
    wide_door = (
        span_mm > MAX_SINGLE_DOOR_SPAN_MM
        if span_mm is not None
        else thicknesses is not None and thicknesses > MAX_SINGLE_DOOR_THICKNESSES
    )
    if (
        connects_rooms
        and wide_door
        and shallow_frontage_share >= MIN_GLAZED_FACADE_FRONTAGE_SHARE
    ):
        return "sliding_door"
    if connects_rooms and walkable and not single:
        return "passage"
    if EXTERIOR in candidate.connects:
        return "window"
    return "unknown_symbol"


def merge_openings(
    candidates: list[OpeningCandidate],
    read: OpeningsRead | None,
    image_shape: tuple[int, int],
    rooms: list[RoomDraft] | None = None,
    scale: ScaleDraft | None = None,
    segments: list[WallSegment] | None = None,
) -> tuple[list[OpeningDraft], list[ElementDraft], list[ParseWarning], list[Unresolved]]:
    drafts: list[OpeningDraft] = []
    elements: list[ElementDraft] = []
    warnings: list[ParseWarning] = []
    unresolved: list[Unresolved] = []
    entries = read.candidates if read is not None else {}
    leaf_spans = [
        candidate.arc.radius_px
        for candidate in candidates
        if candidate.arc is not None and candidate.arc.radius_px is not None
    ]
    wide_track_px = 1.5 * statistics.median(leaf_spans) if len(leaf_spans) >= 2 else math.inf

    sidelights = {
        candidate.marker
        for candidate in candidates
        if candidate.kind_hint == "window"
        and candidate.arc is None
        and any(
            _window_beside_swing(candidate, other, segments)
            for other in candidates
            if other.arc is not None
        )
    }
    cv_rooms = rooms or []
    cv_types = [_cv_opening_type(cand, cv_rooms, scale, segments) for cand in candidates]
    railing_rooms = {
        side
        for candidate, cv_type in zip(candidates, cv_types, strict=True)
        if cv_type == "railing"
        for side in candidate.connects
        if isinstance(side, int) and 0 <= side < len(cv_rooms)
    }
    exterior_axes: dict[int, set[str]] = {}
    for candidate, cv_type in zip(candidates, cv_types, strict=True):
        if (
            EXTERIOR in candidate.connects
            and cv_type in _WINDOWS | {"railing"}
            and candidate.axis in {"h", "v"}
        ):
            for side in candidate.connects:
                if isinstance(side, int) and 0 <= side < len(cv_rooms):
                    exterior_axes.setdefault(side, set()).add(candidate.axis)
    total_room_area = sum(room.area_px for room in cv_rooms)
    sliding_widths = {
        room_index: max(
            (
                candidate.width_px
                for candidate, cv_type in zip(candidates, cv_types, strict=True)
                if room_index in candidate.connects and cv_type == "sliding_door"
            ),
            default=0.0,
        )
        for room_index in exterior_axes
        if cv_rooms[room_index].area_px <= 0.05 * total_room_area
    }
    track_rooms = railing_rooms | {
        room_index
        for room_index, axes in exterior_axes.items()
        if axes == {"h", "v"}
        and cv_rooms[room_index].area_px <= 0.05 * total_room_area
        and sum(
            room_index in candidate.connects and cv_type in _WALK_THROUGH
            for candidate, cv_type in zip(candidates, cv_types, strict=True)
        )
        == 1
    }
    for cand, cv_type in zip(candidates, cv_types, strict=True):
        zone_divider = (
            cv_type == "single_door"
            and cand.kind_hint == "doorlike"
            and cand.arc is None
            and _both_sides_are_rooms(cand.connects, cv_rooms)
            and all(cv_rooms[side].zone_bounded for side in cand.connects if isinstance(side, int))
        )
        balcony_track = (
            cv_type == "single_door"
            and cand.arc is None
            and cand.marker not in sidelights
            and any(side in track_rooms for side in cand.connects if isinstance(side, int))
        )
        balcony_window = (
            cv_type == "single_door"
            and cand.kind_hint == "doorlike"
            and cand.arc is None
            and any(
                isinstance(side, int)
                and cand.width_px < 0.75 * sliding_widths.get(side, 0.0)
                for side in cand.connects
            )
        )
        if zone_divider:
            cv_type = "passage"
        elif balcony_window:
            cv_type = "window"
        elif balcony_track:
            cv_type = "sliding_door"
        if (
            cv_type == "single_door"
            and cand.kind_hint == "window"
            and cand.arc is None
            and cand.width_px > wide_track_px
            and _both_sides_are_rooms(cand.connects, rooms or [])
        ):
            cv_type = "sliding_door"
        if cv_type == "single_door" and cand.marker in sidelights:
            cv_type = "window"
        span_mm = _opening_span_mm(cand, scale, segments)
        entry = entries.get(cand.marker)
        if entry is None:
            element_type = cv_type
            source, confidence, raw_text = "cv", 0.35, None
            if element_type == "unknown_symbol" and cand.arc is None:
                unresolved.append(
                    Unresolved(
                        path=f"openings/{cand.marker}/element_type",
                        reason="no VLM classification for this opening",
                    )
                )
        elif not entry.is_real:
            warnings.append(
                ParseWarning(
                    code="opening_vetoed",
                    message=f"VLM judged candidate {cand.marker} not to be a real opening",
                    ref=cand.marker,
                )
            )
            continue
        else:
            element_type = entry.element_type
            source, confidence, raw_text = "cv+vlm", entry.confidence, entry.raw_text

        if element_type not in _WINDOWS | _WALK_THROUGH | {"railing", "unknown_symbol"}:
            warnings.append(
                ParseWarning(
                    code="opening_legend_conflict",
                    message=(
                        f"candidate {cand.marker} lies in a measured wall opening but was "
                        f"read as free-standing {element_type}; class left unresolved"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "unknown_symbol"
            confidence = min(confidence, 0.3)
            unresolved.append(
                Unresolved(
                    path=f"openings/{cand.marker}/element_type",
                    reason="a wall opening cannot be a free-standing legend element",
                )
            )

        if cand.wall_index < 0 and element_type != "passage":
            warnings.append(
                ParseWarning(
                    code="opening_reclassified_by_geometry",
                    message=(
                        f"candidate {cand.marker} was read as {element_type}, but it is "
                        "a drawn functional divider with no host wall; reclassified passage"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "passage"
            confidence = min(confidence, 0.7)

        swing = hinge = None
        if cand.arc is not None:
            # A drawn quarter-disc at a jamb is a door leaf sweeping the
            # floor: windows, sliding leaves and plain passages do not have
            # one, and the arc is measured from pixels at leaf scale while
            # the class is a reading of a small crop. The drawing wins.
            if element_type not in _SWINGING or (
                cv_type == "double_door" and element_type != "double_door"
            ):
                physical_swinging = (
                    cv_type if cv_type in _SWINGING else "single_door"
                )
                if source != "cv":  # a disagreement, not merely a CV-only read
                    warnings.append(
                        ParseWarning(
                            code="opening_reclassified_by_arc",
                            message=(
                                f"candidate {cand.marker} was read as {element_type} but "
                                "the plan draws a door-leaf swing sector at its jamb; "
                                f"reclassified {physical_swinging}"
                            ),
                            ref=cand.marker,
                        )
                    )
                element_type = physical_swinging
                confidence = min(confidence, 0.7) if source != "cv" else 0.5
            swing, hinge = cand.arc.swing, cand.arc.hinge
        elif source == "cv+vlm" and (
            (cv_type == "sliding_door" and element_type in {"window", "passage", "unknown_symbol"})
            or (balcony_track and element_type != "sliding_door")
            or (balcony_window and element_type != "window")
            or (zone_divider and element_type != "passage")
            or (
                cv_type == "passage"
                and cand.kind_hint == "passage"
                and element_type != "passage"
            )
            or (
                cv_type == "single_door"
                and (
                    element_type == "unknown_symbol"
                    or (
                        element_type in _WINDOWS
                        and _both_sides_indoor(cand.connects, rooms or [])
                    )
                )
            )
            or (
                cv_type == "railing"
                and element_type not in {"railing", "floor_to_ceiling_window"}
            )
            or (
                cv_type == "floor_to_ceiling_window"
                and element_type in {"window", "unknown_symbol"}
            )
        ):
            warnings.append(
                ParseWarning(
                    code="opening_reclassified_by_geometry",
                    message=(
                        f"candidate {cand.marker} was read as {element_type}, but its "
                        f"measured span and wall-side geometry identify a {cv_type}; "
                        "reclassified"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = cv_type
            confidence = min(confidence, 0.7)
        # A sliding door and a sliding window are the SAME drawn symbol —
        # parallel overlapping leaves — so only adjacency separates them, and
        # between two indoor rooms there is no exterior for a window to face.
        # The prompt says so; this makes it hold (a 1.5m 客厅↔门厅 sliding
        # door came back as sliding_window at 0.9 confidence).
        if element_type == "sliding_window" and (
            _both_sides_indoor(cand.connects, rooms or []) or cv_type == "sliding_door"
        ):
            warnings.append(
                ParseWarning(
                    code="opening_reclassified_by_adjacency",
                    message=(
                        f"candidate {cand.marker} was read as sliding_window but its "
                        "parallel tracks span a walk-through between two rooms; "
                        "reclassified sliding_door"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "sliding_door"
            confidence = min(confidence, 0.7)
        elif (
            source == "cv+vlm"
            and element_type in _WINDOWS
            and _window_conflicts_with_privacy_room(cand.connects, rooms or [])
        ):
            warnings.append(
                ParseWarning(
                    code="opening_habitability_conflict",
                    message=(
                        f"candidate {cand.marker} was read as {element_type} between "
                        "indoor rooms including a bedroom or bathroom; class left "
                        "unresolved instead of asserting an interior privacy window"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "unknown_symbol"
            confidence = min(confidence, 0.3)
            unresolved.append(
                Unresolved(
                    path=f"openings/{cand.marker}/element_type",
                    reason="indoor window reading conflicts with residential privacy",
                )
            )
        elif (
            element_type == "passage"
            and "exterior" in cand.connects
            and rooms is not None
            and any(
                isinstance(side, int)
                and 0 <= side < len(rooms)
                and rooms[side].room_type not in OUTDOOR_ROOM_TYPES
                for side in cand.connects
            )
        ):
            warnings.append(
                ParseWarning(
                    code="opening_habitability_conflict",
                    message=(
                        f"candidate {cand.marker} was read as an open passage through the "
                        "dwelling envelope; class left unresolved because an exterior "
                        "residential opening needs a door or window"
                    ),
                    ref=cand.marker,
                )
            )
            element_type = "unknown_symbol"
            confidence = min(confidence, 0.3)
            unresolved.append(
                Unresolved(
                    path=f"openings/{cand.marker}/element_type",
                    reason="open passage through the dwelling envelope is not habitable",
                )
            )
        if span_mm is not None and element_type != "unknown_symbol":
            reason = None
            connected_types = [
                cv_rooms[side].room_type
                for side in cand.connects
                if isinstance(side, int) and 0 <= side < len(cv_rooms)
            ]
            wet_zone_door = (
                cand.arc is not None
                and len(connected_types) == 2
                and all(room_type == "bathroom" for room_type in connected_types)
            )
            multi_panel_balcony_door = (
                element_type == "sliding_door"
                and len(connected_types) == 2
                and any(room_type in OUTDOOR_ROOM_TYPES for room_type in connected_types)
                and any(room_type not in OUTDOOR_ROOM_TYPES for room_type in connected_types)
            )
            if (
                element_type in _WALK_THROUGH
                and span_mm < MIN_WALK_THROUGH_SPAN_MM
                and not wet_zone_door
            ):
                reason = f"spans only {span_mm:.0f}mm — too narrow for habitable circulation"
            elif (
                element_type in _WALK_THROUGH
                and element_type != "passage"
                and span_mm > MAX_DOOR_SPAN_MM
                and not multi_panel_balcony_door
            ):
                reason = (
                    f"spans {span_mm:.0f}mm — wider than any door leaf can be built, "
                    "so the drawing is a glazed facade or a merged wall run"
                )
            elif (
                span_mm > MAX_INTERIOR_SPAN_MM
                and _both_sides_indoor(cand.connects, rooms or [])
                and (
                    element_type not in _WALK_THROUGH
                    or not _both_sides_open_plan(cand.connects, rooms or [])
                )
            ):
                reason = (
                    f"spans {span_mm:.0f}mm between two indoor rooms — too wide for "
                    "one opening, and a partition the wall mask missed reads identically"
                )
            if reason is not None:
                warnings.append(
                    ParseWarning(
                        code="opening_span_implausible",
                        message=(
                            f"candidate {cand.marker} was read as {element_type} but it "
                            f"{reason}; class left unresolved"
                        ),
                        ref=cand.marker,
                    )
                )
                element_type = "unknown_symbol"
                confidence = min(confidence, 0.3)
                unresolved.append(
                    Unresolved(
                        path=f"openings/{cand.marker}/element_type",
                        reason=f"measured span contradicts the reading: {reason}",
                    )
                )
        if element_type in _SWINGING and cand.arc is None:
            unresolved.append(
                Unresolved(
                    path=f"openings/{cand.marker}/swing",
                    reason="no swing-arc evidence in the image",
                )
            )

        drafts.append(
            OpeningDraft(
                marker=cand.marker,
                element_type=element_type,
                raw_text=raw_text,
                bbox=cand.bbox,
                center=cand.center,
                axis=cand.axis,
                width_px=cand.width_px,
                wall_index=cand.wall_index,
                connects=cand.connects,
                swing=swing,
                hinge=hinge,
                source=source,
                confidence=confidence,
            )
        )

    h, w = image_shape
    for extra in read.extra_elements if read is not None else []:
        if not extra.is_real:
            continue
        y0, x0, y1, x1 = (v / 1000.0 for v in extra.box_2d)
        bbox = (x0 * w, y0 * h, x1 * w, y1 * h)
        approximate_opening = extra.element_type not in _FREE_STANDING
        if approximate_opening:
            warnings.append(
                ParseWarning(
                    code="opening_without_physical_evidence",
                    message=(
                        f"VLM suggested an unmarked {extra.element_type}, but no measured "
                        "wall opening supports it; it is not emitted"
                    ),
                )
            )
            continue
        elements.append(
            ElementDraft(
                element_type=extra.element_type,
                raw_text=extra.raw_text,
                bbox=bbox,
                source="vlm",
                confidence=min(extra.confidence, 0.5),
            )
        )
    return drafts, elements, warnings, unresolved


def _window_beside_swing(
    window: OpeningCandidate,
    door: OpeningCandidate,
    segments: list[WallSegment] | None,
) -> bool:
    if (
        window.axis not in {"h", "v"}
        or window.axis != door.axis
        or frozenset(window.connects) != frozenset(door.connects)
        or segments is None
        or not 0 <= window.wall_index < len(segments)
    ):
        return False
    thickness = segments[window.wall_index].thickness_px
    along = (0, 2) if window.axis == "h" else (1, 3)
    normal = 1 if window.axis == "h" else 0
    gap = max(
        0.0,
        max(window.bbox[along[0]], door.bbox[along[0]])
        - min(window.bbox[along[1]], door.bbox[along[1]]),
    )
    return (
        abs(window.center[normal] - door.center[normal]) <= 1.5 * thickness
        and gap <= 4.0 * thickness
    )


def apply_area_checks(rooms: list[RoomDraft], scale: ScaleDraft | None) -> MergeOutcome:
    """Drop segmentation junk and compute deviation-vs-printed per room.

    Deviations are returned via RoomDraft-adjacent metadata at assembly time;
    here the tiny-room filter needs the scale, so it runs after estimation.
    """
    out = MergeOutcome(rooms=[])
    for draft in rooms:
        if scale is not None and draft.name is None:
            area_sqm = draft.area_px / (scale.px_per_mm_x * scale.px_per_mm_y) / 1e6
            if area_sqm < MIN_ROOM_SQM:
                out.warnings.append(
                    ParseWarning(
                        code="tiny_room_dropped",
                        message=(
                            f"unnamed region of {area_sqm:.2f}㎡ dropped as segmentation junk"
                        ),
                        ref=draft.marker,
                    )
                )
                continue
        out.rooms.append(draft)
    return out
