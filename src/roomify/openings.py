"""Opening candidates, swing-arc evidence, and wall segments.

Openings are found by SCANNING WALLS, not by morphological gap-closing:
wall segments are derived from the room polygons (shared edges between
adjacent rooms, leftover edges toward everything else), and every interval
along a segment where the solid mask has no wall is an opening candidate.
Closing-based detection was tried first and fails structurally — door gaps
that open onto corridors narrower than the closing kernel get subsumed into
corridor-sized blobs. Scanning also yields ``connects`` and the hosting
wall for free.

CV records the observable legend cues (parallel tracks and swing arcs); room
adjacency and physical span resolve the common door/window classes during
merge, with the VLM supplying finer vocabulary when it is available.

Swing/hinge facts are the opposite: they come EXCLUSIVELY from pixel
evidence (a quarter-disc sector at a jamb, drawn either as a stroked arc or
as a tinted fill), never from the VLM. Plans that draw no swing arc get
``swing=None`` — reporting the unobservable would be invention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box

from roomify.merge import (
    MAX_DOOR_THICKNESSES,
    MAX_SINGLE_DOOR_SPAN_MM,
    MAX_SINGLE_DOOR_THICKNESSES,
    MIN_OPENING_ROOM_SHARE,
    MIN_WHOLE_HOST_OPENING_SHARE,
    RoomDraft,
)
from roomify.rooms import _boundary_fraction, building_silhouette
from roomify.walls import MAX_CHROMA, WallExtraction, _arc_stroke_coverage

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
UNKNOWN_ARC_STROKE_MARGIN = 0.38  # concave-facade speckle tops out at 0.366
DOUBLE_ARC_STROKE_COVERAGE = 0.70
DOUBLE_ARC_STROKE_MARGIN = 0.15
MIN_SECTOR_PX = 100  # smaller samples (tiny rooms) give no verdict
# A swing sector's radius IS the leaf, and no door leaf is 1.4m wide. Beyond
# that the sector test only ever fires on a floor-tint change spanning a wide
# mouth (measured: 1.6-2.4m "arcs" on balcony and corridor openings, while
# every genuinely arc-drawn door on the corpus sits at 0.57-1.33m).
MAX_HOST_GAP_THICKNESSES = 4.0
MIN_LEAF_GAP_THICKNESSES = 2.0


@dataclass(frozen=True)
class ArcEvidence:
    hinge: tuple[float, float]  # working px
    swing: str  # "clockwise" | "counterclockwise" (image coords, y down)
    opens_into: int | None  # room index the sector lies in
    radius_px: float | None = None  # measured stroked leaf; tint-only arcs have none


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
    axis: str  # "h": horizontal wall; "v": vertical; "d": diagonal
    width_px: float  # break length along the wall
    kind_hint: str  # "window" (tracks) | "doorlike" (leaf gap) | "passage"
    connects: tuple[int | str, int | str]  # room indices / "exterior" / "unknown"
    wall_index: int  # index into the wall-segment list
    arc: ArcEvidence | None


def find_openings(
    walls: WallExtraction,
    rooms: list[RoomDraft],
    bgr: np.ndarray,
    door_px: float | None = None,
    *,
    ignored_rooms: list[RoomDraft] | None = None,
) -> tuple[list[OpeningCandidate], list[WallSegment]]:
    from roomify.vlm import marker_ids

    if door_px is None:
        x0, y0, x1, y1 = walls.footprint
        door_px = 0.06 * max(x1 - x0, y1 - y0)  # ~900mm on a 10-15m dwelling

    segments = derive_wall_segments(rooms, walls, ignored_rooms=ignored_rooms)
    room_masks = _room_masks(rooms, walls.solid.shape)
    silhouette, _ = building_silhouette(walls.union, walls.footprint)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )

    min_len = max(6.0, 0.4 * door_px)
    max_len = 4.6 * door_px  # wide balcony mouths still count; whole façades don't
    raw: list[OpeningCandidate] = []
    for si, seg in enumerate(segments):
        raw.extend(
            _scan_segment(
                si,
                seg,
                walls,
                room_masks,
                silhouette,
                min_len,
                max_len,
                bgr=bgr,
            )
        )

    raw = _resolve_unknown_adjacencies(_dedupe(raw, walls.thickness_px), rooms, walls.thickness_px)
    # T-junction closure can extend a paired wall beyond one room's actual
    # face. A whole-host candidate in that extension must re-prove both rooms
    # locally; otherwise dimension leaders become interior doors.
    for index, cand in enumerate(raw):
        host = segments[cand.wall_index]
        host_length = float(np.hypot(host.end[0] - host.start[0], host.end[1] - host.start[1]))
        if (
            all(isinstance(side, int) for side in cand.connects)
            and cand.width_px >= MIN_WHOLE_HOST_OPENING_SHARE * host_length
        ):
            connects = _resolve_connects(
                replace(host, rooms=cand.connects), walls, room_masks, silhouette, cand.center
            )
            raw[index] = replace(cand, connects=connects)
    if len(raw) > MAX_CANDIDATES:
        raw.sort(key=lambda c: c.width_px, reverse=True)
        raw = raw[:MAX_CANDIDATES]

    classified: list[OpeningCandidate] = []
    max_leaf_px = door_px * MAX_SINGLE_DOOR_SPAN_MM / 1000.0
    min_opening_room_area = MIN_OPENING_ROOM_SHARE * sum(room.area_px for room in rooms)
    weak_rooms = {
        i
        for i, room in enumerate(rooms)
        if room.area_px < min_opening_room_area
        and _boundary_fraction(room, walls.solid) < 0.25
    }
    ordered = sorted(raw, key=lambda c: (c.center[1], c.center[0]))
    for cand in ordered:
        coarse_kind = "window" if _looks_like_window(cand, walls) else "doorlike"
        kind = "window" if _looks_like_window(cand, walls, bgr) else "doorlike"
        fx0, fy0, fx1, fy1 = walls.footprint
        facade_distance = min(
            cand.center[0] - fx0,
            fx1 - cand.center[0],
            cand.center[1] - fy0,
            fy1 - cand.center[1],
        )
        arc = None
        exterior_track = coarse_kind == "window" and EXTERIOR in cand.connects
        strict_arc = exterior_track or (cand.axis == "d" and EXTERIOR in cand.connects)
        if (
            cand.width_px > MIN_LEAF_GAP_THICKNESSES * walls.thickness_px
            and (
                cand.width_px <= max_leaf_px
                or (
                    strict_arc
                    and cand.width_px <= MAX_DOOR_THICKNESSES * walls.thickness_px
                )
            )
            and all(
                not isinstance(side, int)
                or (0 <= side < len(rooms) and rooms[side].area_px >= min_opening_room_area)
                for side in cand.connects
            )
            and (
                coarse_kind == "doorlike"
                or all(isinstance(side, int) for side in cand.connects)
                or exterior_track
            )
        ):
            arc = _detect_arc(
                replace(cand, kind_hint=kind),
                walls,
                room_masks,
                bgr,
                binary,
                segments[cand.wall_index],
                require_coherent_stroke=strict_arc,
            )
            if kind == "window" and arc is not None and arc.radius_px is None:
                arc = None
        if (
            kind == "window"
            and arc is None
            and EXTERIOR in cand.connects
            and cand.width_px >= 2.5 * walls.thickness_px
        ):
            double_arc = _detect_double_arc(cand, walls, room_masks, binary)
            if double_arc is not None and double_arc.opens_into is None:
                kind, arc = "double_door", double_arc
        if kind == "window" and arc is None and all(
            isinstance(side, int) for side in cand.connects
        ):
            paired = _paired_zone_door(cand, walls, room_masks, binary)
            if paired is not None:
                cand, kind, arc = paired, paired.kind_hint, paired.arc
        if kind == "window" and arc is None and cand.width_px > max_leaf_px:
            mixed = _split_swing_beside_window(
                cand, segments[cand.wall_index], walls, room_masks, bgr, binary
            )
            if mixed is not None:
                window_part, cand = mixed
                classified.append(window_part)
                kind, arc = cand.kind_hint, cand.arc
        if (
            kind == "doorlike"
            and arc is None
            and cand.width_px <= MIN_LEAF_GAP_THICKNESSES * walls.thickness_px
        ):
            continue  # wall-junction notch, too narrow to be a usable opening
        if kind == "doorlike" and arc is None and _plain_internal_gap(cand, walls):
            kind = "passage"
        host = segments[cand.wall_index]
        whole_host = cand.width_px >= MIN_WHOLE_HOST_OPENING_SHARE * float(
            np.hypot(host.end[0] - host.start[0], host.end[1] - host.start[1])
        )
        if (
            kind == "window"
            and cand.axis in {"h", "v"}
            and cand.width_px <= MIN_LEAF_GAP_THICKNESSES * walls.thickness_px
            and all(isinstance(side, int) for side in cand.connects)
        ):
            continue  # short room-corner stub, not usable glazing
        weak_side = UNKNOWN in cand.connects or any(
            isinstance(side, int) and side in weak_rooms for side in cand.connects
        )
        undersized_side = any(
            isinstance(side, int) and rooms[side].area_px < min_opening_room_area
            for side in cand.connects
        )
        if (
            kind == "doorlike"
            and arc is None
            and EXTERIOR in cand.connects
            and undersized_side
        ):
            continue
        usable_exterior_door = (
            EXTERIOR in cand.connects
            and cand.width_px <= MAX_SINGLE_DOOR_THICKNESSES * host.thickness_px
        )
        if (
            kind == "doorlike"
            and arc is None
            and whole_host
            and weak_side
            and not usable_exterior_door
        ):
            continue  # closed furniture/fixture contour, not a wall opening
        if (
            kind == "window"
            and arc is None
            and whole_host
            and weak_side
            and facade_distance > cand.width_px
        ):
            continue  # interior furniture frame, too deep inside for facade glazing
        resolved = replace(cand, kind_hint=kind, arc=arc)
        if arc is not None and kind != "double_door":
            resolved = _fit_stroked_leaf(resolved, segments[cand.wall_index])
        classified.append(resolved)

    initial_candidate_hosts = {candidate.wall_index for candidate in classified}
    classified = _recover_parallel_swing(classified, walls.thickness_px)
    classified = _recover_corner_swing(classified, walls, room_masks, bgr, binary)
    classified = [
        cand
        for cand in classified
        if not _looks_like_solid_wall(cand, bgr, cand.kind_hint, walls.thickness_px)
    ]
    classified = _collapse_thin_regions(classified, rooms, walls)
    classified = [
        _recover_facade_candidate(cand, rooms, walls, min_opening_room_area)
        for cand in classified
    ]
    furniture = [candidate for candidate in classified if _opening_over_furniture(candidate, bgr)]
    discarded_hosts = {
        candidate.wall_index
        for candidate in furniture
        if 0 <= candidate.wall_index < len(segments)
        and LineString(
            (segments[candidate.wall_index].start, segments[candidate.wall_index].end)
        ).length
        <= 3.0 * walls.thickness_px
    }
    classified = [candidate for candidate in classified if candidate not in furniture]
    for index, cand in enumerate(classified):
        if cand.kind_hint != "doorlike" or cand.arc is not None:
            continue
        arc = _detect_double_arc(cand, walls, room_masks, binary)
        if arc is not None:
            classified[index] = replace(cand, kind_hint="double_door", arc=arc)
    # A swung-open leaf is perpendicular to its wall and can itself look like
    # a second wall opening. Keep the candidate whose hinge sits at their
    # shared corner; that is the actual threshold, not the leaf.
    classified = _dedupe_corner_swings(
        classified, walls.thickness_px, segments=segments, walls=walls
    )
    classified = _recover_corner_glazing(classified, segments, walls)
    classified = _merge_window_panels(classified, segments)
    classified, segments = _merge_zone_facade_panels(
        classified, segments, rooms, walls, room_masks, bgr, binary
    )
    classified, segments = _recover_isolated_room_openings(
        classified,
        segments,
        rooms,
        walls,
        room_masks,
        silhouette,
        bgr,
        binary,
        min_len,
        max_len,
        max_leaf_px,
    )
    exterior_rooms = {
        side
        for segment in segments
        if EXTERIOR in segment.rooms
        for side in segment.rooms
        if isinstance(side, int)
    }
    classified = _resolve_unknown_adjacencies(
        classified,
        rooms,
        walls.thickness_px,
        exterior_rooms=exterior_rooms,
    )
    for candidate in classified:
        if not 0 <= candidate.wall_index < len(segments):
            continue
        host = segments[candidate.wall_index]
        if UNKNOWN in host.rooms and UNKNOWN not in candidate.connects:
            segments[candidate.wall_index] = replace(host, rooms=candidate.connects)
    remaining_hosts = {candidate.wall_index for candidate in classified}
    fx0, fy0, fx1, fy1 = walls.footprint
    for wall_index, segment in enumerate(segments):
        line = LineString((segment.start, segment.end))
        if wall_index not in remaining_hosts and any(
            candidate.arc is not None
            and candidate.axis in {"h", "v"}
            and (
                (candidate.axis == "h" and abs(segment.end[0] - segment.start[0])
                 <= 0.2 * abs(segment.end[1] - segment.start[1]))
                or (candidate.axis == "v" and abs(segment.end[1] - segment.start[1])
                    <= 0.2 * abs(segment.end[0] - segment.start[0]))
            )
            and 0.65 * (candidate.arc.radius_px or candidate.width_px)
            <= line.length
            <= 1.35 * (candidate.arc.radius_px or candidate.width_px)
            and min(
                Point(segment.start).distance(Point(candidate.arc.hinge)),
                Point(segment.end).distance(Point(candidate.arc.hinge)),
            )
            <= 0.75 * walls.thickness_px
            and {
                side for side in segment.rooms if isinstance(side, int)
            }
            & {side for side in candidate.connects if isinstance(side, int)}
            and _ink_share(segment, walls.solid) < 0.20
            for candidate in classified
        ):
            discarded_hosts.add(wall_index)
            continue
        if (
            wall_index in remaining_hosts
            or UNKNOWN not in segment.rooms
            or EXTERIOR in segment.rooms
            or _ink_share(segment, walls.solid) >= 0.10
        ):
            continue
        mx, my = line.interpolate(0.5, normalized=True).coords[0]
        facade_distance = min(mx - fx0, fx1 - mx, my - fy0, fy1 - my)
        if facade_distance > min(1.25 * line.length, 8.0 * walls.thickness_px):
            discarded_hosts.add(wall_index)
    for wall_index in initial_candidate_hosts - remaining_hosts:
        segment = segments[wall_index]
        if (
            UNKNOWN in segment.rooms
            and _ink_share(segment, walls.solid) < 0.30
            and LineString((segment.start, segment.end)).length
            <= 8.0 * walls.thickness_px
        ):
            discarded_hosts.add(wall_index)
    if discarded_hosts:
        remap: dict[int, int] = {}
        kept_segments: list[WallSegment] = []
        for old_index, segment in enumerate(segments):
            if old_index not in discarded_hosts:
                remap[old_index] = len(kept_segments)
                kept_segments.append(segment)
        classified = [
            replace(candidate, wall_index=remap[candidate.wall_index])
            for candidate in classified
            if candidate.wall_index in remap
        ]
        segments = kept_segments
    classified, segments = _merge_diagonal_window_panels(
        classified, segments, walls.thickness_px
    )
    ids = marker_ids(len(classified))
    out = [replace(cand, marker=marker) for marker, cand in zip(ids, classified, strict=True)]
    return out, segments


def _recover_isolated_room_openings(
    candidates: list[OpeningCandidate],
    segments: list[WallSegment],
    rooms: list[RoomDraft],
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    silhouette: np.ndarray,
    bgr: np.ndarray,
    binary: np.ndarray,
    min_len: float,
    max_len: float,
    max_leaf_px: float,
) -> tuple[list[OpeningCandidate], list[WallSegment]]:
    """Recover the best drawn doorway on a room left structurally isolated."""
    out = list(candidates)
    hosts = list(segments)
    total_area = sum(room.area_px for room in rooms)

    def gives_access(candidate: OpeningCandidate, room_index: int) -> bool:
        if room_index not in candidate.connects:
            return False
        host = hosts[candidate.wall_index] if 0 <= candidate.wall_index < len(hosts) else None
        thickness = host.thickness_px if host is not None else walls.thickness_px
        walkable = (
            2.5 * thickness
            <= candidate.width_px
            <= MAX_DOOR_THICKNESSES * thickness
        ) or candidate.kind_hint == "passage"
        return bool(
            candidate.arc is not None
            or candidate.kind_hint == "double_door"
            or (
                walkable
                and candidate.kind_hint in {"doorlike", "window", "passage"}
                and all(isinstance(side, int) for side in candidate.connects)
            )
        )

    accessed = {
        index
        for index in range(len(rooms))
        if any(gives_access(candidate, index) for candidate in out)
    }

    def parallel_edge_distance(room: RoomDraft, candidate: OpeningCandidate) -> float:
        along = (candidate.bbox[0], candidate.bbox[2]) if candidate.axis == "h" else (
            candidate.bbox[1],
            candidate.bbox[3],
        )
        best = float("inf")
        for start, end in zip(
            room.polygon,
            np.roll(room.polygon, -1, axis=0),
            strict=True,
        ):
            dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
            edge_axis = "h" if dx >= dy else "v"
            if edge_axis != candidate.axis or min(dx, dy) > 0.09 * max(dx, dy):
                continue
            edge_along = (
                sorted((start[0], end[0]))
                if edge_axis == "h"
                else sorted((start[1], end[1]))
            )
            overlap = min(along[1], edge_along[1]) - max(along[0], edge_along[0])
            if overlap < 0.65 * candidate.width_px:
                continue
            fixed = (start[1] + end[1]) / 2 if edge_axis == "h" else (
                start[0] + end[0]
            ) / 2
            candidate_fixed = candidate.center[1] if edge_axis == "h" else candidate.center[0]
            best = min(best, abs(fixed - candidate_fixed))
        return best

    # A door can open across a narrow unlabelled swing pocket. In that case
    # local mask probing reaches a room only at the jamb corner and assigns
    # the wrong side; parallel span overlap identifies the room it actually serves.
    for room_index, room in enumerate(rooms):
        if room_index in accessed:
            continue
        for candidate_index, candidate in enumerate(out):
            if (
                candidate.axis not in {"h", "v"}
                or candidate.kind_hint == "window"
                or not all(isinstance(side, int) for side in candidate.connects)
            ):
                continue
            distance = parallel_edge_distance(room, candidate)
            if distance > 1.25 * candidate.width_px:
                continue
            side_distances = [
                parallel_edge_distance(rooms[side], candidate)
                for side in candidate.connects
                if isinstance(side, int)
            ]
            local = min(range(2), key=side_distances.__getitem__)
            stale = 1 - local
            if (
                side_distances[local] > 2.5 * walls.thickness_px
                or side_distances[stale] <= 1.25 * candidate.width_px
            ):
                continue
            updated_connects = list(candidate.connects)
            updated_connects[stale] = room_index
            out[candidate_index] = replace(
                candidate,
                connects=(updated_connects[0], updated_connects[1]),
            )
            accessed.add(room_index)
            break

    neighbours: dict[int, set[int]] = {index: set() for index in range(len(rooms))}
    outside: set[int] = set()
    for candidate in out:
        sides = [side for side in candidate.connects if isinstance(side, int)]
        if EXTERIOR in candidate.connects:
            outside.update(side for side in sides if gives_access(candidate, side))
        if len(sides) == 2 and all(gives_access(candidate, side) for side in sides):
            neighbours[sides[0]].add(sides[1])
            neighbours[sides[1]].add(sides[0])
    second_door_partner: dict[int, int] = {}
    for room_index, connected in neighbours.items():
        if len(connected) != 1:
            continue
        partner = next(iter(connected))
        if (
            room_index < partner
            and neighbours[partner] == {room_index}
            and not {room_index, partner} & outside
            and not rooms[room_index].zone_bounded
            and not rooms[partner].zone_bounded
        ):
            larger = max((room_index, partner), key=lambda index: rooms[index].area_px)
            second_door_partner[larger] = partner if larger == room_index else room_index

    for room_index, room in enumerate(rooms):
        width, height = np.ptp(room.polygon, axis=0)
        compact_fixture = (
            room.name is None
            and room.area_px < 0.01 * total_area
            and max(width, height) <= 10.0 * walls.thickness_px
        )
        needs_second_door = room_index in second_door_partner
        if (room_index in accessed and not needs_second_door) or compact_fixture:
            continue

        choices: list[
            tuple[tuple[bool, bool, bool, bool, float], OpeningCandidate, WallSegment]
        ] = []
        for start, end in zip(
            room.polygon,
            np.roll(room.polygon, -1, axis=0),
            strict=True,
        ):
            dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
            if max(dx, dy) < min_len or min(dx, dy) > 0.09 * max(dx, dy):
                continue
            edge = _center_open_segment(
                WallSegment(
                    (float(start[0]), float(start[1])),
                    (float(end[0]), float(end[1])),
                    walls.thickness_px,
                    (room_index, UNKNOWN),
                ),
                room_masks,
            )
            for candidate in _scan_segment(
                -1,
                edge,
                walls,
                room_masks,
                silhouette,
                min_len,
                max_len,
                bgr=bgr,
            ):
                if room_index not in candidate.connects:
                    continue
                kind = "window" if _looks_like_window(candidate, walls, bgr) else "doorlike"
                candidate = replace(candidate, kind_hint=kind)
                internal = all(isinstance(side, int) for side in candidate.connects)
                if (
                    candidate.width_px > MIN_LEAF_GAP_THICKNESSES * walls.thickness_px
                    and (
                        candidate.width_px <= max_leaf_px
                        or (
                            internal
                            and candidate.width_px
                            <= MAX_SINGLE_DOOR_THICKNESSES * walls.thickness_px
                        )
                    )
                    and (kind == "doorlike" or internal)
                ):
                    arc = _detect_arc(candidate, walls, room_masks, bgr, binary)
                    if arc is not None:
                        candidate = _fit_stroked_leaf(replace(candidate, arc=arc), edge)
                if candidate.arc is None and candidate.kind_hint in {"doorlike", "window"}:
                    edge_along = sorted(
                        (edge.start[0], edge.end[0])
                        if candidate.axis == "h"
                        else (edge.start[1], edge.end[1])
                    )
                    lo, hi = (
                        (candidate.bbox[0], candidate.bbox[2])
                        if candidate.axis == "h"
                        else (candidate.bbox[1], candidate.bbox[3])
                    )
                    leaf = candidate.width_px - 1.0
                    sibling_spans = ((lo - leaf, lo), (hi, hi + leaf))
                    for sibling_lo, sibling_hi in sibling_spans:
                        if (
                            sibling_lo < edge_along[0] - 1.5 * walls.thickness_px
                            or sibling_hi > edge_along[1] + 1.5 * walls.thickness_px
                            or min(
                                abs(sibling_lo - edge_along[0]),
                                abs(sibling_hi - edge_along[1]),
                            )
                            > 1.5 * walls.thickness_px
                        ):
                            continue
                        x0, y0, x1, y1 = candidate.bbox
                        sibling_bbox = (
                            (sibling_lo, y0, sibling_hi, y1)
                            if candidate.axis == "h"
                            else (x0, sibling_lo, x1, sibling_hi)
                        )
                        sibling = replace(
                            candidate,
                            bbox=sibling_bbox,
                            center=(
                                (sibling_bbox[0] + sibling_bbox[2]) / 2,
                                (sibling_bbox[1] + sibling_bbox[3]) / 2,
                            ),
                        )
                        sibling_arc = _detect_arc(
                            sibling, walls, room_masks, bgr, binary
                        )
                        if sibling_arc is None:
                            continue
                        bbox = (
                            min(x0, sibling_bbox[0]),
                            min(y0, sibling_bbox[1]),
                            max(x1, sibling_bbox[2]),
                            max(y1, sibling_bbox[3]),
                        )
                        candidate = replace(
                            candidate,
                            bbox=bbox,
                            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                            width_px=2.0 * candidate.width_px - 1.0,
                            kind_hint="double_door",
                            arc=sibling_arc,
                        )
                        break
                if _looks_like_solid_wall(
                    candidate, bgr, candidate.kind_hint, walls.thickness_px
                ) or _opening_over_furniture(candidate, bgr):
                    continue
                if candidate.arc is None and candidate.kind_hint == "doorlike":
                    arc = _detect_double_arc(candidate, walls, room_masks, binary)
                    if arc is not None:
                        candidate = replace(candidate, kind_hint="double_door", arc=arc)
                known_pair = all(isinstance(side, int) for side in candidate.connects)
                narrow_passage = (
                    candidate.arc is None
                    and candidate.kind_hint == "doorlike"
                    and known_pair
                    and candidate.width_px >= max(min_len, walls.thickness_px)
                )
                if narrow_passage:
                    candidate = replace(candidate, kind_hint="passage")
                walkable = (
                    candidate.arc is not None
                    or 2.5 * walls.thickness_px
                    <= candidate.width_px
                    <= MAX_DOOR_THICKNESSES * walls.thickness_px
                    or narrow_passage
                )
                if (
                    not walkable
                    or candidate.connects[0] == candidate.connects[1]
                    or (UNKNOWN in candidate.connects and candidate.arc is None)
                    or (EXTERIOR in candidate.connects and candidate.kind_hint == "window")
                ):
                    continue
                if needs_second_door:
                    if (
                        candidate.arc is None
                        or second_door_partner[room_index] in candidate.connects
                    ):
                        continue
                    filtered_connects = tuple(
                        side
                        if not isinstance(side, int)
                        or side == room_index
                        or parallel_edge_distance(rooms[side], candidate)
                        <= 2.5 * walls.thickness_px
                        else UNKNOWN
                        for side in candidate.connects
                    )
                    candidate = replace(
                        candidate,
                        connects=(filtered_connects[0], filtered_connects[1]),
                        kind_hint="doorlike",
                    )
                known_pair = all(isinstance(side, int) for side in candidate.connects)
                score = (
                    candidate.arc is not None,
                    any(
                        isinstance(side, int) and side != room_index and side in accessed
                        for side in candidate.connects
                    ),
                    known_pair,
                    candidate.kind_hint != "window",
                    -abs(candidate.width_px / walls.thickness_px - 4.5),
                )
                choices.append((score, candidate, edge))
        if not choices:
            continue

        _score, candidate, edge = max(choices, key=lambda item: item[0])
        axis = candidate.axis
        along = (candidate.bbox[0], candidate.bbox[2]) if axis == "h" else (
            candidate.bbox[1],
            candidate.bbox[3],
        )
        containing: list[tuple[float, int, WallSegment]] = []
        partial = False
        for index, segment in enumerate(hosts):
            if room_index not in segment.rooms:
                continue
            segment_axis = "h" if abs(segment.end[0] - segment.start[0]) >= abs(
                segment.end[1] - segment.start[1]
            ) else "v"
            if segment_axis != axis:
                continue
            fixed = (
                (segment.start[1] + segment.end[1]) / 2
                if axis == "h"
                else (segment.start[0] + segment.end[0]) / 2
            )
            candidate_fixed = candidate.center[1] if axis == "h" else candidate.center[0]
            if abs(fixed - candidate_fixed) > 2.5 * walls.thickness_px:
                continue
            segment_along = (
                sorted((segment.start[0], segment.end[0]))
                if axis == "h"
                else sorted((segment.start[1], segment.end[1]))
            )
            overlap = min(along[1], segment_along[1]) - max(along[0], segment_along[0])
            partial |= overlap >= -2.0 * walls.thickness_px
            if segment_along[0] <= along[0] and segment_along[1] >= along[1]:
                containing.append((abs(fixed - candidate_fixed), index, segment))

        zones = _zone_mask(walls)
        zone_boundary = zones is not None and _ink_share(
            edge,
            cv2.dilate(
                zones,
                np.ones(
                    (2 * max(1, round(walls.thickness_px)) + 1,) * 2,
                    np.uint8,
                ),
            ),
        ) >= 0.50
        if zone_boundary and candidate.arc is None:
            candidate = replace(candidate, kind_hint="passage", wall_index=-1)
        elif containing:
            _, wall_index, host = min(containing)
            fixed = (
                (host.start[1] + host.end[1]) / 2
                if axis == "h"
                else (host.start[0] + host.end[0]) / 2
            )
            x0, y0, x1, y1 = candidate.bbox
            bbox = (
                (x0, fixed - host.thickness_px / 2, x1, fixed + host.thickness_px / 2)
                if axis == "h"
                else (fixed - host.thickness_px / 2, y0, fixed + host.thickness_px / 2, y1)
            )
            candidate = replace(
                candidate,
                bbox=bbox,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                wall_index=wall_index,
            )
        else:
            if partial:
                x0, y0, x1, y1 = candidate.bbox
                host = WallSegment(
                    ((x0, candidate.center[1]) if axis == "h" else (candidate.center[0], y0)),
                    ((x1, candidate.center[1]) if axis == "h" else (candidate.center[0], y1)),
                    walls.thickness_px,
                    candidate.connects,
                )
            else:
                host = replace(edge, rooms=candidate.connects)
            hosts.append(host)
            candidate = replace(candidate, wall_index=len(hosts) - 1)
        out.append(candidate)
        accessed.add(room_index)
    return out, hosts


def _resolve_unknown_adjacencies(
    candidates: list[OpeningCandidate],
    rooms: list[RoomDraft],
    thickness: float,
    *,
    exterior_rooms: set[int] | None = None,
) -> list[OpeningCandidate]:
    """Fill an unknown side only when one nearby room uniquely lies opposite."""
    polygons = [Polygon(room.polygon) if len(room.polygon) >= 3 else None for room in rooms]
    out: list[OpeningCandidate] = []
    for candidate in candidates:
        if candidate.axis == "d":
            out.append(candidate)
            continue
        known = [side for side in candidate.connects if isinstance(side, int)]
        if UNKNOWN not in candidate.connects or len(known) != 1 or not 0 <= known[0] < len(rooms):
            out.append(candidate)
            continue
        current = known[0]
        coordinate = 1 if candidate.axis == "h" else 0
        current_side = rooms[current].seed[coordinate] - candidate.center[coordinate]
        opening = box(*candidate.bbox)
        nearby = sorted(
            (
                opening.distance(polygon),
                index,
            )
            for index, polygon in enumerate(polygons)
            if index != current
            and polygon is not None
            and current_side * (rooms[index].seed[coordinate] - candidate.center[coordinate]) < 0
        )
        neighbour = None
        if nearby and nearby[0][0] <= 1.5 * thickness:
            close = [item for item in nearby if item[0] <= 1.5 * thickness]
            internal = (
                [index for _, index in close if index not in exterior_rooms]
                if exterior_rooms is not None
                else []
            )
            if len(nearby) == 1 or nearby[1][0] > nearby[0][0] + 0.5 * thickness:
                neighbour = nearby[0][1]
            else:
                tied = [item for item in nearby if item[0] <= nearby[0][0] + 0.5 * thickness]
                circulation = [
                    index for _, index in tied if rooms[index].room_type in {"hallway", "entrance"}
                ]
                if len(circulation) == 1:
                    neighbour = circulation[0]
            # A recovered door can open into a small unclaimed landing. Its
            # closest polygon may be a facade bedroom/bathroom around the
            # corner; the only nearby non-facade room is the circulation
            # space that owns the landing.
            if len(internal) == 1:
                neighbour = internal[0]
        if neighbour is not None:
            connects = (
                neighbour if candidate.connects[0] == UNKNOWN else candidate.connects[0],
                neighbour if candidate.connects[1] == UNKNOWN else candidate.connects[1],
            )
            candidate = replace(candidate, connects=connects)
        out.append(candidate)
    return out


_OPEN_ZONE_TYPES = {
    "living_room",
    "living_dining",
    "dining_room",
    "kitchen",
    "hallway",
    "entrance",
    "study",
    "closet",
    "storage",
    "multipurpose",
}


def find_zone_passages(
    walls: WallExtraction,
    rooms: list[RoomDraft],
    segments: list[WallSegment],
) -> list[OpeningCandidate]:
    """Turn drawn dashed room-zone boundaries into explicit passages.

    Room extraction deliberately seals these dividers so each labelled zone
    can be measured, while wall extraction deliberately removes them because
    they are not structure.  Preserve the missing topological fact here: the
    two compatible zones remain openly connected.
    """
    zones = _zone_mask(walls)
    if zones is None:
        return []
    masks = _room_masks(rooms, zones.shape)
    radius = max(3, int(round(1.5 * walls.thickness_px)))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    near = [cv2.dilate(mask, kernel) if mask is not None else None for mask in masks]
    track_hosts = {
        frozenset((a, b)): index
        for index, segment in enumerate(segments)
        for a, b in [segment.rooms]
        if isinstance(a, int)
        and isinstance(b, int)
        and a != b
        and _is_zone_track(segment, walls)
    }
    physical_pairs = {
        frozenset((a, b))
        for segment in segments
        for a, b in [segment.rooms]
        if isinstance(a, int) and isinstance(b, int) and a != b
    } - track_hosts.keys()
    min_length = max(12, int(round(2 * walls.thickness_px)))
    passages: list[OpeningCandidate] = []
    for i, room in enumerate(rooms):
        near_i = near[i]
        if near_i is None:
            continue
        for j in range(i + 1, len(rooms)):
            near_j = near[j]
            if (
                near_j is None
                or not (room.zone_bounded or rooms[j].zone_bounded)
                or frozenset((i, j)) in physical_pairs
                or not _open_zone_types_compatible(room, rooms[j], walls.thickness_px)
            ):
                continue
            bridge = cv2.bitwise_and(zones, cv2.bitwise_and(near_i, near_j))
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
                bridge, connectivity=8
            )
            choices = [
                k
                for k in range(1, count)
                if max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) >= min_length
            ]
            if not choices:
                continue
            k = max(
                choices,
                key=lambda n: max(stats[n, cv2.CC_STAT_WIDTH], stats[n, cv2.CC_STAT_HEIGHT]),
            )
            x, y, width, height = (int(v) for v in stats[k, :4])
            axis = "h" if width >= height else "v"
            track_host = track_hosts.get(frozenset((i, j)))
            passages.append(
                OpeningCandidate(
                    marker=f"zone_{i + 1}_{j + 1}",
                    bbox=(float(x), float(y), float(x + width), float(y + height)),
                    center=(float(centroids[k, 0]), float(centroids[k, 1])),
                    axis=axis,
                    width_px=float(max(width, height)),
                    kind_hint="sliding_door" if track_host is not None else "doorlike",
                    connects=(i, j),
                    wall_index=track_host if track_host is not None else -1,
                    arc=None,
                )
            )
    # ponytail: room count is bounded, so pairwise trimming is cheaper than
    # building a sweep-line just to divide one dashed T-junction.
    for i, left in enumerate(passages):
        for j in range(i + 1, len(passages)):
            right = passages[j]
            if left.axis != right.axis:
                continue
            along = (0, 2) if left.axis == "h" else (1, 3)
            perpendicular = 1 if left.axis == "h" else 0
            if (
                abs(left.center[perpendicular] - right.center[perpendicular])
                > 1.5 * walls.thickness_px
            ):
                continue
            overlap_lo = max(left.bbox[along[0]], right.bbox[along[0]])
            overlap_hi = min(left.bbox[along[1]], right.bbox[along[1]])
            if overlap_hi <= overlap_lo:
                continue
            split = (overlap_lo + overlap_hi) / 2
            first, second = (left, right)
            first_index, second_index = (i, j)
            if left.center[along[0]] > right.center[along[0]]:
                first, second = right, left
                first_index, second_index = j, i
            first_box, second_box = list(first.bbox), list(second.bbox)
            first_box[along[1]], second_box[along[0]] = split, split
            for index, candidate, bbox in (
                (first_index, first, first_box),
                (second_index, second, second_box),
            ):
                passages[index] = replace(
                    candidate,
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    center=(
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2,
                    ),
                    width_px=bbox[along[1]] - bbox[along[0]],
                )
            left, right = passages[i], passages[j]
    return passages


def _open_zone_types_compatible(left: RoomDraft, right: RoomDraft, thickness: float) -> bool:
    for room in (left, right):
        if room.room_type in {"closet", "storage", "unknown_space"}:
            width, height = np.ptp(room.polygon, axis=0)
            if min(width, height) < 4.0 * thickness:
                return False  # pipe/utility shaft, not walkable floor
    types = {left.room_type, right.room_type}
    if left.room_type == right.room_type == "bathroom":  # wet/dry bathroom zones
        return True
    if left.room_type == right.room_type == "bedroom":
        return False
    if "unknown_space" in types:
        unknown = left if left.room_type == "unknown_space" else right
        # An unlabelled region bounded by a dashed divider is circulation,
        # while a named unknown (管道/烟道/井) must not become walkable.
        return unknown.name is None and "bedroom" not in types
    if "bedroom" in types:
        # A long dashed divider is still physically open even when the plan's
        # bedroom layout is unconventional; habitability is warned downstream.
        return bool((types - {"bedroom"}) & (_OPEN_ZONE_TYPES | {"living_room"}))
    return types.issubset(_OPEN_ZONE_TYPES | {"bathroom"})


# ------------------------------------------------------------- wall scanning


def _scan_segment(
    seg_index: int,
    seg: WallSegment,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    silhouette: np.ndarray,
    min_len: float,
    max_len: float,
    *,
    bgr: np.ndarray | None = None,
) -> list[OpeningCandidate]:
    if seg.rooms[0] == seg.rooms[1]:
        # A stub the same room wraps around: there is nothing on either side
        # to connect, so any break in it is the stub's own end, not a door.
        return []
    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    dx, dy = sx1 - sx0, sy1 - sy0
    diagonal = not (abs(dx) < abs(dy) * 0.2 or abs(dy) < abs(dx) * 0.2)
    axis = ("h" if abs(dx) >= abs(dy) else "v") if not diagonal else "d"
    if diagonal and not (
        (all(isinstance(side, int) for side in seg.rooms) and seg.rooms[0] != seg.rooms[1])
        or EXTERIOR in seg.rooms
    ):
        return []

    solid = walls.solid
    h, w = solid.shape
    t = seg.thickness_px
    band = max(3, int(round(1.3 * t)))  # covers the wall body on either side

    if axis == "h":
        c = int(round((sy0 + sy1) / 2))
        lo, hi = int(round(min(sx0, sx1))), int(round(max(sx0, sx1)))
        lo, hi = max(0, lo), min(w - 1, hi)
        solid_strip = solid[max(0, c - band) : min(h, c + band + 1), lo : hi + 1]
        line_strip = walls.lines[max(0, c - band) : min(h, c + band + 1), lo : hi + 1]
    elif axis == "v":
        c = int(round((sx0 + sx1) / 2))
        lo, hi = int(round(min(sy0, sy1))), int(round(max(sy0, sy1)))
        lo, hi = max(0, lo), min(h - 1, hi)
        solid_strip = solid[lo : hi + 1, max(0, c - band) : min(w, c + band + 1)].T
        line_strip = walls.lines[lo : hi + 1, max(0, c - band) : min(w, c + band + 1)].T
    else:
        solid_strip = _diagonal_strip(solid, seg, band)
        line_strip = _diagonal_strip(walls.lines, seg, band)
        lo, hi, c = 0, solid_strip.shape[1] - 1, 0
    coverage = solid_strip.any(axis=0)
    assert isinstance(coverage, np.ndarray)
    if coverage.size == 0:
        return []

    connects = _resolve_connects(seg, walls, room_masks, silhouette)
    # Windows do not always interrupt the wall: some plans draw the window
    # box ON a continuous wall run. Multi-stroke intervals in the lines mask
    # are the second, independent opening signal.
    window_runs = _multi_stroke_runs(line_strip, min_len)
    runs = _opening_runs(coverage, window_runs, min_len, max_len, axis != "d")
    seg_len = float(np.hypot(dx, dy))
    if axis == "d" and float(coverage.mean()) < 0.2:
        runs = window_runs
    elif axis != "d":
        neutral_wide = any(
            run_hi - run_lo + 1 >= 0.85 * seg_len for run_lo, run_hi in runs
        )
        if (
            neutral_wide
            and bgr is not None
            and not window_runs
            and UNKNOWN not in connects
        ):
            if axis == "h":
                color_strip = bgr[
                    max(0, c - band) : min(h, c + band + 1), lo : hi + 1
                ]
            else:
                color_strip = bgr[
                    lo : hi + 1, max(0, c - band) : min(w, c + band + 1)
                ].transpose(1, 0, 2)
            mean = color_strip.mean(axis=2)
            chroma = color_strip.max(axis=2) - color_strip.min(axis=2)
            wall_tone = (chroma < MAX_CHROMA) & (
                mean <= max(high for _, high in walls.bands)
            )
            neutral_coverage = wall_tone.mean(axis=0) >= 0.12
            refined_coverage = np.logical_or(coverage, neutral_coverage)
            runs = _opening_runs(
                refined_coverage,
                window_runs,
                min_len,
                max_len,
                True,
            )
        wide_gap = any(run_hi - run_lo + 1 >= 0.65 * seg_len for run_lo, run_hi in runs)
        if wide_gap and _ink_share(seg, walls.lines) >= 0.5:
            # A thin partition can be absent from ``solid`` while remaining clear
            # in ``lines``. Re-scan that apparent whole-wall opening against both
            # masks; true door/window intervals remain as gaps or parallel strokes.
            refined_coverage = np.logical_or(solid_strip, line_strip).any(axis=0)
            refined = _opening_runs(
                refined_coverage,
                window_runs,
                min_len,
                max_len,
                True,
            )
            if refined:
                runs = refined

    out = []
    for run_lo, run_hi in runs:
        a, b = lo + run_lo, lo + run_hi
        center_along = (a + b) / 2
        if axis == "h":
            bbox = (float(a), float(c - t / 2), float(b), float(c + t / 2))
            center = (center_along, float(c))
        elif axis == "v":
            bbox = (float(c - t / 2), float(a), float(c + t / 2), float(b))
            center = (float(c), center_along)
        else:
            start = np.asarray(seg.start, dtype=np.float64)
            direction = np.asarray((dx, dy), dtype=np.float64) / max(seg_len, 1e-6)
            normal = np.asarray((-direction[1], direction[0]))
            pa, pb = start + direction * run_lo, start + direction * run_hi
            corners = np.vstack(
                (pa + normal * t / 2, pa - normal * t / 2, pb + normal * t / 2, pb - normal * t / 2)
            )
            bbox = (
                float(corners[:, 0].min()),
                float(corners[:, 1].min()),
                float(corners[:, 0].max()),
                float(corners[:, 1].max()),
            )
            midpoint = (pa + pb) / 2
            center = (float(midpoint[0]), float(midpoint[1]))
        # One segment can have different neighbours along its length (a wall
        # that runs past two rooms), so each opening asks at its own centre
        # rather than inheriting the midpoint's answer.
        width = float(run_hi - run_lo + 1)
        here = (
            connects
            if all(isinstance(side, int) for side in connects)
            else _resolve_connects(seg, walls, room_masks, silhouette, center)
        )
        out.append(
            OpeningCandidate(
                marker="",
                bbox=bbox,
                center=center,
                axis=axis,
                width_px=width,
                kind_hint=(
                    "window"
                    if any(
                        min(run_hi, window_hi) - max(run_lo, window_lo)
                        >= 0.5 * min(run_hi - run_lo + 1, window_hi - window_lo + 1)
                        for window_lo, window_hi in window_runs
                    )
                    else "doorlike"
                ),
                connects=here,
                wall_index=seg_index,
                arc=None,
            )
        )
    return out


def _diagonal_strip(mask: np.ndarray, seg: WallSegment, band: int) -> np.ndarray:
    """Sample a wall-aligned strip: rows cross the wall, columns run along it."""
    start = np.asarray(seg.start, dtype=np.float64)
    delta = np.asarray(seg.end, dtype=np.float64) - start
    length = float(np.linalg.norm(delta))
    if length == 0:
        return np.empty((0, 0), dtype=mask.dtype)
    direction = delta / length
    normal = np.asarray((-direction[1], direction[0]))
    along = np.linspace(0.0, length, max(2, int(round(length)) + 1))
    offsets = np.arange(-band, band + 1, dtype=np.float64)
    map_x = (start[0] + normal[0] * offsets[:, None] + direction[0] * along).astype(np.float32)
    map_y = (start[1] + normal[1] * offsets[:, None] + direction[1] * along).astype(np.float32)
    return cv2.remap(
        mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _multi_stroke_runs(strip: np.ndarray, min_len: float) -> list[tuple[int, int]]:
    """Intervals along the segment where ≥2 separated strokes run in the
    wall band — the window signature, valid even on unbroken walls."""
    strokes = (strip > 0).astype(np.int8)
    transitions = (np.diff(strokes, axis=0) == 1).sum(axis=0) + strokes[0]
    multi = (transitions >= 2).astype(np.int8)
    out = []
    for start, end, value in _runs(multi):
        if value == 1 and end - start + 1 >= min_len:
            out.append((start, end))
    return out


def _merge_runs(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of two interval lists, coalescing overlaps."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(a + b):
        if merged and lo <= merged[-1][1] + 2:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _opening_runs(
    coverage: np.ndarray,
    window_runs: list[tuple[int, int]],
    min_len: float,
    max_len: float,
    include_wide_gaps: bool,
) -> list[tuple[int, int]]:
    """Use a measured wall gap for extent; window strokes only prove type."""
    gaps = _absent_runs(coverage, min_len, max_len)
    all_gaps = (
        _absent_runs(coverage, min_len, float(coverage.size)) if include_wide_gaps else gaps
    )
    sized_windows: list[tuple[int, int]] = []
    for window in window_runs:
        matches: list[tuple[int, int]] = []
        for gap in all_gaps:
            overlap = min(window[1], gap[1]) - max(window[0], gap[0]) + 1
            if overlap >= 0.5 * min(window[1] - window[0] + 1, gap[1] - gap[0] + 1):
                matches.append(gap)
        if matches:
            sized_windows.append((min(gap[0] for gap in matches), max(gap[1] for gap in matches)))
        elif not any(gap[0] <= window[1] + 2 and window[0] <= gap[1] + 2 for gap in all_gaps):
            sized_windows.append(window)
    return _merge_runs(gaps, sized_windows)


def _absent_runs(coverage: np.ndarray, min_len: float, max_len: float) -> list[tuple[int, int]]:
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
    silhouette: np.ndarray,
    at: tuple[float, float] | None = None,
) -> tuple[int | str, int | str]:
    """Room indices pass through unless ``at`` disproves a stale paired edge;
    the non-room side of a leftover edge is resolved by probing beyond the
    wall at ``at`` (the segment's midpoint by default, an opening's own centre
    when one is being classified).

    The far side is whatever floor is actually there. A ROOM behind the wall
    is the answer even when the two polygons' edges never paired — offset
    faces, a chamfered neighbour, a wall thicker than the pairing window all
    leave a one-room leftover edge, and calling its far side "unknown" cuts
    the doorway's link. The plan then falls apart into groups of rooms with
    no path between them (12 such groups on the corpus, one plan split 7/7),
    which no dwelling can be.

    Failing that, outside-ness is the filled wall silhouette, never its
    bounding box: an L-shaped, notched or diagonal footprint keeps most of
    its facade well inside that box, so a box test called the open air behind
    those walls "unknown" (33 of fp6's 49 walls). Either mislabel also
    reaches the classifier prompt as the wrong adjacency prior — the
    strongest one it has.
    """
    a, b = seg.rooms
    if isinstance(a, int) and isinstance(b, int):
        if at is None:
            return (a, b)
        delta = np.subtract(seg.end, seg.start)
        length = float(np.linalg.norm(delta))
        normal = np.array([-delta[1], delta[0]]) / max(length, 1e-6)
        seen: set[int] = set()
        for reach in (0.75, 1.5, 2.5, 4.0):
            pair = tuple(
                _room_at(
                    room_masks,
                    tuple(np.asarray(at) + sign * normal * reach * seg.thickness_px),
                )
                for sign in (-1, 1)
            )
            if set(pair) == {a, b}:
                return (a, b)
            seen.update(side for side in pair if side is not None)
        seen_room = a if a in seen else b if b in seen else None
        if seen_room is None:
            return (a, b)
        return (seen_room, UNKNOWN) if seen_room == a else (UNKNOWN, seen_room)
    else:
        room = a if isinstance(a, int) else b
    mask = room_masks[room] if isinstance(room, int) else None
    (sx0, sy0), (sx1, sy1) = seg.start, seg.end
    if at is None:
        at = ((sx0 + sx1) / 2, (sy0 + sy1) / 2)
    outward = _room_outward_normal(seg, mask)
    far: int | str = UNKNOWN
    if outward is not None:
        # The room polygons trace inner faces, so the neighbour's floor
        # starts about one wall thickness out; look no further, or a probe
        # jumps a narrow closet into the room beyond it.
        for reach in (1.5, 2.5):
            probe = (
                at[0] + outward[0] * reach * seg.thickness_px,
                at[1] + outward[1] * reach * seg.thickness_px,
            )
            neighbour = _room_at(room_masks, probe)
            if neighbour is not None and neighbour != room:
                far = neighbour
                break
        else:
            fx = at[0] + outward[0] * 2.5 * seg.thickness_px
            fy = at[1] + outward[1] * 2.5 * seg.thickness_px
            xi, yi = int(round(fx)), int(round(fy))
            h, w = silhouette.shape
            inside_building = 0 <= yi < h and 0 <= xi < w and silhouette[yi, xi] > 0
            far = UNKNOWN if inside_building else EXTERIOR
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
        if (room.source == "vlm" and not room.spatially_grounded) or len(room.polygon) < 3:
            masks.append(None)
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(room.polygon).astype(np.int32)], 255)
        masks.append(mask)
    return masks


def _looks_like_window(
    candidate: OpeningCandidate,
    walls: WallExtraction,
    bgr: np.ndarray | None = None,
) -> bool:
    """Windows carry 2-3 thin strokes ALONG the wall axis across the break
    (legend cue B28). Count distinct stroke rows/columns in the lines mask.
    """
    if candidate.axis == "d":
        return candidate.kind_hint == "window"
    x0, y0, x1, y1 = (int(round(v)) for v in candidate.bbox)
    h, w = walls.lines.shape
    pad = 2
    crop = walls.lines[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]
    if crop.size == 0:
        return False
    # per-row (h) / per-column (v) fill fraction along the wall direction
    coverage = (crop > 0).mean(axis=1 if candidate.axis == "h" else 0)
    stroke_rows = coverage >= 0.5
    runs = int(np.count_nonzero(np.diff(np.concatenate(([0], stroke_rows.view(np.int8)))) == 1))
    if runs >= 2 or bgr is None:
        return runs >= 2

    # Short windows can fall below the long-line extractor's minimum span.
    # Their three parallel neutral tracks are still explicit in the source.
    source = bgr[max(0, y0) : min(h, y1 + 1), max(0, x0) : min(w, x1 + 1)]
    if source.size == 0:
        return False
    fx0, fy0, fx1, fy1 = walls.footprint
    facade_distance = (
        min(abs(candidate.center[1] - fy0), abs(candidate.center[1] - fy1))
        if candidate.axis == "h"
        else min(abs(candidate.center[0] - fx0), abs(candidate.center[0] - fx1))
    )
    if facade_distance > 1.5 * walls.thickness_px and not (
        UNKNOWN in candidate.connects and _clear_normal_to_border(candidate, walls)
    ):
        return False
    track_pad = max(2, round(0.5 * walls.thickness_px))
    if candidate.axis == "h":
        track_source = bgr[
            max(0, y0 - track_pad) : min(h, y1 + track_pad + 1),
            max(0, x0) : min(w, x1 + 1),
        ]
        profile_axis = 1
    else:
        track_source = bgr[
            max(0, y0) : min(h, y1 + 1),
            max(0, x0 - track_pad) : min(w, x1 + track_pad + 1),
        ]
        profile_axis = 0
    track_ink = (
        (track_source.max(axis=2) - track_source.min(axis=2) < MAX_CHROMA)
        & (track_source.mean(axis=2) < 235)
    )
    track_profile = track_ink.mean(axis=profile_axis) >= 0.75
    starts = np.flatnonzero(np.diff(np.concatenate(([0], track_profile.view(np.int8)))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((track_profile.view(np.int8), [0]))) == -1)
    centers = [(start + end) / 2 for start, end in zip(starts, ends, strict=True)]
    if any(
        right - left <= 0.6 * walls.thickness_px
        for left, right in zip(centers, centers[1:], strict=False)
    ):
        return True
    neutral_ink = (
        (source.max(axis=2) - source.min(axis=2) < MAX_CHROMA)
        & (source.mean(axis=2) < 220)
    )
    coverage = neutral_ink.mean(axis=1 if candidate.axis == "h" else 0)
    tracks = coverage >= 0.75
    runs = int(
        np.count_nonzero(np.diff(np.concatenate(([0], tracks.view(np.int8)))) == 1)
    )
    return runs >= 3


def _paired_zone_door(
    candidate: OpeningCandidate,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    binary: np.ndarray,
) -> OpeningCandidate | None:
    """Recover paired swing leaves when their thin chord shortens a wide frame."""
    if walls.zones is None or candidate.axis not in {"h", "v"}:
        return None
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        walls.zones, connectivity=8
    )
    candidate_lo, candidate_hi = (
        (candidate.bbox[0], candidate.bbox[2])
        if candidate.axis == "h"
        else (candidate.bbox[1], candidate.bbox[3])
    )
    candidate_fixed = candidate.center[1] if candidate.axis == "h" else candidate.center[0]
    for index in range(1, count):
        x, y, width, height, _area = (int(value) for value in stats[index])
        axis = "h" if width >= height else "v"
        length = max(width, height)
        if (
            axis != candidate.axis
            or not 2.5 * walls.thickness_px
            <= length
            <= MAX_DOOR_THICKNESSES * walls.thickness_px
        ):
            continue
        low, high = (x, x + width - 1) if axis == "h" else (y, y + height - 1)
        fixed = float(centroids[index, 1] if axis == "h" else centroids[index, 0])
        if (
            min(high, candidate_hi) - max(low, candidate_lo) < 0.75 * length
            or abs(fixed - candidate_fixed) > walls.thickness_px
        ):
            continue
        x0, y0, x1, y1 = candidate.bbox
        bbox = (low, y0, high, y1) if axis == "h" else (x0, low, x1, high)
        probe = replace(
            candidate,
            bbox=bbox,
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            width_px=float(length),
        )
        arc = _detect_double_arc(probe, walls, room_masks, binary)
        if arc is not None:
            return replace(probe, kind_hint="double_door", arc=arc)
    return None


def _plain_internal_gap(candidate: OpeningCandidate, walls: WallExtraction) -> bool:
    """A room-to-room break with neither threshold stroke nor door-leaf ink."""
    if candidate.axis not in {"h", "v"} or not all(
        isinstance(side, int) for side in candidate.connects
    ):
        return False
    x0, y0, x1, y1 = (int(round(value)) for value in candidate.bbox)
    height, width = walls.lines.shape
    crop = walls.lines[
        max(0, y0 - 2) : min(height, y1 + 3),
        max(0, x0 - 2) : min(width, x1 + 3),
    ]
    if crop.size == 0:
        return False
    coverage = (crop > 0).mean(axis=1 if candidate.axis == "h" else 0)
    if (coverage >= 0.5).any():
        return False
    if walls.door_hints is None:
        return True
    radius = max(1, round(candidate.width_px))
    cx, cy = (int(round(value)) for value in candidate.center)
    hints = walls.door_hints[
        max(0, cy - radius) : min(height, cy + radius + 1),
        max(0, cx - radius) : min(width, cx + radius + 1),
    ]
    return not hints.any()


def _clear_normal_to_border(candidate: OpeningCandidate, walls: WallExtraction) -> bool:
    """Does an unresolved wall side face a concave part of the exterior?"""
    x0, y0, x1, y1 = (int(round(value)) for value in candidate.bbox)
    height, width = walls.union.shape
    x = int(np.clip(round(candidate.center[0]), 0, width - 1))
    y = int(np.clip(round(candidate.center[1]), 0, height - 1))
    if candidate.axis == "h":
        return not walls.union[: max(0, y0), x].any() or not walls.union[
            min(height, y1 + 1) :, x
        ].any()
    if candidate.axis == "v":
        return not walls.union[y, : max(0, x0)].any() or not walls.union[
            y, min(width, x1 + 1) :
        ].any()
    return False


def _looks_like_solid_wall(
    candidate: OpeningCandidate, bgr: np.ndarray, kind: str, thickness: float
) -> bool:
    """Reject continuous neutral wall fill that escaped the colour mask."""
    if candidate.axis == "d":
        return False
    x0, y0, x1, y1 = (int(round(v)) for v in candidate.bbox)
    crop = bgr[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
    if crop.size == 0:
        return False
    mean = crop.mean(axis=2)
    chroma = crop.max(axis=2) - crop.min(axis=2)
    neutral = float(((chroma < 12) & (mean < 235)).mean())
    dark = float(((chroma < 15) & (mean < 210)).mean())
    if candidate.arc is not None:
        return neutral >= 0.50 and dark >= 0.50
    if (
        kind == "window"
        and candidate.width_px > MAX_DOOR_THICKNESSES * thickness
        and neutral >= 0.65
        and dark >= 0.28
    ):
        return True
    threshold = 0.28 if kind == "doorlike" else 0.85
    return neutral >= threshold and dark >= threshold


def _opening_over_furniture(candidate: OpeningCandidate, bgr: np.ndarray) -> bool:
    """Reject an unresolved facade gap occupied entirely by furniture ink."""
    if UNKNOWN not in candidate.connects:
        return False
    x0, y0, x1, y1 = (round(value) for value in candidate.bbox)
    crop = bgr[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
    if crop.size == 0:
        return False
    mean = crop.mean(axis=2)
    chroma = crop.max(axis=2) - crop.min(axis=2)
    return bool(
        (mean > 245).mean() < 0.10
        and ((mean < 80).mean() >= 0.50 or (chroma > 30).mean() >= 0.30)
    )


def _recover_facade_candidate(
    candidate: OpeningCandidate,
    rooms: list[RoomDraft],
    walls: WallExtraction,
    min_room_area: float,
) -> OpeningCandidate:
    """Resolve a candidate that reaches the outer footprint as exterior."""
    fx0, fy0, fx1, fy1 = walls.footprint
    distance = min(
        candidate.center[0] - fx0,
        fx1 - candidate.center[0],
        candidate.center[1] - fy0,
        fy1 - candidate.center[1],
    )
    if distance > 1.25 * candidate.width_px and not (
        UNKNOWN in candidate.connects and _clear_normal_to_border(candidate, walls)
    ):
        return candidate
    if UNKNOWN in candidate.connects:
        weak_tint_arc = (
            candidate.kind_hint == "doorlike"
            and candidate.arc is not None
            and candidate.arc.radius_px is None
        )
        return replace(
            candidate,
            kind_hint="window" if weak_tint_arc else candidate.kind_hint,
            connects=(
                EXTERIOR if candidate.connects[0] == UNKNOWN else candidate.connects[0],
                EXTERIOR if candidate.connects[1] == UNKNOWN else candidate.connects[1],
            ),
            arc=None if weak_tint_arc else candidate.arc,
        )
    if candidate.kind_hint != "doorlike" or not all(
        isinstance(side, int) for side in candidate.connects
    ):
        return candidate
    undersized = [
        side
        for side in candidate.connects
        if isinstance(side, int) and rooms[side].area_px < min_room_area
    ]
    if len(undersized) != 1:
        return candidate
    return replace(
        candidate,
        kind_hint="window",
        connects=(
            EXTERIOR if candidate.connects[0] == undersized[0] else candidate.connects[0],
            EXTERIOR if candidate.connects[1] == undersized[0] else candidate.connects[1],
        ),
    )


def _collapse_thin_regions(
    candidates: list[OpeningCandidate], rooms: list[RoomDraft], walls: WallExtraction
) -> list[OpeningCandidate]:
    """Collapse window-frame/furniture voids into at most one facade opening."""
    total_area = sum(room.area_px for room in rooms)
    fixture: dict[int, tuple[float, str, float]] = {}
    fx0, fy0, fx1, fy1 = walls.footprint
    for index, room in enumerate(rooms):
        lo = np.min(room.polygon, axis=0)
        hi = np.max(room.polygon, axis=0)
        width, height = hi - lo
        edge_distance = min(lo[0] - fx0, lo[1] - fy0, fx1 - hi[0], fy1 - hi[1])
        if (
            room.area_px < 0.04 * total_area
            and min(width, height) < 4.0 * walls.thickness_px
            and _boundary_fraction(room, walls.solid) < 0.5
        ):
            fixture[index] = (
                max(width, height),
                "h" if width >= height else "v",
                edge_distance,
            )
    if not fixture:
        return candidates

    dropped: set[int] = set()
    replacements: dict[int, OpeningCandidate] = {}
    for room_index, (length, axis, edge_distance) in fixture.items():
        touching = [
            i
            for i, candidate in enumerate(candidates)
            if room_index in candidate.connects
        ]
        if not touching:
            continue
        long_faces = [
            i
            for i in touching
            if candidates[i].axis == axis and candidates[i].width_px >= 0.6 * length
        ]
        outer = [i for i in long_faces if EXTERIOR in candidates[i].connects]
        inner = [
            i
            for i in long_faces
            if any(
                isinstance(side, int) and side != room_index and side not in fixture
                for side in candidates[i].connects
            )
        ]
        if outer and inner:
            outer_index = max(outer, key=lambda i: candidates[i].width_px)
            inner_candidate = candidates[max(inner, key=lambda i: candidates[i].width_px)]
            real_room = next(
                side
                for side in inner_candidate.connects
                if isinstance(side, int) and side != room_index and side not in fixture
            )
            candidate = candidates[outer_index]
            replacements[outer_index] = replace(
                candidate,
                connects=(
                    real_room if candidate.connects[0] == room_index else candidate.connects[0],
                    real_room if candidate.connects[1] == room_index else candidate.connects[1],
                ),
            )
            dropped.update(i for i in touching if i != outer_index)
        elif edge_distance > 3.0 * walls.thickness_px:
            dropped.update(touching)
    return [
        replacements.get(i, candidate)
        for i, candidate in enumerate(candidates)
        if i not in dropped
    ]


def _recover_parallel_swing(
    candidates: list[OpeningCandidate], thickness: float
) -> list[OpeningCandidate]:
    """Move an arc from an offset door-leaf line to the real threshold."""
    recovered = list(candidates)
    dropped: set[int] = set()
    for i, threshold in enumerate(recovered):
        if threshold.arc is not None or threshold.axis not in {"h", "v"}:
            continue
        known = {side for side in threshold.connects if isinstance(side, int)}
        if len(known) < 2:
            continue
        for j, leaf in enumerate(recovered):
            if i == j or leaf.arc is None or leaf.axis != threshold.axis:
                continue
            leaf_known = {side for side in leaf.connects if isinstance(side, int)}
            if len(leaf_known) >= len(known) or not known & leaf_known:
                continue
            coordinate = 0 if threshold.axis == "h" else 1
            perpendicular = 1 - coordinate
            t0, t1 = threshold.bbox[coordinate], threshold.bbox[coordinate + 2]
            l0, l1 = leaf.bbox[coordinate], leaf.bbox[coordinate + 2]
            overlap = min(t1, l1) - max(t0, l0)
            if (
                overlap < 0.7 * min(t1 - t0, l1 - l0)
                or abs(threshold.center[perpendicular] - leaf.center[perpendicular])
                > 3.0 * thickness
                or not 0.7 <= threshold.width_px / leaf.width_px <= 1.3
            ):
                continue
            x0, y0, x1, y1 = threshold.bbox
            old = leaf.arc
            if threshold.axis == "h":
                hinge = (
                    x0 if abs(old.hinge[0] - x0) <= abs(old.hinge[0] - x1) else x1,
                    threshold.center[1],
                )
            else:
                hinge = (
                    threshold.center[0],
                    y0 if abs(old.hinge[1] - y0) <= abs(old.hinge[1] - y1) else y1,
                )
            recovered[i] = replace(threshold, arc=replace(old, hinge=hinge))
            dropped.add(j)
            break
    return [candidate for i, candidate in enumerate(recovered) if i not in dropped]


def _dedupe_corner_swings(
    candidates: list[OpeningCandidate],
    thickness: float,
    *,
    segments: list[WallSegment] | None = None,
    walls: WallExtraction | None = None,
) -> list[OpeningCandidate]:
    """Drop a perpendicular door-leaf duplicate at the same wall corner."""

    def nearest_mid(a0: float, a1: float, b0: float, b1: float) -> float:
        if a1 < b0:
            return (a1 + b0) / 2
        if b1 < a0:
            return (b1 + a0) / 2
        return (max(a0, b0) + min(a1, b1)) / 2

    dropped: set[int] = set()
    for swung in candidates:
        if (
            swung.arc is None
            or swung.arc.radius_px is None
            or swung.kind_hint != "window"
            or swung.axis not in {"h", "v"}
        ):
            continue
        for leaf_index, leaf in enumerate(candidates):
            if (
                leaf.arc is not None
                or leaf.kind_hint != "window"
                or frozenset(leaf.connects) != frozenset(swung.connects)
                or {leaf.axis, swung.axis} != {"h", "v"}
                or box(*leaf.bbox).distance(box(*swung.bbox)) > thickness
            ):
                continue
            joint = np.array(
                [
                    nearest_mid(leaf.bbox[0], leaf.bbox[2], swung.bbox[0], swung.bbox[2]),
                    nearest_mid(leaf.bbox[1], leaf.bbox[3], swung.bbox[1], swung.bbox[3]),
                ]
            )
            if (
                0.7 <= leaf.width_px / swung.arc.radius_px <= 1.3
                and np.linalg.norm(np.asarray(swung.arc.hinge) - joint) <= 0.75 * thickness
            ):
                dropped.add(leaf_index)
    for i, left in enumerate(candidates):
        if i in dropped or left.arc is None or left.axis not in {"h", "v"}:
            continue
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            same_rooms = frozenset(left.connects) == frozenset(right.connects)
            shared_rooms = {
                side for side in left.connects if isinstance(side, int)
            } & {side for side in right.connects if isinstance(side, int)}
            if not same_rooms and all(
                isinstance(side, int) for side in (*left.connects, *right.connects)
            ):
                if segments is None or walls is None:
                    continue
                raw_support = [
                    _ink_share(segments[candidate.wall_index], walls.solid)
                    for candidate in (left, right)
                    if 0 <= candidate.wall_index < len(segments)
                ]
                if len(raw_support) == 2 and min(raw_support) >= 0.10:
                    continue  # two supported hosts are two real corner doors
            if (
                j in dropped
                or right.arc is None
                or {left.axis, right.axis} != {"h", "v"}
                or (not same_rooms and not shared_rooms)
                or (
                    not same_rooms
                    and np.linalg.norm(np.subtract(left.arc.hinge, right.arc.hinge))
                    > 0.75 * thickness
                )
                or box(*left.bbox).distance(box(*right.bbox)) > thickness
            ):
                continue
            joint = np.array(
                [
                    nearest_mid(left.bbox[0], left.bbox[2], right.bbox[0], right.bbox[2]),
                    nearest_mid(left.bbox[1], left.bbox[3], right.bbox[1], right.bbox[3]),
                ]
            )
            left_distance = float(np.linalg.norm(np.asarray(left.arc.hinge) - joint))
            right_distance = float(np.linalg.norm(np.asarray(right.arc.hinge) - joint))
            # ponytail: close scores stay unresolved; add door-leaf line fitting
            # only when a real corpus case contains two genuine corner doors.
            if abs(left_distance - right_distance) >= 0.25 * thickness:
                dropped.add(i if left_distance > right_distance else j)
            elif not same_rooms and segments is not None and walls is not None:
                support = []
                mask = cv2.dilate(
                    walls.solid,
                    np.ones((2 * max(1, round(thickness)) + 1,) * 2, np.uint8),
                )
                for candidate in (left, right):
                    if 0 <= candidate.wall_index < len(segments):
                        support.append(_ink_share(segments[candidate.wall_index], mask))
                    else:
                        support.append(0.0)
                if abs(support[0] - support[1]) < 0.15:
                    continue
                dropped.add(i if support[0] < support[1] else j)
            else:
                continue
            if i in dropped:
                break
    return [candidate for i, candidate in enumerate(candidates) if i not in dropped]


def _recover_corner_glazing(
    candidates: list[OpeningCandidate],
    segments: list[WallSegment],
    walls: WallExtraction,
) -> list[OpeningCandidate]:
    """Keep a shallow exterior corner frame from becoming two doors."""
    out = list(candidates)
    for i, left in enumerate(out):
        for j in range(i + 1, len(out)):
            right = out[j]
            if (
                left.kind_hint != right.kind_hint
                or left.kind_hint != "doorlike"
                or {left.axis, right.axis} != {"h", "v"}
                or frozenset(left.connects) != frozenset(right.connects)
                or EXTERIOR not in left.connects
                or box(*left.bbox).distance(box(*right.bbox)) > walls.thickness_px
                or any(
                    candidate.arc is not None and candidate.arc.radius_px is not None
                    for candidate in (left, right)
                )
            ):
                continue
            support = [
                _ink_share(segments[candidate.wall_index], walls.solid)
                for candidate in (left, right)
            ]
            if min(support) > 0.05:
                continue
            out[i] = replace(left, kind_hint="window", arc=None)
            out[j] = replace(right, kind_hint="window", arc=None)
    return out


def _merge_window_panels(
    candidates: list[OpeningCandidate], segments: list[WallSegment]
) -> list[OpeningCandidate]:
    """Join split panels that together span one interior sliding track."""
    out = list(candidates)
    dropped: set[int] = set()
    for i, left in enumerate(out):
        if i in dropped or left.kind_hint != "window" or left.axis not in {"h", "v"}:
            continue
        for j in range(i + 1, len(out)):
            right = out[j]
            if (
                j in dropped
                or right.kind_hint != "window"
                or right.axis != left.axis
                or right.wall_index != left.wall_index
                or frozenset(right.connects) != frozenset(left.connects)
                or not all(isinstance(side, int) for side in left.connects)
            ):
                continue
            host = segments[left.wall_index]
            host_length = float(np.hypot(host.end[0] - host.start[0], host.end[1] - host.start[1]))
            lo_index, hi_index = (0, 2) if left.axis == "h" else (1, 3)
            lo = min(left.bbox[lo_index], right.bbox[lo_index])
            hi = max(left.bbox[hi_index], right.bbox[hi_index])
            if hi - lo < 0.8 * host_length:
                continue
            bbox = (
                min(left.bbox[0], right.bbox[0]),
                min(left.bbox[1], right.bbox[1]),
                max(left.bbox[2], right.bbox[2]),
                max(left.bbox[3], right.bbox[3]),
            )
            out[i] = replace(
                left,
                bbox=bbox,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                width_px=hi - lo + 1,
                arc=left.arc if left.arc is not None and left.arc.radius_px is not None else None,
            )
            dropped.add(j)
            break
    return [candidate for i, candidate in enumerate(out) if i not in dropped]


def _merge_zone_facade_panels(
    candidates: list[OpeningCandidate],
    segments: list[WallSegment],
    rooms: list[RoomDraft],
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    bgr: np.ndarray,
    binary: np.ndarray,
) -> tuple[list[OpeningCandidate], list[WallSegment]]:
    """Rejoin one facade opening divided by an inferred open-zone cut."""
    if walls.inferred_zones is None:
        return candidates, segments
    near_zone = cv2.dilate(
        walls.inferred_zones,
        np.ones((2 * max(1, round(walls.thickness_px)) + 1,) * 2, np.uint8),
    )
    for i, left in enumerate(candidates):
        if left.kind_hint != "window" or left.axis not in {"h", "v"}:
            continue
        left_rooms = [side for side in left.connects if isinstance(side, int)]
        if EXTERIOR not in left.connects or len(left_rooms) != 1:
            continue
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            right_rooms = [side for side in right.connects if isinstance(side, int)]
            if (
                right.kind_hint != "window"
                or right.axis != left.axis
                or EXTERIOR not in right.connects
                or len(right_rooms) != 1
                or right_rooms == left_rooms
                or left.wall_index == right.wall_index
                or any(
                    index not in {i, j}
                    and candidate.wall_index in {left.wall_index, right.wall_index}
                    for index, candidate in enumerate(candidates)
                )
            ):
                continue
            along = (0, 2) if left.axis == "h" else (1, 3)
            normal = 1 if left.axis == "h" else 0
            gap = max(
                0.0,
                max(left.bbox[along[0]], right.bbox[along[0]])
                - min(left.bbox[along[1]], right.bbox[along[1]]),
            )
            if (
                gap > walls.thickness_px
                or abs(left.center[normal] - right.center[normal]) > walls.thickness_px
            ):
                continue
            join = (
                (left.bbox[along[1]] + right.bbox[along[0]]) / 2
                if left.center[along[0]] <= right.center[along[0]]
                else (right.bbox[along[1]] + left.bbox[along[0]]) / 2
            )
            point = (
                (round(join), round((left.center[1] + right.center[1]) / 2))
                if left.axis == "h"
                else (round((left.center[0] + right.center[0]) / 2), round(join))
            )
            if not near_zone[
                int(np.clip(point[1], 0, near_zone.shape[0] - 1)),
                int(np.clip(point[0], 0, near_zone.shape[1] - 1)),
            ]:
                continue

            bbox = (
                float(min(left.bbox[0], right.bbox[0])),
                float(min(left.bbox[1], right.bbox[1])),
                float(max(left.bbox[2], right.bbox[2])),
                float(max(left.bbox[3], right.bbox[3])),
            )
            open_lo, open_hi = bbox[along[0]], bbox[along[1]]
            hosts = [segments[left.wall_index], segments[right.wall_index]]
            host_intervals = [
                sorted(
                    (host.start[0], host.end[0])
                    if left.axis == "h"
                    else (host.start[1], host.end[1])
                )
                for host in hosts
            ]
            host_lo = min(interval[0] for interval in host_intervals)
            host_hi = max(interval[1] for interval in host_intervals)
            fixed = float((left.center[normal] + right.center[normal]) / 2)
            chosen_room = max(left_rooms + right_rooms, key=lambda index: rooms[index].area_px)
            obsolete = {left.wall_index, right.wall_index}
            remap: dict[int, int] = {}
            merged_segments: list[WallSegment] = []
            for old_index, segment in enumerate(segments):
                if old_index not in obsolete:
                    remap[old_index] = len(merged_segments)
                    merged_segments.append(segment)
            for lo, hi in ((host_lo, open_lo), (open_hi, host_hi)):
                if hi - lo < 0.25 * walls.thickness_px:
                    continue
                midpoint = (lo + hi) / 2
                owner = next(
                    (
                        host.rooms
                        for host, interval in zip(hosts, host_intervals, strict=True)
                        if interval[0] <= midpoint <= interval[1]
                    ),
                    (chosen_room, EXTERIOR),
                )
                merged_segments.append(
                    _segment(left.axis, fixed, lo, hi, walls.thickness_px, owner)
                )
            opening_host = len(merged_segments)
            merged_segments.append(
                _segment(
                    left.axis,
                    fixed,
                    open_lo,
                    open_hi,
                    walls.thickness_px,
                    (chosen_room, EXTERIOR),
                )
            )
            merged = replace(
                left,
                bbox=bbox,
                center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                width_px=open_hi - open_lo + 1,
                connects=(chosen_room, EXTERIOR),
                wall_index=opening_host,
                arc=None,
            )
            arc = _detect_double_arc(merged, walls, room_masks, binary)
            if arc is not None:
                merged = replace(merged, kind_hint="double_door", arc=arc)
            else:
                arc = _detect_arc(
                    merged,
                    walls,
                    room_masks,
                    bgr,
                    binary,
                    require_coherent_stroke=True,
                )
                if arc is not None:
                    merged = replace(merged, kind_hint="doorlike", arc=arc)
            out = []
            for index, candidate in enumerate(candidates):
                if index == j:
                    continue
                if index == i:
                    out.append(merged)
                else:
                    out.append(
                        replace(
                            candidate,
                            wall_index=(
                                remap[candidate.wall_index]
                                if candidate.wall_index >= 0
                                else candidate.wall_index
                            ),
                        )
                    )
            return out, merged_segments
    return candidates, segments


def _merge_diagonal_window_panels(
    candidates: list[OpeningCandidate],
    segments: list[WallSegment],
    thickness: float,
) -> tuple[list[OpeningCandidate], list[WallSegment]]:
    """Join adjacent panels split across near-collinear diagonal hosts."""
    eligible = {
        index
        for index, candidate in enumerate(candidates)
        if candidate.axis == "d"
        and candidate.kind_hint == "window"
        and candidate.arc is None
        and 0 <= candidate.wall_index < len(segments)
    }
    groups: list[set[int]] = []
    while eligible:
        group = {eligible.pop()}
        changed = True
        while changed:
            changed = False
            for index in list(eligible):
                candidate = candidates[index]
                if any(
                    frozenset(candidate.connects) == frozenset(candidates[other].connects)
                    and box(*candidate.bbox).distance(box(*candidates[other].bbox)) <= thickness
                    and _near_collinear_hosts(
                        segments[candidate.wall_index],
                        segments[candidates[other].wall_index],
                        thickness,
                    )
                    for other in group
                ):
                    eligible.remove(index)
                    group.add(index)
                    changed = True
        if len(group) > 1:
            groups.append(group)
    if not groups:
        return candidates, segments

    work_segments = list(segments)
    replacements: dict[int, OpeningCandidate] = {}
    dropped: set[int] = set()
    obsolete_hosts: set[int] = set()
    for group in groups:
        hosts = {candidates[index].wall_index for index in group}
        if any(
            index not in group and candidate.wall_index in hosts
            for index, candidate in enumerate(candidates)
        ):
            continue
        first_host = segments[next(iter(hosts))]
        if any(set(segments[index].rooms) != set(first_host.rooms) for index in hosts):
            continue
        endpoints = [
            np.asarray(point, dtype=np.float64)
            for index in hosts
            for point in (segments[index].start, segments[index].end)
        ]
        start, end = max(
            ((left, right) for left in endpoints for right in endpoints),
            key=lambda pair: float(np.linalg.norm(pair[1] - pair[0])),
        )
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length == 0:
            continue
        direction = delta / length
        normal = np.asarray((-direction[1], direction[0]))
        intervals = []
        for index in group:
            candidate = candidates[index]
            middle = float(np.dot(np.asarray(candidate.center) - start, direction))
            half = max(0.0, (candidate.width_px - 1.0) / 2.0)
            intervals.append((middle - half, middle + half))
        lo = min(interval[0] for interval in intervals)
        hi = max(interval[1] for interval in intervals)
        pa, pb = start + direction * lo, start + direction * hi
        corners = np.vstack(
            (
                pa + normal * thickness / 2,
                pa - normal * thickness / 2,
                pb + normal * thickness / 2,
                pb - normal * thickness / 2,
            )
        )
        bbox = (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )
        keep = min(group)
        wall_index = len(work_segments)
        work_segments.append(
            WallSegment(
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                first_host.thickness_px,
                first_host.rooms,
            )
        )
        replacements[keep] = replace(
            candidates[keep],
            bbox=bbox,
            center=(float((pa[0] + pb[0]) / 2), float((pa[1] + pb[1]) / 2)),
            width_px=hi - lo + 1.0,
            wall_index=wall_index,
        )
        dropped.update(group - {keep})
        obsolete_hosts.update(hosts)

    if not replacements:
        return candidates, segments
    merged = [
        replacements.get(index, candidate)
        for index, candidate in enumerate(candidates)
        if index not in dropped
    ]
    remap: dict[int, int] = {}
    kept_segments: list[WallSegment] = []
    for index, segment in enumerate(work_segments):
        if index not in obsolete_hosts:
            remap[index] = len(kept_segments)
            kept_segments.append(segment)
    return (
        [replace(candidate, wall_index=remap[candidate.wall_index]) for candidate in merged],
        kept_segments,
    )


def _near_collinear_hosts(left: WallSegment, right: WallSegment, thickness: float) -> bool:
    left_delta = np.subtract(left.end, left.start)
    right_delta = np.subtract(right.end, right.start)
    norms = float(np.linalg.norm(left_delta) * np.linalg.norm(right_delta))
    return (
        norms > 0
        and abs(float(np.dot(left_delta, right_delta))) / norms >= np.cos(np.deg2rad(8.0))
        and LineString((left.start, left.end)).distance(LineString((right.start, right.end)))
        <= thickness
    )


def _recover_corner_swing(
    candidates: list[OpeningCandidate],
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    bgr: np.ndarray,
    binary: np.ndarray,
) -> list[OpeningCandidate]:
    """Move a corner leaf's arc to the threshold and remove its wall echo."""
    recovered = list(candidates)
    dropped: set[int] = set()
    for i, threshold in enumerate(recovered):
        if (
            threshold.arc is not None
            or threshold.kind_hint != "doorlike"
            or threshold.axis not in {"h", "v"}
        ):
            continue
        for j, leaf in enumerate(recovered):
            same_rooms = frozenset(threshold.connects) == frozenset(leaf.connects)
            shared_rooms = {
                side for side in threshold.connects if isinstance(side, int)
            } & {side for side in leaf.connects if isinstance(side, int)}
            if (
                i == j
                or j in dropped
                or {threshold.axis, leaf.axis} != {"h", "v"}
                or (
                    not same_rooms
                    and (not shared_rooms or UNKNOWN not in leaf.connects)
                )
                or box(*threshold.bbox).distance(box(*leaf.bbox)) > 1.5 * walls.thickness_px
                or not 0.75 <= threshold.width_px / leaf.width_px <= 1.25
            ):
                continue
            if EXTERIOR in threshold.connects and leaf.kind_hint == "window":
                arc = leaf.arc or _detect_arc(leaf, walls, room_masks, bgr, binary)
                if arc is None:
                    continue
                recovered[i] = replace(leaf, kind_hint="doorlike", arc=arc)
                dropped.add(j)
                break
            if leaf.arc is None or (not same_rooms and leaf.arc.radius_px is None):
                continue
            arc = leaf.arc
            x0, y0, x1, y1 = threshold.bbox
            hinge = (
                x0 if abs(arc.hinge[0] - x0) <= abs(arc.hinge[0] - x1) else x1,
                y0 if abs(arc.hinge[1] - y0) <= abs(arc.hinge[1] - y1) else y1,
            )
            # Preserve this list slot so stable downstream IDs do not change
            # when the actual threshold precedes the drawn open leaf.
            recovered[i] = replace(threshold, arc=replace(arc, hinge=hinge))
            dropped.add(j)
            if not same_rooms and all(isinstance(side, int) for side in threshold.connects):
                perpendicular = 0 if threshold.axis == "v" else 1
                along = 1 - perpendicular
                for k, echo in enumerate(recovered):
                    if (
                        k in {i, j}
                        or echo.arc is None
                        or echo.axis != threshold.axis
                        or UNKNOWN not in echo.connects
                        or box(*echo.bbox).distance(box(*leaf.bbox))
                        > walls.thickness_px
                        or np.linalg.norm(np.subtract(echo.arc.hinge, leaf.arc.hinge))
                        > 0.75 * walls.thickness_px
                        or abs(
                            abs(echo.center[perpendicular] - threshold.center[perpendicular])
                            - leaf.width_px
                        )
                        > 1.5 * walls.thickness_px
                        or abs(echo.center[along] - threshold.center[along])
                        > 0.5 * max(echo.width_px, threshold.width_px)
                    ):
                        continue
                    dropped.add(k)
            break
    return [candidate for i, candidate in enumerate(recovered) if i not in dropped]


def _fit_stroked_leaf(candidate: OpeningCandidate, host: WallSegment) -> OpeningCandidate:
    """Restore a door gap eroded by a perpendicular wall junction."""
    arc = candidate.arc
    if arc is None or arc.radius_px is None:
        return candidate
    host_width = float(np.hypot(host.end[0] - host.start[0], host.end[1] - host.start[1]))
    width = min(float(round(arc.radius_px)), host_width)
    x0, y0, x1, y1 = candidate.bbox
    if candidate.axis == "h":
        host_lo, host_hi = sorted((host.start[0], host.end[0]))
        if abs(arc.hinge[0] - x0) <= abs(arc.hinge[0] - x1):
            x1 = min(host_hi, x0 + width - 1)
        else:
            x0 = max(host_lo, x1 - width + 1)
        width = x1 - x0 + 1
    elif candidate.axis == "v":
        host_lo, host_hi = sorted((host.start[1], host.end[1]))
        if abs(arc.hinge[1] - y0) <= abs(arc.hinge[1] - y1):
            y1 = min(host_hi, y0 + width - 1)
        else:
            y0 = max(host_lo, y1 - width + 1)
        width = y1 - y0 + 1
    if width - candidate.width_px < 0.5 * host.thickness_px:
        return candidate
    return replace(
        candidate,
        bbox=(x0, y0, x1, y1),
        center=((x0 + x1) / 2, (y0 + y1) / 2),
        width_px=width,
    )


def _split_swing_beside_window(
    candidate: OpeningCandidate,
    host: WallSegment,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    bgr: np.ndarray,
    binary: np.ndarray,
) -> tuple[OpeningCandidate, OpeningCandidate] | None:
    """Split a fixed panel and a pale swing leaf merged into one wall gap."""
    if candidate.axis not in {"h", "v"} or not all(
        isinstance(side, int) for side in candidate.connects
    ):
        return None
    coordinate = 0 if candidate.axis == "h" else 1
    candidate_lo = candidate.bbox[coordinate]
    candidate_hi = candidate.bbox[coordinate + 2]
    host_lo, host_hi = sorted(
        (host.start[coordinate], host.end[coordinate])
    )
    for high_end in (True, False):
        host_end = host_hi if high_end else host_lo
        candidate_end = candidate_hi if high_end else candidate_lo
        if abs(host_end - candidate_end) > host.thickness_px:
            continue
        for factor in (3.5, 3.25, 3.75, 3.0, 4.0):
            leaf_width = factor * host.thickness_px
            door_lo, door_hi = (
                (host_hi - leaf_width + 1, host_hi)
                if high_end
                else (host_lo, host_lo + leaf_width - 1)
            )
            remainder = (
                door_lo - candidate_lo if high_end else candidate_hi - door_hi
            )
            if remainder < 1.5 * host.thickness_px:
                continue
            x0, y0, x1, y1 = candidate.bbox
            door_bbox = (
                (door_lo, y0, door_hi, y1)
                if candidate.axis == "h"
                else (x0, door_lo, x1, door_hi)
            )
            door = replace(
                candidate,
                bbox=door_bbox,
                center=(
                    (door_bbox[0] + door_bbox[2]) / 2,
                    (door_bbox[1] + door_bbox[3]) / 2,
                ),
                width_px=leaf_width,
                kind_hint="window",
            )
            arc = _detect_arc(
                door,
                walls,
                room_masks,
                bgr,
                binary,
                host,
                require_coherent_stroke=True,
            )
            if arc is None:
                continue
            split = door_lo if high_end else door_hi
            window_bbox = (
                (x0, y0, split, y1)
                if candidate.axis == "h" and high_end
                else (split, y0, x1, y1)
                if candidate.axis == "h"
                else (x0, y0, x1, split)
                if high_end
                else (x0, split, x1, y1)
            )
            window_width = (
                window_bbox[2] - window_bbox[0] + 1
                if candidate.axis == "h"
                else window_bbox[3] - window_bbox[1] + 1
            )
            window = replace(
                candidate,
                bbox=window_bbox,
                center=(
                    (window_bbox[0] + window_bbox[2]) / 2,
                    (window_bbox[1] + window_bbox[3]) / 2,
                ),
                width_px=window_width,
            )
            return window, replace(door, kind_hint="doorlike", arc=arc)
    return None


# ---------------------------------------------------------------- swing arcs


def _detect_arc(
    candidate: OpeningCandidate,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    bgr: np.ndarray,
    binary: np.ndarray,
    host: WallSegment | None = None,
    *,
    require_coherent_stroke: bool = False,
) -> ArcEvidence | None:
    """Quarter-disc evidence at either jamb, on either side of the wall.

    Two independent detectors, either suffices:
    - tinted sector: the disc's fill differs in color from the same room's
      floor just beyond it (listing plans draw the sweep as a lighter tint);
    - stroked arc: dark non-wall pixels concentrated on the quarter-circle
      at radius = opening width.
    Both compare strictly WITHIN the room the sector opens into — comparing
    across rooms would mistake a flooring change for a door arc. Exterior
    sectors use only the stricter stroked-arc cue. Ambiguous evidence yields
    None: swing is then honestly unresolved rather than invented.
    """
    x0, y0, x1, y1 = candidate.bbox
    radius = candidate.width_px
    if radius < 8:
        return None
    if candidate.axis == "d" and not require_coherent_stroke:
        return None
    stroke_margin = (
        UNKNOWN_ARC_STROKE_MARGIN if UNKNOWN in candidate.connects else ARC_STROKE_MARGIN
    )
    if candidate.axis == "h":
        jambs = [(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)]
        normals = [(0.0, -1.0), (0.0, 1.0)]
    elif candidate.axis == "v":
        jambs = [((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)]
        normals = [(-1.0, 0.0), (1.0, 0.0)]
    elif host is not None:
        direction = np.subtract(host.end, host.start).astype(np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        half = max(0.0, (candidate.width_px - 1.0) / 2)
        center = np.asarray(candidate.center)
        jambs = [tuple(center - direction * half), tuple(center + direction * half)]
        normal_pair = (-float(direction[1]), float(direction[0]))
        normals = [normal_pair, (-normal_pair[0], -normal_pair[1])]
    else:
        return None

    pale_strokes = None
    if require_coherent_stroke:
        mean = bgr.mean(axis=2)
        chroma = bgr.max(axis=2) - bgr.min(axis=2)
        pale_strokes = (
            (binary > 0) & (mean >= 190) & (chroma < 15)
        ).astype(np.uint8)

    exterior_valid = None
    if EXTERIOR in candidate.connects:
        exterior_valid = walls.solid == 0
        for mask in room_masks:
            if mask is not None:
                exterior_valid &= mask == 0

    hits = []  # (jamb_idx, normal_idx, room_idx, score, stroked_radius)
    for j, jamb in enumerate(jambs):
        other = jambs[1 - j]
        along = np.array([other[0] - jamb[0], other[1] - jamb[1]])
        norm = float(np.linalg.norm(along))
        if norm == 0:
            continue
        along = along / norm
        for k, normal_t in enumerate(normals):
            normal_vec = np.array(normal_t)
            room_idx = _room_at(
                room_masks,
                (
                    jamb[0] + (along[0] + normal_vec[0]) * radius * 0.45,
                    jamb[1] + (along[1] + normal_vec[1]) * radius * 0.45,
                ),
            )
            if room_idx is None:
                if exterior_valid is None:
                    continue
                valid = exterior_valid
                tinted = fill = 0.0
            else:
                mask = room_masks[room_idx]
                assert mask is not None
                valid = (mask > 0) & (walls.solid == 0)
                tinted, fill = _sector_color_delta(
                    bgr, valid, jamb, along, normal_vec, radius
                )
            stroke_score = 0.0
            stroke_radius = None
            coherent_stroke = not require_coherent_stroke
            if pale_strokes is not None:
                for radius_factor in (
                    0.85,
                    0.9,
                    0.95,
                    1.0,
                    1.05,
                    1.1,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.35,
                    1.4,
                ):
                    leaf_radius = radius * radius_factor
                    longest, total = _arc_stroke_run(
                        pale_strokes, valid, jamb, along, normal_vec, leaf_radius
                    )
                    if longest >= 13 and longest >= 0.8 * total:
                        coherent_stroke = True
                        stroke_score = max(stroke_score, longest / 32)
                        stroke_radius = leaf_radius
            # The scanned wall break includes frame/reveal pixels, so its
            # width and the drawn leaf radius differ by up to about 10%.
            for radius_factor in (0.9, 0.95, 1.0, 1.05, 1.1):
                leaf_radius = radius * radius_factor
                stroked = _arc_stroke_coverage(
                    binary, valid, jamb, along, normal_vec, leaf_radius
                )
                baseline = _arc_stroke_coverage(
                    binary, valid, jamb, along, normal_vec, leaf_radius * 1.3
                )
                # Text and speckle textures light up any circle equally; a
                # real drawn arc lights up only the circle at the leaf radius.
                if (
                    stroked >= ARC_STROKE_COVERAGE
                    and stroked - baseline >= stroke_margin
                    and stroked > stroke_score
                ):
                    stroke_score, stroke_radius = stroked, leaf_radius
            if (
                room_idx is None
                and stroke_score == 0
                and candidate.kind_hint != "window"
            ):
                # Pale outward-swing leaves can be longer than the eroded
                # facade gap. Keep this relaxed cue exterior-only: flooring
                # texture inside rooms is much more likely to trace an arc.
                for radius_factor in (1.25, 1.3, 1.35, 1.4):
                    leaf_radius = radius * radius_factor
                    stroked = _arc_stroke_coverage(
                        binary, valid, jamb, along, normal_vec, leaf_radius
                    )
                    baseline = _arc_stroke_coverage(
                        binary, valid, jamb, along, normal_vec, leaf_radius * 1.3
                    )
                    if stroked >= 0.25 and stroked - baseline >= 0.18:
                        stroke_score, stroke_radius = stroked, radius
            if coherent_stroke and (tinted >= SECTOR_COLOR_DELTA or stroke_score > 0):
                # A perpendicular jamb can hide more of the threshold than
                # the normal reveal allowance. Once the arc itself is proven,
                # two wider probes recover that leaf radius without creating
                # new arc detections from room texture.
                for radius_factor in (1.15, 1.2):
                    leaf_radius = radius * radius_factor
                    stroked = _arc_stroke_coverage(
                        binary, valid, jamb, along, normal_vec, leaf_radius
                    )
                    baseline = _arc_stroke_coverage(
                        binary, valid, jamb, along, normal_vec, leaf_radius * 1.3
                    )
                    if (
                        stroked >= ARC_STROKE_COVERAGE
                        and stroked - baseline >= stroke_margin
                        and stroked > stroke_score
                    ):
                        stroke_score, stroke_radius = stroked, leaf_radius
                # fill breaks the jamb tie: both jambs' sectors overlap the
                # same tinted disc, but only the true hinge's annulus is
                # covered edge to edge.
                hits.append(
                    (
                        j,
                        k,
                        room_idx,
                        fill if tinted >= SECTOR_COLOR_DELTA else stroke_score,
                        stroke_radius,
                    )
                )

    if not hits:
        return None
    hits.sort(key=lambda item: item[3], reverse=True)
    j, k, room_idx, _, stroke_radius = hits[0]
    jamb, normal_vec = jambs[j], np.array(normals[k])
    other = jambs[1 - j]
    along = np.array([other[0] - jamb[0], other[1] - jamb[1]])
    along = along / np.linalg.norm(along)
    # The closed leaf lies along the wall (toward the far jamb); a positive
    # cross product means it sweeps clockwise (image coords, y down) to
    # reach the sector side.
    cross = along[0] * normal_vec[1] - along[1] * normal_vec[0]
    swing = "clockwise" if cross > 0 else "counterclockwise"
    return ArcEvidence(
        hinge=(float(jamb[0]), float(jamb[1])),
        swing=swing,
        opens_into=room_idx,
        radius_px=stroke_radius,
    )


def _detect_double_arc(
    candidate: OpeningCandidate,
    walls: WallExtraction,
    room_masks: list[np.ndarray | None],
    binary: np.ndarray,
) -> ArcEvidence | None:
    """Detect two mirrored half-width swing arcs on the same wall side."""
    if candidate.axis == "d" or candidate.width_px < 16:
        return None
    x0, y0, x1, y1 = candidate.bbox
    radius = candidate.width_px / 2
    if candidate.axis == "h":
        leaves = [
            ((x0, candidate.center[1]), np.array((1.0, 0.0))),
            ((x1, candidate.center[1]), np.array((-1.0, 0.0))),
        ]
        normals = [np.array((0.0, -1.0)), np.array((0.0, 1.0))]
    else:
        leaves = [
            ((candidate.center[0], y0), np.array((0.0, 1.0))),
            ((candidate.center[0], y1), np.array((0.0, -1.0))),
        ]
        normals = [np.array((-1.0, 0.0)), np.array((1.0, 0.0))]

    valid = walls.solid == 0
    best: tuple[float, np.ndarray, float] | None = None
    for normal in normals:
        for factor in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25):
            leaf_radius = radius * factor
            scores = []
            for hinge, along in leaves:
                stroke = _arc_stroke_coverage(
                    binary, valid, hinge, along, normal, leaf_radius
                )
                baseline = _arc_stroke_coverage(
                    binary, valid, hinge, along, normal, leaf_radius * 1.3
                )
                scores.append((stroke, stroke - baseline))
            if (
                min(score[0] for score in scores) >= DOUBLE_ARC_STROKE_COVERAGE
                and min(score[1] for score in scores) >= DOUBLE_ARC_STROKE_MARGIN
            ):
                score = sum(value for pair in scores for value in pair)
                if best is None or score > best[0]:
                    best = (score, normal, leaf_radius)
    if best is None and walls.door_hints is not None:
        for normal in normals:
            for factor in np.arange(0.70, 1.41, 0.05):
                leaf_radius = radius * factor
                scores = []
                for hinge, along in leaves:
                    stroke = _arc_stroke_coverage(
                        walls.door_hints, valid, hinge, along, normal, leaf_radius
                    )
                    baseline = _arc_stroke_coverage(
                        walls.door_hints, valid, hinge, along, normal, leaf_radius * 1.3
                    )
                    scores.append((stroke, stroke - baseline))
                if min(score[0] for score in scores) >= 0.35 and min(
                    score[1] for score in scores
                ) >= 0.25:
                    best = (
                        sum(value for pair in scores for value in pair),
                        normal,
                        leaf_radius,
                    )
                    break
            if best is not None:
                break
    if best is None:
        return None
    _, normal, leaf_radius = best
    hinge, along = leaves[0]
    room_idx = _room_at(
        room_masks,
        (
            hinge[0] + (along[0] + normal[0]) * leaf_radius * 0.45,
            hinge[1] + (along[1] + normal[1]) * leaf_radius * 0.45,
        ),
    )
    cross = along[0] * normal[1] - along[1] * normal[0]
    return ArcEvidence(
        hinge=(float(hinge[0]), float(hinge[1])),
        swing="clockwise" if cross > 0 else "counterclockwise",
        opens_into=room_idx,
        radius_px=leaf_radius,
    )


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
    inner = _sector_mask(bgr.shape[:2], hinge, along, normal, radius * 0.2, radius * 0.85) & valid
    control = _sector_mask(bgr.shape[:2], hinge, along, normal, radius * 1.15, radius * 1.6) & valid
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


def _arc_stroke_run(
    binary: np.ndarray,
    valid: np.ndarray,
    hinge: tuple[float, float],
    along: np.ndarray,
    normal: np.ndarray,
    radius: float,
) -> tuple[int, int]:
    """Longest continuous arc stroke and total hits across 32 samples."""
    h, w = binary.shape
    longest = run = hits = 0
    for angle in np.linspace(0.08, np.pi / 2 - 0.08, 32):
        direction = along * np.cos(angle) + normal * np.sin(angle)
        px = int(round(hinge[0] + direction[0] * radius))
        py = int(round(hinge[1] + direction[1] * radius))
        present = (
            0 <= px < w
            and 0 <= py < h
            and valid[py, px]
            and binary[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2].any()
        )
        hits += int(present)
        run = run + 1 if present else 0
        longest = max(longest, run)
    return longest, hits


# ------------------------------------------------------------- wall segments


def derive_wall_segments(
    rooms: list[RoomDraft],
    walls: WallExtraction,
    *,
    ignored_rooms: list[RoomDraft] | None = None,
) -> list[WallSegment]:
    """Centerline wall segments from room-polygon edges.

    Room polygons trace inner wall faces, so two adjacent rooms leave a
    2×(half thickness) gap between their facing edges — the wall. Facing
    edge pairs merge into one shared segment on the centerline; leftover
    edge intervals face the exterior or uncovered circulation space.
    Diagonal edges are emitted as-is and sampled in their own wall-aligned
    strip. Exterior leftovers move half a measured wall thickness outward and
    extend at both ends so adjacent facade centerlines still meet.
    """
    t = walls.thickness_px
    room_masks = _room_masks(rooms, walls.solid.shape)
    edges = []  # (room_idx, axis, fixed_coord, lo, hi)
    diagonals: list[WallSegment] = []
    for idx, room in enumerate(rooms):
        if room.source == "vlm" and not room.spatially_grounded:
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
                # notches, not architecture. A long door leaf can also pass
                # that size gate, so require an independently detected wall
                # family at the same folded angle.
                angle = (np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) + 45) % 90 - 45
                if not any(
                    abs((angle - wall_angle + 45) % 90 - 45) <= 8 for wall_angle in walls.angles
                ):
                    continue
                diagonals.append(
                    WallSegment(
                        start=(float(a[0]), float(a[1])),
                        end=(float(b[0]), float(b[1])),
                        thickness_px=t,
                        rooms=(idx, EXTERIOR),
                    )
                )

    # Every stretch of wall belongs to exactly ONE facing pair, the nearest
    # one. Emitting a segment per qualifying partner drew the same wall twice
    # wherever three edges sit within a thickness of each other — a recess, or
    # a stub whose face is also another room's boundary. Nearest-first with
    # claimed intervals makes the pairing a partition instead of a fan-out.
    min_pair = max(6.0, t)
    candidates: list[tuple[float, int, int, float, float]] = []
    for i, (room_i, axis, coord_i, lo_i, hi_i) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            room_j, axis_j, coord_j, lo_j, hi_j = edges[j]
            if axis_j != axis or abs(coord_j - coord_i) > 2.5 * t:
                continue
            lo, hi = max(lo_i, lo_j), min(hi_i, hi_j)
            if hi - lo < min_pair:
                continue
            # Facing edges of the SAME room are the two faces of a stub the
            # room wraps around — one wall, so they pair like any other
            # facing pair; skipping them drew a line down each face instead.
            # A room's own two far sides can also land within 2.5t on a
            # narrow closet, so the gap between them has to be wall: floor
            # between them means they are not one wall's faces.
            if room_j == room_i and not _ink_between(walls, axis, coord_i, coord_j, lo, hi):
                continue
            candidates.append((abs(coord_j - coord_i), i, j, lo, hi))

    segments: list[WallSegment] = list(diagonals)
    claimed: dict[int, list[tuple[float, float]]] = {}
    for _, i, j, lo, hi in sorted(candidates):
        room_i, axis, coord_i = edges[i][0], edges[i][1], edges[i][2]
        room_j, coord_j = edges[j][0], edges[j][2]
        free = _subtract_intervals((lo, hi), claimed.get(i, []) + claimed.get(j, []))
        for a, b in free:
            if b - a < min_pair:
                continue
            center = (coord_i + coord_j) / 2
            segments.append(_segment(axis, center, a, b, t, (room_i, room_j)))
            claimed.setdefault(i, []).append((a, b))
            claimed.setdefault(j, []).append((a, b))
    for i, (room_i, axis, coord_i, lo_i, hi_i) in enumerate(edges):
        for lo, hi in _subtract_intervals((lo_i, hi_i), claimed.get(i, [])):
            if hi - lo >= max(6.0, 1.5 * t):
                segments.append(_segment(axis, coord_i, lo, hi, t, (room_i, EXTERIOR)))
    silhouette, _ = building_silhouette(walls.union, walls.footprint)
    resolved = [
        replace(seg, rooms=_resolve_connects(seg, walls, room_masks, silhouette))
        for seg in segments
    ]
    if ignored_rooms:
        resolved = [
            seg
            for seg in resolved
            if not _is_ignored_region_edge(seg, ignored_rooms, walls)
        ]
    centered = [_center_open_segment(seg, room_masks) for seg in resolved]
    walled = [seg for seg in centered if _rides_on_wall(seg, walls)]
    walled = _fuse_diagonal_segments(walled, t)
    result = _partition_collinear_overlaps(
        _close_tee_junctions(_fuse_collinear_segments(walled, t), t),
        t,
    )
    result = _drop_unsupported_loops(result, walls)
    if ignored_rooms:
        result = [
            seg
            for seg in result
            if not _is_ignored_region_edge(seg, ignored_rooms, walls)
        ]
    return result


def _partition_collinear_overlaps(
    segments: list[WallSegment], thickness: float
) -> list[WallSegment]:
    """Partition one physical wall run instead of drawing overlapping owners."""

    def geometry(segment: WallSegment) -> tuple[str, float, float, float] | None:
        dx, dy = segment.end[0] - segment.start[0], segment.end[1] - segment.start[1]
        if abs(dy) <= 0.09 * abs(dx):
            return (
                "h",
                (segment.start[1] + segment.end[1]) / 2,
                min(segment.start[0], segment.end[0]),
                max(segment.start[0], segment.end[0]),
            )
        if abs(dx) <= 0.09 * abs(dy):
            return (
                "v",
                (segment.start[0] + segment.end[0]) / 2,
                min(segment.start[1], segment.end[1]),
                max(segment.start[1], segment.end[1]),
            )
        return None

    def priority(segment: WallSegment) -> tuple[int, int, float]:
        known = sum(isinstance(side, int) for side in segment.rooms)
        return known, int(UNKNOWN not in segment.rooms), LineString(
            (segment.start, segment.end)
        ).length

    out: list[WallSegment] = []
    for segment in sorted(segments, key=priority, reverse=True):
        info = geometry(segment)
        if info is None:
            out.append(segment)
            continue
        axis, fixed, lo, hi = info
        claimed = []
        for other in out:
            other_info = geometry(other)
            if (
                other_info is not None
                and other_info[0] == axis
                and abs(other_info[1] - fixed) <= 1.0
            ):
                claimed.append((other_info[2], other_info[3]))
        for start, end in _subtract_intervals((lo, hi), claimed):
            if end - start >= max(2.0, 0.25 * thickness):
                out.append(
                    _segment(axis, fixed, start, end, segment.thickness_px, segment.rooms)
                )
    return out


def _is_ignored_region_edge(
    segment: WallSegment, ignored_rooms: list[RoomDraft], walls: WallExtraction
) -> bool:
    """Drop an interior furniture notch while preserving real facade frames."""
    if UNKNOWN not in segment.rooms or EXTERIOR in segment.rooms:
        return False
    line = LineString((segment.start, segment.end))
    if line.length == 0:
        return False
    required = min(0.5 * line.length, walls.thickness_px)
    fx0, fy0, fx1, fy1 = walls.footprint
    mx, my = line.interpolate(0.5, normalized=True).coords[0]
    facade_distance = min(mx - fx0, fx1 - mx, my - fy0, fy1 - my)
    segment_axis = "h" if abs(segment.end[0] - segment.start[0]) >= abs(
        segment.end[1] - segment.start[1]
    ) else "v"
    for room in ignored_rooms:
        if len(room.polygon) < 3:
            continue
        overlap = line.intersection(
            Polygon(room.polygon).boundary.buffer(0.35 * walls.thickness_px)
        ).length
        if overlap < required:
            continue
        width, height = np.ptp(room.polygon, axis=0)
        frame_axis = "h" if width >= height else "v"
        if (
            facade_distance <= min(1.25 * line.length, 8.0 * walls.thickness_px)
            and segment_axis == frame_axis
        ):
            continue  # shallow facade frame/reveal, not interior furniture
        return True
    return False


def _rides_on_wall(seg: WallSegment, walls: WallExtraction) -> bool:
    """Is this edge structure, or a dashed functional split?

    Room extraction seals along dashed zone dividers (玄关/走廊/餐厅), so the
    room polygon has an edge there and the pairing turns it into a wall
    segment — but you cannot lean on a dashed line. A segment whose ink is
    the divider mask rather than the wall mask is a zone boundary; the split
    is still reported, through the room's ``zone_bounded`` flag and its
    ``room_zone_boundary`` warning, which is where it belongs.
    """
    solid_support = _ink_share(seg, walls.solid)
    dx = abs(seg.end[0] - seg.start[0])
    dy = abs(seg.end[1] - seg.start[1])
    if min(dx, dy) > 0.09 * max(dx, dy) and _ink_share(seg, walls.union) < 0.20:
        return False  # unsupported diagonal room edge/door leaf, not structure
    if seg.rooms[0] == seg.rooms[1] and solid_support < 0.20:
        return False
    if walls.inferred_zones is not None:
        reach = 2 * max(1, int(round(walls.thickness_px))) + 1
        if _ink_share(
            seg, cv2.dilate(walls.inferred_zones, np.ones((reach, reach), np.uint8))
        ) >= 0.50:
            return False
    if walls.zones is None:
        return True
    radius = max(3, round(0.35 * walls.thickness_px))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    on_zone = _ink_share(seg, cv2.dilate(walls.zones, kernel))
    if _is_zone_track(seg, walls) or on_zone < 0.50:
        return True
    solid_radius = max(1, round(walls.thickness_px))
    on_solid = _ink_share(
        seg,
        cv2.dilate(
            walls.solid,
            np.ones((2 * solid_radius + 1, 2 * solid_radius + 1), np.uint8),
        ),
    )
    if on_solid < 0.30:
        return False
    on_structure = _ink_share(seg, cv2.dilate(walls.union, np.ones((5, 5), np.uint8)))
    return on_structure >= on_zone


def _zone_mask(walls: WallExtraction) -> np.ndarray | None:
    if walls.inferred_zones is None:
        return walls.zones
    if walls.zones is None:
        return walls.inferred_zones
    return cv2.bitwise_or(walls.zones, walls.inferred_zones)


def _is_zone_track(seg: WallSegment, walls: WallExtraction) -> bool:
    if walls.zones is None:
        return False
    if not (
        _ink_share(seg, cv2.dilate(walls.zones, np.ones((7, 7), np.uint8))) >= 0.50
        and _ink_share(seg, cv2.erode(walls.lines, np.ones((3, 3), np.uint8))) >= 0.30
        and _ink_share(
            seg, cv2.bitwise_and(walls.lines, cv2.bitwise_not(walls.zones))
        )
        >= 0.30
    ):
        return False
    if _ink_share(seg, walls.solid) < 0.50:
        return True

    # A track drawn over grey poche has two parallel rails; one zone trace
    # through otherwise solid poche is just the wall's printed finish edge.
    direction = np.subtract(seg.end, seg.start)
    normal = np.array((-direction[1], direction[0])) / np.linalg.norm(direction)
    radius = max(1, int(round(seg.thickness_px)))
    traces = []
    for offset in range(-radius, radius + 1):
        shift = offset * normal
        shifted = replace(
            seg,
            start=tuple(np.add(seg.start, shift)),
            end=tuple(np.add(seg.end, shift)),
        )
        traces.append(_ink_share(shifted, walls.zones) >= 0.50)
    return (
        sum(
            hit and (index == 0 or not traces[index - 1])
            for index, hit in enumerate(traces)
        )
        >= 2
    )


def _drop_unsupported_loops(
    segments: list[WallSegment], walls: WallExtraction
) -> list[WallSegment]:
    """Remove small low-mass closed contours made by fixtures/furniture."""
    suspect = {
        index
        for index, segment in enumerate(segments)
        if UNKNOWN in segment.rooms
        and _ink_share(segment, walls.solid) < 0.40
        and LineString((segment.start, segment.end)).length <= 8.0 * walls.thickness_px
    }
    if len(suspect) < 3:
        return segments
    lines = {index: LineString((segments[index].start, segments[index].end)) for index in suspect}
    tolerance = 0.6 * walls.thickness_px
    neighbours: dict[int, set[int]] = {index: set() for index in suspect}
    for index in suspect:
        endpoints = [Point(lines[index].coords[0]), Point(lines[index].coords[-1])]
        for other in suspect:
            if other <= index:
                continue
            other_endpoints = [Point(lines[other].coords[0]), Point(lines[other].coords[-1])]
            if min(
                *(point.distance(lines[other]) for point in endpoints),
                *(point.distance(lines[index]) for point in other_endpoints),
            ) <= tolerance:
                neighbours[index].add(other)
                neighbours[other].add(index)
    cyclic = set(suspect)
    while leaves := {index for index in cyclic if len(neighbours[index] & cyclic) < 2}:
        cyclic -= leaves
    return [segment for index, segment in enumerate(segments) if index not in cyclic]


def _ink_share(seg: WallSegment, mask: np.ndarray) -> float:
    (x0, y0), (x1, y1) = seg.start, seg.end
    n = max(2, int(round(float(np.hypot(x1 - x0, y1 - y0)))))
    h, w = mask.shape
    hits = 0
    for xf, yf in zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n), strict=True):
        xi, yi = int(round(xf)), int(round(yf))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            hits += 1
    return hits / n


def _ink_between(
    walls: WallExtraction, axis: str, coord_a: float, coord_b: float, lo: float, hi: float
) -> bool:
    """Is the strip between two facing edges wall, or floor?

    Sampled on the midline: a stub's two faces have wall between them, a
    narrow room's two sides have its own floor.
    """
    mid = (coord_a + coord_b) / 2
    h, w = walls.union.shape
    n = max(4, int(round(hi - lo)))
    hits = 0
    for k in range(n):
        along = lo + (hi - lo) * k / max(n - 1, 1)
        x, y = (along, mid) if axis == "h" else (mid, along)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and walls.union[yi, xi]:
            hits += 1
    return hits >= 0.7 * n


def _fuse_collinear_segments(segments: list[WallSegment], t: float) -> list[WallSegment]:
    """Merge same-axis segments that are really one wall.

    Sub-thickness face jitter in the traced polygons splits a straight wall
    into two pieces offset by a few pixels; renderers show the offset as a
    crack. Pieces with the same rooms on both sides, nearly the same
    centerline (≤0.6t apart) and at most one door-sized gap (≤4t) fuse into
    one host wall. The opening scanner then distinguishes solid continuation
    from an actual pixel gap; larger open-plan breaks stay split."""

    def geometry(seg: WallSegment) -> tuple[str, float, float, float] | None:
        dx, dy = seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]
        if abs(dy) <= 0.09 * abs(dx):
            return (
                "h",
                (seg.start[1] + seg.end[1]) / 2,
                min(seg.start[0], seg.end[0]),
                max(seg.start[0], seg.end[0]),
            )
        if abs(dx) <= 0.09 * abs(dy):
            return (
                "v",
                (seg.start[0] + seg.end[0]) / 2,
                min(seg.start[1], seg.end[1]),
                max(seg.start[1], seg.end[1]),
            )
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
                    or o_info[2] > hi + MAX_HOST_GAP_THICKNESSES * t
                    or o_info[3] < lo - MAX_HOST_GAP_THICKNESSES * t
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


def _fuse_diagonal_segments(
    segments: list[WallSegment], thickness: float
) -> list[WallSegment]:
    """Pair the two traced faces of a diagonal wall and join its fragments."""
    diagonal = {
        index
        for index, segment in enumerate(segments)
        if abs(segment.end[0] - segment.start[0]) > 0.09 * abs(
            segment.end[1] - segment.start[1]
        )
        and abs(segment.end[1] - segment.start[1]) > 0.09 * abs(
            segment.end[0] - segment.start[0]
        )
        and UNKNOWN not in segment.rooms
    }
    components: list[set[int]] = []
    remaining = set(diagonal)
    while remaining:
        group = {remaining.pop()}
        changed = True
        while changed:
            changed = False
            for index in list(remaining):
                if any(
                    set(segments[index].rooms) == set(segments[other].rooms)
                    and _near_collinear_hosts(
                        segments[index], segments[other], thickness
                    )
                    for other in group
                ):
                    remaining.remove(index)
                    group.add(index)
                    changed = True
        components.append(group)

    fused: dict[int, WallSegment] = {}
    consumed: set[int] = set()
    for group in components:
        first = min(group)
        consumed.update(group)
        if len(group) == 1:
            fused[first] = segments[first]
            continue
        points = np.asarray(
            [
                point
                for index in group
                for point in (segments[index].start, segments[index].end)
            ],
            dtype=np.float64,
        )
        center = points.mean(axis=0)
        _u, _s, axes = np.linalg.svd(points - center, full_matrices=False)
        direction = axes[0]
        projection = (points - center) @ direction
        start = center + direction * projection.min()
        end = center + direction * projection.max()
        template = segments[first]
        fused[first] = WallSegment(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            template.thickness_px,
            template.rooms,
        )

    return [
        fused[index]
        for index in range(len(segments))
        if index in fused
    ] + [
        segment
        for index, segment in enumerate(segments)
        if index not in consumed
        and not (
            UNKNOWN in segment.rooms
            and LineString((segment.start, segment.end)).length <= 9.0 * thickness
            and abs(segment.end[0] - segment.start[0])
            > 0.09 * abs(segment.end[1] - segment.start[1])
            and abs(segment.end[1] - segment.start[1])
            > 0.09 * abs(segment.end[0] - segment.start[0])
        )
    ]


def _close_tee_junctions(segments: list[WallSegment], t: float) -> list[WallSegment]:
    """Extend wall ends to meet the centerline of a crossing perpendicular
    wall. Segments derived from room-polygon edges stop at the polygon's
    inner corners, leaving half-thickness notches at every T and L junction;
    renderers show them as broken walls."""
    reach = 2.0 * t

    def geometry(seg: WallSegment) -> tuple[str, float, float, float] | None:
        dx, dy = seg.end[0] - seg.start[0], seg.end[1] - seg.start[1]
        if abs(dy) <= 0.09 * abs(dx):
            return (
                "h",
                (seg.start[1] + seg.end[1]) / 2,
                min(seg.start[0], seg.end[0]),
                max(seg.start[0], seg.end[0]),
            )
        if abs(dx) <= 0.09 * abs(dy):
            return (
                "v",
                (seg.start[0] + seg.end[0]) / 2,
                min(seg.start[1], seg.end[1]),
                max(seg.start[1], seg.end[1]),
            )
        return None

    infos = [geometry(seg) for seg in segments]
    out: list[WallSegment] = []
    for seg, info in zip(segments, infos, strict=True):
        if info is None:
            out.append(seg)
            continue
        axis, fixed, lo, hi = info
        lo_junctions: list[float] = []
        hi_junctions: list[float] = []
        for other in infos:
            if other is None or other[0] == axis:
                continue
            o_fixed, o_lo, o_hi = other[1], other[2], other[3]
            if not (o_lo - 0.6 * t <= fixed <= o_hi + 0.6 * t):
                continue  # the perpendicular wall doesn't cross our line
            if abs(o_fixed - lo) <= reach:
                lo_junctions.append(o_fixed)
            if abs(o_fixed - hi) <= reach:
                hi_junctions.append(o_fixed)
        if lo_junctions:
            lo = min(lo_junctions, key=lambda value: abs(value - lo))
        if hi_junctions:
            hi = min(hi_junctions, key=lambda value: abs(value - hi))
        out.append(
            _segment(axis, fixed, lo, hi, seg.thickness_px, seg.rooms)
            if (lo, hi) != (info[2], info[3])
            else seg
        )
    return out


def _center_open_segment(seg: WallSegment, room_masks: list[np.ndarray | None]) -> WallSegment:
    """Move a one-room segment from the room's inner face to the wall centre.

    Two facing rooms already average onto the centreline. A leftover edge has
    only its own room's face, so it stayed half a thickness off — breaking the
    centreline convention and drawing a second line beside every shared wall
    it ran alongside (31 such pairs on the corpus). Exterior segments also
    grow half a thickness at each end so adjacent facade centrelines meet;
    interior ones must not, or they overrun the neighbour they abut.
    """
    if not any(side in (EXTERIOR, UNKNOWN) for side in seg.rooms):
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
    grow = half if EXTERIOR in seg.rooms else 0.0
    start = start + outward * half - along * grow
    end = end + outward * half + along * grow
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
