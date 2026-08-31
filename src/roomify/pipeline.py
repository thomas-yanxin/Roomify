"""The parse() orchestrator: load → CV geometry → VLM semantics → FloorPlan.

Geometry never blocks on the VLM: every VLM call that fails (or is disabled
via use_vlm=False) routes to a degradation path that still yields a valid —
if less semantic — document, with the degradation recorded in ``warnings``
and ``unresolved``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from roomify import debug as debug_mod
from roomify.io import SourceImage, derotate, load
from roomify.merge import (
    DEVIATION_FLAG_THRESHOLD,
    MIN_ROOM_SQM,
    ElementDraft,
    OpeningDraft,
    RoomDraft,
    ScaleDraft,
    apply_area_checks,
    estimate_scale,
    ground_room_inventory,
    merge_openings,
    merge_rooms,
    pixels_per_mm_in_direction,
    reconcile_rooms,
)
from roomify.openings import (
    WallSegment,
    find_openings,
    find_zone_passages,
    measure_bay_protrusion,
)
from roomify.rooms import (
    _boundary_fraction,
    clean_room_ring,
    detect_rooms,
    uncovered_floor,
)
from roomify.schema import (
    DEFAULT_DOOR_HEIGHT_MM,
    DEFAULT_LEVEL_HEIGHT_MM,
    DEFAULT_WINDOW_HEIGHT_MM,
    DEFAULT_WINDOW_SILL_HEIGHT_MM,
    OUTDOOR_ROOM_TYPES,
    BBox,
    Element,
    FloorPlan,
    Opening,
    ParseWarning,
    Room,
    Unresolved,
    Wall,
)
from roomify.schema import (
    Point as SchemaPoint,
)
from roomify.walls import WallExtraction, estimate_plan_rotation, extract_walls

logger = logging.getLogger("roomify.pipeline")


def _inferred_zone_mask(
    walls: WallExtraction,
    boundaries: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray | None:
    """Keep functional cuts only where the drawing has no physical wall."""
    radius = max(1, round(0.25 * walls.thickness_px))
    supported = cv2.dilate(
        walls.solid, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    )
    inferred = np.zeros_like(walls.union)
    for start, end in boundaries:
        points = tuple(round(value) for point in (start, end) for value in point)
        trace = np.zeros_like(walls.union)
        cv2.line(trace, points[:2], points[2:], 255, 1)
        count = cv2.countNonZero(trace)
        if count and cv2.countNonZero(cv2.bitwise_and(trace, supported)) / count < 0.5:
            cv2.line(
                inferred,
                points[:2],
                points[2:],
                255,
                max(1, round(0.2 * walls.thickness_px)),
            )
    return inferred if inferred.any() else None


def parse(
    path: str | Path,
    *,
    page: int | None = None,
    use_vlm: bool = True,
    debug_dir: str | Path | None = None,
    max_dim: int = 2000,
    level_height_mm: float = DEFAULT_LEVEL_HEIGHT_MM,
    door_height_mm: float = DEFAULT_DOOR_HEIGHT_MM,
    window_sill_height_mm: float = DEFAULT_WINDOW_SILL_HEIGHT_MM,
    window_height_mm: float = DEFAULT_WINDOW_HEIGHT_MM,
) -> FloorPlan:
    """Parse a floor-plan image or PDF into a structured FloorPlan.

    use_vlm=False produces a pixel-only document (no names, no mm) from CV
    alone. debug_dir writes per-stage overlay PNGs — with this many tuned
    thresholds they are part of the algorithm, not a nicety.
    """
    vertical = (level_height_mm, door_height_mm, window_sill_height_mm, window_height_mm)
    if not all(math.isfinite(value) for value in vertical):
        raise ValueError("vertical dimensions must be finite")
    if min(level_height_mm, door_height_mm, window_height_mm) <= 0:
        raise ValueError("level, door, and window heights must be positive")
    if window_sill_height_mm < 0:
        raise ValueError("window sill height must be non-negative")
    if (
        door_height_mm > level_height_mm
        or window_sill_height_mm + window_height_mm > level_height_mm
    ):
        raise ValueError("door and window extents must fit within the level height")

    src = load(path, page=page, max_dim=max_dim)
    warnings: list[ParseWarning] = []
    unresolved: list[Unresolved] = []

    # Whole-sheet rotations (decorative listing exports): parse in a
    # derotated frame where walls are axis-aligned — every CV stage and the
    # VLM overlays run on ``work_bgr`` — and map emitted pixel coordinates
    # back through ``back``. Millimetre coordinates stay in the derotated,
    # wall-aligned frame (the only frame where they are rectilinear).
    rotation = estimate_plan_rotation(src.bgr)
    work_bgr = src.bgr
    back: np.ndarray | None = None
    if abs(rotation) >= 3.0:
        work_bgr, back = derotate(src.bgr, rotation)
        warnings.append(
            ParseWarning(
                code="plan_derotated",
                message=(
                    f"plan drawn rotated {rotation:.1f}°; parsed in a derotated "
                    "frame (polygon_mm axes are wall-aligned)"
                ),
            )
        )

    client = None
    if use_vlm:
        from roomify.vlm import VLMClient, VLMUnavailable

        try:
            client = VLMClient()
        except VLMUnavailable as exc:
            warnings.append(ParseWarning(code="vlm_unavailable", message=str(exc)))

    # The plan-read call runs concurrently with the CV stages and the
    # room-semantics call: none of them depend on it, and this endpoint's
    # reasoning latency (~45s/call) would otherwise serialize.
    plan_future = None
    executor = None
    if client is not None:
        from concurrent.futures import ThreadPoolExecutor

        from roomify.vlm import PLAN_WIRE, PlanRead, plan_read_prompt

        # 2 workers, measured: this endpoint's multi-image requests start
        # timing out under 3-way concurrency.
        executor = ThreadPoolExecutor(max_workers=2)
        plan_future = executor.submit(
            client.call, plan_read_prompt(), [work_bgr], PlanRead, wire_schema=PLAN_WIRE
        )

    walls = extract_walls(work_bgr)
    if walls.band_fallback:
        warnings.append(
            ParseWarning(
                code="wall_band_fallback",
                message="no wall-grey histogram peak found; wide fallback band used",
            )
        )

    detection = detect_rooms(walls)
    if detection.strategy != "close5":
        warnings.append(
            ParseWarning(
                code="room_gap_fallback",
                message=f"rooms sealed with {detection.strategy} (thin seal strokes absent)",
            )
        )

    room_read = None
    if client is not None:
        from roomify.vlm import (
            ROOMS_WIRE,
            RoomRead,
            render_room_sheet,
            room_semantics_prompt,
        )

        overlay = render_room_sheet(work_bgr, detection.rooms)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_room_overlay.png", overlay)
        # 1.5× upscale: printed ㎡ labels sit at the OCR limit on ~700px
        # listing exports; measured to fix small-text misreads at no latency
        # cost (the extra image tokens don't move this model's runtime).
        upscaled = cv2.resize(overlay, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        total_area = sum(r.area_px for r in detection.rooms) or 1.0
        shares = {
            str(i + 1): r.area_px / total_area for i, r in enumerate(detection.rooms)
        }
        room_prompt = room_semantics_prompt(len(detection.rooms), shares, review_sheet=True)
        room_read = client.call(
            room_prompt,
            [upscaled],
            RoomRead,
            wire_schema=ROOMS_WIRE,
        )
        if room_read is None or (not room_read.rooms and not room_read.extra_rooms):
            # Constrained decoding occasionally echoes an empty schema. One
            # unconstrained retry recovers the actual inventory.
            if room_read is None and plan_future is not None:
                plan_future.result()
            room_read = client.call(
                room_prompt,
                [upscaled],
                RoomRead,
            )
        if room_read is not None and len(room_read.rooms) < len(detection.rooms):
            retry = client.call(room_prompt, [upscaled], RoomRead)
            if retry is not None and len(retry.rooms) > len(room_read.rooms):
                room_read = retry
        if room_read is not None:
            from roomify.vlm import remap_room_sheet_extras

            room_read = remap_room_sheet_extras(room_read, work_bgr.shape[1])
        from roomify.vlm import (
            SPATIAL_ROOMS_WIRE,
            SpatialRoomRead,
            spatial_room_inventory_prompt,
        )

        spatial_prompt = spatial_room_inventory_prompt()
        spatial_images = [
            cv2.resize(
                work_bgr,
                None,
                fx=1.5,
                fy=1.5,
                interpolation=cv2.INTER_CUBIC,
            )
        ]
        spatial_read = client.call(
            spatial_prompt,
            spatial_images,
            SpatialRoomRead,
            wire_schema=SPATIAL_ROOMS_WIRE,
            max_tokens=5000,
        )
        if spatial_read is not None and not spatial_read.rooms:
            spatial_read = client.call(
                spatial_prompt, spatial_images, SpatialRoomRead, max_tokens=5000
            )
        room_read = ground_room_inventory(
            detection.rooms, room_read, spatial_read, work_bgr.shape[:2]
        )
        if room_read is None:
            warnings.append(
                ParseWarning(code="vlm_call_failed", message="room-semantics call failed; "
                             "rooms keep CV geometry with unknown names")
            )

    outcome = merge_rooms(
        detection.rooms,
        room_read,
        work_bgr.shape[:2],
        uncovered_floor(walls, detection.rooms),
    )
    if outcome.zone_boundaries:
        walls = replace(
            walls,
            inferred_zones=_inferred_zone_mask(walls, outcome.zone_boundaries),
        )
    opening_support_rooms = list(outcome.rooms)
    outcome.rooms, dropped_rooms = _drop_unlabelled_fixture_regions(
        outcome.rooms, walls, work_bgr, semantic_review=room_read is not None
    )
    warnings += outcome.warnings

    plan_read = plan_future.result() if plan_future is not None else None
    if plan_future is not None and plan_read is None:
        warnings.append(
            ParseWarning(code="vlm_call_failed", message="plan-read call failed; "
                         "scale falls back to printed areas, footprint to CV")
        )
    elif (
        plan_read is not None
        and not plan_read.dimension_chains
        and plan_read.footprint_box_2d is None
    ):
        assert client is not None
        plan_read = client.call(plan_read_prompt(), [work_bgr], PlanRead)

    chains = plan_read.dimension_chains if plan_read is not None else []
    scale_draft, scale_warnings = estimate_scale(chains, walls, outcome.rooms)
    warnings += scale_warnings

    # printed labels are ink truth; with a scale in hand, repair the
    # label↔polygon pairings the VLM fumbled and merge fragments whose sum
    # matches an otherwise-unplaceable label (every repair is logged)
    reconciled, reconcile_warnings = reconcile_rooms(
        outcome.rooms, scale_draft, walls.thickness_px
    )
    warnings += reconcile_warnings

    door_px = 1000.0 * scale_draft.px_per_mm_x if scale_draft else None
    candidates, segments = find_openings(
        walls, reconciled, work_bgr, door_px, ignored_rooms=dropped_rooms
    )
    reconciled, isolated_fixtures = _drop_isolated_fixture_regions(
        reconciled, candidates, walls
    )
    if isolated_fixtures:
        dropped_rooms.extend(isolated_fixtures)
        candidates, segments = find_openings(
            walls, reconciled, work_bgr, door_px, ignored_rooms=dropped_rooms
        )
    if dropped_rooms:
        support_candidates, _ = find_openings(
            walls, opening_support_rooms, work_bgr, door_px
        )
        current_by_marker = {
            room.marker: index
            for index, room in enumerate(reconciled)
            if room.marker is not None
        }
        candidates = _upgrade_openings_from_support(
            candidates,
            support_candidates,
            [
                current_by_marker.get(room.marker) if room.marker is not None else None
                for room in opening_support_rooms
            ],
            walls.thickness_px,
        )
        segments = _extend_short_opening_hosts(candidates, segments)
    dropped_room_markers = {
        room.marker for room in dropped_rooms if room.marker is not None
    }
    if dropped_room_markers:
        outcome.unresolved = [
            item
            for item in outcome.unresolved
            if not any(
                item.path.startswith(f"rooms/{marker}/")
                for marker in dropped_room_markers
            )
        ]
        warnings.append(
            ParseWarning(
                code="non_room_regions_dropped",
                message=(
                    f"{len(dropped_room_markers)} narrow or unsupported furniture/frame "
                    "regions were excluded from room and wall geometry"
                ),
            )
        )
    unresolved += outcome.unresolved
    cleaned = _clean_room_drafts(reconciled, walls)
    checked = apply_area_checks(cleaned, scale_draft)
    warnings += checked.warnings
    rooms = checked.rooms

    openings_read = None
    if client is not None and candidates:
        from roomify.vlm import (
            EXTRAS_WIRE,
            OpeningsRead,
            extra_elements_prompt,
            openings_prompt,
            render_candidate_sheet,
            render_openings_overlay,
        )

        overlay = render_openings_overlay(work_bgr, candidates)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_openings_overlay.png", overlay)
        context = _opening_context(candidates, rooms, scale_draft, segments)
        # Keep each response short enough for reliable JSON generation. All
        # visual evidence for a chunk is composited into one image because
        # this endpoint times out on multi-image opening requests.
        chunk_size = 3
        chunks = [candidates[s : s + chunk_size] for s in range(0, len(candidates), chunk_size)]
        merged = OpeningsRead()
        got_any = False
        failed: list[list] = []
        # The reference endpoint returns empty responses under concurrent
        # vision requests. Three-candidate serial batches measured faster
        # end-to-end than retrying nominally parallel work.
        for chunk in chunks:
            part = client.call(
                openings_prompt(
                    [c.marker for c in chunk],
                    include_extras=False,
                    context=context,
                    contact_sheet=True,
                ),
                [render_candidate_sheet(work_bgr, chunk, context_candidates=candidates)],
                OpeningsRead,
                # This endpoint accepts the opening schema but times out
                # while constrained-decoding it. The validated JSON-object
                # path completes reliably for the same image and prompt.
                max_tokens=1000,
            )
            if part is None:
                failed.append(chunk)
                continue
            got_any = True
            merged.candidates.update(part.candidates)
        extras_read = client.call(
            extra_elements_prompt(),
            [overlay],
            OpeningsRead,
            wire_schema=EXTRAS_WIRE,
            max_tokens=1000,
        )
        if extras_read is not None:
            got_any = True
            merged.extra_elements.extend(extras_read.extra_elements)
        # Concurrency can transiently saturate the endpoint. Retry each failed
        # single-image batch once, serially; never explode one failure into a
        # dozen slow per-crop calls.
        for chunk in failed:
            part = client.call(
                openings_prompt(
                    [cand.marker for cand in chunk],
                    include_extras=False,
                    context=context,
                    contact_sheet=True,
                ),
                [render_candidate_sheet(work_bgr, chunk, context_candidates=candidates)],
                OpeningsRead,
                max_tokens=1000,
            )
            if part is None:
                for cand in chunk:
                    warnings.append(
                        ParseWarning(
                            code="vlm_call_failed",
                            message=f"openings call failed for marker {cand.marker}; "
                            "CV kind hint kept",
                            ref=cand.marker,
                        )
                    )
                continue
            got_any = True
            merged.candidates.update(part.candidates)
        openings_read = merged if got_any else None

    if executor is not None:
        executor.shutdown(wait=False)

    opening_drafts, element_drafts, op_warnings, op_unresolved = merge_openings(
        candidates, openings_read, work_bgr.shape[:2], rooms, scale_draft, segments
    )
    warnings += op_warnings
    unresolved += op_unresolved
    connected_pairs = {
        frozenset(op.connects)
        for op in opening_drafts
        if _is_walk_through(op.element_type)
        and all(isinstance(side, int) for side in op.connects)
    }
    for zone in find_zone_passages(walls, rooms, segments):
        pair = frozenset(zone.connects)
        zone_type = "sliding_door" if zone.kind_hint == "sliding_door" else "passage"
        along = (0, 2) if zone.axis == "h" else (1, 3)
        recovered = next(
            (
                opening
                for opening in opening_drafts
                if zone_type == "sliding_door"
                and frozenset(opening.connects) == pair
                and opening.axis == zone.axis
                and (
                    opening.element_type == "passage"
                    or opening.element_type == "window"
                    or opening.element_type.endswith("_window")
                )
                and min(opening.bbox[along[1]], zone.bbox[along[1]])
                - max(opening.bbox[along[0]], zone.bbox[along[0]])
                >= 0.5 * min(opening.width_px, zone.width_px)
            ),
            None,
        )
        if recovered is not None:
            recovered.element_type = zone_type
            recovered.wall_index = zone.wall_index
            recovered.confidence = max(recovered.confidence, 0.7)
            opening_drafts = [
                opening
                for opening in opening_drafts
                if opening is recovered
                or frozenset(opening.connects) != pair
                or _is_walk_through(opening.element_type)
                or opening.axis != recovered.axis
                or min(opening.bbox[along[1]], recovered.bbox[along[1]])
                - max(opening.bbox[along[0]], recovered.bbox[along[0]])
                < 0.5 * min(opening.width_px, recovered.width_px)
            ]
        elif pair in connected_pairs:
            continue
        else:
            opening_drafts.append(
                OpeningDraft(
                    marker=zone.marker,
                    element_type=zone_type,
                    raw_text=None,
                    bbox=zone.bbox,
                    center=zone.center,
                    axis=zone.axis,
                    width_px=zone.width_px,
                    wall_index=zone.wall_index,
                    connects=zone.connects,
                    swing=None,
                    hinge=None,
                    source="cv",
                    confidence=0.7,
                )
            )
        connected_pairs.add(pair)
        warnings.append(
            ParseWarning(
                code="open_zone_connection",
                message=(
                    "continuous track across the zone divider retained as a sliding door"
                    if zone_type == "sliding_door"
                    else "dashed functional divider retained as an open passage, not a wall"
                ),
                ref=zone.marker,
            )
        )
        zone_types = {rooms[side].room_type for side in zone.connects if isinstance(side, int)}
        private_zone = zone_types & {"bedroom", "bathroom"}
        expected_private_zone = zone_types == {"bathroom"} or zone_types.issubset(
            {"bedroom", "closet", "study", "storage"}
        )
        if zone_type == "passage" and private_zone and not expected_private_zone:
            warnings.append(
                ParseWarning(
                    code="open_zone_privacy_conflict",
                    message=(
                        "a dashed divider leaves a private room open to circulation; "
                        "physical passage retained, residential privacy should be reviewed"
                    ),
                    ref=zone.marker,
                )
            )
    warnings.append(
        ParseWarning(
            code="vertical_defaults_assumed",
            message=(
                "2D plans contain no vertical dimensions; configured defaults used "
                f"(level {level_height_mm:g}mm, doors {door_height_mm:g}mm, "
                f"window sill/height {window_sill_height_mm:g}/{window_height_mm:g}mm)"
            ),
        )
    )

    if debug_dir:
        debug_mod.save(debug_dir, "01_walls.png", debug_mod.walls_overlay(work_bgr, walls))
        debug_mod.save(
            debug_dir,
            "02_rooms.png",
            debug_mod.polygons_overlay(
                work_bgr,
                [r.polygon for r in rooms],
                [f"{i + 1}:{r.name or '?'}" for i, r in enumerate(rooms)],
            ),
        )
        from roomify.vlm import render_openings_overlay as _roo

        debug_mod.save(debug_dir, "03_openings.png", _roo(work_bgr, candidates))

    plan = _assemble(
        src,
        work_bgr,
        back,
        walls,
        rooms,
        segments,
        opening_drafts,
        element_drafts,
        scale_draft,
        plan_read,
        warnings,
        unresolved,
        level_height_mm,
        door_height_mm,
        window_sill_height_mm,
        window_height_mm,
    )
    if plan.scale is not None:
        from roomify.node_graph import embed_node_graph

        plan = embed_node_graph(plan)
    return plan


def _drop_unlabelled_fixture_regions(
    rooms: list[RoomDraft],
    walls: WallExtraction,
    bgr: np.ndarray,
    *,
    semantic_review: bool = True,
) -> tuple[list[RoomDraft], list[RoomDraft]]:
    """Remove small CV voids bounded by furniture/window-frame ink, not walls."""
    total_area = sum(room.area_px for room in rooms)
    kept: list[RoomDraft] = []
    dropped: list[RoomDraft] = []
    for room in rooms:
        width, height = np.ptp(room.polygon, axis=0)
        share = room.area_px / max(total_area, 1.0)
        wall_support = _boundary_fraction(room, walls.solid)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(room.polygon).astype(np.int32)], 255)
        interior = cv2.erode(mask, np.ones((7, 7), np.uint8))
        pixels = bgr[interior > 0]
        paper = 0.0
        if len(pixels):
            mean = pixels.mean(axis=1)
            chroma = pixels.max(axis=1) - pixels.min(axis=1)
            paper = float(((mean > 245) & (chroma < 10)).mean())
        unsupported = share < 0.02 and wall_support < 0.30
        vetoed_small = room.vlm_vetoed and share < 0.02
        narrow = (
            share < 0.04
            and min(width, height) < 4.0 * walls.thickness_px
            and wall_support < 0.50
        )
        door_pocket = (
            share < 0.04
            and min(width, height) < 4.0 * walls.thickness_px
            and walls.door_hints is not None
            and _boundary_fraction(room, walls.door_hints) >= 0.25
        )
        blank_exterior = paper >= 0.97 and wall_support < 0.50
        if room.printed_area_sqm is None and (
            (semantic_review and room.zone_bounded and room.vlm_vetoed)
            or vetoed_small
            or unsupported
            or narrow
            or door_pocket
            or blank_exterior
        ):
            dropped.append(room)
            continue
        kept.append(room)
    return kept, dropped


def _upgrade_openings_from_support(
    candidates: list,
    support_candidates: list,
    room_index_map: list[int | None],
    wall_thickness: float,
) -> list:
    """Use dropped voids to complete a door span, never to add an opening."""

    def interval(candidate):
        x0, y0, x1, y1 = candidate.bbox
        return (x0, x1, (y0 + y1) / 2) if candidate.axis == "h" else (
            y0,
            y1,
            (x0 + x1) / 2,
        )

    out = list(candidates)
    for support in support_candidates:
        mapped_connects = []
        for side in support.connects:
            if not isinstance(side, int):
                mapped_connects.append(side)
            elif side >= len(room_index_map) or room_index_map[side] is None:
                break
            else:
                mapped_connects.append(room_index_map[side])
        if len(mapped_connects) != 2:
            continue
        support = replace(support, connects=tuple(mapped_connects))
        s_lo, s_hi, s_perp = interval(support)
        for index, candidate in enumerate(out):
            if candidate.axis != support.axis:
                continue
            c_lo, c_hi, c_perp = interval(candidate)
            overlap = min(c_hi, s_hi) - max(c_lo, s_lo)
            if (
                abs(c_perp - s_perp) > 2.0 * wall_thickness
                or overlap < 0.5 * min(c_hi - c_lo, s_hi - s_lo)
            ):
                continue
            stronger = (
                support.kind_hint == "double_door",
                support.arc is not None,
                support.width_px,
            ) > (
                candidate.kind_hint == "double_door",
                candidate.arc is not None,
                candidate.width_px,
            )
            if stronger:
                out[index] = replace(
                    support,
                    marker=candidate.marker,
                    wall_index=candidate.wall_index,
                )
            break
    return out


def _extend_short_opening_hosts(candidates: list, segments: list[WallSegment]):
    """Keep a support-completed opening inside its emitted host wall."""
    out = list(segments)
    for candidate in candidates:
        if candidate.axis not in {"h", "v"} or not 0 <= candidate.wall_index < len(out):
            continue
        segment = out[candidate.wall_index]
        length = math.dist(segment.start, segment.end)
        if candidate.width_px <= length + 3:
            continue
        x0, y0, x1, y1 = candidate.bbox
        if candidate.axis == "h":
            fixed = (segment.start[1] + segment.end[1]) / 2
            lo = min(segment.start[0], segment.end[0], x0)
            hi = max(segment.start[0], segment.end[0], x1)
            out[candidate.wall_index] = replace(
                segment, start=(lo, fixed), end=(hi, fixed)
            )
        else:
            fixed = (segment.start[0] + segment.end[0]) / 2
            lo = min(segment.start[1], segment.end[1], y0)
            hi = max(segment.start[1], segment.end[1], y1)
            out[candidate.wall_index] = replace(
                segment, start=(fixed, lo), end=(fixed, hi)
            )
    return out


def _drop_isolated_fixture_regions(
    rooms: list[RoomDraft], candidates: list, walls: WallExtraction
) -> tuple[list[RoomDraft], list[RoomDraft]]:
    """Drop compact frame/cabinet voids that have no walkable connection."""
    total_area = sum(room.area_px for room in rooms)
    connected = {
        side
        for candidate in candidates
        if (
            candidate.arc is not None
            or candidate.kind_hint != "window"
            or all(isinstance(value, int) for value in candidate.connects)
        )
        for side in candidate.connects
        if isinstance(side, int)
    }
    kept: list[RoomDraft] = []
    dropped: list[RoomDraft] = []
    for index, room in enumerate(rooms):
        width, height = np.ptp(room.polygon, axis=0)
        compact = (
            room.printed_area_sqm is None
            and room.area_px < 0.01 * total_area
            and max(width, height) <= 10.0 * walls.thickness_px
        )
        isolated = index not in connected
        (dropped if isolated and (compact or room.source == "vlm") else kept).append(room)
    return kept, dropped


def _clean_room_drafts(
    rooms: list[RoomDraft], walls: WallExtraction
) -> list[RoomDraft]:
    """Clean output polygons after raw contours have served opening detection."""
    polygons = [
        Polygon(clean_room_ring(room.polygon, walls.thickness_px, walls.solid))
        for room in rooms
    ]
    # ponytail: O(n²) is simpler and bounded by the handful of rooms in a
    # floor plan; use a spatial index only if plans grow to hundreds of rooms.
    for left in range(len(polygons)):
        for right in range(left + 1, len(polygons)):
            if polygons[left].intersection(polygons[right]).area <= 1.0:
                continue
            loser, winner = (
                (left, right)
                if polygons[left].area >= polygons[right].area
                else (right, left)
            )
            difference = polygons[loser].difference(polygons[winner])
            parts = [difference] if isinstance(difference, Polygon) else [
                part
                for part in getattr(difference, "geoms", ())
                if isinstance(part, Polygon)
            ]
            replacement = max(parts, key=lambda part: part.area, default=None)
            if replacement is not None and replacement.area > 0:
                polygons[loser] = replacement

    out: list[RoomDraft] = []
    for room, polygon in zip(rooms, polygons, strict=True):
        if not polygon.is_valid or polygon.area <= 0:
            out.append(room)
            continue
        ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        closed = np.vstack([ring, ring[:1]])
        edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        seed = polygon.representative_point()
        out.append(
            replace(
                room,
                polygon=ring,
                area_px=float(polygon.area),
                perimeter_px=float(edges.sum()),
                edge_lengths_px=[float(edge) for edge in edges],
                seed=(float(seed.x), float(seed.y)),
            )
        )
    return out


def _opening_context(
    candidates: list,
    rooms: list[RoomDraft],
    scale_draft: ScaleDraft | None,
    segments: list[WallSegment] | None = None,
) -> dict[str, str]:
    """Per-marker structural one-liners for the openings prompt: what the
    break connects and its measured width. Adjacency is invisible in a tight
    crop but is the strongest classification prior available."""

    def side_label(side: int | str) -> str:
        # Balconies and AC platforms are OUTDOOR spaces the plan draws inside
        # the footprint. Calling them "interior room" made the strongest prior
        # in the openings prompt ("interior↔interior is a door") both wrong
        # for them and untrustworthy everywhere else — a bedroom's glazed
        # balcony wall really is a window, and a model that has to overrule
        # the hint there stops honouring it at a bathroom door.
        if isinstance(side, int):
            room = rooms[side]
            label = room.name or room.room_type
            if room.room_type in OUTDOOR_ROOM_TYPES:
                return (
                    f"{label} (an OUTDOOR space, not an interior room: it is "
                    "entered through a door — often sliding — and glazed "
                    "elsewhere)"
                )
            return f"{label} (interior room)"
        if side == "exterior":
            return "the exterior"
        return "interior circulation space"

    context: dict[str, str] = {}
    for cand in candidates:
        if scale_draft is not None:
            if (
                cand.axis == "d"
                and segments is not None
                and 0 <= cand.wall_index < len(segments)
            ):
                segment = segments[cand.wall_index]
                per_mm = pixels_per_mm_in_direction(
                    segment.end[0] - segment.start[0],
                    segment.end[1] - segment.start[1],
                    scale_draft,
                )
            else:
                per_mm = (
                    scale_draft.px_per_mm_x
                    if cand.axis == "h"
                    else scale_draft.px_per_mm_y
                )
            width = f"width ≈ {cand.width_px / per_mm:.0f}mm"
        else:
            width = f"width ≈ {cand.width_px:.0f}px"
        context[cand.marker] = (
            f"connects {side_label(cand.connects[0])} ↔ {side_label(cand.connects[1])}, {width}"
        )
    return context


# ------------------------------------------------------------------ assembly


def _assemble(
    src: SourceImage,
    work_bgr: np.ndarray,
    back: np.ndarray | None,
    walls: WallExtraction,
    rooms: list[RoomDraft],
    segments: list[WallSegment],
    opening_drafts: list[OpeningDraft],
    element_drafts: list[ElementDraft],
    scale_draft: ScaleDraft | None,
    plan_read: object | None,
    warnings: list[ParseWarning],
    unresolved: list[Unresolved],
    level_height_mm: float,
    door_height_mm: float,
    window_sill_height_mm: float,
    window_height_mm: float,
) -> FloorPlan:
    f = src.scale_to_original  # working px -> original px
    scale = scale_draft.to_schema(f) if scale_draft else None
    origin = (walls.footprint[0] * f, walls.footprint[1] * f)  # mm origin

    def pt(x: float, y: float) -> SchemaPoint:
        # pixel outputs live in the SOURCE image frame: undo the derotation
        if back is not None:
            x, y = (
                back[0, 0] * x + back[0, 1] * y + back[0, 2],
                back[1, 0] * x + back[1, 1] * y + back[1, 2],
            )
        return SchemaPoint(x=x * f, y=y * f)

    def bbox_px(x0: float, y0: float, x1: float, y1: float) -> BBox:
        # a derotated-frame box maps to a rotated quad; emit its source-frame
        # axis-aligned bounds
        corners = [pt(x0, y0), pt(x1, y0), pt(x0, y1), pt(x1, y1)]
        return BBox(
            x0=min(c.x for c in corners),
            y0=min(c.y for c in corners),
            x1=max(c.x for c in corners),
            y1=max(c.y for c in corners),
        )

    def pt_mm(x: float, y: float) -> SchemaPoint:
        assert scale is not None
        return SchemaPoint(
            x=(x * f - origin[0]) / scale.px_per_mm_x,
            y=(y * f - origin[1]) / scale.px_per_mm_y,
        )

    room_ids: list[str] = [f"room_{i + 1}" for i in range(len(rooms))]
    opening_ids: list[str] = [f"op_{i + 1}" for i in range(len(opening_drafts))]
    ids_by_collection: dict[str, dict[str | None, str]] = {
        "rooms": {draft.marker: room_ids[i] for i, draft in enumerate(rooms)},
        "openings": {draft.marker: opening_ids[i] for i, draft in enumerate(opening_drafts)},
    }
    rooms_by_marker = {draft.marker: draft for draft in rooms}
    marker_to_id = ids_by_collection["rooms"] | ids_by_collection["openings"]
    warnings = [
        warning.model_copy(update={"ref": marker_to_id.get(warning.ref, warning.ref)})
        for warning in warnings
        if warning.code != "room_semantics_missing"
        or (
            warning.ref in rooms_by_marker
            and rooms_by_marker[warning.ref].name is None
        )
    ]
    remapped_unresolved: list[Unresolved] = []
    for item in unresolved:
        parts = item.path.split("/", 2)
        if len(parts) == 3 and parts[0] in ids_by_collection:
            marker = parts[1]
            if marker not in ids_by_collection[parts[0]]:
                continue
            if parts[0] == "rooms" and parts[2] == "name":
                draft = rooms_by_marker[marker]
                if draft.name is not None:
                    continue
            parts[1] = ids_by_collection[parts[0]][marker]
        remapped_unresolved.append(item.model_copy(update={"path": "/".join(parts)}))
    unresolved = remapped_unresolved

    schema_rooms: list[Room] = []
    for i, draft in enumerate(rooms):
        if draft.recovered:
            warnings.append(
                ParseWarning(
                    code="room_recovered",
                    message="room reclaimed from floor space no sealing candidate "
                    "resolved (decor strokes had chopped it); boundary follows "
                    "the surrounding structure",
                    ref=room_ids[i],
                )
            )
        if draft.zone_bounded:
            warnings.append(
                ParseWarning(
                    code="room_zone_boundary",
                    message="room boundary includes a dashed zone divider "
                    "(functional split, not a physical wall)",
                    ref=room_ids[i],
                )
            )
        ring = draft.polygon
        polygon_px = [pt(x, y) for x, y in ring]
        closed = np.vstack([ring, ring[:1]]) * f
        edge_lengths_px = [float(np.linalg.norm(d)) for d in np.diff(closed, axis=0)]

        polygon_mm = area_sqm = perimeter_mm = edge_lengths_mm = None
        deviation = None
        flag = False
        if scale is not None:
            polygon_mm = [pt_mm(x, y) for x, y in ring]
            mm = np.array([[p.x, p.y] for p in polygon_mm])
            closed_mm = np.vstack([mm, mm[:1]])
            edge_lengths_mm = [float(np.linalg.norm(d)) for d in np.diff(closed_mm, axis=0)]
            perimeter_mm = float(sum(edge_lengths_mm))
            area_sqm = float(
                abs(
                    np.sum(
                        mm[:, 0] * np.roll(mm[:, 1], -1) - np.roll(mm[:, 0], -1) * mm[:, 1]
                    )
                )
                / 2
                / 1e6
            )
            if draft.printed_area_sqm:
                if draft.printed_area_sqm < MIN_ROOM_SQM:
                    # 0.01㎡ duct labels state the DUCT's area; the smallest
                    # void CV can resolve is orders of magnitude larger, so a
                    # percentage against such a label is a category error,
                    # not a measurement verdict.
                    warnings.append(
                        ParseWarning(
                            code="printed_area_below_resolution",
                            message=(
                                f"printed area {draft.printed_area_sqm}㎡ is below "
                                "the CV measurement resolution; deviation not "
                                "comparable"
                            ),
                            ref=room_ids[i],
                        )
                    )
                else:
                    deviation = (area_sqm - draft.printed_area_sqm) / draft.printed_area_sqm
                    flag = abs(deviation) > DEVIATION_FLAG_THRESHOLD

        schema_rooms.append(
            Room(
                id=room_ids[i],
                name=draft.name,
                room_type=draft.room_type,  # type: ignore[arg-type]
                polygon_px=polygon_px,
                area_px=draft.area_px * f * f,
                perimeter_px=draft.perimeter_px * f,
                edge_lengths_px=edge_lengths_px,
                polygon_mm=polygon_mm,
                area_sqm=area_sqm,
                perimeter_mm=perimeter_mm,
                edge_lengths_mm=edge_lengths_mm,
                printed_area_sqm=draft.printed_area_sqm,
                area_deviation=deviation,
                area_deviation_flag=flag,
                source=draft.source,  # type: ignore[arg-type]
                confidence=draft.confidence,
            )
        )

    def side_name(side: int | str) -> str:
        if isinstance(side, int):
            return room_ids[side]
        return side  # "exterior" | "unknown"

    schema_walls: list[Wall] = []
    wall_id_of_segment: dict[int, str] = {}
    for si, seg in enumerate(segments):
        start, end = pt(*seg.start), pt(*seg.end)
        if math.isclose(start.x, end.x) and math.isclose(start.y, end.y):
            continue
        wall_id = f"wall_{len(schema_walls) + 1}"
        wall_id_of_segment[si] = wall_id
        thickness_mm = None
        start_mm = end_mm = None
        if scale is not None:
            start_mm, end_mm = pt_mm(*seg.start), pt_mm(*seg.end)
            assert scale_draft is not None
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            thickness_mm = seg.thickness_px / pixels_per_mm_in_direction(
                -dy, dx, scale_draft
            )
        schema_walls.append(
            Wall(
                id=wall_id,
                start_px=start,
                end_px=end,
                thickness_px=seg.thickness_px * f,
                rooms=(side_name(seg.rooms[0]), side_name(seg.rooms[1])),
                start_mm=start_mm,
                end_mm=end_mm,
                thickness_mm=thickness_mm,
                source="cv",
                confidence=0.6,
            )
        )

    schema_openings: list[Opening] = []
    for i, op in enumerate(opening_drafts):
        x0, y0, x1, y1 = op.bbox
        width_mm = None
        if scale is not None:
            assert scale_draft is not None
            if op.axis == "d" and 0 <= op.wall_index < len(segments):
                segment = segments[op.wall_index]
                per_mm = pixels_per_mm_in_direction(
                    segment.end[0] - segment.start[0],
                    segment.end[1] - segment.start[1],
                    scale_draft,
                )
            else:
                per_mm = (
                    scale_draft.px_per_mm_x
                    if op.axis == "h"
                    else scale_draft.px_per_mm_y
                )
            width_mm = op.width_px / per_mm
        is_door = op.element_type == "passage" or op.element_type.endswith("_door")
        is_window = op.element_type == "window" or op.element_type.endswith("_window")
        sill_height_mm = height_mm = None
        if is_door:
            sill_height_mm, height_mm = 0.0, door_height_mm
        elif is_window:
            if op.element_type == "floor_to_ceiling_window":
                sill_height_mm, height_mm = 0.0, level_height_mm
            else:
                sill_height_mm, height_mm = window_sill_height_mm, window_height_mm

        protrusion = None
        if op.element_type == "bay_window" and 0 <= op.wall_index < len(segments):
            protrusion = measure_bay_protrusion(
                segments[op.wall_index], op.bbox, walls, rooms, work_bgr
            )
        protrusion_px = [pt(x, y) for x, y in protrusion] if protrusion else None
        protrusion_mm = (
            [pt_mm(x, y) for x, y in protrusion] if protrusion and scale is not None else None
        )
        if op.element_type == "bay_window" and protrusion is None:
            warnings.append(
                ParseWarning(
                    code="bay_window_geometry_unresolved",
                    message="bay-window facade strokes did not form a measurable protrusion",
                    ref=opening_ids[i],
                )
            )
            unresolved.append(
                Unresolved(
                    path=f"openings/{opening_ids[i]}/protrusion_polygon_px",
                    reason="no closed bay-window facade stroke evidence in the image",
                )
            )
        schema_openings.append(
            Opening(
                id=opening_ids[i],
                element_type=op.element_type,  # type: ignore[arg-type]
                raw_text=op.raw_text,
                bbox_px=bbox_px(x0, y0, x1, y1),
                center_px=pt(*op.center),
                width_px=op.width_px * f,
                width_mm=width_mm,
                sill_height_mm=sill_height_mm,
                height_mm=height_mm,
                wall_id=wall_id_of_segment.get(op.wall_index),
                connects=(side_name(op.connects[0]), side_name(op.connects[1])),
                swing=op.swing,  # type: ignore[arg-type]
                hinge_px=pt(*op.hinge) if op.hinge else None,
                protrusion_polygon_px=protrusion_px,
                protrusion_polygon_mm=protrusion_mm,
                source=op.source,  # type: ignore[arg-type]
                confidence=op.confidence,
            )
        )

    room_polys = [
        (room_ids[i], Polygon(r.polygon)) for i, r in enumerate(rooms) if len(r.polygon) >= 3
    ]
    schema_elements: list[Element] = []
    for i, el in enumerate(element_drafts):
        x0, y0, x1, y1 = el.bbox
        center = Point((x0 + x1) / 2, (y0 + y1) / 2)
        room_id = next((rid for rid, poly in room_polys if poly.contains(center)), None)
        schema_elements.append(
            Element(
                id=f"el_{i + 1}",
                element_type=el.element_type,  # type: ignore[arg-type]
                raw_text=el.raw_text,
                bbox_px=bbox_px(x0, y0, x1, y1),
                room_id=room_id,
                source=el.source,  # type: ignore[arg-type]
                confidence=el.confidence,
            )
        )

    north = getattr(plan_read, "north_angle_deg", None) if plan_read else None
    if north is not None:
        north = float(north)
        if back is not None:
            # the VLM read the arrow in the derotated frame; report in the
            # source frame (derotation turned content by -rotation)
            north += float(np.degrees(np.arctan2(back[1, 0], back[0, 0])))
        north %= 360.0

    warnings.extend(_habitability_warnings(schema_rooms, schema_openings))

    return FloorPlan(
        source_file=src.source_file,
        source_sha256=src.sha256,
        page=src.page,
        image_width_px=src.original_width,
        image_height_px=src.original_height,
        north_angle_deg=north,
        level_height_mm=level_height_mm,
        scale=scale,
        rooms=schema_rooms,
        walls=schema_walls,
        openings=schema_openings,
        elements=schema_elements,
        warnings=warnings,
        unresolved=unresolved,
    )


_CRITICAL_ROOM_TYPES = {
    "living_room",
    "living_dining",
    "dining_room",
    "bedroom",
    "kitchen",
    "bathroom",
    "hallway",
    "entrance",
    "study",
    "multipurpose",
}


def _is_walk_through(element_type: str) -> bool:
    return element_type == "passage" or element_type.endswith("_door")


def _habitability_warnings(rooms: list[Room], openings: list[Opening]) -> list[ParseWarning]:
    """Report topology contradictions without inventing missing doors."""
    room_by_id = {room.id: room for room in rooms}
    indoor = {
        room.id for room in rooms if room.room_type not in OUTDOOR_ROOM_TYPES
    }
    graph: dict[str, set[str]] = {room_id: set() for room_id in indoor}
    for opening in openings:
        if not _is_walk_through(opening.element_type) or opening.connects is None:
            continue
        left, right = opening.connects
        if left in graph and right in graph:
            graph[left].add(right)
            graph[right].add(left)

    components: list[set[str]] = []
    unseen = set(indoor)
    while unseen:
        component: set[str] = set()
        stack = [unseen.pop()]
        while stack:
            room_id = stack.pop()
            component.add(room_id)
            neighbours = graph[room_id] & unseen
            unseen -= neighbours
            stack.extend(neighbours)
        if any(room_by_id[room_id].room_type in _CRITICAL_ROOM_TYPES for room_id in component):
            components.append(component)

    warnings: list[ParseWarning] = []
    if len(components) > 1:
        order = {room.id: i for i, room in enumerate(rooms)}
        groups = [
            ", ".join(
                room_by_id[room_id].name or room_id
                for room_id in sorted(component, key=order.__getitem__)
                if room_by_id[room_id].room_type in _CRITICAL_ROOM_TYPES
            )
            for component in components
        ]
        groups.sort()
        warnings.append(
            ParseWarning(
                code="dwelling_circulation_disconnected",
                message=(
                    f"walk-through openings leave occupied rooms in {len(groups)} "
                    f"disconnected groups: {' | '.join(groups)}"
                ),
            )
        )

    accessible_balconies = {
        side
        for opening in openings
        if _is_walk_through(opening.element_type) and opening.connects is not None
        for side, other in (opening.connects, opening.connects[::-1])
        if side in room_by_id
        and room_by_id[side].room_type == "balcony"
        and other in indoor
    }
    inaccessible = [
        room.name or room.id
        for room in rooms
        if room.room_type == "balcony" and room.id not in accessible_balconies
    ]
    if inaccessible:
        warnings.append(
            ParseWarning(
                code="balcony_access_unresolved",
                message=f"no indoor door or passage was found for: {', '.join(inaccessible)}",
            )
        )

    unknown = [
        opening.id
        for opening in openings
        if opening.connects is not None and "unknown" in opening.connects
    ]
    if unknown:
        warnings.append(
            ParseWarning(
                code="opening_adjacency_unresolved",
                message=f"{len(unknown)} openings have an unresolved indoor side",
                ref=unknown[0],
            )
        )
    return warnings
