"""The parse() orchestrator: load → CV geometry → VLM semantics → FloorPlan.

Geometry never blocks on the VLM: every VLM call that fails (or is disabled
via use_vlm=False) routes to a degradation path that still yields a valid —
if less semantic — document, with the degradation recorded in ``warnings``
and ``unresolved``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

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
    merge_openings,
    merge_rooms,
    reconcile_rooms,
)
from roomify.openings import (
    WallSegment,
    find_openings,
    find_zone_passages,
    measure_bay_protrusion,
)
from roomify.rooms import detect_rooms, uncovered_floor
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
            render_room_overlay,
            room_semantics_prompt,
        )

        overlay = render_room_overlay(work_bgr, detection.rooms)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_room_overlay.png", overlay)
        # 1.5× upscale: printed ㎡ labels sit at the OCR limit on ~700px
        # listing exports; measured to fix small-text misreads at no latency
        # cost (the extra image tokens don't move this model's runtime).
        import cv2

        upscaled = cv2.resize(overlay, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        total_area = sum(r.area_px for r in detection.rooms) or 1.0
        shares = {
            str(i + 1): r.area_px / total_area for i, r in enumerate(detection.rooms)
        }
        room_read = client.call(
            room_semantics_prompt(len(detection.rooms), shares),
            [upscaled],
            RoomRead,
            wire_schema=ROOMS_WIRE,
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
    warnings += outcome.warnings
    unresolved += outcome.unresolved

    plan_read = plan_future.result() if plan_future is not None else None
    if plan_future is not None and plan_read is None:
        warnings.append(
            ParseWarning(code="vlm_call_failed", message="plan-read call failed; "
                         "scale falls back to printed areas, footprint to CV")
        )

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

    checked = apply_area_checks(reconciled, scale_draft)
    warnings += checked.warnings
    rooms = checked.rooms

    door_px = 1000.0 * scale_draft.px_per_mm_x if scale_draft else None
    candidates, segments = find_openings(walls, rooms, work_bgr, door_px)

    openings_read = None
    if client is not None and candidates:
        from roomify.vlm import (
            OPENINGS_WIRE,
            OpeningsRead,
            openings_prompt,
            render_candidate_crop,
            render_openings_overlay,
        )

        overlay = render_openings_overlay(work_bgr, candidates)
        if debug_dir:
            debug_mod.save(debug_dir, "vlm_openings_overlay.png", overlay)
        context = _opening_context(candidates, rooms, scale_draft)
        # Small chunks, in parallel: this endpoint 502s beyond ~6 images per
        # request, and at ~45s per call serial chunks would dominate runtime.
        chunk_size = 5
        chunks = [candidates[s : s + chunk_size] for s in range(0, len(candidates), chunk_size)]
        futures = []
        assert executor is not None
        for idx, chunk in enumerate(chunks):
            crops = [render_candidate_crop(work_bgr, c) for c in chunk]
            futures.append(
                executor.submit(
                    client.call,
                    openings_prompt(
                        [c.marker for c in chunk],
                        include_extras=idx == 0,
                        context=context,
                    ),
                    [overlay, *crops],
                    OpeningsRead,
                    wire_schema=OPENINGS_WIRE,
                )
            )
        merged = OpeningsRead()
        got_any = False
        failed: list[list] = []
        for chunk, future in zip(chunks, futures, strict=True):
            part = future.result()
            if part is None:
                failed.append(chunk)
                continue
            got_any = True
            merged.candidates.update(part.candidates)
            merged.extra_elements.extend(part.extra_elements)
        # Reasoning latency is content-driven: ambiguous crops can stall the
        # model past the upstream's own ~240s kill switch, so retrying a
        # failed chunk at the same size can never succeed. Retry as
        # single-crop requests instead — the smallest possible reasoning
        # load — and accept the degradation only per candidate.
        retry = [c for chunk in failed for c in chunk][:12]  # bound worst-case time
        skipped = [c for chunk in failed for c in chunk][12:]
        for cand in retry:
            part = client.call(
                openings_prompt([cand.marker], include_extras=False, context=context),
                [overlay, render_candidate_crop(work_bgr, cand)],
                OpeningsRead,
                wire_schema=OPENINGS_WIRE,
            )
            if part is None:
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
        for cand in skipped:
            warnings.append(
                ParseWarning(
                    code="vlm_call_failed",
                    message=f"openings retry budget exhausted for marker {cand.marker}; "
                    "CV kind hint kept",
                    ref=cand.marker,
                )
            )
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
        if frozenset(zone.connects) in connected_pairs:
            continue
        opening_drafts.append(
            OpeningDraft(
                marker=zone.marker,
                element_type="passage",
                raw_text=None,
                bbox=zone.bbox,
                center=zone.center,
                axis=zone.axis,
                width_px=zone.width_px,
                wall_index=-1,
                connects=zone.connects,
                swing=None,
                hinge=None,
                source="cv",
                confidence=0.7,
            )
        )
        connected_pairs.add(frozenset(zone.connects))
        warnings.append(
            ParseWarning(
                code="open_zone_connection",
                message="dashed functional divider retained as an open passage, not a wall",
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


def _opening_context(
    candidates: list,
    rooms: list[RoomDraft],
    scale_draft: ScaleDraft | None,
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
            per_mm = (
                scale_draft.px_per_mm_x if cand.axis == "h" else scale_draft.px_per_mm_y
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
        horizontal = abs(end.x - start.x) >= abs(end.y - start.y)
        thickness_mm = None
        start_mm = end_mm = None
        if scale is not None:
            start_mm, end_mm = pt_mm(*seg.start), pt_mm(*seg.end)
            thickness_mm = seg.thickness_px * f / (
                scale.px_per_mm_y if horizontal else scale.px_per_mm_x
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
            width_mm = op.width_px * f / (
                scale.px_per_mm_x if op.axis == "h" else scale.px_per_mm_y
            )
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
